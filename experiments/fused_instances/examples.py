"""fused_examples.jpg: two-animal instances (by-eye label T in FI/eye_labels.json), per instance three tiles: the photo,
our shadow instance (red, other instances yellow), and what the candidates do with it (split pieces filled: A_G blue /
green, A_S as separate tiles if asked; flag letters in the caption). Output to a scratch / untracked folder only.

    PYTHONPATH=. .venv/bin/python experiments/fused_instances/examples.py --out SHEET.jpg [--n 20] [--mode G|S|both] [--keys TSV]
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import fcommon as F  # noqa: E402
import split_arm as SA  # noqa: E402
import tiles as T  # noqa: E402


def full(m, x0y0s, hw):
    x0, y0, s = x0y0s
    Minv = np.array([[1 / s, 0, x0], [0, 1 / s, y0]], np.float64)
    return cv2.warpAffine(m.astype(np.uint8), Minv, (hw[1], hw[0]), flags=cv2.INTER_NEAREST) > 0


def pieces(day, key, mode, root=F.SR):
    stem, k = key.rsplit("~", 1); k = int(k)
    base, new = SA.new_instances(day, root / day, stem, mode)
    if new is None or base is None:
        return None
    z = F.cse(root / day, stem)
    img = cv2.imread(str(root / day / "images" / f"{stem}.jpg")); hw = img.shape[:2]
    ps = [new[k]] + [m for m in new[len(base):] if (m & base[k]).any()]
    return [full(p, z["x0y0s"], hw) for p in ps] if len(ps) > 1 else None


def main() -> None:
    ap = argparse.ArgumentParser(); ap.add_argument("--out", type=Path, required=True); ap.add_argument("--n", type=int, default=20)
    ap.add_argument("--mode", default="both"); ap.add_argument("--keys", type=Path, default=None, help="tsv day <tab> key (else the first n T labels)")
    ap.add_argument("--root", type=Path, default=F.SR); ap.add_argument("--flags", default="", help="caption flag source: FI/flags/<name>")
    ap.add_argument("--title", default="two-animal instances (by eye): photo + ours | candidate")
    ap.add_argument("--guard", action="store_true", help="captions for the guard B_G (with --mode G --flags P)")
    a = ap.parse_args()
    if a.keys:
        rows = [l.rstrip("\n").split("\t")[:2] for l in open(a.keys)]
    else:
        E = json.load(open(F.FI / "eye_labels.json"))
        rows = [(r["day"], r["key"]) for r in E if r["eye"] == "T"][: a.n]
    tiles = []
    for day, key in rows:
        F.check_day(day)
        seq = key.split("__")[-1][-12:]
        fl = ""
        if a.flags:
            ff = F.FI / "flags" / a.flags / f"{day}.json"
            if ff.exists():
                v = json.loads(ff.read_text())["flags"].get(key, {})
                fl = ("G" if v.get("G") else "") + ("S" if v.get("S") else "")
        tiles.append(T.tile(day, key, f"{day[5:]} {seq}\nphoto + our instance (red)", a.root, 300, sam=False, pad=1.2))
        modes = ["G", "S"] if a.mode == "both" else [a.mode]
        for m in modes:
            ps = pieces(day, key, m, a.root)
            if a.guard:
                cap = "guard B_G: " + ("FLAGGED, no auto link" if ("G" in fl or (m == "G" and ps)) else "not flagged") + "\nfilled: the components flag_G sees"
            else:
                cap = f"split {m}: " + (f"{len(ps)} pieces" if ps else "no split") + (f"\nflags {fl or '-'}" if a.flags else "\nthin lines: SAM 3")
            tiles.append(T.tile(day, key, cap, a.root, 300, pieces=ps, sam=(m == "S"), pad=1.2))
    T.sheet(tiles, (4 if a.mode != "both" else 6), a.out, a.title)


if __name__ == "__main__":
    main()
