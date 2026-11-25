"""The two fused-instance signals of rules_fused.md and the two split rules, on the CSE crop grid.

    flag_G(inst, comps, u)   geometry: a fluke-like component and a non-fluke component of one instance far apart
    flag_S(inst, sam)        SAM 3: two or more confident whale instances each holding >= 15 % of ours
    split_G / split_S        the pieces a split arm makes of a flagged instance
    arm_instances(D, stem)   an arm's instances of a photo: DAY/parts_hr/split/<stem>.npz when the arm split it, else
                             parts.split_instances as parts.py --instances makes them (keys "<stem>~k" by list index)

    PYTHONPATH=. .venv/bin/python experiments/fused_instances/flags.py --root DAYS --day DAY --out FLAGS.json
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from multiprocessing import Pool
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import fcommon as F  # noqa: E402

G_GAP = 0.20; U_FLUKE = 0.15; U_OTHER = 0.30
S_SHARE = 0.15; S_MIN_PX = 400; MIN_PIECE = 400


def _ext(c):
    ys, xs = np.nonzero(c)
    return max(np.ptp(xs), np.ptp(ys)) + 1


def flag_G(inst: np.ndarray, comps: list, u: np.ndarray):
    """-> list of the fluke-like components that are far from a non-fluke component of the same instance ([] = no flag)."""
    cs = [c for c in comps if (c & inst).any()]
    if len(cs) < 2:
        return []
    med = [float(np.median(u[c])) for c in cs]
    flk = [i for i, m in enumerate(med) if m < U_FLUKE]; oth = [i for i, m in enumerate(med) if m >= U_OTHER]
    out = []
    for i in flk:
        dt = cv2.distanceTransform((~cs[i]).astype(np.uint8), cv2.DIST_L2, 3)
        for j in oth:
            gap = float(dt[cs[j]].min())
            if gap >= G_GAP * max(_ext(cs[i]), _ext(cs[j])):
                out.append(cs[i]); break
    return out


def flag_S(inst: np.ndarray, sam: list | None):
    """-> the SAM 3 instance masks (crop grid) that each hold >= S_SHARE of ours ([] = no flag; None input = no SAM run)."""
    if not sam:
        return []
    hits = F.sam_split(inst, sam, S_SHARE, S_MIN_PX)
    return [m for _, m, _ in hits] if len(hits) >= 2 else []


def split_G(inst, flagged):
    a = inst.copy()
    f = np.zeros_like(inst)
    for c in flagged:
        f |= c
    a &= ~f
    pieces = [p for p in (a, f) if p.sum() >= MIN_PIECE]
    return pieces if len(pieces) == 2 else [inst]


def split_S(inst, sams):
    """our pixels to the SAM instance they fall in, the rest to the nearest one; small pieces back to the nearest."""
    k = len(sams)
    lab = np.full(inst.shape, -1, np.int32)
    for i, m in enumerate(sams):
        lab[inst & m & (lab < 0)] = i
    if (inst & (lab < 0)).any():
        d = np.stack([cv2.distanceTransform((~(sams[i] & inst)).astype(np.uint8), cv2.DIST_L2, 3) for i in range(k)])
        near = d.argmin(0)
        rest = inst & (lab < 0)
        lab[rest] = near[rest]
    while True:
        sizes = [(lab == i).sum() for i in range(k)]
        live = [i for i in range(k) if sizes[i] > 0]
        small = [i for i in live if sizes[i] < MIN_PIECE]
        if not small or len(live) <= 1:
            break
        i = small[0]; others = [j for j in live if j != i]
        d = np.stack([cv2.distanceTransform((lab != j).astype(np.uint8), cv2.DIST_L2, 3) for j in others])
        sel = lab == i
        lab[sel] = np.array(others)[d[:, sel].argmin(0)]
    pieces = [lab == i for i in range(k) if (lab == i).sum() >= MIN_PIECE]
    pieces.sort(key=lambda p: -p.sum())
    return pieces if len(pieces) >= 2 else [inst]


def apply_split(base: list, pieces_of: dict) -> list:
    """base instances, {index: [pieces, largest first]} -> new list: the largest piece keeps its index, the rest appended."""
    out = list(base); extra = []
    for i, ps in pieces_of.items():
        out[i] = ps[0]; extra += ps[1:]
    return out + extra


def arm_instances(day_dir: Path, stem: str, z=None):
    from model_fine_tuning.cluster.parts import split_instances
    if z is None:
        z = F.cse(day_dir, stem)
    if z is None or z["use"].sum() < 400:
        return [], z
    f = day_dir / "parts_hr" / "split" / f"{stem}.npz"
    if f.exists():
        L = np.load(f)["lab"]
        return [L == (i + 1) for i in range(int(L.max()))], z
    return split_instances(z["use"], 0.1, u=z["u"]), z


def photo_flags(day: str, day_dir: Path, stem: str) -> dict:
    insts, z = arm_instances(day_dir, stem)
    if not insts:
        return {}
    comps = F.components(z["use"], 0.1, 400)
    sam = F.sam_in_crop(day, stem, z["x0y0s"], z["use"].shape)
    out = {}
    for k, m in enumerate(insts):
        out[f"{stem}~{k}"] = {"G": bool(flag_G(m, comps, z["u"])), "S": None if sam is None else bool(flag_S(m, sam))}
    return out


def _pf(a):
    return photo_flags(*a)


def day_flags(root: Path, day: str, workers: int = 8) -> dict:
    F.check_day(day)
    D = root / day
    t = time.time(); res = {}
    with Pool(workers) as p:
        for r in p.imap(_pf, [(day, D, s) for s in F.stems(D)], chunksize=8):
            res.update(r)
    return {"flags": res, "seconds": round(time.time() - t, 1), "photos": len(F.stems(D))}


def main() -> None:
    ap = argparse.ArgumentParser(); ap.add_argument("--root", type=Path, required=True); ap.add_argument("--day", required=True); ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args()
    r = day_flags(a.root, a.day)
    a.out.parent.mkdir(parents=True, exist_ok=True); a.out.write_text(json.dumps(r))
    fl = r["flags"]
    print(a.day, len(fl), "instances; G", sum(v["G"] for v in fl.values()), "S", sum(bool(v["S"]) for v in fl.values()), "no SAM", sum(v["S"] is None for v in fl.values()))


if __name__ == "__main__":
    main()
