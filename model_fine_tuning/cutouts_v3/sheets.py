"""By-eye sheets of the v3 cut-outs (written to a folder OUTSIDE the repository: photos may show crew members).

    PYTHONPATH=. .venv/bin/python -m model_fine_tuning.cutouts_v3.sheets compare OUT.jpg --days train --n 48 [--seed 0]
        old cut-out (parts_up) next to the v3 variants: flukes (old | v3) and fins (A | B | C), half and half
    PYTHONPATH=. .venv/bin/python -m model_fine_tuning.cutouts_v3.sheets stacks OUT.jpg --days val --n 12
        the fluke cut-outs of n stacks with at least 3 flukes: old row, new row
    PYTHONPATH=. .venv/bin/python -m model_fine_tuning.cutouts_v3.sheets fins OUT.jpg --queries Q.json --n 24
        fins whose closest look-alike changes from right to wrong or back between A and B / C (score.py --dump)
"""
from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import cv2
import numpy as np

from . import CACHE, DAYS, day_list

T = 200


def tile(path: Path | None, label: str = "", color=(255, 255, 255)) -> np.ndarray:
    im = cv2.imread(str(path)) if path is not None and Path(path).exists() else None
    t = np.full((T, T, 3), 40, np.uint8) if im is None else cv2.resize(im, (T, T), interpolation=cv2.INTER_AREA)
    if label:
        cv2.putText(t, label, (4, T - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (0, 0, 0), 3, cv2.LINE_AA)
        cv2.putText(t, label, (4, T - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.42, color, 1, cv2.LINE_AA)
    return t


def up(day, key, part):
    return DAYS / day / "parts_up" / "charts" / f"{key}__{part}.jpg"


def meta(day):
    f = CACHE / day / "meta.jsonl"
    return [json.loads(l) for l in open(f)] if f.exists() else []


def grid(rows: list[list[np.ndarray]], cols_per_row: int = 1, gap: int = 6) -> np.ndarray:
    w = max(len(r) for r in rows)
    rr = [np.hstack(r + [np.zeros((T, T, 3), np.uint8)] * (w - len(r))) for r in rows]
    blocks = []
    for i in range(0, len(rr), cols_per_row):
        chunk = rr[i:i + cols_per_row]
        chunk += [np.zeros_like(rr[0])] * (cols_per_row - len(chunk))
        blocks.append(np.hstack([np.pad(c, ((gap, 0), (gap, 0), (0, 0))) for c in chunk]))
    return np.vstack(blocks)


def cmd_compare(a):
    rng = random.Random(a.seed); flukes, fins = [], []
    for day in day_list(a.days):
        for r in meta(day):
            (flukes if r["variant"] == "fluke" else fins).append((day, r))
    nf = a.n // 2
    pick_f = rng.sample(flukes, min(nf, len(flukes)))
    # fins: over-sample the fallbacks so that each kind is seen
    heads = [x for x in fins if x[1]["src"] == "head"]; As = [x for x in fins if x[1]["src"] == "A"]; regs = [x for x in fins if x[1]["src"] == "fin"]
    pick_g = rng.sample(heads, min(5, len(heads))) + rng.sample(As, min(3, len(As)))
    pick_g += rng.sample(regs, a.n - nf - len(pick_g))
    rows = []
    for day, r in pick_f:
        k = r["key"]; lab = f"{r['pixels'][0]}/{r['stock'][:2]}" + (f" j{r['joined']}" if r["joined"] else "")
        rows.append([tile(up(day, k, "fluke"), "old"), tile(CACHE / day / "fluke" / f"{k}.jpg", "v3 " + lab, (120, 255, 120))])
    for day, r in pick_g:
        k = r["key"]; s = r["src"] + (" cap" if r.get("capped") else "")
        rows.append([tile(up(day, k, "fin"), "A"), tile(CACHE / day / "finB" / f"{k}.jpg", "B " + s, (120, 255, 120)), tile(CACHE / day / "finC" / f"{k}.jpg", "C " + s, (120, 255, 120))])
    parts = [grid(rows[:nf], 4)] + ([grid(rows[nf:], 3)] if rows[nf:] else [])
    w = max(x.shape[1] for x in parts)
    sheet = np.vstack([np.pad(x, ((0, 12), (0, w - x.shape[1]), (0, 0))) for x in parts])
    cv2.imwrite(str(a.out), sheet, [cv2.IMWRITE_JPEG_QUALITY, 85]); print(a.out, sheet.shape)


def cmd_stacks(a):
    from .points import load as load_points
    rng = random.Random(a.seed); cand = []
    for day in day_list(a.days):
        pts = load_points(day)["fluke"]; by = {}
        for k, (st, cat) in pts.items():
            if (CACHE / day / "fluke" / f"{k}.jpg").exists():
                by.setdefault(st, []).append(k)
        cand += [(day, st, sorted(ks)) for st, ks in by.items() if len(ks) >= 3]
    pick = rng.sample(cand, min(a.n, len(cand)))
    rows = []
    for day, st, ks in pick:
        ks = ks[:6]
        rows.append([tile(up(day, k, "fluke"), f"old {day[5:]} s{st}") for k in ks])
        rows.append([tile(CACHE / day / "fluke" / f"{k}.jpg", "v3", (120, 255, 120)) for k in ks])
    sheet = grid(rows, 1)
    cv2.imwrite(str(a.out), sheet, [cv2.IMWRITE_JPEG_QUALITY, 85]); print(a.out, sheet.shape)


def cmd_fins(a):
    raw = json.loads(Path(a.queries).read_text()); q = []
    for x in raw:                                       # score.py's form: per query nn / hit / set per arm -> one row per change against G
        for v in a.variants.split(","):
            if v in x["hit"] and x["hit"]["G"] != x["hit"][v] and x["set"].get(v, "up") != "up":
                q.append(dict(day=x["day"], key=x["key"], variant=v, set=x["set"][v], change="found" if x["hit"][v] else "lost",
                              nn_base=x["nn"]["G"], nn_var=x["nn"][v], hit_base=x["hit"]["G"], hit_var=x["hit"][v]))
    print(f"{len(q)} changes: found {sum(r['change'] == 'found' for r in q)}, lost {sum(r['change'] == 'lost' for r in q)}")
    rng = random.Random(a.seed); found = [x for x in q if x["change"] == "found"]; lost = [x for x in q if x["change"] == "lost"]
    pick = rng.sample(found, min(a.n // 2, len(found))) + rng.sample(lost, min(a.n - min(a.n // 2, len(found)), len(lost)))
    rows = []
    for x in pick:
        day, k, v = x["day"], x["key"], x["variant"]; f = CACHE / day / x["set"] / f"{k}.jpg"
        lab = f"{x['change']} {v}"
        col = (120, 255, 120) if x["change"] == "found" else (120, 120, 255)
        nn_a, nn_v = x["nn_base"], x["nn_var"]
        rows.append([tile(up(day, k, "fin"), "A query"), tile(up(day, nn_a, "fin"), "A: closest " + ("ok" if x["hit_base"] else "wrong")),
                     tile(f, lab, col), tile(CACHE / day / f.parent.name / f"{nn_v}.jpg" if (CACHE / day / f.parent.name / f"{nn_v}.jpg").exists() else up(day, nn_v, "fin"),
                                             v + ": closest " + ("ok" if x["hit_var"] else "wrong"))])
    sheet = grid(rows, 2)
    cv2.imwrite(str(a.out), sheet, [cv2.IMWRITE_JPEG_QUALITY, 85]); print(a.out, sheet.shape)


def main(argv=None):
    ap = argparse.ArgumentParser(); ap.add_argument("cmd", choices=["compare", "stacks", "fins"]); ap.add_argument("out", type=Path)
    ap.add_argument("--days", default="train"); ap.add_argument("--n", type=int, default=48); ap.add_argument("--seed", type=int, default=0); ap.add_argument("--queries", default="")
    ap.add_argument("--variants", default="G.B,G.C", help="fins: the arms compared with G (score.py arm names)")
    a = ap.parse_args(argv)
    if "Dinov3-WhaleID" in str(a.out.resolve()):
        raise SystemExit("sheets never go into the repository")
    {"compare": cmd_compare, "stacks": cmd_stacks, "fins": cmd_fins}[a.cmd](a)


if __name__ == "__main__":
    main()
