"""Counts of how the v3 cut-outs were made, and the fluke mirror / turn consistency inside stacks, from the saved
transforms (no catalogue).

    PYTHONPATH=. .venv/bin/python -m model_fine_tuning.cutouts_v3.stats --days val [--json OUT]

Mirror state of a cut-out = sign of the determinant of its whole transform (old fluke: M then the vertical flip of
parts.py when flipped_v; v3: M, never flipped). Within the stacks (safe clusters, CACHE/points) with >= 3 fluke cut-outs
the share mirrored against the stack's majority; for v3 also the share turned against the majority (the chart's x
axis more than 90 deg from the stack's circular mean in the photo: the stock-up choice went the other way).
"""
import argparse, json
from collections import Counter
from pathlib import Path

import numpy as np

from . import CACHE, DAYS, day_list
from .points import load as load_points


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--days", default="val"); ap.add_argument("--json", type=Path, default=None)
    a = ap.parse_args(); out = {}
    fl, fi = [], []
    mir = {"old": [0, 0], "v3": [0, 0]}; turn = [0, 0]; stacks_n = 0; queries_only = {"old": [0, 0], "v3": [0, 0]}
    for day in day_list(a.days):
        rows = [json.loads(l) for l in open(CACHE / day / "meta.jsonl")]
        fl += [r for r in rows if r["variant"] == "fluke"]; fi += [r for r in rows if r["variant"] == "fin"]
        pts = load_points(day)["fluke"]; by = {}
        for k, (st, cat) in pts.items():
            by.setdefault(st, []).append(k)
        for st, ks in by.items():
            old, new, ang = [], [], []
            for k in ks:
                fo = DAYS / day / "parts_up" / f"{k}__fluke.npz"; fn = CACHE / day / "fluke" / f"{k}.npz"
                if not fo.exists() or not fn.exists():
                    continue
                zo = np.load(fo); Mo = zo["M"]
                old.append(np.sign(np.linalg.det(Mo[:, :2])) * (-1 if bool(zo["flipped_v"]) else 1))
                Mn = np.load(fn)["M"]; new.append(np.sign(np.linalg.det(Mn[:, :2])))
                ex = Mn[0, :2] / np.hypot(*Mn[0, :2]); ang.append(np.arctan2(ex[1], ex[0]))
            if len(old) < 3:
                continue
            stacks_n += 1
            for name, v in (("old", np.array(old)), ("v3", np.array(new))):
                maj = 1 if (v > 0).sum() >= (v < 0).sum() else -1
                mir[name][0] += int((v != maj).sum()); mir[name][1] += len(v)
            ang = np.array(ang); m = np.arctan2(np.sin(ang).sum(), np.cos(ang).sum())
            turn[0] += int((np.cos(ang - m) < 0).sum()); turn[1] += len(ang)
    out["flukes"] = dict(n=len(fl), pixels=dict(Counter(r["pixels"] for r in fl)), stock=dict(Counter(r["stock"] for r in fl)),
                         stock_partmap_agrees_photo=dict(Counter(str(r.get("stock_agree_photo")) for r in fl if r["stock"] == "partmap")),
                         joined_keys=dict(Counter(str(r["joined"]) for r in fl)), joined_cutouts=len({tuple(r["group"]) for r in fl if r["joined"]}),
                         det_positive=float(np.mean([r["det"] > 0 for r in fl])) if fl else None,
                         stacks_ge3=stacks_n, mirrored_vs_stack_majority={k: round(v[0] / max(v[1], 1), 4) for k, v in mir.items()}, n_in_stacks=mir["v3"][1],
                         v3_turned_vs_stack_majority=round(turn[0] / max(turn[1], 1), 4))
    out["fins"] = dict(n=len(fi), src=dict(Counter(r["src"] for r in fi)), capped=int(sum(bool(r.get("capped")) for r in fi)),
                       head_checked_rejected=int(sum(r["src"] == "A" and r.get("head_check") is not None for r in fi)),
                       no_head_patch=int(sum(r["src"] == "A" and r.get("head_check") is None for r in fi)))
    print(json.dumps(out, indent=1))
    if a.json:
        a.json.write_text(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
