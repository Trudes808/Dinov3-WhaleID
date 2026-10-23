"""Bursts: runs of consecutive shots from one camera that are almost surely the same animal (the photographer holds
the shutter on one whale). A burst = photos of the same camera folder ordered by capture time with gaps <= `gap`
seconds. These are the natural small clusters that the day clustering starts from.

    python -m model_fine_tuning.cluster.bursts --day-dir experiments/days_all/2024-12-18 [--labels labels.json] [--gap 2.0] [--out bursts.json]

With labels: the purity of the bursts against the animal truth (how often a burst holds one animal), their size
distribution, and the share of labelled photos covered by bursts of two or more.
"""
from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np


def make_bursts(timestamps: dict, gap: float = 2.0) -> list[dict]:
    """[{"id", "camera", "photos": [stems in time order], "start", "end"}]"""
    by_cam = defaultdict(list)
    for stem, t in timestamps.items():
        if t.get("time") is None:
            continue
        by_cam[(t.get("folder", "?"), t.get("camera", "?"))].append((float(t["time"]), stem))
    bursts = []
    for cam, items in by_cam.items():
        items.sort()
        cur = [items[0]]
        for prev, nxt in zip(items, items[1:]):
            if nxt[0] - prev[0] <= gap:
                cur.append(nxt)
            else:
                bursts.append(cur); cur = [nxt]
        bursts.append(cur)
    out = []
    for k, b in enumerate(sorted(bursts, key=lambda b: b[0][0])):
        out.append({"id": k, "camera": "|".join(str(c) for c in cam), "photos": [s for _, s in b], "start": b[0][0], "end": b[-1][0]})
    return out


def burst_report(bursts: list[dict], labels: dict | None) -> dict:
    sizes = np.array([len(b["photos"]) for b in bursts])
    rep = {"bursts": len(bursts), "photos": int(sizes.sum()), "size_median": float(np.median(sizes)), "size_p90": float(np.percentile(sizes, 90)),
           "singletons": int((sizes == 1).sum()), "photos_in_multi_bursts": int(sizes[sizes >= 2].sum())}
    if labels:
        pure = mixed = 0; covered = 0; n_lab = 0
        for b in bursts:
            ids = [labels.get(s.split("__")[-1]) for s in b["photos"]]; ids = [i for i in ids if i]
            if len(ids) >= 2:
                covered += len(ids); (pure, mixed) = (pure + 1, mixed) if len(set(ids)) == 1 else (pure, mixed + 1)
            n_lab += len(ids)
        rep.update({"multi_bursts_pure": pure, "multi_bursts_mixed": mixed, "burst_purity": pure / max(pure + mixed, 1), "labelled_photos": n_lab,
                    "labelled_photos_in_multi_bursts": covered})
    return rep


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--day-dir", type=Path, required=True); p.add_argument("--labels", type=Path, default=None); p.add_argument("--gap", type=float, default=2.0)
    p.add_argument("--out", type=Path, default=None)
    a = p.parse_args(argv)
    ts = json.loads((a.day_dir / "timestamps.json").read_text())
    bursts = make_bursts(ts, a.gap)
    labels = json.loads(a.labels.read_text())["days"].get(a.day_dir.name, {}).get("photos") if a.labels else None
    rep = burst_report(bursts, labels)
    print(json.dumps(rep, indent=1))
    if a.out:
        a.out.write_text(json.dumps({"gap_s": a.gap, "report": rep, "bursts": bursts}, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
