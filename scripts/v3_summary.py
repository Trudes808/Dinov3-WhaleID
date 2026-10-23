"""One table over all day-clustering runs: per-day animal ARI, eval / val means, group ARI (eval), oracle means.

    .venv/bin/python scripts/v3_summary.py [--runs label=dir_or_tag ...]

Old pipeline runs live in experiments/days_all/<day>/cluster_bursts_<name>/metrics.json, v3 runs in
experiments/days_all/<day>/cluster_v3/metrics<tag>.json.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

EVAL = ("2024-12-18", "2024-12-26", "2025-02-19", "2025-03-21"); VAL = ("2025-01-15", "2025-02-27", "2025-03-14")
DEFAULT = [
    ("old: parts MegaDescriptor", "cluster_bursts_md"), ("old: whole-crop MegaDescriptor", "cluster_bursts_whole"),
    ("old: id_v2 parts", "cluster_bursts_id_v2"), ("old: id_v3 parts", "cluster_bursts_id_v3"),
    ("v3: id_v3, bursts", "_v3id"), ("v3: id_v4, bursts", "_v4id"), ("v3: id_v4, tracklets+fin CL", "_v4id_t"),
    ("v3: id_v2+id_v3 mean z", "_ens_b"), ("v3: fusion_B (learned)", "_fB_b"), ("v3: fusion_C (learned, no ind.)", "_fC_b"),
    ("v3: hi-res DINO global", "_hr"), ("v3: id_v4 + hi-res DINO", "_v4hr"), ("v3: distilled student (pairnet)", "_student"),
]


def load(root: Path, day: str, run: str) -> dict | None:
    f = root / day / run / "metrics.json" if run.startswith("cluster_bursts") else root / day / "cluster_v3" / f"metrics{run}.json"
    return json.loads(f.read_text()) if f.exists() else None


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--root", type=Path, default=Path("experiments/days_all")); p.add_argument("--runs", nargs="*", default=None)
    a = p.parse_args(argv)
    runs = [tuple(r.split("=", 1)) for r in a.runs] if a.runs else DEFAULT
    days = EVAL + VAL
    print(f"{'method':34s} " + " ".join(f"{d[5:]:>6s}" for d in days) + f" | {'eval':>6s} {'val':>6s} | {'orEval':>6s} {'grpEv':>6s}")
    for label, run in runs:
        ms = {d: load(a.root, d, run) for d in days}
        if not any(ms.values()):
            continue
        ari = {d: (m["animals"]["ari"] if m and m.get("animals") else np.nan) for d, m in ms.items()}
        orc = [max(s["ari"] for s in ms[d]["oracle_sweep_animals"]) for d in EVAL if ms[d]]
        grp = [ms[d]["groups"]["ari"] for d in EVAL if ms[d] and ms[d].get("groups")]
        ev = np.nanmean([ari[d] for d in EVAL]); va = np.nanmean([ari[d] for d in VAL])
        print(f"{label:34s} " + " ".join(f"{ari[d]:6.3f}" for d in days) + f" | {ev:6.3f} {va:6.3f} | {np.mean(orc):6.3f} {np.mean(grp):6.3f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
