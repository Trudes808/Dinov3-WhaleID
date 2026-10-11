"""Train / val / test split of a Whale-Image-Data-Factory run, stratified by pose category.

    python -m model_fine_tuning.common.splits make --run-dir RUN --out model_fine_tuning/splits/run01.json

Every shot is placed in exactly one split (negatives and empty-mask shots included; the
body-coordinate model filters by visible pixels at load time). The JSON records the run,
seed and fractions so the split is reproducible and auditable.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

from . import paths


def load_manifest(run_dir: Path) -> list[dict]:
    with open(run_dir / "dataset" / "manifest.jsonl") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def make_split(rows: list[dict], fractions=(0.7, 0.15, 0.15), seed: int = 0, strata=("category",)) -> dict:
    rng = np.random.default_rng(seed)
    groups: dict[tuple, list[str]] = defaultdict(list)
    for row in rows:
        groups[tuple(row[k] for k in strata)].append(row["id"])
    out = {"train": [], "val": [], "test": []}
    for key in sorted(groups):
        ids = sorted(groups[key])
        rng.shuffle(ids)
        n = len(ids)
        n_val = int(round(fractions[1] * n))
        n_test = int(round(fractions[2] * n))
        out["test"].extend(ids[:n_test])
        out["val"].extend(ids[n_test:n_test + n_val])
        out["train"].extend(ids[n_test + n_val:])
    return {k: sorted(v) for k, v in out.items()}


DEFECT_FLAGS = {"empty_mask", "visible_outside_amodal"}


def filter_rows(rows: list[dict], min_visible_px: int = 0, drop_flagged: bool = False) -> list[dict]:
    """Keep negatives (no whale by design) and whale shots with at least min_visible_px visible
    pixels; optionally drop shots the factory flagged as defective (empty mask, label outside
    the projection). mask_touches_border is not a defect (a close whale) and is kept."""
    out = []
    for row in rows:
        if drop_flagged and (set(row.get("qa_flags") or []) & DEFECT_FLAGS):
            continue
        if not row["negative"] and row["visible_pixels"] < min_visible_px:
            continue
        out.append(row)
    return out


def summarize(rows: list[dict], split: dict) -> dict:
    by_id = {r["id"]: r for r in rows}
    summary = {}
    for name, ids in split.items():
        sub = [by_id[i] for i in ids]
        summary[name] = {
            "shots": len(sub),
            "negatives": sum(r["negative"] for r in sub),
            "with_whale_pixels": sum(r["visible_pixels"] > 0 for r in sub),
            "categories": dict(sorted(Counter(r["category"] for r in sub).items())),
            "sea_states": dict(sorted(Counter(r["sea_state"] for r in sub).items())),
        }
    return summary


def load_split(path: Path) -> dict:
    return json.loads(Path(path).read_text())


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="cmd", required=True)
    mk = sub.add_parser("make")
    mk.add_argument("--run-dir", type=Path, default=paths.WHALE_RUN_DIR)
    mk.add_argument("--out", type=Path, default=paths.SPLITS_DIR / "run01.json")
    mk.add_argument("--train", type=float, default=0.7)
    mk.add_argument("--val", type=float, default=0.15)
    mk.add_argument("--test", type=float, default=0.15)
    mk.add_argument("--seed", type=int, default=0)
    mk.add_argument("--min-visible-px", type=int, default=0,
                    help="drop whale shots (not negatives) with fewer visible-whale pixels than this")
    mk.add_argument("--drop-flagged", action="store_true", help="drop shots with factory QA flags")
    args = parser.parse_args(argv)
    rows = load_manifest(args.run_dir)
    before = len(rows)
    rows = filter_rows(rows, args.min_visible_px, args.drop_flagged)
    split = make_split(rows, (args.train, args.val, args.test), args.seed)
    summary = summarize(rows, split)
    summary["filtered_out"] = {"shots": before - len(rows), "min_visible_px": args.min_visible_px, "drop_flagged": args.drop_flagged}
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps({"run_dir": str(args.run_dir), "seed": args.seed, "fractions": [args.train, args.val, args.test],
                                    "strata": ["category"], "summary": summary, **split}, indent=1))
    print(json.dumps(summary, indent=1))
    print("wrote", args.out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
