"""Per-day tracklet-pair link AUC of every signature variant (experiments/days_all/<day>/signature*/summary.json)."""
import json, sys
from pathlib import Path
root = Path("experiments/days_all"); days = ["2024-12-18", "2024-12-26", "2025-02-19", "2025-03-21", "2025-01-15", "2025-02-27", "2025-03-14"]
variants = sys.argv[1:] or ["signature", "signature_local_v1", "signature_local_v2"]
keys = [("hr_global", "global"), ("signature", "sig"), ("dense", "dense"), ("combined", "rank"), ("shape_fin", "fin-shape"), ("shape_fluke", "fluke-shape")]
for v in variants:
    print(f"== {v}"); print(f"{'day':11s}" + "".join(f"{k:>12s}" for _, k in keys))
    for d in days:
        f = root / d / v / "summary.json"
        if not f.exists():
            print(f"{d:11s}  missing"); continue
        m = json.loads(f.read_text())["metrics"]
        print(f"{d:11s}" + "".join(f"{m.get('tracklet_pair_auc_' + k, float('nan')):12.3f}" for k, _ in keys))
