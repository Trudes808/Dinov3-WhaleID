"""Fins on finless backs: "other" views (by eye O, or the classifier's confident other) on the validation days that a
model paints a dorsal-fin region on, and that its three-way rule (re-chosen on the training days, compare.py) calls a
dorsal view. Paired bootstrap over photos against the first model.

    PYTHONPATH=. .venv/bin/python experiments/parts_weak/finless.py --models parts_seg_r05_a ... --out OUT.json
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import compare as CP  # noqa: E402
from model_fine_tuning.common import paths  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="+", required=True)
    ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args()
    sp = json.loads((paths.SPLITS_DIR / "days.json").read_text())
    days = sp["train"] + sp["val"]
    rows = {m: CP.rows_of(m, days) for m in a.models}
    res = {}
    for var in ("hr", "full"):
        res[var] = {}
        outc = {}
        for m in a.models:
            rule = CP.choose(rows[m], var, "two_part")
            o = {}
            for truth in ("t_eye", "t_vc"):
                sel = [r for r in rows[m] if r["split"] == "val" and r.get(truth) == "other" and var in r["pred"]]
                region = {r["key"]: (r["stem"], int("fin" in r["pred"][var])) for r in sel}
                called = {r["key"]: (r["stem"], int(CP.ER.rule_three(r["pred"][var], *rule) == "dorsal")) for r in sel}
                o[truth] = {"n": len(sel), "with_fin_region": round(sum(v for _, v in region.values()) / max(len(sel), 1), 3),
                            "called_dorsal": round(sum(v for _, v in called.values()) / max(len(sel), 1), 3)}
                outc[(m, truth, "region")] = region
                outc[(m, truth, "called")] = called
            res[var][m] = o
        b = a.models[0]
        res[var]["paired_vs_" + b] = {m: {f"{t}_{k}": CP.paired(outc[(b, t, k)], outc[(m, t, k)]) for t in ("t_eye", "t_vc") for k in ("region", "called")}
                                      for m in a.models[1:]}
    a.out.write_text(json.dumps(res, indent=1))
    for var in ("hr", "full"):
        for m in a.models:
            print(var, f"{m:26s}", res[var][m])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
