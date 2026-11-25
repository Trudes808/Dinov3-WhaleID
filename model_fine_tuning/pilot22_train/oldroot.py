"""Score every model on the day root W3e was predicted and scored on (2026-11-21), not on the live root.

Why: experiments/days_all is since 2026-11-22 22:16 a link to the new mask root (moonshot_cache/switch_root/days: S6h
mask, other instances). The parts_weak shadow (/var/tmp/sat3737-parts-cache/parts_weak/days_shadow) holds W3e's class
maps (made on the OLD instances) next to links named experiments/days_all/<day>/<child>, which now resolve into the
NEW root: scoring W3e there mixes old predictions with new instance indexes, and a new model predicted there would
see other instances than W3e did. Here every reader is pointed at one root made of links to the OLD root
(/var/tmp/sat3737-whale-days, kept by SWITCH.md while the new root is in use), W3e's existing class maps (linked, not
copied) and the new arms' class maps (predicted here). The scripts themselves run unchanged; only their module-level
root constants are set before their main() runs.

    .venv/bin/python -m model_fine_tuning.pilot22_train.oldroot make
    flock LOCK .venv/bin/python -m model_fine_tuning.pilot22_train.oldroot predict --name NAME --checkpoint CK --metrics OUT.json
    .venv/bin/python -m model_fine_tuning.pilot22_train.oldroot compare  <compare.py arguments>
    .venv/bin/python -m model_fine_tuning.pilot22_train.oldroot finless  <finless.py arguments>
    .venv/bin/python -m model_fine_tuning.pilot22_train.oldroot predfill <pred_fill.py score arguments>
    .venv/bin/python -m model_fine_tuning.pilot22_train.oldroot partgap  <part_gaps/report.py arguments>   (rule cache bypassed)
Training and validation days only (the evaluation days are refused by the scripts and never linked here).
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

from ..common import paths

OLD = Path("/var/tmp/sat3737-whale-days")
SH = Path("/var/tmp/sat3737-parts-cache/pilot22/days_shadow")
PW_SHADOW = Path("/var/tmp/sat3737-parts-cache/parts_weak/days_shadow")
BASELINES = ("parts_seg_r05_w3e", "parts_seg_r05_w3es1")
REPO = paths.REPO_ROOT


def days() -> list[str]:
    sp = json.loads((paths.SPLITS_DIR / "days.json").read_text())
    d = sp["train"] + sp["val"]
    assert not set(d) & set(sp["eval"])
    return d


def make() -> None:
    n = 0
    for day in days():
        dst = SH / day
        dst.mkdir(parents=True, exist_ok=True)
        for child in (OLD / day).iterdir():
            link = dst / child.name
            if not link.exists() and not link.is_symlink():
                os.symlink(child, link)
                n += 1
        for m in BASELINES:
            link = dst / m
            if not link.exists() and not link.is_symlink():
                os.symlink(PW_SHADOW / day / m, link)
                n += 1
    print(f"{SH}: {n} links made")


def patch_er():
    from ..parts_seg import evaluate_real as ER
    from ..parts_seg import predict as P
    ER.DAYS = SH
    P.DAYS = SH
    return ER, P


def cmd_predict(argv) -> int:
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", required=True)
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--metrics", type=Path, required=True)
    ap.add_argument("--days", default="train,val")
    a = ap.parse_args(argv)
    ER, P = patch_er()
    P.main(["--checkpoint", a.checkpoint, "--days", a.days, "--out-name", a.name])
    ER.main(["--out", str(a.metrics), "--name", a.name, "--days", a.days])
    return 0


def import_compare():
    sys.path.insert(0, str(REPO / "experiments" / "parts_weak"))
    import compare as CP
    CP.SHADOW = SH
    CP.DAYS_ALL = SH
    patch_er()
    return CP


def cmd_script(mod, script_name, argv) -> int:
    sys.argv = [script_name] + list(argv)
    return mod.main()


def main() -> int:
    if len(sys.argv) < 2:
        print(__doc__)
        return 2
    cmd, rest = sys.argv[1], sys.argv[2:]
    if cmd == "make":
        make()
        return 0
    if cmd == "predict":
        return cmd_predict(rest)
    if cmd == "compare":
        CP = import_compare()
        return cmd_script(CP, "compare.py", rest)
    if cmd == "finless":
        import_compare()
        import finless as FL
        return cmd_script(FL, "finless.py", rest)
    if cmd == "predfill":
        patch_er()
        sys.path.insert(0, str(REPO / "experiments" / "parts_tri_expand"))
        import pred_fill as PF
        PF.SHADOW = SH
        return cmd_script(PF, "pred_fill.py", rest)
    if cmd == "partgap":
        patch_er()
        from ..part_gaps import common as C
        from ..part_gaps import report as R
        C.DAYS_ALL = SH
        C.SHADOW_ROOTS = [SH]
        C.arm_root = lambda name: SH if (SH / C.VAL_DAYS[0] / name / "instances.json").exists() else None

        def rules_nocache(name, rows):          # report.rules_for without its cache (the cache holds live-root rules)
            rules = {}
            for variant in ("full", "hr"):
                if any(variant in r["pred"] for r in rows):
                    tf, td = C.choose_rule(rows, variant)
                    rules[variant] = (tuple(tf), td)
            return rules
        R.rules_for = rules_nocache
        return R.main(rest)
    print(f"unknown command {cmd}")
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
