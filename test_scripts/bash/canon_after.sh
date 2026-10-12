#!/usr/bin/env bash
# Detached follow-up to canon_pipeline.sh: waits for it, then writes the before/after comparison
# (synthetic metrics old vs new, real-photo mirror test) to experiments/canon_r02_ft4/summary.txt.
set -uo pipefail
cd "$(dirname "$0")/../.."
NAME="${1:-canon_r02_ft4}"
L=experiments/canon_pipeline.log
while ! grep -q "^== done" "$L" 2>/dev/null; do
    if grep -q "Traceback" "$L" 2>/dev/null; then echo "pipeline error"; exit 1; fi
    if ! pgrep -f canon_pipeline.sh >/dev/null; then echo "pipeline gone"; exit 1; fi
    sleep 60
done
OUT=experiments/$NAME/summary.txt
{
echo "# canonical-surface model $NAME vs (u,theta) body model body_r02_ft4 -- $(date -u +%FT%TZ)"
echo; echo "## synthetic run02 test split"
.venv/bin/python - <<PY
import json
new = json.load(open("experiments/$NAME/eval_test/metrics.json"))
old = json.load(open("experiments/body_r02_ft4/eval_test/metrics.json"))
for tag in ("oracle_crops", "seg_crops"):
    n, o = new[tag], old[tag]
    print(f"[{tag}] images new={n['images_evaluated']} old={o['images_evaluated']}")
    print(f"  NEW surface error mean/median {n.get('surface_err_m_mean'):.3f}/{n.get('surface_err_median_m_median'):.3f} m  torso {n.get('surface_err_torso_m_mean')} fluke {n.get('surface_err_fluke_m_mean')} fin {n.get('surface_err_fin_m_mean')}")
    print(f"  NEW mirror: folded disagreement {n.get('flip_surface_disagreement_m_mean'):.3f} m, side agreement {n.get('side_flip_agreement_mean')}, unstable crops {n.get('side_unstable_crops_frac')}")
    print(f"  NEW side accuracy vs GT {n.get('side_acc_mean')}  margin median {n.get('side_margin_median')}")
    print(f"  NEW rotation direct median {n.get('rot_direct_deg_median'):.1f} (<15: {n.get('rot_direct_deg_under_15')})  PnP median {n.get('rot_pnp_deg_median')} (<15: {n.get('rot_pnp_deg_under_15')})")
    print(f"  NEW points of interest from the map: PCK@0.05 {n.get('lm_map_pck_0.05_mean')}  PCK@0.1 {n.get('lm_map_pck_0.1_mean')}  (PnP-reprojected PCK@0.1 {n.get('lm_pnp_pck_0.1_mean')})")
    print(f"  OLD u MAE {o.get('u_mae_mean'):.4f} ({o.get('u_mae_metres_mean'):.2f} m along the body), angle MAE {o.get('angle_mae_deg_mean'):.1f} deg, rotation direct median {o.get('rot_direct_deg_median'):.1f}, PnP median {o.get('rot_pnp_deg_median'):.1f}, PnP-reprojected PCK@0.1 {o.get('pck_0.1_mean')}")
PY
echo; echo "## real photos (2024-12-18)"
.venv/bin/python test_scripts/python/compare_real_before_after.py experiments/real_2024-12-18/side_diag/real_side.json experiments/real_2024-12-18/canon_$NAME experiments/real_2024-12-18/canon_${NAME}_compare
} > "$OUT" 2>&1
echo "wrote $OUT"; cat "$OUT"
