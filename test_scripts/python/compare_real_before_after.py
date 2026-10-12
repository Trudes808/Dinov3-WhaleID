"""Before/after on the real photos: the (u, theta) body model's mirror test (side_diag json)
against the canonical-surface model's own mirror statistics (points.jsonl), plus a sheet of
the crops that were worst before, rendered by the new model.

    .venv/bin/python test_scripts/python/compare_real_before_after.py BEFORE_SIDE_JSON CANON_PRED_DIR OUT_DIR
"""
from __future__ import annotations
import json, sys
from pathlib import Path
import numpy as np, cv2

before = json.load(open(sys.argv[1]))
pred_dir, out = Path(sys.argv[2]), Path(sys.argv[3]); out.mkdir(parents=True, exist_ok=True)
after = [json.loads(l) for l in open(pred_dir / "points.jsonl")]
after = {Path(r["image"]).stem: r for r in after if r.get("status") == "ok"}

def stat(v): v = np.array([x for x in v if x is not None and np.isfinite(x)]); return (np.median(v), np.mean(v), len(v)) if len(v) else (np.nan, np.nan, 0)
b_dis = [r["side_disagree_flipTTA"] for r in before]
a_agree = [r.get("side_flip_agreement") for r in after.values()]
a_dis = [None if a is None else 1 - a for a in a_agree]
print("== real photos: fraction of whale pixels whose LEFT/RIGHT changes when the crop is mirrored")
print(f"  before ((u,theta) head, {len(before)} crops): median {stat(b_dis)[0]:.3f} mean {stat(b_dis)[1]:.3f}  crops >20%: {np.mean(np.array([x for x in b_dis if np.isfinite(x)])>0.2):.2%}")
print(f"  after  (canonical surface, {len(after)} crops): median {stat(a_dis)[0]:.3f} mean {stat(a_dis)[1]:.3f}  crops >20%: {np.mean(np.array([x for x in a_dis if x is not None])>0.2):.2%}")
print("== after: folded-coordinate disagreement under mirroring (metres on the whale), median over crops:", f"{stat([r.get('flip_disagreement_median_m') for r in after.values()])[0]:.3f} m")
print("== after: side margin (relative cost gap between the two global hypotheses): median", f"{stat([r.get('side_margin') for r in after.values()])[0]:.3f}",
      " crops with margin < 0.1:", f"{np.mean(np.array([r.get('side_margin') or 0 for r in after.values()]) < 0.1):.2%}")
print("== after: direct-head rotation change under mirroring, median deg:", f"{stat([r.get('rot_direct_flip_consistency_deg') for r in after.values()])[0]:.1f}",
      "  direct vs PnP (resolved side), median deg:", f"{stat([r.get('rot_direct_vs_pnp_deg') for r in after.values()])[0]:.1f}")
found = [sum(1 for v in r["landmarks"].values() if v["uv_full"] is not None) for r in after.values()]
print("== after: points of interest found per photo (of", len(next(iter(after.values()))["landmarks"]), "landmarks): median", np.median(found))
# sheet: the six worst "before" crops, as the new model sees them (its own overlay panels)
worst = [r["name"] for r in sorted(before, key=lambda r: -(r["side_disagree_flipTTA"] if np.isfinite(r["side_disagree_flipTTA"]) else -1))[:6]]
tiles = []
for name in worst:
    f = pred_dir / f"{name}_overlay.jpg"
    if not f.exists(): continue
    t = cv2.imread(str(f)); t = cv2.resize(t, (t.shape[1] // 2, t.shape[0] // 2))
    r = after.get(name, {})
    cv2.putText(t, f"{name[-22:]} before: {[x for x in before if x['name']==name][0]['side_disagree_flipTTA']:.2f} of pixels changed side | after: side agree={r.get('side_flip_agreement')} margin={r.get('side_margin')}",
                (6, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1, cv2.LINE_AA)
    tiles.append(t)
if tiles:
    W = max(t.shape[1] for t in tiles)
    cv2.imwrite(str(out / "worst_before_as_seen_after.jpg"), np.concatenate([np.pad(t, ((0, 0), (0, W - t.shape[1]), (0, 0))) for t in tiles], 0), [cv2.IMWRITE_JPEG_QUALITY, 85])
    print("sheet:", out / "worst_before_as_seen_after.jpg")
