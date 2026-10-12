"""Collapse check for a mirror-consistency fine-tune: on the real photos, the spread of the
predicted canonical coordinates within each crop must not shrink relative to the reference
model (a model can be trivially mirror-consistent by predicting the same point everywhere).

    .venv/bin/python test_scripts/python/canon_spread_check.py REF_PRED_DIR NEW_PRED_DIR
"""
import sys, numpy as np
from pathlib import Path
ref, new = Path(sys.argv[1]), Path(sys.argv[2])
def spreads(d):
    out = {}
    for f in sorted(d.glob("*_canon.npz")):
        z = np.load(f); use = z["use"]
        if use.sum() < 500: continue
        pts = z["signed_mm"][use].astype(float) / 1000.0
        out[f.stem] = (pts.std(0).mean(), np.ptp(pts[:, 0]), np.median(z["disagree_mm"][use]) / 1000.0)
    return out
a, b = spreads(ref), spreads(new)
common = sorted(set(a) & set(b))
sa = np.array([a[k][0] for k in common]); sb = np.array([b[k][0] for k in common])
ra = np.array([a[k][1] for k in common]); rb = np.array([b[k][1] for k in common])
da = np.array([a[k][2] for k in common]); db = np.array([b[k][2] for k in common])
print(f"crops compared: {len(common)}")
print(f"per-crop coordinate spread (mean std over xyz, m): ref median {np.median(sa):.3f}  new median {np.median(sb):.3f}  ratio new/ref median {np.median(sb / np.maximum(sa, 1e-6)):.3f}")
print(f"per-crop body-fraction range (m along the body): ref median {np.median(ra):.3f}  new median {np.median(rb):.3f}")
print(f"mirror disagreement (m): ref median {np.median(da):.3f}  new median {np.median(db):.3f}")
print("collapse warning" if np.median(sb / np.maximum(sa, 1e-6)) < 0.7 else "no collapse: spread preserved")
