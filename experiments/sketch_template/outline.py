"""sketch_template: outlines as sampled curves in the fin_sketch picture frame (u along the body, head toward -u; v down;
fin heights; tip at (0, -1), back base near (0, 0), the same for every drawing), and the noted-shape templates built
from them.

curves(view)  -> dict(top = v(u) on UG (the back and hump in front of the fin and the leading edge up to the tip),
                       trail = u(v) on VG (the trailing edge), back = v(u) on BG (the back behind the fin, from the whale
                       raster)), NaN where not seen
lead(top)     -> u(v) on VG: the leading edge, read off the top curve (from the tip forward, kept monotone)
dist(A, B)    -> median |difference| over the hump, the leading edge and the trailing edge (fin heights)
"""
import numpy as np

from common import FS

UG, VG, BG = FS.PAD_UG, FS.PAD_VG, FS.PAD_BG      # the sampling lives in fin_sketch (the page uses the same)
HUMP = UG <= -0.8                     # the back and hump ahead of the fin (no leading edge)
FIN_U = UG > -0.8
sample, curves, lead, back_curve = FS.pad_sample, FS.pad_curves, FS.pad_lead, FS.pad_back


def dist(A, B, parts=('top', 'trail')):
    ds = []
    for p in parts:
        d = np.abs(A[p] - B[p]); ds.append(d[~np.isnan(d)])
    d = np.concatenate(ds)
    return float(np.median(d)) if len(d) >= 15 else None


def nanmed(stack, min_n=1):
    S = np.stack(stack)
    n = (~np.isnan(S)).sum(0)
    with np.errstate(all='ignore'):
        import warnings
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            m = np.nanmedian(S, 0)
    m[n < min_n] = np.nan
    return m


def fill_nan(x, grid):
    ok = ~np.isnan(x)
    if ok.sum() < 2:
        return x
    return np.interp(grid, grid[ok], x[ok])


def smooth(x, s):
    from scipy.ndimage import gaussian_filter1d
    ok = ~np.isnan(x)
    if ok.sum() < 3:
        return x
    g = np.arange(len(x))
    f = gaussian_filter1d(np.interp(g, g[ok], x[ok]), s, mode='nearest')
    return np.where(ok, f, np.nan)
