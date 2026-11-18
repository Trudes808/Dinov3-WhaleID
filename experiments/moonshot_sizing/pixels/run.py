"""pixels: feature variants from the extract.py caches through the shared bench (CPU).

    OMP_NUM_THREADS=8 python run.py VARIANT [VARIANT ...]     (or `list`)

A variant = a list of blocks (job, view, transform, layer, pooling); every block is scaled to unit length and the
blocks are concatenated, like the live view features. `{p}` in a view / pooling name is the bench part (fluke /
dorsal), `{c}` the part's chart name. Results: ../results/pixels_<variant>__<part>.json; text in logs/.
"""
import functools
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
import bench  # noqa: E402

CACHE = Path('/home/sat3737/NOAA_whales/moonshot_cache/pixels')


def B(view, t, pool, layer=11, job='base'):
    return (job, view, t, layer, pool)


def four(t_cls, t_tok, pool, job='base', layer=11, views=('sq', 'ar')):
    out = []
    for v in views:
        out += [B(v, t_cls, 'cls', layer, job), B(v, t_tok, pool, layer, job)]
    return out


V = {
    'base_repro': four('rgb', 'rgb', 'mean'),
    'wpool_canon': four('rgb', 'rgb', 'wc'),                      # pooling only: whale-weighted mean, cls as BASE
    'wpool_inst': four('rgb', 'rgb', 'wi'),
    'fillcls': four('fill', 'rgb', 'mean'),                       # cls of the filled crop, mean as BASE
    'whale': four('fill', 'rgb', 'wc'),                           # whale-only: filled cls + whale-weighted mean
    'whale_inst': four('fill', 'rgb', 'wi'),
    'fill_all': four('fill', 'fill', 'wc'),                       # everything from the filled crop
    'fill_mean': four('fill', 'fill', 'mean'),
    'part': four('fill', 'rgb', 'part_{p}'),                      # part region tokens + filled cls
    'part_only': [B('sq', 'rgb', 'part_{p}'), B('ar', 'rgb', 'part_{p}')],
    'whale_part': four('fill', 'rgb', 'wc') + [B('sq', 'rgb', 'part_{p}'), B('ar', 'rgb', 'part_{p}')],
    # input normalisation, alone and with whale-only pooling
    'grey': four('grey', 'grey', 'mean', 'norm'),
    'grey_whale': four('grey_fill', 'grey', 'wc', 'norm'),
    'expo': four('expo', 'expo', 'mean', 'norm'),
    'expo_whale': four('expo_fill', 'expo', 'wc', 'norm'),
    'clahe': four('clahe', 'clahe', 'mean', 'norm'),
    'clahe_whale': four('clahe_fill', 'clahe', 'wc', 'norm'),
    'grey_wpool': four('grey', 'grey', 'wc', 'norm'),
    'expo_wpool': four('expo', 'expo', 'wc', 'norm'),
    'clahe_wpool': four('clahe', 'clahe', 'wc', 'norm'),
    'rgb_plus_bin_w': four('rgb', 'rgb', 'wc') + four('bin', 'bin', 'wc', 'norm'),
    'sq896_wpool': [B('sq896', 'rgb', 'cls', job='res'), B('sq896', 'rgb', 'wc', job='res')],
    'partsq_w': [B('partsq_{p}', 'rgb', 'cls', job='res'), B('partsq_{p}', 'rgb', 'wc', job='res')],
    'chart_w': [B('chart_{p}', 'rgb', 'cls', job='res'), B('chart_{p}', 'rgb', 'wc', job='res')],
    'chart_p': [B('chart_{p}', 'rgb', 'cls', job='res'), B('chart_{p}', 'rgb', 'part_{p}', job='res')],
    'wpool_partsq': four('rgb', 'rgb', 'wc') + [B('partsq_{p}', 'rgb', 'cls', job='res'), B('partsq_{p}', 'rgb', 'wc', job='res')],
    'wpool_chart': four('rgb', 'rgb', 'wc') + [B('chart_{p}', 'rgb', 'cls', job='res'), B('chart_{p}', 'rgb', 'wc', job='res')],
    'wpool_sq896': four('rgb', 'rgb', 'wc') + [B('sq896', 'rgb', 'cls', job='res'), B('sq896', 'rgb', 'wc', job='res')],
    'clahe_wpool_plus_part': four('clahe', 'clahe', 'wc', 'norm') + [B('sq', 'clahe', 'part_{p}', 11, 'norm'), B('ar', 'clahe', 'part_{p}', 11, 'norm')],
    'grey_clahe_wpool': four('clahe', 'clahe', 'wc', 'norm') + four('grey', 'grey', 'wc', 'norm'),
    'wpool_partsq_p': four('rgb', 'rgb', 'wc') + [B('partsq_{p}', 'rgb', 'cls', job='res'), B('partsq_{p}', 'rgb', 'part_{p}', job='res')],
    'wpool_partsq_chart': four('rgb', 'rgb', 'wc') + [B('partsq_{p}', 'rgb', 'cls', job='res'), B('partsq_{p}', 'rgb', 'wc', job='res'),
                                                      B('chart_{p}', 'rgb', 'cls', job='res'), B('chart_{p}', 'rgb', 'wc', job='res')],
    # fin + hump zoom where the part model finds the part, the plain whale-pooled crop where it does not (all instances)
    'wpool_partsq_p_full': four('rgb', 'rgb', 'wc') + [[B('partsq_{p}', 'rgb', 'cls', job='res'), B('sq', 'rgb', 'cls')],
                                                       [B('partsq_{p}', 'rgb', 'part_{p}', job='res'), B('sq', 'rgb', 'wc')]],
    'bin': four('bin', 'bin', 'wc', 'norm'),
    'rgb_plus_bin': four('rgb', 'rgb', 'mean') + four('bin', 'bin', 'wc', 'norm'),
    'whale_plus_bin': four('fill', 'rgb', 'wc') + four('bin', 'bin', 'wc', 'norm'),
    # resolution
    'sq896': [B('sq896', 'rgb', 'cls', job='res'), B('sq896', 'rgb', 'mean', job='res')],
    'sq896_whale': [B('sq896', 'fill', 'cls', job='res'), B('sq896', 'rgb', 'wc', job='res')],
    'partsq': [B('partsq_{p}', 'fill', 'cls', job='res'), B('partsq_{p}', 'rgb', 'wc', job='res')],
    'partsq_part': [B('partsq_{p}', 'fill', 'cls', job='res'), B('partsq_{p}', 'rgb', 'part_{p}', job='res')],
    'chart': [B('chart_{p}', 'rgb', 'cls', job='res'), B('chart_{p}', 'rgb', 'mean', job='res')],
    'chart_whale': [B('chart_{p}', 'fill', 'cls', job='res'), B('chart_{p}', 'rgb', 'wc', job='res')],
    'chart_part': [B('chart_{p}', 'fill', 'cls', job='res'), B('chart_{p}', 'rgb', 'part_{p}', job='res')],
    'whale_partsq': four('fill', 'rgb', 'wc') + [B('partsq_{p}', 'fill', 'cls', job='res'), B('partsq_{p}', 'rgb', 'wc', job='res')],
    'whale_chart': four('fill', 'rgb', 'wc') + [B('chart_{p}', 'fill', 'cls', job='res'), B('chart_{p}', 'rgb', 'wc', job='res')],
    'wpool_part': four('rgb', 'rgb', 'part_{p}'),
    'wpool_plus_part': four('rgb', 'rgb', 'wc') + [B('sq', 'rgb', 'part_{p}'), B('ar', 'rgb', 'part_{p}')],
    'wc_only': [B('sq', 'rgb', 'wc'), B('ar', 'rgb', 'wc')],
    'cls_mean_wc': four('rgb', 'rgb', 'mean') + [B('sq', 'rgb', 'wc'), B('ar', 'rgb', 'wc')],
    'wpool_l7': four('rgb', 'rgb', 'wc') + four('rgb', 'rgb', 'wc', layer=7),
    # layers
    'whale_l7': four('fill', 'rgb', 'wc') + four('fill', 'rgb', 'wc', layer=7),
    'whale_l7only': four('fill', 'rgb', 'wc', layer=7),
    'base_l7': four('rgb', 'rgb', 'mean') + four('rgb', 'rgb', 'mean', layer=7),
}


@functools.lru_cache(None)
def load(job, day):
    p = CACHE / job / f'{day}.npz'
    if not p.exists():
        return None
    z = np.load(p)
    return {n: (z[n + '#keys'], z[n]) for n in z.files if not n.endswith('#keys')}


def block(job, view, t, layer, pool, part, day):
    z = load(job, day)
    if z is None:
        return {}
    name = f'{view}|{t}|{layer}|{pool}'.replace('{p}', part)
    if name not in z:
        return {}
    keys, X = z[name]
    X = X.astype(np.float32)
    X /= np.maximum(np.linalg.norm(X, axis=1, keepdims=True), 1e-6)
    return dict(zip([str(k) for k in keys], X))


def features(variant, part):
    out = {}
    for d in bench.TRAIN + bench.VAL:
        blocks = []
        for b in V[variant]:
            if isinstance(b, list):                 # [first choice, fallback]: fallback fills the keys the first lacks
                first, fb = block(*b[0], part, d), block(*b[1], part, d)
                blocks.append({**fb, **first})
            else:
                blocks.append(block(*b, part, d))
        keys = set.intersection(*[set(b) for b in blocks]) if blocks else set()
        out[d] = {k: np.concatenate([b[k] for b in blocks]) for k in keys}
    return out


def main():
    args = sys.argv[1:]
    if not args or args[0] == 'list':
        print('\n'.join(V))
        return
    parts = bench.PARTS
    if args[0] in bench.PARTS:
        parts, args = (args[0],), args[1:]
    for v in args:
        for p in parts:
            bench.evaluate(f'pixels_{v}', p, features(v, p))


if __name__ == '__main__':
    main()
