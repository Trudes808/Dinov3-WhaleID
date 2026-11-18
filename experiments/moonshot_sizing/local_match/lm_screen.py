"""Fast val screen: BASE fitted once on the 21 train days, every score variant measured on the same val pairs.
    python lm_screen.py VARIANT PART [geo] [nofb] [npick=N] [mirror=plain|max|mirror]"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import lm_score as SC  # noqa: E402
from lm_lib import bench  # noqa: E402

variant, part = sys.argv[1:3]
opt = dict(a.split('=') for a in sys.argv[3:] if '=' in a)
npick = int(opt.get('npick', 2))
mirrors = [opt['mirror']] if 'mirror' in opt else (['plain', 'max'] if part == 'dorsal' else ['plain'])
B = bench.base_scores(part, bench.VAL, bench.TRAIN)
for mirror in mirrors:
    R = SC.cluster_scores(variant, part, bench.VAL, mirror, 'geo' in sys.argv, 'nofb' in sys.argv, npick)
    for name, S in R.items():
        only = {d: {p for p in S.get(d, {}) if p in B.get(d, {})} for d in bench.VAL}
        c, b = bench.measure(part, S, bench.VAL, only), bench.measure(part, B, bench.VAL, only)
        fl = bench._flank_rows(part, S, bench.VAL, only)
        fls = '  '.join(f"{k[:8]} {v['auc']:.3f}/{100 * v['r90']:.1f}" for k, v in sorted(fl.items()) if k in ('same flank', 'opposite flanks'))
        print(f"{variant:13s} {part:6s} {mirror:6s} {name:13s} AUC {c['auc']:.3f} AP {c['ap']:.3f} P95 {100*c['r95']:5.1f} P90 {100*c['r90']:5.1f}"
              f" | BASE {b['auc']:.3f} {b['ap']:.3f} {100*b['r95']:5.1f} {100*b['r90']:5.1f} | {c['same']}+{c['diff']} {fls}", flush=True)
