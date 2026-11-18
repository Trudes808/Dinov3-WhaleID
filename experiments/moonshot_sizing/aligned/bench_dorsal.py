"""Dorsal runs on the bench (CPU).  python bench_dorsal.py STEP
  pairs    pair scorers (no fitting): pos, posc, posc_nb, posc_top, outline (quality-filtered fins) and posc without filter
  fuse     z-mean with BASE: posc, outline, posc + outline
  vectors  label-free vectors through the bench head: aligned grid (tok11 2x2 cells, PCA 24 per cell), outline profile
"""
import sys

import common  # noqa: F401  (puts the bench on sys.path)
import bench
import fuse
import grid
import run_pairs

step = sys.argv[1]
P = 'dorsal'
if step == 'pairs':
    for v in sys.argv[2:] or ['posc', 'outline', 'pos', 'posc_nb', 'posc_top']:
        bench.evaluate_pairs(f'aligned_{v}', P, run_pairs.pair_scores(P, v))
    bench.evaluate_pairs('aligned_posc_qnone', P, run_pairs.pair_scores(P, 'posc', 'none'))
elif step == 'fuse':
    S = {v: run_pairs.pair_scores(P, v) for v in ('posc', 'outline')}
    fuse.fuse('aligned_fuse_posc_base', P, {'posc': S['posc']}, [1.0])
    fuse.fuse('aligned_fuse_outline_base', P, {'outline': S['outline']}, [1.0])
    fuse.fuse('aligned_fuse_posc_outline_base', P, {'posc': S['posc'], 'outline': S['outline']}, [1.0, 1.0])
elif step == 'vectors':
    bench.evaluate('aligned_outlinevec', P, grid.outline_vectors(qual=run_pairs.score.QUAL))
    bench.evaluate('aligned_grid_tok11_p2_24', P, grid.build(P, 'tok11', 2, 24))
