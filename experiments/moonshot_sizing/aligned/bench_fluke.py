"""Fluke runs on the bench (CPU): pair scorers posc / edge (quality-filtered and not), fusion with BASE."""
import common  # noqa: F401
import bench
import fuse
import run_pairs

P = 'fluke'
S = {}
for v in ('posc', 'edge'):
    for q in ('default', 'none'):
        S[(v, q)] = run_pairs.pair_scores(P, v, q)
        bench.evaluate_pairs(f'aligned_{v}' + ('' if q == 'default' else '_qnone'), P, S[(v, q)])
fuse.fuse('aligned_fuse_posc_base', P, {'posc': S[('posc', 'none')]}, [1.0])
fuse.fuse('aligned_fuse_posc_q_base', P, {'posc': S[('posc', 'default')]}, [1.0])
fuse.fuse('aligned_fuse_edge_base', P, {'edge': S[('edge', 'none')]}, [1.0])
