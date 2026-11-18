"""pixels: resolution variants on the days whose res cache exists (17 of 24: all 3 val + 14 train; the GPU queue did
not give the last 7 train days a turn). NOT the bench's standard protocol: 20 repetitions, 4 of the 14 train days set
aside, head fitted on the other 10 for the candidate AND for BASE (bench.embed_scores / bench.base_scores), scored with
bench.measure on val + the 4 set-aside days, on the pairs both score. Paired difference per repetition.

    OMP_NUM_THREADS=8 python res_eval.py VARIANT [VARIANT ...]
"""
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
import bench  # noqa: E402
import run  # noqa: E402

CACHE = run.CACHE / 'res'
AVAIL = [d for d in bench.TRAIN if (CACHE / f'{d}.npz').exists()]


def main():
    print(f'train days with res cache: {len(AVAIL)} of {len(bench.TRAIN)}; val {bench.VAL}')
    for v in sys.argv[1:]:
        for part in bench.PARTS:
            X = {d: x for d, x in run.features(v, part).items() if d in AVAIL + bench.VAL}
            n = sum(len(x) for d, x in X.items())
            ntot = sum(len(bench.instances(d, part)) for d in X)
            rows = []
            for r in range(20):
                rng = np.random.default_rng(2000 + r)
                order = [AVAIL[i] for i in rng.permutation(len(AVAIL))]
                hold, fit = order[:4], order[4:14]
                test = bench.VAL + hold
                S = bench.embed_scores(part, X, test, fit)
                Bs = bench.base_scores(part, test, fit)
                only = {d: {p for p in S.get(d, {}) if p in Bs.get(d, {})} for d in test}
                rows.append((bench.measure(part, S, test, only), bench.measure(part, Bs, test, only)))
            c = {k: np.mean([a[k] for a, b in rows]) for k in ('auc', 'ap', 'r95', 'r90')}
            b = {k: np.mean([b[k] for a, b in rows]) for k in ('auc', 'ap', 'r95', 'r90')}
            d = np.array([a['auc'] - b['auc'] for a, b in rows])
            d90 = np.array([a['r90'] - b['r90'] for a, b in rows]) * 100
            print(f'pixels_{v:<14s} {part:<6s} inst {n}/{ntot}  cand AUC {c["auc"]:.3f} AP {c["ap"]:.3f} P95 {100 * c["r95"]:.1f} P90 {100 * c["r90"]:.1f} | '
                  f'BASE same pairs AUC {b["auc"]:.3f} AP {b["ap"]:.3f} P95 {100 * b["r95"]:.1f} P90 {100 * b["r90"]:.1f} | dAUC {d.mean():+.3f} up {(d > 0).sum()}/20 '
                  f'dP90 {d90.mean():+.1f} | pairs {rows[0][0]["same"]}+{rows[0][0]["diff"]} (rep 0)', flush=True)


if __name__ == '__main__':
    main()
