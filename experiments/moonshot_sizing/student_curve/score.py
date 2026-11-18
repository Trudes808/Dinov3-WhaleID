"""Score one embedded network on the shared bench (fixed split), dorsal from the fin embedding, fluke from the fluke one.

    OMP_NUM_THREADS=8 .venv/bin/python experiments/moonshot_sizing/student_curve/score.py NAME EMB_DIR --test val|valhold \
        --fit train|pool:N
  test val      = bench.VAL;             fit train  = bench.TRAIN (21 days)
  test valhold  = bench.VAL + hold;      fit pool:N = pool[:N]    (hold, pool = bench.splits()[0])
Result: experiments/moonshot_sizing/results/student_curve_NAME__<part>.json (bench.compare)
"""
import argparse
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
import bench  # noqa: E402

EMB = Path('/home/sat3737/NOAA_whales/moonshot_cache/student_curve/emb')


def load(emb_dir, days, part):
    out = {}
    for d in days:
        z = np.load(emb_dir / f'{d}.npz')
        out[d] = {str(k): v.astype(np.float32) for k, v in zip(z[f'{part}_keys'], z[f'{part}_E'])}
    return out


def main():
    ap = argparse.ArgumentParser(); ap.add_argument('name'); ap.add_argument('emb'); ap.add_argument('--test', required=True); ap.add_argument('--fit', required=True)
    a = ap.parse_args()
    hold, pool = bench.splits()[0]
    test = bench.VAL if a.test == 'val' else bench.VAL + hold
    fit = bench.TRAIN if a.fit == 'train' else pool[: int(a.fit.split(':')[1])]
    emb = Path(a.emb) if Path(a.emb).is_absolute() else EMB / a.emb
    for part, net_part in (('dorsal', 'fin'), ('fluke', 'fluke')):
        X = load(emb, test, net_part)
        S = bench.embed_scores(part, X, test)
        bench.compare(f'student_curve_{a.name}', part, S, test, fit)


if __name__ == '__main__':
    main()
