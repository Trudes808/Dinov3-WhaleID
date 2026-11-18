"""Table of the student_curve results (results/student_curve_*.json): candidate next to BASE on the same pairs, per run,
plus mean and seed spread per size, and the 5 -> 10 -> 16 gains.
    .venv/bin/python experiments/moonshot_sizing/student_curve/summarize.py > experiments/moonshot_sizing/student_curve/summary.txt
"""
import json
from pathlib import Path

import numpy as np

RES = Path(__file__).resolve().parent.parent / 'results'
KEYS = (('auc', 'AUC', 1), ('ap', 'AP', 1), ('r95', 'P.95 %', 100), ('r90', 'P.90 %', 100))


def row(tag, part, r):
    c, b = r['cand'], r['base']
    return (f"{tag:<26s} {part:<6s} " + '  '.join(f"{c[k] * s:6.3f}" if s == 1 else f"{c[k] * s:6.1f}" for k, _, s in KEYS) + '   | BASE '
            + '  '.join(f"{b[k] * s:6.3f}" if s == 1 else f"{b[k] * s:6.1f}" for k, _, s in KEYS) + f"   ({c['same']} same / {c['diff']} diff, {r['pairs_scored']} of {r['pairs_total']})")


def main():
    print(f"{'model':<26s} {'part':<6s} " + '  '.join(f'{n:>6s}' for _, n, _ in KEYS) + '   | BASE same columns')
    R = {}
    for f in sorted(RES.glob('student_curve_*.json')):
        r = json.load(open(f)); name = r['name'].replace('student_curve_', '')
        R[(name, r['part'])] = r
    for part in ('dorsal', 'fluke'):
        for name in sorted({n for n, p in R if p == part}, key=lambda n: (n.startswith('s'), n.endswith('idstage'), n)):
            print(row(name, part, R[(name, part)]))
        print()
    for kind in ('', '_idstage'):
        print(f'== per size, mean over seeds (spread = |seed0 - seed1| / 2), {"student" if not kind else "id stage alone"}')
        for part in ('dorsal', 'fluke'):
            means = {}
            for n in (5, 10, 16):
                rs = [R[(f's{n}_{s}{kind}', part)] for s in (0, 1) if (f's{n}_{s}{kind}', part) in R]
                if not rs:
                    continue
                line = []
                for k, nm, sc in KEYS:
                    c = np.array([r['cand'][k] for r in rs]) * sc; b = np.array([r['base'][k] for r in rs]) * sc
                    means[(n, k)] = (c.mean(), b.mean(), (c.max() - c.min()) / 2 if len(c) > 1 else float('nan'))
                    line.append(f'{nm} {c.mean():.3f} +- {means[(n, k)][2]:.3f} (BASE {b.mean():.3f})')
                print(f'  {part:<6s} {n:>2d} days ({len(rs)} seeds): ' + ' | '.join(line))
            for a, b_ in ((5, 10), (10, 16), (5, 16)):
                if (a, 'auc') in means and (b_, 'auc') in means:
                    print(f'  {part:<6s} gain {a} -> {b_}: ' + ' | '.join(
                        f"{nm} cand {means[(b_, k)][0] - means[(a, k)][0]:+.3f} BASE {means[(b_, k)][1] - means[(a, k)][1]:+.3f}" for k, nm, _ in KEYS))
        print()


if __name__ == '__main__':
    main()
