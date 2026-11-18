"""Table of all fin_tune runs from fin_tune/results/<run>.json (written by score.py). CPU, no GPU.
    .venv/bin/python experiments/moonshot_sizing/fin_tune/summarize.py > experiments/moonshot_sizing/fin_tune/summary.txt
Also writes results/summary.json. Every number is pooled over the 8 test days, same pairs as raw HW-b6 (public_reid cache).
"""
import json

import numpy as np

from common import HERE

R = HERE / 'results'
RUNS = [('raw', 'raw HW-b6, this pipeline (bf16)', None)] + [(f's{n}_{s}', f'{n} days, seed {s}', 1500) for n in (5, 10, 16) for s in (0, 1)] + \
       [(f'e5_{s}', f'5 days, seed {s}, equal epochs', 278) for s in (0, 1)]
SUB = ['same flank', 'opposite flanks', 'flank unknown', 'gap > 1 h', 'val days', 'set-aside days']


def main():
    rows, out = [], {}
    j0 = json.load(open(R / 'raw.json'))['vs_raw']
    b = j0['all']['raw']
    print('raw HW-b6 (public_reid cache) on the 8 test days: AUC %.3f  AP %.3f  found at P 0.95 %.1f %%  at 0.90 %.1f %%  (%d same / %d different pairs)'
          % (b['auc'], b['ap'], 100 * b['r95'], 100 * b['r90'], b['same'], b['diff']))
    print('   raw by subset (AUC): ' + ', '.join(f'{t} {j0[t]["raw"]["auc"]:.3f}' for t in SUB))
    print()
    print(f'{"run":<34s} {"steps":>5s} {"AUC":>6s} {"AP":>6s} {"P95%":>6s} {"P90%":>6s} | {"dAUC":>7s} {"dAP":>7s} {"dP95":>6s} {"dP90":>6s} | bootstrap 95 % dAUC  | dAUC: ' + ' '.join(f'{t[:10]:>10s}' for t in SUB) + ' | dP90 gap>1h')
    for name, tag, steps in RUNS:
        f = R / f'{name}.json'
        if not f.exists():
            continue
        v = json.load(open(f))['vs_raw']
        a, d, bt = v['all']['cand'], v['all']['delta'], v['all']['boot95']
        out[name] = dict(tag=tag, steps=steps, all=v['all'], **{t: v[t] for t in SUB if t in v})
        print(f'{tag:<34s} {steps or 0:5d} {a["auc"]:6.3f} {a["ap"]:6.3f} {100 * a["r95"]:6.1f} {100 * a["r90"]:6.1f} | {d["auc"]:+7.3f} {d["ap"]:+7.3f} {100 * d["r95"]:+6.1f} {100 * d["r90"]:+6.1f} | '
              f'[{bt["auc"][0]:+.3f}, {bt["auc"][1]:+.3f}]  | ' + ' '.join(f'{v[t]["delta"]["auc"]:+10.3f}' if t in v else f'{"-":>10s}' for t in SUB)
              + f' | {100 * v["gap > 1 h"]["delta"]["r90"]:+.1f}')
        rows.append((name, v))
    print('\nseed means (difference to raw HW-b6; +- = half the spread of the two seeds)')
    for pre, tag in (('s5', '5 days, 1500 steps'), ('s10', '10 days, 1500 steps'), ('s16', '16 days, 1500 steps'), ('e5', '5 days, 278 steps (equal epochs)')):
        vs = [v for n, v in rows if n.startswith(pre + '_')]
        if len(vs) < 2:
            continue
        line = []
        for k, sc, nm in (('auc', 1, 'dAUC'), ('ap', 1, 'dAP'), ('r95', 100, 'dP95'), ('r90', 100, 'dP90')):
            x = np.array([v['all']['delta'][k] for v in vs]) * sc
            line.append(f'{nm} {x.mean():+.3f} +- {abs(x[0] - x[1]) / 2:.3f}')
        x = np.array([v['opposite flanks']['delta']['auc'] for v in vs])
        y = np.array([v['gap > 1 h']['delta']['auc'] for v in vs])
        print(f'   {tag:<34s} ' + ' | '.join(line) + f' | opposite flanks dAUC {x.mean():+.3f} | gap > 1 h dAUC {y.mean():+.3f}')
    json.dump(out, open(R / 'summary.json', 'w'), indent=1)


if __name__ == '__main__':
    main()
