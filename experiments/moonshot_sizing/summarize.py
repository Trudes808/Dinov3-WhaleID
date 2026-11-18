"""One line per saved bench result (results/<name>__<part>.json), read straight from the files (2026-11-18).

Columns, on val + the 5 set-aside train days with 16 fit days, mean over the 20 fixed splits: candidate and BASE AUC,
paired AUC difference (splits in which the candidate is ahead), share of same-animal pairs found at precision 0.95 /
0.90 for candidate and BASE, paired difference at 0.90, the 3 val days with all 21 train days (candidate / BASE), and
the gain per step in fit days (5 -> 10 -> 16) where the result has it. `cov` = share of the bench's pairs the candidate
scored; BASE is measured on the same pairs. A star marks the bar set before the agent round: +0.03 AUC in >= 17 of 20
splits, or found at precision 0.90 at least doubled.

    python summarize.py [name prefix ...]
"""
import glob
import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent


def col(rows, key):
    return np.array([r[key] for r in rows], float)


def held16(d):
    """(candidate rows, base rows, {fit days: (candidate rows, base rows)}) of one result."""
    h, kind = d.get('held'), d.get('kind')
    if h is None:
        return None, None, {}
    if kind == 'features':
        by = {n: (h[f"('cand', {n})"], h[f"('base', {n})"]) for n in (5, 10, 16) if f"('cand', {n})" in h}
        return by[16][0], by[16][1], by
    if kind in ('fitted_pairs', 'fusion_curve'):
        by = {int(n): (v['cand'], v['base']) for n, v in h.items()}
        return by[16][0], by[16][1], by
    return h['cand'], h['base'], {}


def line(d, tag=''):
    c, b, by = held16(d)
    v = d.get('val') or (d if 'cand' in d and 'base' in d else None)
    name = d['name'] + tag
    out = f'{name[:44]:<44s} {d["part"]:<6s}'
    star = ' '
    if c:
        da, d90 = col(c, 'auc') - col(b, 'auc'), (col(c, 'r90') - col(b, 'r90')) * 100
        up = int((da > 0).sum())
        if (da.mean() >= 0.03 and up >= 17) or col(c, 'r90').mean() >= 2 * col(b, 'r90').mean():
            star = '*'
        out += (f' {col(c, "auc").mean():.3f} {col(b, "auc").mean():.3f} {da.mean():+.3f} ({up:2d})'
                f'  {col(c, "r95").mean() * 100:4.1f}/{col(b, "r95").mean() * 100:4.1f}'
                f'  {col(c, "r90").mean() * 100:4.1f}/{col(b, "r90").mean() * 100:4.1f} {d90.mean():+5.1f} ({int((d90 > 0).sum()):2d})')
    else:
        out += ' ' * 56
    if v and 'cand' in v:
        out += (f'  {v["cand"]["auc"]:.3f}/{v["base"]["auc"]:.3f}'
                f'  {v["cand"]["r90"] * 100:4.1f}/{v["base"]["r90"] * 100:4.1f}')
        f = v.get('flank')
        if f and 'same flank' in f.get('cand', {}):
            out += f'  {f["cand"]["same flank"]["auc"]:.3f}/{f["cand"]["opposite flanks"]["auc"]:.3f}'
        else:
            out += ' ' * 13
    else:
        out += ' ' * 39
    if 'pairs_scored' in d and d.get('pairs_total'):
        out += f'  {100 * d["pairs_scored"] / d["pairs_total"]:3.0f}%'
    elif 'instances' in d and d.get('instances_total'):
        out += f'  {100 * d["instances"] / d["instances_total"]:3.0f}%'
    else:
        out += '      '
    if len(by) == 3:
        g = [col(by[n][0], 'auc').mean() for n in (5, 10, 16)]
        gb = [col(by[n][1], 'auc').mean() for n in (5, 10, 16)]
        out += f'  {g[1] - g[0]:+.3f} {g[2] - g[1]:+.3f} (BASE {gb[1] - gb[0]:+.3f} {gb[2] - gb[1]:+.3f})'
    return star + out


if __name__ == '__main__':
    want = sys.argv[1:]
    print(f' {"result":<44s} {"part":<6s} {"AUC":>5s} {"BASE":>5s} {"diff (up/20)":>12s}  {"P.95 c/B":>9s}  {"P.90 c/B":>9s} {"diff (up)":>10s}'
          f'  {"val AUC c/B":>11s}  {"val P.90":>9s}  {"same/opp fl":>11s}  {"cov":>4s}  gain 5>10 10>16')
    for f in sorted(glob.glob(str(HERE / 'results' / '*.json'))):
        d = json.load(open(f))
        stem = Path(f).stem
        if not isinstance(d, dict) or 'part' not in d:
            d = dict(d if isinstance(d, dict) else {}, part=stem.rsplit('__', 1)[-1])
        d.setdefault('name', stem.rsplit('__', 1)[0])
        if want and not any(d['name'].startswith(w) for w in want):
            continue
        try:
            print(line(d))
        except Exception as e:                          # a file an agent wrote in a layout of its own
            print(f' {d["name"][:44]:<44s} {d["part"]:<6s} not in the bench layout ({type(e).__name__}: {e}); keys {list(d)[:8]}')
            continue
        if 'fuse' in d:
            print(line(dict(d['fuse'], name=d['name'], part=d['part'], kind='fusion'), tag=' + BASE'))
