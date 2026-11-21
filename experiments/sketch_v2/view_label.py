"""sketch_v2 B (2026-11-19): can a drawing say on its own whether it shows the direct side profile or an angled view?
No ground truth of the angle exists; the check uses the animal itself: for two stacks of the SAME catalogue animal and
side (bench.pairs, same group, > 60 s apart), the one seen more side-on shows the wider fin halfway up (a fin turned
toward or away from the camera is foreshortened along the body; its height is kept). A cue is useful when, for the
same-animal pairs it orders, the stack it calls more side-on has the wider fin. Cues (each from the stack alone):
profile (the photo pose model's |nose right|, parts_up npz, median over the frames) and bperp (keypoint base ratio
square to the base line). Width halfway up in the anchored frame (back base and tip fixed, fin height 1).
21 train + 3 val days.     python view_label.py"""
import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import straighten as ST  # noqa: E402
import bench  # noqa: E402


def width(c):
    lead = ST.sample(c['top'], 0, np.arange(-2.5, 0.05, 0.01))
    ug = np.arange(-2.5, 0.05, 0.01)
    # leading edge at v = -0.5: the top edge crossing v = -0.5 in front of the tip
    ok = ~np.isnan(lead)
    xs = ug[ok][np.where(np.diff(np.sign(lead[ok] + 0.5)) != 0)[0]] if ok.sum() > 2 else []
    tr = ST.sample(c['trail'], 1, np.array([-0.5]))
    if not len(xs) or np.isnan(tr[0]):
        return np.nan
    return float(tr[0] - xs[-1])


def main():
    rows = []
    for d in bench.TRAIN + bench.VAL:
        C = ST.load(d)
        for a, b, same in bench.pairs(d, 'dorsal'):
            if not same or a not in C or b not in C:
                continue
            for s in ('L', 'R'):
                if s in C[d if False else a] and s in C[b]:
                    A, B = C[a][s], C[b][s]
                    wa, wb = width(C[a][s + '_bt']), width(C[b][s + '_bt'])
                    if np.isnan(wa) or np.isnan(wb) or abs(np.log(wa / wb)) < 0.1:
                        continue
                    rows.append((wa > wb, A['prof'], B['prof'], A['bperp'], B['bperp'], A['n'] >= 2 and B['n'] >= 2, d in bench.VAL))
    R = np.array(rows, float)
    wider_a = R[:, 0] > 0
    out = {'pairs': len(R)}
    for name, i, j in (('profile', 1, 2), ('bperp', 3, 4)):
        ok = ~np.isnan(R[:, i]) & ~np.isnan(R[:, j]) & (R[:, i] != R[:, j])
        says_a = R[ok, i] > R[ok, j]
        out[name] = dict(n=int(ok.sum()), agree=round(float(np.mean(says_a == wider_a[ok])), 3))
        big = ok & (np.abs(R[:, i] - R[:, j]) >= (0.3 if name == 'profile' else 0.8))
        out[name + '_big_gap'] = dict(n=int(big.sum()), agree=round(float(np.mean((R[big, i] > R[big, j]) == wider_a[big])), 3))
        v = ok & (R[:, 6] > 0)
        out[name + '_val'] = dict(n=int(v.sum()), agree=round(float(np.mean((R[v, i] > R[v, j]) == wider_a[v])), 3))
    # the label rule: one 'side' (profile >= 0.85) against one 'angled' (profile < 0.5)
    for lo, hi in ((0.5, 0.85), (0.4, 0.9), (0.6, 0.85)):
        m = ((R[:, 1] >= hi) & (R[:, 2] < lo)) | ((R[:, 2] >= hi) & (R[:, 1] < lo))
        says_a = R[m, 1] > R[m, 2]
        out[f'rule_side>={hi}_angled<{lo}'] = dict(n=int(m.sum()), agree=round(float(np.mean(says_a == wider_a[m])), 3))
    print(json.dumps(out, indent=1))
    json.dump(out, open(HERE / 'view_label.json', 'w'), indent=1)


if __name__ == '__main__':
    main()


def rules():
    """the label rule (profile and bperp together) on the same-animal pairs: 'side' vs 'angled' labelled pairs."""
    import itertools
    rows = []
    for d in bench.TRAIN + bench.VAL:
        C = ST.load(d)
        for a, b, same in bench.pairs(d, 'dorsal'):
            if not same or a not in C or b not in C:
                continue
            for s in ('L', 'R'):
                if s in C[a] and s in C[b]:
                    wa, wb = width(C[a][s + '_bt']), width(C[b][s + '_bt'])
                    if np.isnan(wa) or np.isnan(wb) or wa <= 0 or wb <= 0 or abs(np.log(wa / wb)) < 0.1:
                        continue
                    rows.append((wa > wb, C[a][s]['prof'], C[b][s]['prof'], C[a][s]['bperp'], C[b][s]['bperp'], d in bench.VAL))
    R = np.array(rows, float)
    res = []
    for ps, bs, pa, ba in itertools.product((0.8, 0.85, 0.9), (1.6, 2.0, 2.5), (0.5, 0.6), (1.2, 1.4, 1.6)):
        lab = lambda p, b: 'side' if (p >= ps and b >= bs) else ('angled' if (p < pa or b < ba) else '')
        la = [lab(p, b) for p, b in R[:, [1, 3]]]; lb = [lab(p, b) for p, b in R[:, [2, 4]]]
        m = np.array([{x, y} == {'side', 'angled'} for x, y in zip(la, lb)])
        if m.sum() < 30:
            continue
        a_side = np.array([x == 'side' for x in la])
        ok = (a_side == (R[:, 0] > 0))
        tr, va = m & (R[:, 5] == 0), m & (R[:, 5] > 0)
        res.append(dict(side_profile=ps, side_bperp=bs, angled_profile=pa, angled_bperp=ba, n_train=int(tr.sum()), agree_train=round(float(ok[tr].mean()), 3),
                        n_val=int(va.sum()), agree_val=round(float(ok[va].mean()), 3) if va.any() else None,
                        share_side=round(float(np.mean([x == 'side' for x in la + lb])), 3), share_angled=round(float(np.mean([x == 'angled' for x in la + lb])), 3)))
    res.sort(key=lambda r: -r['agree_train'])
    for r in res[:8]:
        print(r)
    json.dump(res, open(HERE / 'view_label_rules.json', 'w'), indent=1)
