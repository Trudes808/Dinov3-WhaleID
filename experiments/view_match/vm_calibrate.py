"""Calibrate the blend of the two part similarities (student part cosine + view-specific embedding) -> view_match.pkl "mix"
    python vm_calibrate.py [--write]
The page shows each part's similarity as P(same whale) and its thresholds (merge hints, "lead") read that scale, so the
blend must stay a probability. Per stack pair the two calibrated logits are averaged (where both exist); at the
stack-animal level (best pair of the stack against the animal's stacks, as rank() takes it) a temperature and a shift
are fitted on the VAL days, where both sources are out of sample: P = sigmoid(T x mean logit + c).
Reads the production embeddings (DAY/safe/view_match.npz) and the page's inputs (inputs_gated)."""
import sys, pickle, warnings
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from vm_common import *
warnings.filterwarnings('ignore')
MODEL = f'{R}/experiments/safe_clusters/view_match.pkl'; M = pickle.load(open(MODEL, 'rb'))
PAGE = json.load(open('/home/sat3737/NOAA_whales/Whale-Review-Web/backend/labeling_model.json'))['blend']
sig = lambda z: 1 / (1 + np.exp(-z))


def day_rows(day, part):
    d = load_day(day); z = np.load(f'{DAYS_ROOT}/{day}/safe/view_match.npz'); keys = [str(k) for k in z[f'{part}|keys']]; E = z[f'{part}|E'].astype(np.float32); kid = {k: i for i, k in enumerate(keys)}
    a, b = M['parts'][part]['cal']; bl = PAGE['fluke' if part == 'fluke' else 'dorsal']; student = (d.F if part == 'fluke' else d.Dn).astype(np.float32)
    idx = [[kid[k] for k in d.keys[s] if k in kid] for s in range(d.n)]; S = np.full((d.n, d.n), np.nan, np.float32); has = [s for s in range(d.n) if idx[s] and d.kind[s] != 'extra']
    for s in has:
        for t in has:
            if s < t: S[s, t] = S[t, s] = float((E[idx[s]] @ E[idx[t]].T).max())
    ln = a * S + b; ls = np.where(student > 0, bl['coef'] * (student - 1) / 254 + bl['intercept'], np.nan); out = []
    for g, cs in d.scope.items():
        for c in cs:
            for s in d.animals[c]:
                for c2 in cs:
                    T = [t for t in d.animals[c2] if t != s and (s, t) not in d.conf]
                    if not T: continue
                    n_, s_ = ln[s, T], ls[s, T]; both = np.isfinite(n_) & np.isfinite(s_)
                    mean = np.where(both, (n_ + s_) / 2, np.nan)
                    out.append((c == c2, np.nanmax(mean) if both.any() else np.nan, np.nanmax(n_) if np.isfinite(n_).any() else np.nan, np.nanmax(s_) if np.isfinite(s_).any() else np.nan))
    return out


mix = {}
for part in ('fluke', 'dorsal'):
    rows = {day: day_rows(day, part) for day in TRAIN + VAL}; A = np.array([r for day in TRAIN + VAL for r in rows[day]], float); isval = np.concatenate([[day in VAL] * len(rows[day]) for day in TRAIN + VAL]).astype(bool)
    y, mean, new, stu = A[:, 0], A[:, 1], A[:, 2], A[:, 3]; b = np.isfinite(mean)
    lr = LogisticRegression(C=100, max_iter=500).fit(mean[b & isval, None], y[b & isval]); T, c = float(lr.coef_[0, 0]), float(lr.intercept_[0]); mix[part] = (T, c)
    print(f'{part}: stack-animal rows with both sources: val {int((b & isval).sum())}, train {int((b & ~isval).sum())}; fitted on val: P = sigmoid({T:.3f} x mean logit {c:+.3f})')
    for tag, m in (('val', b & isval), ('train (student in-sample)', b & ~isval)):
        p_mean = sig(mean[m]); p_mix = sig(T * mean[m] + c); p_new = sig(new[m]); p_stu = sig(stu[m])
        print(f'   {tag:28s} same {y[m].mean():.3f} | mean predicted: plain mean of logits {p_mean.mean():.3f}, mixed {p_mix.mean():.3f}, new alone {p_new.mean():.3f}, student alone {p_stu.mean():.3f} | AUC mean {roc_auc_score(y[m], mean[m]):.3f} new {roc_auc_score(y[m], new[m]):.3f} student {roc_auc_score(y[m], stu[m]):.3f}')
        for lo, hi in ((0, 0.1), (0.1, 0.3), (0.3, 0.5), (0.5, 0.7), (0.7, 1.01)):
            k = (p_mix >= lo) & (p_mix < hi)
            if k.sum() >= 15: print(f'      mixed P {lo:.1f}-{hi:.1f}: {int(k.sum()):5d} rows, same {y[m][k].mean():.2f}')
    only = np.isfinite(new) & ~b
    if only.sum() > 30: print(f'   rows the student cannot score ({int(only.sum())}): same {y[only].mean():.3f}, new alone predicts {sig(new[only]).mean():.3f}, AUC {roc_auc_score(y[only], new[only]):.3f}')
if '--write' in sys.argv:
    for part in mix: M['parts'][part]['mix'] = mix[part]
    pickle.dump(M, open(MODEL, 'wb')); print('written to', MODEL)
