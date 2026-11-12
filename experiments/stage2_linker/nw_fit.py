"""Fit the non-whale probe on the train days' by-eye labels, score every train and val instance, write contact sheets of
the unlabelled train instances the probe is least sure are whales (for the next round of labels by eye).
usage: nw_fit.py ROUND [n_sheet_tiles]     -> nw_scores_ROUND.pkl, sheets/nw_rROUND_k.jpg + sheets/nw_rROUND.json (tile -> day, key)"""
import sys, json, pickle, os
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from PIL import Image, ImageDraw
from nw_common import *
rnd = sys.argv[1]; NT = int(sys.argv[2]) if len(sys.argv) > 2 else 144
L = load_labels(); data = {d: load_day(d) for d in TRAIN + VAL}
def split(days):
    X, y, ref = [], [], []
    for d in days:
        dd = data[d]; lab = L.get(d, {}); pos = set(lab.get('pos', [])); neg = set(lab.get('neg', [])); skip = set(lab.get('skip', []))
        for i, k in enumerate(dd['keys']):
            m = dd['meta'][i]
            t = 1 if k in pos else 0 if (k in neg or (m['focal'] >= 100 and k not in skip)) else None       # unlabelled telephoto instances count as whales
            if t is not None: X.append(dd['X'][i]); y.append(t); ref.append((d, k))
    return np.array(X), np.array(y), ref
Xt, yt, _ = split(TRAIN); print(f'round {rnd}: training on {len(yt)} instances, {int(yt.sum())} labelled not-a-whale, from {sum(1 for d in TRAIN if L.get(d, {}).get("pos"))} days with positives')
sc = StandardScaler().fit(Xt); clf = LogisticRegression(C=0.01, class_weight='balanced', max_iter=2000).fit(sc.transform(Xt), yt)
S = {d: clf.decision_function(sc.transform(data[d]['X'])) for d in data}
pickle.dump(dict(scores={d: dict(zip(data[d]['keys'], S[d].tolist())) for d in data}, mean=sc.mean_, scale=sc.scale_, coef=clf.coef_[0], intercept=float(clf.intercept_[0])), open(f'nw_scores_{rnd}.pkl', 'wb'))
# val check against the by-eye labels of the val days
for d in VAL:
    lab = L.get(d, {}); pos = set(lab.get('pos', [])); s = S[d]; k = data[d]['keys']; isp = np.array([x in pos for x in k])
    line = f'  val {d}: instances {len(k)}, labelled not-a-whale {int(isp.sum())}'
    for th in (0.0, 2.0, 4.0):
        f = s > th; line += f' | score > {th:.0f}: flagged {int(f.sum())}, of them labelled {int((f & isp).sum())}'
    if isp.any(): line += f' | lowest labelled score {s[isp].min():.2f}, highest unlabelled {s[~isp].max():.2f}'
    print(line)
# sheets: unlabelled train instances, highest score first
cand = []
for d in TRAIN:
    lab = L.get(d, {}); seen = set(lab.get('pos', [])) | set(lab.get('neg', [])) | set(lab.get('skip', []))
    for i, k in enumerate(data[d]['keys']):
        if k not in seen: cand.append((float(S[d][i]), d, k, data[d]['meta'][i]))
cand.sort(key=lambda t: -t[0]); cand = cand[:NT]
print('  score range on the sheets: %.2f .. %.2f; unlabelled train instances with score > 0: %d' % (cand[0][0], cand[-1][0], sum(1 for d in TRAIN for i, k in enumerate(data[d]['keys']) if S[d][i] > 0 and k not in (set(L.get(d, {}).get('pos', [])) | set(L.get(d, {}).get('neg', [])) | set(L.get(d, {}).get('skip', []))))))
os.makedirs('sheets', exist_ok=True); T = 250; cols = 8; per = cols * 6; index = {}
for s0 in range(0, len(cand), per):
    part = cand[s0:s0 + per]; rows = (len(part) + cols - 1) // cols; sheet = Image.new('RGB', (cols * (T + 4), rows * (T + 20)), (20, 20, 20))
    for i, (sco, d, k, m) in enumerate(part):
        n = s0 + i; index[n] = (d, k); stem = k.split('~')[0]; im = Image.open(f'{D}/{d}/images/{stem}.jpg').convert('RGB')
        x0, y0, x1, y1 = m['bbox']; side = max(max(x1 - x0, y1 - y0) * 2.2, 420); cx, cy = (x0 + x1) / 2, (y0 + y1) / 2
        ImageDraw.Draw(im).rectangle((x0, y0, x1, y1), outline=(255, 220, 0), width=max(3, int(side / 120)))
        c = im.crop((int(cx - side / 2), int(cy - side / 2), int(cx + side / 2), int(cy + side / 2))).resize((T, T), Image.LANCZOS)
        t = Image.new('RGB', (T, T + 16), (20, 20, 20)); t.paste(c, (0, 16)); ImageDraw.Draw(t).text((2, 2), f"{n}  s{sco:.1f}  {m['focal']:.0f}mm  {d[5:]}", fill=(255, 255, 255))
        sheet.paste(t, ((i % cols) * (T + 4) + 2, (i // cols) * (T + 20) + 2))
    fn = f'sheets/nw_r{rnd}_{s0 // per + 1}.jpg'; sheet.save(fn, quality=82); print('  saved', fn, sheet.size, len(part))
json.dump(index, open(f'sheets/nw_r{rnd}.json', 'w'))
