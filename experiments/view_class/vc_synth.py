"""Synthetic shots (Whale-Image-Data-Factory run04 + run05) for the view-class experiment: exact labels from the render
metadata and the same frozen features as the real instances.
    python vc_synth.py            -> feats/synth.npz (keys run/id, y [fluke, dorsal] 1 / 0 / -1, cat, pec, features)
Labels (the main whale of the shot; its visible-mask box is the instance box):
  fluke   1: a fluke tip is above the water, in frame and unoccluded, and the blades hold >= 4 % of the whale's visible
             pixels (>= 300 px); 0: no fluke tip or notch visible and < 0.5 % of the pixels; else -1 (no loss)
  dorsal  1: the fin tip and one of its base points are visible and the fin stands >= 2 % of the box's long side
             (>= 6 px) above its base line; 0: the tip is under water, out of frame or hidden and no base point shows;
             else -1"""
import json, sys, time
from collections import defaultdict, Counter
from pathlib import Path
import numpy as np, torch
from PIL import Image
sys.path.insert(0, str(Path(__file__).resolve().parents[2])); sys.path.insert(0, str(Path(__file__).resolve().parent))
from model_fine_tuning.common.backbone import load_backbone
from vc_feats import cut, shaped, BUCKETS, OUT

RENDERS = Path('/home/sat3737/NOAA_whales/Whale-Image-Data-Factory/data/renders')


def label(m):
    v = m.get('visible') or {}; bb = v.get('bbox_xyxy'); px = v.get('visible_pixels', 0)
    if not bb or px < 1500 or max(bb[2] - bb[0], bb[3] - bb[1]) < 60: return None
    kp = m['keypoints']; vis = lambda n: kp.get(n, {}).get('state') == 'visible'; st = lambda n: kp.get(n, {}).get('state')
    fl = m['parts']['fluke']; fpx = fl['amodal_pixels'] * fl['visible_fraction']
    tip = vis('fluke_tip_left') or vis('fluke_tip_right')
    f = 1 if tip and fpx >= max(300, 0.04 * px) else 0 if (not tip and not vis('fluke_notch') and fpx < max(0.005 * px, 80)) else -1
    d = -1
    if vis('dorsal_fin_tip') and (vis('dorsal_fin_front') or vis('dorsal_fin_back')):
        t = np.array(kp['dorsal_fin_tip']['uv']); base = [np.array(kp[n]['uv']) for n in ('dorsal_fin_front', 'dorsal_fin_back') if vis(n)]
        hgt = float(np.linalg.norm(t - np.mean(base, 0))) if len(base) == 1 else float(abs((base[1] - base[0])[0] * (t - base[0])[1] - (base[1] - base[0])[1] * (t - base[0])[0]) / max(np.linalg.norm(base[1] - base[0]), 1e-6))
        d = 1 if hgt >= max(6.0, 0.02 * max(bb[2] - bb[0], bb[3] - bb[1])) else -1
    elif st('dorsal_fin_tip') in ('under_water', 'out_of_frame', 'hidden') and not vis('dorsal_fin_front') and not vis('dorsal_fin_back'):
        d = 0
    pec = any(vis(n) for n in ('pectoral_tip_left', 'pectoral_tip_right'))
    return dict(bbox=[float(x) for x in bb], y=[f, d], pec=pec, cat=m['category'], px=px)


class Shots(torch.utils.data.Dataset):
    def __init__(self, rows): self.rows = rows
    def __len__(self): return len(self.rows)
    def __getitem__(self, j):
        r = self.rows[j]; im = Image.open(r['path']).convert('RGB'); b = r['bbox']
        cx, cy = (b[0] + b[2]) / 2, (b[1] + b[3]) / 2; s = max(max(b[2] - b[0], b[3] - b[1]) * 1.15, 64.0); k, w, h = shaped(b)
        return j, cut(im, cx, cy, s, s, 224, 224), cut(im, cx, cy, s, s, 448, 448), k, cut(im, cx, cy, w, h, BUCKETS[k][1], BUCKETS[k][0])


if __name__ == '__main__':
    rows = []
    for run in ('run04', 'run05'):
        for line in open(RENDERS / run / 'dataset' / 'manifest.jsonl'):
            r = json.loads(line)
            if r['negative']: continue
            m = json.load(open(RENDERS / run / 'dataset' / 'meta' / f"{r['id']}.json")); L = label(m)
            if L: rows.append(dict(L, key=f"{run}/{r['id']}", path=str(RENDERS / run / 'dataset' / r['image'])))
    print(len(rows), 'synthetic shots;', 'fluke', dict(Counter(r['y'][0] for r in rows)), 'dorsal', dict(Counter(r['y'][1] for r in rows)), '| neither', sum(1 for r in rows if r['y'] == [0, 0]), '| pec visible', sum(r['pec'] for r in rows), flush=True)
    print({c: dict(Counter(tuple(r['y']) for r in rows if r['cat'] == c).most_common(4)) for c in sorted({r['cat'] for r in rows})}, flush=True)
    if '--labels-only' in sys.argv: raise SystemExit
    model = load_backbone('vitb16').cuda().eval().half(); n = len(rows); t = time.time()
    F = {k: np.zeros((n, 768), np.float16) for k in ('b_cls', 'b_mean', 'sq_cls', 'sq_mean', 'sq_max', 'sq_mid', 'ar_cls', 'ar_mean', 'ar_max')}
    dl = torch.utils.data.DataLoader(Shots(rows), batch_size=16, num_workers=10, collate_fn=lambda b: b); pend = defaultdict(list)

    def flush(k, force=False):
        while pend[k] and (force or len(pend[k]) >= 48):
            chunk, pend[k] = pend[k][:48], pend[k][48:]
            x = torch.from_numpy(np.stack([c[1] for c in chunk])).cuda().half(); idx = [c[0] for c in chunk]
            with torch.no_grad():
                (p8, _), (pt, cl) = model.get_intermediate_layers(x, n=[8, 11], reshape=False, return_class_token=True, norm=True)
            pre = {'b': 'b', 'sq': 'sq'}.get(k, 'ar')
            F[pre + '_cls'][idx] = cl.float().cpu().numpy(); F[pre + '_mean'][idx] = pt.float().mean(1).cpu().numpy()
            if pre != 'b': F[pre + '_max'][idx] = pt.float().amax(1).cpu().numpy()
            if pre == 'sq': F['sq_mid'][idx] = p8.float().mean(1).cpu().numpy()
    for batch in dl:
        for j, b224, sq, k, ar in batch:
            pend['b'].append((j, b224)); pend['sq'].append((j, sq)); pend[k].append((j, ar))
        for k in list(pend): flush(k)
    for k in list(pend): flush(k, True)
    np.savez(OUT / 'synth.tmp.npz', keys=np.array([r['key'] for r in rows]), y=np.array([r['y'] for r in rows], np.int8), cat=np.array([r['cat'] for r in rows]), pec=np.array([r['pec'] for r in rows]),
             bbox=np.array([r['bbox'] for r in rows], np.float32), **F)
    (OUT / 'synth.tmp.npz').replace(OUT / 'synth.npz'); print(f'done in {time.time() - t:.0f} s')
