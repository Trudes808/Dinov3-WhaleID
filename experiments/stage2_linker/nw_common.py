"""Non-whale instance filter: shared loading. An instance is one mask component of a photo (cluster/parts.py); on
wide-angle frames people on the bow, the deck and camera gear get a whale mask and are tracked as whales.
Labels by eye in nw_labels.json: {day: {"pos": [instance keys that are not a whale], "neg": [whales], "skip": [unclear]}}."""
import json, os
import numpy as np
R = '/home/sat3737/NOAA_whales/Dinov3-WhaleID'; D = f'{R}/experiments/days_all'
EVAL = ["2024-12-18", "2024-12-26", "2025-02-19", "2025-03-21"]; VAL = ["2025-01-15", "2025-02-27", "2025-03-14"]
DAYS = sorted(d for d in os.listdir(D) if os.path.exists(f'{D}/{d}/safe/photos.json'))
TRAIN = [d for d in DAYS if d not in EVAL and d not in VAL]
def load_day(day):
    z = np.load(f'{D}/{day}/safe/inst_feat.npz'); keys = [str(k) for k in z['keys']]
    P = json.load(open(f'{D}/{day}/safe/photos.json')); C = json.load(open(f'{D}/{day}/safe/clusters.json'))['clusters']
    cl = {k: c['id'] for c in C for k in c['keys']}; meta = {}
    for stem, p in P.items():
        for j, it in enumerate(p['instances']):
            x0, y0, x1, y1 = it['bbox']
            meta[it['key']] = dict(focal=p.get('focal_mm') or np.nan, w=x1 - x0, h=y1 - y0, asp=(y1 - y0) / max(x1 - x0, 1.0), rank=j, n=len(p['instances']), L=(it.get('lab_whale') or [np.nan])[0], bbox=it['bbox'], cl=cl.get(it['key']), parts='+'.join(it.get('parts', [])))
    def unit(a):
        a = a.astype(np.float32); return a / np.maximum(np.linalg.norm(a, axis=1, keepdims=True), 1e-6)
    return dict(day=day, keys=keys, X=np.hstack([unit(z['cls']), unit(z['mean'])]), meta=[meta[k] for k in keys])
def load_labels():
    f = 'nw_labels.json'
    return json.load(open(f)) if os.path.exists(f) else {}
def save_labels(L):
    json.dump(L, open('nw_labels.json', 'w'), indent=1)
