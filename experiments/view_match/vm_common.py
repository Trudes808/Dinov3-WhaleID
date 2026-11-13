"""View-aware matching experiments: shared loading.

inputs_gated/cache, inputs_ungated/cache = the Photo Labeling page's own inputs for every day (payload + the six
stack-pair matrices), fetched from a test server with the view classifier on (fluke / dorsal similarities gated by the
classifier / not gated); the chart-flag inputs of the same days are in ../labeling_runs/merge_hints/cache.
Day (from ../labeling_runs/merge_hints/sim.py) rebuilds the page's scores; here it also carries, per stack, the
instance keys, the view classifier's probabilities and the frozen view features of its instances."""
import json, os, sys, math, collections
import numpy as np

R = '/home/sat3737/NOAA_whales/Dinov3-WhaleID'; HERE = f'{R}/experiments/view_match'; MH = f'{R}/experiments/labeling_runs/merge_hints'; DAYS_ROOT = f'{R}/experiments/days_all'
sys.path.insert(0, MH)
SPLIT = json.load(open(f'{R}/model_fine_tuning/splits/days.json')); EVAL, VAL, TRAIN = SPLIT['eval'], SPLIT['val'], SPLIT['train']


def load_day(day, inputs='inputs_gated'):
    """sim.Day on the given inputs + keys / view probabilities per stack. Day reads cache/<day>.* relative to the cwd."""
    import sim
    cwd = os.getcwd(); os.chdir(f'{HERE}/{inputs}' if not inputs.startswith('/') else inputs)
    try:
        d = sim.Day(day)
    finally:
        os.chdir(cwd)
    P = d.D['photos']; d.keys = []; d.pf = []; d.pd = []
    for x in d.D['stacks']:
        ks, f, dd = [], [], []
        for p, k in x['items']:
            if k < 0: continue
            ks.append(f"{P[p]['stem']}~{k}"); vw = (P[p].get('vw') or [None] * (k + 1))[k]
            f.append(vw[0] / 100 if vw else np.nan); dd.append(vw[1] / 100 if vw else np.nan)
        d.keys.append(ks); d.pf.append(np.array(f)); d.pd.append(np.array(dd))
    d.kind = [x['kind'] for x in d.D['stacks']]
    return d


def view_feats(day):
    z = np.load(f'{DAYS_ROOT}/{day}/safe/view_feat.npz'); X = z['X'].astype(np.float32).reshape(len(z['keys']), 4, 768)
    X = (X / np.maximum(np.linalg.norm(X, axis=2, keepdims=True), 1e-6)).reshape(len(X), -1)
    return {str(k): X[i] for i, k in enumerate(z['keys'])}


def scope_pairs(d):
    """Stack pairs inside a catalogue encounter (both with a catalogue animal, not two whales of one frame): (a, b, same animal)."""
    out = []
    for g, cs in d.scope.items():
        ids = [i for c in cs for i in d.animals[c]]
        for x in range(len(ids)):
            for y in range(x + 1, len(ids)):
                a, b = ids[x], ids[y]
                if (a, b) in d.conf: continue
                out.append((a, b, d.cat[a] == d.cat[b]))
    return out
