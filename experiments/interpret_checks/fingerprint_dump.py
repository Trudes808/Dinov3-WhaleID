"""Read-only: the fingerprints the page lays out (same cut-outs, same order), with the page's own PCA and t-SNE
positions, each point's stack and catalogue animal -> fp/<day>_<part>.npz"""
import json, sys, os, urllib.request, numpy as np
from pathlib import Path
sys.path.insert(0, '/home/sat3737/NOAA_whales/Dinov3-WhaleID')
from model_fine_tuning.cluster.safe_clusters import load_embeddings
D = Path('/home/sat3737/NOAA_whales/Dinov3-WhaleID/experiments/days_all'); os.makedirs('fp', exist_ok=True)
for day in sys.argv[1:]:
    emb = load_embeddings(D / day)
    for part in ('fin', 'fluke', 'body'):
        g = lambda lay: json.load(urllib.request.urlopen(f'http://127.0.0.1:8765/api/interpret/{day}/fingerprints?part={part}&layout={lay}', timeout=600))
        a = g('pca'); b = g('tsne'); keys = [p[0] for p in a['points']]; assert keys == [p[0] for p in b['points']]
        if len(keys) < 30: continue
        E = np.asarray([emb[part][k] for k in keys], np.float32)
        np.savez(f'fp/{day}_{part}.npz', E=E, pca=np.asarray(a['xyz'], np.float32).reshape(-1, 3), tsne=np.asarray(b['xyz'], np.float32).reshape(-1, 3), stack=np.asarray([p[1] for p in a['points']]),
                 cat=np.asarray([p[4] or '' for p in a['points']]), keys=np.asarray(keys))
        print(day, part, len(keys), flush=True)
