"""Random instances for a labelling round: vc_sample.py NAME PER_DAY SEED [split,...] -> sheets/NAME_keys.json (unlabelled only)."""
import sys, json, random
from vc_common import *
name, per, seed = sys.argv[1], int(sys.argv[2]), int(sys.argv[3]); splits = sys.argv[4].split(',') if len(sys.argv) > 4 else ['train', 'val', 'eval']
L = load_labels(); rng = random.Random(seed); out = []
for d in DAYS:
    if split_of(d) not in splits: continue
    ks = [k for k in load_day(d, feats=False)['keys'] if k not in L]; rng.shuffle(ks); out += [[d, k] for k in ks[:per]]
rng.shuffle(out); json.dump(out, open(f'{HERE}/sheets/{name}_keys.json', 'w')); print(len(out), 'instances')
