"""Cluster-pair tables (link_scores.pair_table) of the days given, one pickle per day in tables/. Needs the day's
safe/window_scores.npz (link_scores scores). usage: tables.py DAY [DAY ...]"""
import sys, pickle, os, time
from pathlib import Path
sys.path.insert(0, '/home/sat3737/NOAA_whales/Dinov3-WhaleID')
from model_fine_tuning.cluster.link_scores import pair_table
D = Path('/home/sat3737/NOAA_whales/Dinov3-WhaleID/experiments/days_all'); os.makedirs('tables', exist_ok=True)
for day in sys.argv[1:]:
    t = time.time(); rows = pair_table(D / day); pickle.dump(rows, open(f'tables/{day}.pkl', 'wb'))
    print(day, 'pairs', len(rows), 'catalog same / diff / ?', sum(r['cat'] == 'same' for r in rows), sum(r['cat'] == 'diff' for r in rows), sum(r['cat'] == '?' for r in rows), f'({time.time() - t:.0f} s)', flush=True)
