"""Fill the matrix cache for one variant over every day whose tokens exist (waits for missing days)."""
import sys
import time

import common  # noqa: F401  (puts the bench on sys.path)
import bench
import run_pairs
from common import CACHE

part, var = sys.argv[1], sys.argv[2]
todo = list(bench.TRAIN + bench.VAL)
while todo:
    for d in list(todo):
        if (CACHE / part / f'{d}_info.json').exists():
            t = time.time(); run_pairs.matrix(part, d, var); todo.remove(d)
            print(var, d, f'{time.time() - t:.0f} s', flush=True)
    if todo:
        time.sleep(30)
print('done', var, flush=True)
