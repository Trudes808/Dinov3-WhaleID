"""CPU: cut the training crops (dorsal instances with a catalogue animal) of the 16 pool days of bench.splits()[0].
-> CACHE/crops/<day>/<n>.jpg (TRAIN_PX, quality 95) and CACHE/crops/<day>/index.json [(n, key, cluster, animal)]
Test days are never cut here.      OMP_NUM_THREADS=8 .venv/bin/python experiments/moonshot_sizing/fin_tune/prep_crops.py
"""
import json
from concurrent.futures import ProcessPoolExecutor

from common import CACHE, CTX, TRAIN_PX, bench, crop, split


def one(args):
    day, n, key = args
    p = CACHE / 'crops' / day / f'{n}.jpg'
    if not p.exists():
        crop(day, key, CTX, TRAIN_PX).save(p, quality=95)
    return n


def main():
    hold, pool, test = split()
    assert not set(pool) & set(test)
    jobs = []
    for day in pool:
        (CACHE / 'crops' / day).mkdir(parents=True, exist_ok=True)
        idx = [(n, k, c, a) for n, (k, c, a) in enumerate(x for x in bench.instances(day, 'dorsal') if x[2] and x[1] is not None)]
        json.dump(idx, open(CACHE / 'crops' / day / 'index.json', 'w'))
        jobs += [(day, n, k) for n, k, c, a in idx]
    with ProcessPoolExecutor(8) as ex:
        list(ex.map(one, jobs, chunksize=8))
    print('crops', len(jobs))


if __name__ == '__main__':
    main()
