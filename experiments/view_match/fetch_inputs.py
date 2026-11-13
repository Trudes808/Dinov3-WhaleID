"""Page inputs of every day from a test server -> OUT/cache/<day>.json, .bin (as labeling_runs/merge_hints/fetch.py)."""
import json, urllib.request, sys
BASE, OUT = sys.argv[1], sys.argv[2]
days = [d['day'] for d in json.load(urllib.request.urlopen(f'{BASE}/api/labeling/days', timeout=120))]
for day in days:
    D = json.load(urllib.request.urlopen(f'{BASE}/api/labeling/{day}', timeout=600)); raw = urllib.request.urlopen(f'{BASE}/api/labeling/{day}/pairs.bin', timeout=600).read()
    n = len(D['stacks']); assert len(raw) == 6 * n * n, (day, len(raw), n)
    D.pop('state', None); json.dump(D, open(f'{OUT}/cache/{day}.json', 'w')); open(f'{OUT}/cache/{day}.bin', 'wb').write(raw)
print(len(days), 'days ->', OUT, '| view model of the last day:', D.get('view_model'))
