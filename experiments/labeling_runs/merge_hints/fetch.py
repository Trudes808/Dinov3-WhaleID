"""The Photo Labeling page's own inputs for every day, read from the running server (GET only): the day payload
(photos with their catalogue animal and part bits, stacks, the proposed boundaries) and the six stack-pair matrices
(link, appearance, fluke, dorsal, body, sequence). Saved without the researcher's state. -> cache/<day>.json, .bin"""
import json, urllib.request, sys
BASE = 'http://127.0.0.1:8765'
days = [d['day'] for d in json.load(urllib.request.urlopen(f'{BASE}/api/labeling/days', timeout=120))]
for day in days:
    D = json.load(urllib.request.urlopen(f'{BASE}/api/labeling/{day}', timeout=300)); raw = urllib.request.urlopen(f'{BASE}/api/labeling/{day}/pairs.bin', timeout=300).read()
    n = len(D['stacks']); assert len(raw) == 6 * n * n, (day, len(raw), n)
    D.pop('state', None); json.dump(D, open(f'cache/{day}.json', 'w')); open(f'cache/{day}.bin', 'wb').write(raw)
    print(day, D['split'], 'photos', len(D['photos']), 'stacks', n, 'catalog', D['has_catalog'], flush=True)
