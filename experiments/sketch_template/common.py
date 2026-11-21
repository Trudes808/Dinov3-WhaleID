"""sketch_template (2026-11-19): shared loading for the pad-style fin drawing experiments. Code only: nothing of the trip
log is written here; per-animal tables and arrays go to CACHE (outside the repository).

noted(day)     {catalogue animal id: "height form"} for a day, from the trip log (ONE switch: the trip-log importer's
               per-day file Whale-Review-Web/data/trip_log/<day>.json when it exists, else the export read by
               moonshot_cache/notes/tables.py). The log's group number = the catalogue's G number, its whale number
               = the catalogue's A number.
drawings(day)  {(cluster, side): fin_sketch view} of the clusters with a catalogue animal (bench refuses eval days)
"""
import functools
import json
import re
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / 'moonshot_sizing'))
sys.path.insert(0, str(HERE.parents[1]))
import bench  # noqa: E402
from model_fine_tuning.cluster import fin_sketch as FS  # noqa: E402

CACHE = Path('/home/sat3737/NOAA_whales/moonshot_cache/sketch_template')
TRIP_LOG = Path('/home/sat3737/NOAA_whales/Whale-Review-Web/data/trip_log')
NOTES = Path('/home/sat3737/NOAA_whales/moonshot_cache/notes')
SUB = 'sketch'
DAYS = bench.TRAIN + bench.VAL
HEIGHTS = ('low', 'moderate', 'high')
FORMS = ('triangle', 'hook', 'round', 'square', 'point', 'pyramid')


def norm(height, form):
    """'moderate hook' or None (unknown, humped, or a form outside the six)."""
    if height in HEIGHTS and form in FORMS:
        return f'{height} {form}'
    return None


def _parse(text):
    t = str(text).strip().lower().replace('miderate', 'moderate').replace('squared', 'square')
    w = re.findall(r'[a-z]+', t)
    h = next((x for x in HEIGHTS if any(v.startswith(x[:3]) for v in w)), None)
    f = next((x for x in FORMS if any(v.startswith(x[:4]) for v in w)), None)
    return norm(h, f)


@functools.lru_cache(None)
def _tables():
    sys.path.insert(0, str(NOTES))
    from tables import tables
    return tables()


@functools.lru_cache(None)
def noted(day):
    bench._ok(day)
    f = TRIP_LOG / f'{day}.json'
    out = {}
    if f.exists():
        doc = json.loads(f.read_text())
        for g in doc.get('groups', []):
            if g.get('humpback') is False:
                continue
            for w in g.get('whales', []):
                dz = w.get('dorsal') or {}
                s = norm(dz.get('height'), dz.get('form'))
                if s and g.get('n') is not None and w.get('n') is not None:
                    out[f"{day}_G{int(g['n']):02d}A{int(w['n']):02d}"] = s
        return out
    import pandas as pd
    T = _tables(); S, A = T['sighting'], T['animals']
    s = S[(S['day'] == day) & S['Common name'].fillna('').str.contains('Humpback')]
    for _, r in s.iterrows():
        for _, a in A[A['Sighting_ID'] == r['sight_id']].iterrows():
            if pd.isna(a['Dorsal fin']) or pd.isna(a['Whale #']):
                continue
            sh = _parse(a['Dorsal fin'])
            if sh:
                out[f"{day}_G{int(r['Sighting#']):02d}A{int(a['Whale #']):02d}"] = sh
    return out


@functools.lru_cache(None)
def drawings(day):
    d = FS.day_dir(day) / SUB
    idx = json.loads((d / 'index.json').read_text())['clusters']
    out = {}
    for c in bench.clusters(day, 'dorsal'):
        for s, x in (idx.get(str(c)) or {}).items():
            v = FS.load_view(d / x['npz'])
            v['meta'].update(view=x.get('view', ''), keys=x['keys'], cl=c, day=day, animal=bench.D(day).maj[c])
            out[(c, s)] = v
    return out
