"""The label sources that existed before this experiment, per instance key -> other_labels.json
  researcher  keypoints the researcher clicked on the Pairs / Bursts pages (Whale-Review-Web/data/<day>/pair_marks.json):
              dorsal = a dorsal_fin_* point, fluke = a fluke_* point; a click belongs to the instance of the photo whose
              box holds it (the record's key is the tracked instance, the clicked part can sit on another mask
              component), else to the nearest box within 5 % of the frame. none = the "no keypoints" tick on the
              record's instance. Records are per photo pair; one photo can be in several.
  catalog     photos the researchers filed in the Lightroom collection "2025 Flukes - ALL" (and the per-photographer fluke
              collections): fluke positives chosen by hand (the best fluke shots, not every photo with a fluke)
"""
import json, sqlite3, glob, os, collections
from PIL import Image
from vc_common import *

out = {'researcher': {}, 'catalog_fluke': {}}; size = {}; P = {}; stats = collections.Counter()
def photo(day, stem):
    if day not in P: P[day] = json.load(open(f'{D}/{day}/safe/photos.json'))
    return P[day].get(stem)
for f in sorted(glob.glob('/home/sat3737/NOAA_whales/Whale-Review-Web/data/*/pair_marks.json')):
    day = f.split('/')[-2]; z = json.load(open(f)); labs = z['labels'] if isinstance(z, dict) else z
    for r in labs:
        for side in 'ab':
            key = r.get(f'key_{side}'); kp = (r.get('keypoints') or {}).get(side) or {}; nk = (r.get('no_keypoints') or {}).get(side)
            if not key: continue
            stem = key.split('~')[0]; p = photo(day, stem)
            if p is None: continue
            if kp:
                if stem not in size: size[stem] = Image.open(f'{D}/{day}/images/{stem}.jpg').size
                W, H = size[stem]
                for name, pt in kp.items():
                    part = 'dorsal' if name.startswith('dorsal_fin') else 'fluke' if name.startswith('fluke_') else None
                    if part is None: continue
                    x, y = pt['x'] * W, pt['y'] * H; best, bd = None, 1e9
                    for it in sorted(p['instances'], key=lambda it: it['key'] != key):          # the record's own instance first: it wins when its box holds the click
                        x0, y0, x1, y1 = it['bbox']; dx = max(x0 - x, 0, x - x1); dy = max(y0 - y, 0, y - y1); d = (dx * dx + dy * dy) ** 0.5
                        if d < bd: best, bd = it['key'], d
                    if best is None or bd > 0.05 * W: stats['click outside every box'] += 1; continue
                    stats['click on the record instance' if best == key else 'click on another instance'] += 1
                    e = out['researcher'].setdefault(best, {'day': day, 'dorsal': 0, 'fluke': 0, 'none': 0, 'seen': 0}); e[part] += 1
                e = out['researcher'].setdefault(key, {'day': day, 'dorsal': 0, 'fluke': 0, 'none': 0, 'seen': 0}); e['seen'] += 1   # the researcher marked this whale's points
            elif nk:
                e = out['researcher'].setdefault(key, {'day': day, 'dorsal': 0, 'fluke': 0, 'none': 0, 'seen': 0}); e['none'] += 1
db = sqlite3.connect('file:/datasets/NOAA_whales_data/Test.lrcat?mode=ro&immutable=1', uri=True)
rows = db.execute("""select f.baseName, c.name from AgLibraryCollectionImage ci join AgLibraryCollection c on c.id_local=ci.collection
    join Adobe_images i on i.id_local=ci.image join AgLibraryFile f on f.id_local=i.rootFile where c.name like '%luke%' and c.name not like '%no fluke%'""").fetchall()
names = collections.defaultdict(set)
for base, cname in rows: names[base].add(cname)
n = 0
for d in DAYS:
    photo(d, '')
    for stem, p in P[d].items():
        base = stem.split('__')[-1]
        if base in names:
            out['catalog_fluke'][stem] = {'day': d, 'collections': sorted(names[base]), 'instances': [i['key'] for i in p['instances']]}; n += 1
json.dump(out, open(f'{HERE}/other_labels.json', 'w'), indent=0)
R_ = out['researcher']; c = collections.Counter((v['day'], 'dorsal' if v['dorsal'] else '', 'fluke' if v['fluke'] else '', 'none' if v['none'] else '') for v in R_.values())
print('researcher instances:', len(R_), dict(stats)); [print('  ', k, v) for k, v in sorted(c.items())]
print('catalog fluke photos:', n, 'among the processed days')
