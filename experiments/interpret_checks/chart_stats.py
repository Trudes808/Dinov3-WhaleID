"""Read-only: how the part cut-outs ("charts") of a day were turned and framed. From each chart's saved transform M
(full-resolution photo -> chart; parts_hr charts are 1536 px, read from the jpg. Until 2026-11-16 this script
sampled a 512-px square, the top-left ninth of the chart, so its 'outside the photo' shares were wrong; angles, mirroring and zoom were not affected): the rotation applied, mirroring, zoom, and how much of the chart lies outside
the photo (black). usage: chart_stats.py DAY [DAY ...] -> stats_<day>.json"""
import sys, json, glob, os, numpy as np
from PIL import Image
D = '/home/sat3737/NOAA_whales/Dinov3-WhaleID/experiments/days_all'
def one(day):
    rows = []; sizes = {}
    for f in sorted(glob.glob(f'{D}/{day}/parts_hr/*__*.npz')):
        name = os.path.basename(f)[:-4]; key, part = name.rsplit('__', 1); stem = key.rsplit('~', 1)[0]
        try: z = np.load(f)
        except Exception: continue
        M = z['M'].astype(float); sf = float(z['scale_full'])
        if stem not in sizes:
            try: sizes[stem] = Image.open(f'{D}/{day}/images/{stem}.jpg').size
            except Exception: sizes[stem] = (2560, 1707)
        W, H = sizes[stem][0] * sf, sizes[stem][1] * sf
        ang = float(np.degrees(np.arctan2(M[0, 1], M[0, 0]))); zoom = float(np.hypot(M[0, 0], M[0, 1]))            # chart pixels per photo pixel
        cs = Image.open(f'{D}/{day}/parts_hr/charts/{name}.jpg').size[0]
        Mi = np.linalg.inv(np.vstack([M, [0, 0, 1]])); g = np.linspace(0, cs - 1, 32); xx, yy = np.meshgrid(g, g); P = Mi @ np.stack([xx.ravel(), yy.ravel(), np.ones(xx.size)])
        out = float(((P[0] < 0) | (P[0] > W) | (P[1] < 0) | (P[1] > H)).mean())
        rows.append(dict(key=key, part=part, angle=round(ang, 1), zoom=round(zoom, 3), outside=round(out, 3), mirrored=bool(z['mirrored']) if 'mirrored' in z else False,
                         flipped_v=bool(z['flipped_v']) if 'flipped_v' in z else False, pixels=int(z['pixels']), profile=float(z['profile']) if 'profile' in z else None))
    json.dump(rows, open(f'stats_{day}.json', 'w')); return rows
for day in sys.argv[1:]:
    rows = one(day); print(f'\n{day}: {len(rows)} charts')
    for part in ('body', 'fin', 'fluke'):
        r = [x for x in rows if x['part'] == part]
        if not r: continue
        a = np.abs(np.array([x['angle'] for x in r])); a = np.minimum(a, 180 - a) if part == 'fluke' else a       # a fluke is level either way up: count the tilt from level
        o = np.array([x['outside'] for x in r]); z = np.array([x['zoom'] for x in r])
        print(f"  {part:5s} n {len(r):4d} | turned by: under 10 deg {np.mean(a < 10):.2f}, 10-30 {np.mean((a >= 10) & (a < 30)):.2f}, 30-60 {np.mean((a >= 30) & (a < 60)):.2f}, 60-120 {np.mean((a >= 60) & (a < 120)):.2f}, over 120 {np.mean(a >= 120):.2f}"
              f" | mirrored {np.mean([x['mirrored'] for x in r]):.2f}" + (f", flipped upside down {np.mean([x['flipped_v'] for x in r]):.2f}" if part == 'fluke' else '')
              + f" | part of the chart outside the photo: any {np.mean(o > 0.02):.2f}, over a quarter {np.mean(o > 0.25):.2f} | enlarged over 2x {np.mean(z > 2):.2f}, over 4x {np.mean(z > 4):.2f} (median zoom {np.median(z):.2f})")
