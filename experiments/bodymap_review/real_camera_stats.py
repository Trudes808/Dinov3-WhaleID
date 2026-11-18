"""Read-only: the cameras, lenses, focal lengths and times of day of the real photos (training + validation days), and the
sun's height over Maui at those times, as targets for the next render run. From DAY/timestamps.json and images/index.jsonl."""
import json, math, sys
from collections import Counter
from datetime import datetime
from pathlib import Path
import numpy as np
ROOT = Path(__file__).resolve().parents[2]; D = ROOT / "experiments" / "days_all"; split = json.loads((ROOT / "model_fine_tuning/splits/days.json").read_text())
LAT, LON, TZ = 20.8, -156.6, -10.0                                 # Maui Nui waters, Hawaii standard time


def sun_elevation(dt: datetime) -> tuple[float, float]:
    """Elevation and azimuth (deg) from the NOAA low-accuracy formulas."""
    doy = dt.timetuple().tm_yday; hour = dt.hour + dt.minute / 60 + dt.second / 3600; g = 2 * math.pi / 365 * (doy - 1 + (hour - 12) / 24)
    eqt = 229.18 * (0.000075 + 0.001868 * math.cos(g) - 0.032077 * math.sin(g) - 0.014615 * math.cos(2 * g) - 0.040849 * math.sin(2 * g))
    dec = 0.006918 - 0.399912 * math.cos(g) + 0.070257 * math.sin(g) - 0.006758 * math.cos(2 * g) + 0.000907 * math.sin(2 * g) - 0.002697 * math.cos(3 * g) + 0.00148 * math.sin(3 * g)
    tst = hour * 60 + eqt + 4 * LON - 60 * TZ; ha = math.radians(tst / 4 - 180); lat = math.radians(LAT)
    cz = math.sin(lat) * math.sin(dec) + math.cos(lat) * math.cos(dec) * math.cos(ha); el = 90 - math.degrees(math.acos(max(-1, min(1, cz))))
    az = (math.degrees(math.atan2(math.sin(ha), math.cos(ha) * math.sin(lat) - math.tan(dec) * math.cos(lat))) + 180) % 360
    return el, az


foc, hours, els, cams, lens = [], [], [], Counter(), Counter()
for day in split["train"] + split["val"]:
    ts = json.loads((D / day / "timestamps.json").read_text()); idx = {Path(json.loads(l)["image"]).stem: json.loads(l) for l in open(D / day / "images" / "index.jsonl")}
    for stem, t in ts.items():
        r = idx.get(stem, {}); cams[r.get("camera") or t.get("camera") or "?"] += 1; lens[r.get("lens") or "?"] += 1
        f = r.get("focal_mm")
        if f: foc.append(float(f))
        if t.get("datetime"):
            try:
                dt = datetime.strptime(t["datetime"], "%Y:%m:%d %H:%M:%S"); hours.append(dt.hour + dt.minute / 60); els.append(sun_elevation(dt)[0])
            except ValueError:
                pass
q = lambda v, p=(5, 25, 50, 75, 95): " / ".join(f"{x:.0f}" for x in np.percentile(v, p))
print(len(foc), "photos with a focal length |", len(hours), "with a time")
print("cameras:", dict(cams.most_common(6))); print("lenses:", dict(lens.most_common(6)))
print("focal length mm (5 / 25 / 50 / 75 / 95 pct):", q(foc), "| share at 300 mm or more:", f"{np.mean(np.array(foc) >= 300):.0%}", "| under 100 mm:", f"{np.mean(np.array(foc) < 100):.0%}")
print("time of day, hour (5 / 25 / 50 / 75 / 95 pct):", " / ".join(f"{x:.1f}" for x in np.percentile(hours, (5, 25, 50, 75, 95))))
print("sun elevation at the photos, deg (5 / 25 / 50 / 75 / 95 pct):", q(els), "| under 20 deg:", f"{np.mean(np.array(els) < 20):.0%}", "| over 40 deg:", f"{np.mean(np.array(els) > 40):.0%}")
