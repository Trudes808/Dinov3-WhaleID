"""Plate index for the MODELLED sea (the offline water engine's plates), kept apart from the real-water index.

    PYTHONPATH=. .venv/bin/python -m model_fine_tuning.synth_aug.fw_plates build
    -> /var/tmp/sat3737-parts-cache/realwater/fw_plates/plate_NNNN.jpg + fw_plates_index.jsonl

Source: /var/tmp/sat3737-parts-cache/water_engine/plates/plate_NNNN.jpg (2560 x 1707) with plate_NNNN.json (horizon row
or null, focal length, sensor width, camera height, distance, pitch, ...). A plate without its JSON is skipped (still
rendering). The plates are water only below the horizon, so no hole and no fill.

Same index fields as plates.py, so composite.Compositor uses them unchanged (Compositor(opts, index=FW_INDEX)):
  * work scale: resized to 1280 px wide (frame width = 1280, as the real plates); plate = the rows from the horizon
    (max row) + 3 % of the frame height down to the bottom (the whole frame when there is no horizon)
  * the "real whale" a real plate carries is replaced by the camera's target: the point at the plate's distance_m
    on the sea, row = cy + f_px * tan(atan(h / d) - camera depression), centre column; its "length" = an assumed
    7 m of visible whale at that distance (f_px = focal_mm / sensor_width_mm x 2560), so the compositor's
    length matching pairs a pasted whale with plates whose camera distance and lens give it that size
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import cv2

from .plates import CACHE, WORK_W

SRC = Path("/var/tmp/sat3737-parts-cache/water_engine/plates")
OUT = CACHE / "fw_plates"
FW_INDEX = CACHE / "fw_plates_index.jsonl"
WHALE_M = 7.0


def plate_record(meta: dict, jpg: Path, whale_m: float = WHALE_M) -> tuple[dict, object] | None:
    img = cv2.imread(str(jpg), cv2.IMREAD_COLOR)
    if img is None:
        return None
    H0, W0 = img.shape[:2]
    s = WORK_W / W0
    Hs = int(round(H0 * s))
    small = cv2.resize(img, (WORK_W, Hs), interpolation=cv2.INTER_AREA)
    f_px = meta["focal_mm"] / meta.get("sensor_width_mm", 22.3) * W0
    hz = meta.get("horizon_row_max", meta.get("horizon_row"))
    top = 0 if hz is None else int(min(Hs - 64, math.ceil((hz * s) + 0.03 * Hs)))
    depress = math.atan2(meta["camera_height_m"], meta["distance_m"])
    pitch = math.radians(-meta.get("pitch_deg", 0.0))                  # camera depression (pitch is negative down)
    row = (H0 / 2 + f_px * math.tan(depress - pitch)) * s
    length = whale_m * f_px / meta["distance_m"] * s                   # work px
    row = min(max(row, top + 0.1 * length), Hs - 0.1 * length) - top
    plate = small[top:]
    ph, pw = plate.shape[:2]
    cx = pw / 2
    wb = [cx - length / 2, row - 0.15 * length, cx + length / 2, row + 0.15 * length]
    rec = dict(day="water_engine", stem=meta["id"], kind="modelled_sea", partner=None, camera=None, focal_mm=meta["focal_mm"],
               time=None, work_scale=s, frame=[WORK_W, Hs], rect=[0, top, pw, top + ph], size=[pw, ph], whale_box=wb,
               length_share=length / WORK_W, length_px=length, hole_box=None, hole_frac=0.0, white_frac=0.0,
               horizon_top=0 if hz is None else int(hz * s), n_instances=0, whale_m_assumed=round(whale_m, 3),
               distance_m=meta["distance_m"], camera_height_m=meta["camera_height_m"], sea_state=meta.get("sea_state"),
               sky=meta.get("sky"), sun_elevation_deg=meta.get("sun_elevation_deg"))
    return rec, plate


def build() -> int:
    import numpy as np

    OUT.mkdir(parents=True, exist_ok=True)
    metas = []
    for jpg in sorted(SRC.glob("plate_*.jpg")):
        js = jpg.with_suffix(".json")
        if not js.exists():
            continue
        try:
            metas.append((json.loads(js.read_text()), jpg))
        except json.JSONDecodeError:
            continue                                                    # JSON being written
    framing = json.loads((CACHE / "real_framing.json").read_text())
    real_med = float(np.median([f["length_share"] for f in framing]))
    px_per_m = [m["focal_mm"] / m.get("sensor_width_mm", 22.3) / m["distance_m"] for m, _ in metas]   # share of width per m
    whale_m = real_med / float(np.median(px_per_m)) if metas else WHALE_M
    print(f"visible whale length set to {whale_m:.2f} m (real median share {real_med:.3f})")
    rows = []
    for meta, jpg in metas:
        r = plate_record(meta, jpg, whale_m)
        if r is None:
            continue
        rec, plate = r
        out = OUT / f"{meta['id']}.jpg"
        cv2.imwrite(str(out), plate, [cv2.IMWRITE_JPEG_QUALITY, 93])
        rec["file"] = str(out)                                          # absolute: CACHE / abs path = abs path
        rows.append(rec)
    with open(FW_INDEX, "w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
    print("modelled-sea plates", len(rows), "->", FW_INDEX)
    return len(rows)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("cmd", choices=["build"])
    ap.parse_args(argv)
    build()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
