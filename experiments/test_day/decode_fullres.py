"""Full-resolution JPEGs and an EXIF table of an unsorted test day (read-only on the RAW folder).

    .venv/bin/python experiments/test_day/decode_fullres.py --day-dir OUT/day --out OUT [--workers 6]

Reads DAY/images/index.jsonl (written by convert_raw.py through process_day_v2.sh, so the frame order and the
2560-px working copies are the pipeline's own), decodes every RAW at the camera's resolution the way the review app's
"full-res whale view" does (rawpy, camera white balance, 8 bit) and writes OUT/fullres/<stem>.jpg (q 90) and
OUT/exif.tsv: seq (000..), frame counter, time, subsecond, lens, focal length, exposure, f-number, ISO, size.
Photo file names stay in the untracked output folder; nothing here names a photo.
"""
from __future__ import annotations

import argparse
import datetime
import json
from multiprocessing import get_context
from pathlib import Path


def exif(src: str) -> dict:
    import exifread
    with open(src, "rb") as fh:
        t = exifread.process_file(fh, details=False)
    g = lambda k: str(t[k]) if k in t else ""
    dto = g("EXIF DateTimeOriginal"); sub = g("EXIF SubSecTimeOriginal").strip() or "0"
    ts = datetime.datetime.strptime(dto, "%Y:%m:%d %H:%M:%S").timestamp() + float("0." + sub) if dto else None
    return {"datetime": dto, "subsec": sub, "time": ts, "camera": g("Image Model"), "lens": g("EXIF LensModel"),
            "focal": g("EXIF FocalLength"), "exposure": g("EXIF ExposureTime"), "fnum": g("EXIF FNumber"), "iso": g("EXIF ISOSpeedRatings")}


def decode(job):
    src, dst = job
    import cv2, rawpy
    if not Path(dst).exists():
        with rawpy.imread(src) as raw:
            rgb = raw.postprocess(use_camera_wb=True, half_size=False, output_bps=8)
        cv2.imwrite(dst + ".tmp.jpg", cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR), [cv2.IMWRITE_JPEG_QUALITY, 90])
        Path(dst + ".tmp.jpg").replace(dst)
    import cv2
    im = cv2.imread(dst, cv2.IMREAD_REDUCED_GRAYSCALE_8)
    return src, exif(src), (im.shape[1] * 8, im.shape[0] * 8)


def main() -> None:
    p = argparse.ArgumentParser(); p.add_argument("--day-dir", type=Path, required=True); p.add_argument("--out", type=Path, required=True)
    p.add_argument("--workers", type=int, default=6); a = p.parse_args()
    rows = [json.loads(l) for l in open(a.day_dir / "images" / "index.jsonl")]
    (a.out / "fullres").mkdir(parents=True, exist_ok=True)
    jobs = [(r["source"], str(a.out / "fullres" / r["image"])) for r in rows]
    with get_context("spawn").Pool(a.workers) as pool:
        res = {s: (e, wh) for s, e, wh in pool.imap_unordered(decode, jobs)}
    with open(a.out / "exif.tsv", "w") as f:
        f.write("seq\tframe\tstem\tdatetime\tsubsec\ttime\tcamera\tlens\tfocal\texposure\tfnum\tiso\tfull_w\tfull_h\n")
        for r in rows:
            e, (w, h) = res[r["source"]]; stem = Path(r["image"]).stem; parts = stem.split("_")
            f.write("\t".join(map(str, [parts[-2], parts[-1], stem, e["datetime"], e["subsec"], e["time"], e["camera"], e["lens"], e["focal"], e["exposure"], e["fnum"], e["iso"], w, h])) + "\n")
    print(len(rows), "photos ->", a.out)


if __name__ == "__main__":
    main()
