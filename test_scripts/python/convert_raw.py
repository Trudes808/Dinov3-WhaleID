"""Decode camera RAW files (Canon CR2/CR3 via LibRaw) to sRGB JPEGs for the predictors.

    .venv/bin/python test_scripts/python/convert_raw.py SRC_DIR OUT_DIR [--long-side 2560]

Writes OUT_DIR/<subfolder>__<stem>.jpg (flattened) and OUT_DIR/index.jsonl with the source
path, camera model and focal length when the EXIF reader can get them (CR2 yes, CR3 no).
The source folder is only read.
"""
from __future__ import annotations
import argparse, json, sys, time
from pathlib import Path
import numpy as np, rawpy, cv2

def exif_info(path: Path) -> dict:
    try:
        import exifread
        with open(path, "rb") as fh:
            tags = exifread.process_file(fh, details=False)
        out = {}
        for k, name in (("Image Model", "camera"), ("EXIF FocalLength", "focal_mm"), ("EXIF LensModel", "lens")):
            if k in tags:
                v = str(tags[k])
                if name == "focal_mm":
                    num = tags[k].values[0]
                    v = float(num.num) / float(num.den) if hasattr(num, "num") else float(v)
                out[name] = v
        return out
    except Exception:
        return {}

def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("src", type=Path); p.add_argument("out", type=Path)
    p.add_argument("--long-side", type=int, default=2560); p.add_argument("--quality", type=int, default=92)
    p.add_argument("--workers", type=int, default=8, help="parallel RAW decodes (the index is written in order afterwards)")
    a = p.parse_args(argv)
    files = sorted(f for f in a.src.rglob("*") if f.suffix.lower() in (".cr2", ".cr3", ".nef", ".arw", ".dng", ".raf"))
    a.out.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    jobs = []
    for f in files:
        rel = f.relative_to(a.src)
        name = "__".join(rel.with_suffix("").parts).replace(" ", "_") + ".jpg"
        jobs.append((str(f), str(a.out / name), a.long_side, a.quality))
    from multiprocessing import Pool
    done = 0
    with Pool(max(1, a.workers)) as pool:
        for ok in pool.imap_unordered(_decode, jobs, chunksize=4):
            done += 1
            if done % 100 == 0: print(f"{done}/{len(files)} {time.time()-t0:.0f}s", flush=True)
    with open(a.out / "index.jsonl", "w") as idx:
        for f, (src, dst, _, _) in zip(files, jobs):
            rel = f.relative_to(a.src)
            if not Path(dst).exists():
                continue
            rec = {"image": Path(dst).name, "source": src, "folder": str(rel.parent), **exif_info(f)}
            idx.write(json.dumps(rec) + "\n")
    print(f"done: {len(files)} files -> {a.out} in {time.time()-t0:.0f}s")
    return 0


def _decode(job) -> bool:
    src, dst, long_side, quality = job
    if Path(dst).exists():
        return True
    try:
        with rawpy.imread(src) as raw:
            rgb = raw.postprocess(use_camera_wb=True, half_size=True, output_bps=8)
        h, w = rgb.shape[:2]; s = long_side / max(h, w)
        if s < 1: rgb = cv2.resize(rgb, (round(w * s), round(h * s)), interpolation=cv2.INTER_AREA)
        cv2.imwrite(dst, cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR), [cv2.IMWRITE_JPEG_QUALITY, quality])
        return True
    except Exception as e:  # noqa: BLE001
        print("decode failed", src, str(e)[:80], flush=True)
        return False

if __name__ == "__main__":
    sys.exit(main())
