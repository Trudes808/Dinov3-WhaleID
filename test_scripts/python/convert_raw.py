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
    a = p.parse_args(argv)
    files = sorted(f for f in a.src.rglob("*") if f.suffix.lower() in (".cr2", ".cr3", ".nef", ".arw", ".dng", ".raf"))
    a.out.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    with open(a.out / "index.jsonl", "w") as idx:
        for i, f in enumerate(files):
            rel = f.relative_to(a.src)
            name = "__".join(rel.with_suffix("").parts).replace(" ", "_") + ".jpg"
            dst = a.out / name
            if not dst.exists():
                with rawpy.imread(str(f)) as raw:
                    rgb = raw.postprocess(use_camera_wb=True, half_size=True, output_bps=8)
                h, w = rgb.shape[:2]; s = a.long_side / max(h, w)
                if s < 1: rgb = cv2.resize(rgb, (round(w * s), round(h * s)), interpolation=cv2.INTER_AREA)
                cv2.imwrite(str(dst), cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR), [cv2.IMWRITE_JPEG_QUALITY, a.quality])
            rec = {"image": name, "source": str(f), "folder": str(rel.parent), **exif_info(f)}
            idx.write(json.dumps(rec) + "\n")
            if i % 25 == 0: print(f"{i}/{len(files)} {time.time()-t0:.0f}s {name}", flush=True)
    print(f"done: {len(files)} files -> {a.out} in {time.time()-t0:.0f}s")
    return 0

if __name__ == "__main__":
    sys.exit(main())
