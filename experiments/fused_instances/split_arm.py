"""Build a SPLIT arm's day folder (rules_fused.md A_G / A_S) from the shadow root, read-only on the source:
images, canon, cse linked; timestamps.json and safe/exif.json copied; parts_hr made of links to the source's files for
the photos the split does not touch and freshly cut charts (cluster.parts.extract_photo, unchanged code; only its
instance list is replaced) for the photos it does. The split photos' instance label maps go to parts_hr/split/<stem>.npz
(crop grid, 1..n = "<stem>~0".."~n-1"), read by flags.arm_instances. The rest of the chain: arm_chain.sh.

    PYTHONPATH=. .venv/bin/python experiments/fused_instances/split_arm.py --mode G|S --src SHADOW_DAYS --dst ARM_DAYS --day DAY
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
import time
from multiprocessing import Pool
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import fcommon as F  # noqa: E402
import flags as FL  # noqa: E402
from model_fine_tuning.cluster import parts as P  # noqa: E402

MODE = {"m": None, "day": None}
_orig_split = P.split_instances
_orig_extract = P.extract_photo
_CUR = {"stem": None, "dir": None}


def new_instances(day: str, D: Path, stem: str, mode: str):
    """(base list, new list) of crop masks, or (base, None) when nothing changes."""
    z = F.cse(D, stem)
    if z is None or z["use"].sum() < 400:
        return None, None
    base = _orig_split(z["use"], 0.1, u=z["u"])
    pieces = {}
    if mode == "G":
        comps = F.components(z["use"], 0.1, 400)
        for i, m in enumerate(base):
            fl = FL.flag_G(m, comps, z["u"])
            if fl:
                ps = FL.split_G(m, fl)
                if len(ps) > 1:
                    pieces[i] = sorted(ps, key=lambda p: -p.sum())
    else:
        sam = F.sam_in_crop(day, stem, z["x0y0s"], z["use"].shape)
        for i, m in enumerate(base):
            s = FL.flag_S(m, sam)
            if s:
                ps = FL.split_S(m, s)
                if len(ps) > 1:
                    pieces[i] = ps
    return base, (FL.apply_split(base, pieces) if pieces else None)


def _patched_split(use, min_frac=0.1, min_px=400, u=None, overlap_px=200):
    f = _CUR["dir"] / "parts_hr" / "split" / f"{_CUR['stem']}.npz"
    if f.exists():
        L = np.load(f)["lab"]
        return [L == (i + 1) for i in range(int(L.max()))]
    return _orig_split(use, min_frac, min_px, u, overlap_px)


def _patched_extract(rec, canon_dir, out, *args, **kw):
    _CUR["stem"] = Path(rec["image"]).stem
    return _orig_extract(rec, canon_dir, out, *args, **kw)


def _probe(a):
    day, D, stem, mode = a
    base, new = new_instances(day, D, stem, mode)
    if new is None:
        return stem, None
    lab = np.zeros(new[0].shape, np.uint8)
    for i, m in enumerate(new):
        lab[m & (lab == 0)] = i + 1
    return stem, lab


def main() -> None:
    ap = argparse.ArgumentParser(); ap.add_argument("--mode", choices=["G", "S"], required=True); ap.add_argument("--src", type=Path, default=F.SR)
    ap.add_argument("--dst", type=Path, required=True); ap.add_argument("--day", required=True); ap.add_argument("--workers", type=int, default=10)
    a = ap.parse_args(); F.check_day(a.day)
    S, D = a.src / a.day, a.dst / a.day
    if (D / "parts_hr" / "index.jsonl").exists():
        print("exists", D); return
    t0 = time.time()
    D.mkdir(parents=True, exist_ok=True)
    for name in ("images", "canon", "cse"):
        if not (D / name).exists():
            os.symlink(os.path.realpath(S / name), D / name)
    shutil.copy2(S / "timestamps.json", D / "timestamps.json")
    (D / "safe").mkdir(exist_ok=True); shutil.copy2(S / "safe" / "exif.json", D / "safe" / "exif.json")
    H = D / "parts_hr"; (H / "charts").mkdir(parents=True, exist_ok=True); (H / "split").mkdir(exist_ok=True)
    stems = F.stems(S)
    with Pool(a.workers) as pool:
        res = dict(pool.imap(_probe, [(a.day, S, s, a.mode) for s in stems], chunksize=8))
    changed = {s: lab for s, lab in res.items() if lab is not None}
    for s, lab in changed.items():
        np.savez_compressed(H / "split" / f"{s}.npz", lab=lab)
    t1 = time.time()
    rows = {json.loads(l)["stem"]: l for l in open(S / "parts_hr" / "index.jsonl")}
    # links for untouched photos
    for f in (S / "parts_hr").iterdir():
        if f.is_file() and f.suffix == ".npz" and f.name.split("~")[0] not in changed:
            os.symlink(os.path.realpath(f), H / f.name)
    for f in (S / "parts_hr" / "charts").iterdir():
        if f.name.split("~")[0] not in changed:
            os.symlink(os.path.realpath(f), H / "charts" / f.name)
    shutil.copy2(S / "parts_hr" / "pose.json", H / "pose.json") if (S / "parts_hr" / "pose.json").exists() else None
    # fresh cuts for the split photos (parts.py's own extract_photo; size / fractions as run_pipeline.sh)
    P.split_instances = _patched_split; P.extract_photo = _patched_extract; _CUR["dir"] = D
    recs = {Path(json.loads(l)["image"]).stem: json.loads(l) for l in open(S / "images" / "index.jsonl")}
    poses = {Path(json.loads(l)["image"]).stem: json.loads(l) for l in open(S / "cse" / "points.jsonl")} if (S / "cse" / "points.jsonl").exists() else {}
    jobs = []
    for s in changed:
        r = dict(recs[s]); r["pose"] = poses.get(s)
        jobs.append((r, str(D / "cse"), str(H), 1536, 0.03, 0.3, True, 0.1, ("body", "fin", "fluke"), None))
    new_rows = {}
    if jobs:
        with Pool(min(a.workers, len(jobs))) as pool:
            for out in pool.imap_unordered(P._worker, jobs):
                if "error" in out:
                    print("error", out["stem"], out["error"][:200])
                new_rows[out["stem"]] = json.dumps(out) + "\n"
    with open(H / "index.jsonl", "w") as h:
        for s, l in rows.items():
            h.write(new_rows.get(s, l))
    info = {"day": a.day, "mode": a.mode, "photos": len(stems), "split_photos": len(changed), "probe_s": round(t1 - t0, 1), "cut_s": round(time.time() - t1, 1),
            "split_stems": sorted(changed)}
    (D / "split_info.json").write_text(json.dumps(info, indent=1))
    print({k: v for k, v in info.items() if k != "split_stems"})


if __name__ == "__main__":
    main()
