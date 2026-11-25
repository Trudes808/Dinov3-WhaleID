"""Pilot r06_pilot22 as a W3e-style label cache at run05's pixel scale. Reads the factory run, never writes it.

    .venv/bin/python -m model_fine_tuning.pilot22_train.prep [--workers 8]
        -> CACHE/images/<id>.png   RGB, 1280 x 854 (2560 x 1707 INTER_AREA, factor 0.5)
           CACHE/labels/<id>.png   palette (parts_seg.labels.PALETTE), 1280 x 854, nearest:
                                   parts_triA/ (0 bg, 1 fluke, 2 dorsal by rule A, 3 pectoral, 4 head, 5 body, 255 undefined
                                   dorsal) with the through-water zone (masks == 2 where parts is background) -> 255,
                                   as run05's cache (labels.py: masks == 2 -> ignore)
           CACHE/ids.json          train / val id lists and what was dropped why
Why 0.5: the median visible whale is sqrt(px) 91 px in run05 (1280 frames) and 202 px in the pilot at 2560 (101 at
1280): at 0.5 the pilot's whales sit in the same pixel-size range as run05's, so the crop scale jitter of the recipe
(0.5 - 1.5 of native) means the same thing for both.
Exclusions: individual i12 (broken generated dorsal blade; `individual` or a `companions` entry in shotlist.jsonl),
gate-rejected shots (manifest `rejected`); training split also run05's split filter (factory defect flags empty_mask /
visible_outside_amodal / projection_not_covered, and whale shots with < 800 visible px at 1280 = 3200 at 2560).
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

from ..parts_seg.labels import IGNORE, PALETTE

RUN = Path("/home/sat3737/NOAA_whales/Whale-Image-Data-Factory/data/renders/r06_pilot22")
DS = RUN / "dataset"
CACHE = Path("/home/sat3737/NOAA_whales/moonshot_cache/pilot22/d/cache")
SIZE = (1280, 854)                 # (W, H) = round(2560 * 0.5), round(1707 * 0.5)
DEFECT = {"empty_mask", "visible_outside_amodal", "projection_not_covered"}
MIN_VIS_2560 = 800 * 4
BAD_INDIVIDUAL = "i12"


def rd(p: Path) -> np.ndarray:
    a = np.asarray(Image.open(p))
    return a[..., 0] if a.ndim == 3 else a


def id_lists() -> dict:
    man = {}
    for line in open(DS / "manifest.jsonl"):
        if line.strip():
            r = json.loads(line)
            man[r["id"]] = r
    bad = set()
    for line in open(RUN / "shotlist.jsonl"):
        if not line.strip():
            continue
        s = json.loads(line)
        comps = [c.get("individual") for c in (s.get("companions") or []) if isinstance(c, dict)]
        if s.get("individual") == BAD_INDIVIDUAL or BAD_INDIVIDUAL in comps:
            bad.add(s["id"])
    sp = json.loads((DS / "splits.json").read_text())
    out = {"i12_shots": sorted(bad), "dropped": {}}
    for k in ("train", "val"):
        ids = [i for i in sp[k] if i in man]
        d = {"i12": [i for i in ids if i in bad], "rejected": [i for i in ids if man[i].get("rejected") and i not in bad]}
        keep = [i for i in ids if i not in bad and not man[i].get("rejected")]
        if k == "train":
            d["defect_flag"] = [i for i in keep if set(man[i].get("qa_flags") or []) & DEFECT]
            keep = [i for i in keep if i not in set(d["defect_flag"])]
            d["min_visible"] = [i for i in keep if not man[i]["negative"] and man[i]["visible_pixels"] < MIN_VIS_2560]
            keep = [i for i in keep if i not in set(d["min_visible"])]
        out[k] = keep
        out["dropped"][k] = {kk: len(v) for kk, v in d.items()} | {"split_size": len(ids)}
    return out


def save_label(a: np.ndarray, path: Path) -> None:
    im = Image.fromarray(a.astype(np.uint8), mode="P")
    im.putpalette(list(PALETTE) + [0] * (768 - len(PALETTE)))
    im.save(path, optimize=True)


def one(sid: str) -> dict:
    oi, ol = CACHE / "images" / f"{sid}.png", CACHE / "labels" / f"{sid}.png"
    parts = rd(DS / "parts_triA" / f"{sid}.png").copy()
    masks = rd(DS / "masks" / f"{sid}.png")
    through = (masks == 2) & (parts == 0)
    vis_unlab = int(((masks == 1) & (parts == 0)).sum())
    parts[through] = IGNORE
    lab = cv2.resize(parts, SIZE, interpolation=cv2.INTER_NEAREST)
    if not ol.exists():
        save_label(lab, ol)
    if not oi.exists():
        bgr = cv2.imread(str(DS / "images" / f"{sid}.png"), cv2.IMREAD_COLOR)
        cv2.imwrite(str(oi), cv2.resize(bgr, SIZE, interpolation=cv2.INTER_AREA))
    cnt = np.bincount(lab.ravel(), minlength=256)
    return {"id": sid, "px": {str(c): int(cnt[c]) for c in (0, 1, 2, 3, 4, 5, 255)}, "visible_unlabelled_2560": vis_unlab}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--workers", type=int, default=8)
    a = ap.parse_args(argv)
    (CACHE / "images").mkdir(parents=True, exist_ok=True)
    (CACHE / "labels").mkdir(parents=True, exist_ok=True)
    ids = id_lists()
    (CACHE / "ids.json").write_text(json.dumps(ids, indent=1))
    todo = ids["train"] + ids["val"]
    from multiprocessing import Pool
    with Pool(a.workers) as pool:
        rows = list(pool.imap(one, todo, chunksize=8))
    tot = {c: sum(r["px"][c] for r in rows) for c in rows[0]["px"]}
    s = {"train": len(ids["train"]), "val": len(ids["val"]), "dropped": ids["dropped"], "label_px_1280": tot,
         "visible_unlabelled_px_2560": sum(r["visible_unlabelled_2560"] for r in rows), "size": SIZE, "source": str(DS)}
    (CACHE / "summary.json").write_text(json.dumps(s, indent=1))
    print(json.dumps(s, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
