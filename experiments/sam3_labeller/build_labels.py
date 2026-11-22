"""Pseudo-label maps (rules_labeller.md A4 / A5) and the S6 real labels with SAM 3 regions ignored (A6).

    .venv/bin/python experiments/sam3_labeller/build_labels.py select            # A4 -> OUT/lists/pseudo.json
    .venv/bin/python experiments/sam3_labeller/build_labels.py pseudo            # A5 maps -> OUT/pseudo/<day>/<stem>.{jpg,png}, index.json, stats.tsv
    .venv/bin/python experiments/sam3_labeller/build_labels.py s6mc              # A6 -> OUT/real_s6mc/<day>/<stem>.png (+ jpg links), index.json
    .venv/bin/python experiments/sam3_labeller/build_labels.py val               # A5 maps on the val days (checks only) -> OUT/pseudo_val/

Classes: 0 water / sky, 1 whale, 2 person, 3 rail, 4 tape, 5 deck incl. bollard, 6 land, 255 ignore.
"""
from __future__ import annotations

import argparse
import json
import os
import random
import sys
import zlib
from multiprocessing import Pool
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
import lab_common as L  # noqa: E402

THIN_BAND = int(os.environ.get("THIN_BAND", 3))      # v1: 3 px for every SAM 3 class; v2 (part B6): 1 px for rail and tape
EXTRA = {"person": 2, "rail": 3, "tape": 4, "deck": 5, "land": 6}
PAINT_ORDER = ["deck", "rail", "tape", "person"]          # A5 step 5, later wins
CHOICE = None


def choice() -> dict:
    global CHOICE
    if CHOICE is None:
        CHOICE = json.loads((L.OUT / "labeller.json").read_text())
    return CHOICE


def raw_dir(day: str) -> Path:
    if day == L.TEST:
        return L.S3RAW / "test"
    if day in L.VAL:
        return L.S3RAW / "val"          # the comparison's run (same model and settings, all 31 phrasings)
    return L.OUT / "raw" / "train"


def raw_rec(day: str, stem: str):
    if day == L.TEST:
        return L.load_raw(L.S3RAW / "test" / f"{stem.split('_')[-2]}.npz")
    return L.load_raw(raw_dir(day) / f"{stem}.npz")


def pindex(day: str) -> dict:
    return {p: i for i, p in enumerate(json.loads((raw_dir(day) / "_prompts.json").read_text()))}


def sam_regions(rec: dict, pidx: dict, hw) -> dict:
    """sure / unsure mask per extra class, plus whale (SAM 3), any-boat and gear regions, at hw."""
    ch = choice(); out = {}
    for c in EXTRA:
        idx = [pidx[p] for p in ch[c]["phrasings"]]; t = ch[c]["thr"]
        out[c] = L.class_mask(rec, idx, t, hw); out[c + "_unsure"] = L.class_mask(rec, idx, 0.15, hw, hi=t)
        out[c + "_hi"] = out[c] if t >= 0.5 else L.class_mask(rec, idx, 0.5, hw)     # part B5: over S6h only score >= 0.5 wins
    out["sam_whale"] = L.class_mask(rec, [pidx[p] for p in ch["whale_ignore"]["phrasings"]], ch["whale_ignore"]["thr"], hw)
    out["boat"] = L.class_mask(rec, [pidx[p] for p in ch["boat_ignore"]["phrasings"]], ch["boat_ignore"]["thr"], hw)
    out["gear"] = L.class_mask(rec, [pidx[p] for p in ch["gear_ignore"]["phrasings"]], ch["gear_ignore"]["thr"], hw)
    return out


def edge_band(m: np.ndarray, r: int) -> np.ndarray:
    k = L.disk(r); u = m.astype(np.uint8)
    return (cv2.dilate(u, k) > 0) & ~(cv2.erode(u, k) > 0)


def label_map(p: np.ndarray, R: dict) -> np.ndarray:
    """A5 steps 1-6 at the 1280 grid of P (S6 whale probability)."""
    s6h = L.hyst(p); lab = np.zeros(p.shape, np.uint8)
    ign = ((p >= 0.1) & ~s6h) | (R["sam_whale"] & ~s6h) | R["boat"] | R["gear"]
    for c in EXTRA:
        ign |= R[c + "_unsure"]
    lab[ign] = L.IGNORE
    lab[R["land"]] = 6
    lab[s6h] = 1; lab[edge_band(s6h, 2)] = L.IGNORE
    for c in PAINT_ORDER:
        lab[R[c] & ~s6h] = EXTRA[c]
        lab[R[c] & s6h & R[c + "_hi"]] = EXTRA[c]                # B5: inside S6h a class wins only at score >= 0.5
        lab[R[c] & s6h & ~R[c + "_hi"]] = L.IGNORE              # B5: else unsure -> ignore
    band = np.zeros(p.shape, bool)
    for c in ["land"] + PAINT_ORDER:
        if R[c].any():
            band |= edge_band(R[c], THIN_BAND if c in ("rail", "tape") else 3)
    lab[band] = L.IGNORE
    return lab


def shares(R: dict) -> dict:
    return {c: float(R[c].mean()) for c in EXTRA}


# ------------------------------------------------------------------------------------------------- select (A4)
def scan_shares(key: str):
    day, stem = key.split("|"); rec = raw_rec(day, stem)
    if rec is None:
        return key, None
    R = sam_regions(rec, pindex(day), (214, 320))
    return key, shares(R)


def cmd_select(a) -> None:
    lm = json.loads((L.OUT / "lists" / "live_minus_s6.json").read_text())
    day_of = json.loads((L.OUT / "lists" / "scan_days.json").read_text())
    keys = [f"{day_of[l.split(chr(9))[0]]}|{l.split(chr(9))[0]}" for l in open(L.OUT / "lists" / "scan_train.tsv")]
    with Pool(12) as P:
        res = dict(P.map(scan_shares, keys, chunksize=16))
    miss = [k for k, v in res.items() if v is None]
    assert not miss, f"{len(miss)} scanned photos without SAM 3 output"
    extra = sorted(k for k, v in res.items() if max(v.values()) >= 0.001 or lm.get(k, -1) >= 0.005)
    plain = sorted(k for k in res if k not in set(extra))
    ext = extra if len(extra) <= 1500 else random.Random(20261121).sample(extra, 1500)
    pl = random.Random(20261122).sample(plain, max(0, 2200 - len(ext)))
    per_class = {c: sum(1 for k in ext if res[k][c] >= 0.001) for c in EXTRA}
    out = {"extra_found": len(extra), "extra_used": len(ext), "plain_used": len(pl), "scanned": len(res), "extra_by_class_ge_0.1pct": per_class,
           "photos": sorted(ext + pl), "extra": sorted(ext), "shares": res}
    (L.OUT / "lists" / "pseudo.json").write_text(json.dumps(out))
    print({k: v for k, v in out.items() if k not in ("photos", "extra", "shares")})


# ------------------------------------------------------------------------------------------------- pseudo maps
def build_one(args):
    day, stem, root = args
    o = Path(root) / day; png = o / f"{stem}.png"
    if png.exists():
        lab = cv2.imread(str(png), cv2.IMREAD_UNCHANGED)
    else:
        rec = raw_rec(day, stem); p = L.s6_prob(day, stem)
        R = sam_regions(rec, pindex(day), p.shape)
        lab = label_map(p, R); o.mkdir(parents=True, exist_ok=True)
        img = L.resize_1280(cv2.imread(str(L.photo_path(day, stem))))
        assert img.shape[:2] == lab.shape, (img.shape, lab.shape)
        cv2.imwrite(str(o / f"{stem}.jpg"), img, [cv2.IMWRITE_JPEG_QUALITY, 95]); cv2.imwrite(str(png), lab)
    rng = np.random.default_rng(zlib.crc32(stem.encode()))
    def pts(mask, n=40):
        ys, xs = np.nonzero(mask)
        if not len(xs):
            return []
        k = rng.choice(len(xs), min(n, len(xs)), replace=False)
        return [[int(xs[i]), int(ys[i])] for i in k]
    ex = np.isin(lab, [2, 3, 4, 5, 6])
    st = {c: float((lab == v).mean()) for c, v in [("water_sky", 0), ("whale", 1), *EXTRA.items(), ("ignore", 255)]}
    pts_cls = {c: pts(lab == v, 20) for c, v in EXTRA.items() if (lab == v).any()}
    return {"day": day, "stem": stem, "H": int(lab.shape[0]), "W": int(lab.shape[1]), "pts_extra": pts(ex), "pts_whale": pts(lab == 1), "pts_cls": pts_cls, "share": st}


def build_set(items, root: Path):
    root.mkdir(parents=True, exist_ok=True)
    with Pool(int(os.environ.get("NPROC", 10))) as P:
        idx = P.map(build_one, [(d, s, str(root)) for d, s in items], chunksize=8)
    (root / "index.json").write_text(json.dumps(idx))
    with open(root / "stats.tsv", "w") as f:
        cols = ["water_sky", "whale", *EXTRA, "ignore"]
        f.write("day\tstem\t" + "\t".join(cols) + "\n")
        for r in idx:
            f.write(f"{r['day']}\t{r['stem']}\t" + "\t".join(f"{r['share'][c]:.5f}" for c in cols) + "\n")
    n = {c: sum(1 for r in idx if r["share"][c] > 0) for c in ["whale", *EXTRA]}
    print(len(idx), "maps; photos holding each class:", n)


def cmd_pseudo(a) -> None:
    sel = json.loads((L.OUT / "lists" / "pseudo.json").read_text())
    build_set([tuple(k.split("|")) for k in sel["photos"]], L.OUT / os.environ.get("PSEUDO_DIR", "pseudo"))


def cmd_test(a) -> None:
    """the labeller's own maps on the test day (its single scored run, rules part C2): comparison raw (all phrasings)."""
    build_set([(L.TEST, s) for s in L.day_stems_any(L.TEST)], L.OUT / "labeller_test")


def cmd_val(a) -> None:
    build_set([(d, s) for d in L.VAL for s in L.day_stems(d)], L.OUT / "pseudo_val")


# ------------------------------------------------------------------------------------------------- real_s6mc
def s6mc_one(args):
    r, src, dst = args
    from model_fine_tuning.seg_real import common as SC
    from model_fine_tuning.seg_real.data import blob_1280
    day, stem = r["day"], r["stem"]
    lab = cv2.imread(str(Path(src) / day / f"{stem}.png"), cv2.IMREAD_UNCHANGED); shape = lab.shape
    blobs = BLOBS.get((day, stem), [])
    Lb = SC.blob_image(day, stem) if blobs else None
    n_bg = 0
    for b in blobs:
        if b["role"] == "background" and b["label"] in ("shore", "boat", "person", "pole", "drone", "bird", "other"):
            m = blob_1280(Lb, b["b"], shape); lab[cv2.dilate(m.astype(np.uint8), L.disk(2)) > 0] = L.IGNORE; n_bg += 1
    rec = raw_rec(day, stem); n_sam = 0
    if rec is not None:
        R = sam_regions(rec, pindex(day), shape)
        reg = R["boat"] | R["gear"]
        for c in EXTRA:
            reg |= R[c] | R[c + "_unsure"]
        reg = cv2.dilate(reg.astype(np.uint8), L.disk(3)) > 0
        n_sam = int(reg.sum()); lab[reg] = L.IGNORE
    o = Path(dst) / day; o.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(o / f"{stem}.png"), lab)
    j = o / f"{stem}.jpg"
    if not j.exists():
        os.symlink(Path(src) / day / f"{stem}.jpg", j)
    return rec is not None, n_bg, n_sam


BLOBS = {}


def cmd_s6mc(a) -> None:
    from model_fine_tuning.seg_real import common as SC
    src = SC.CACHE / "real_s6"; dst = L.OUT / "real_s6mc"
    for b in SC.load_labels():
        if b["split"] == "train":
            BLOBS.setdefault((b["day"], b["stem"]), []).append(b)
    idx = json.loads((src / "index.json").read_text())
    with Pool(10) as P:
        res = P.map(s6mc_one, [(r, str(src), str(dst)) for r in idx], chunksize=8)
    (dst / "index.json").write_text(json.dumps(idx))
    print(len(idx), "photos; with SAM 3 output", sum(r[0] for r in res), "; non-water background blobs set to ignore", sum(r[1] for r in res),
          "; photos with a SAM 3 region set to ignore", sum(1 for r in res if r[2] > 0))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("cmd", choices=["select", "pseudo", "s6mc", "val", "test"]); a = ap.parse_args()
    {"select": cmd_select, "pseudo": cmd_pseudo, "s6mc": cmd_s6mc, "val": cmd_val, "test": cmd_test}[a.cmd](a)


if __name__ == "__main__":
    main()
