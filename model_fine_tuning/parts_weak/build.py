"""Weak labels of the real TRAINING-day photos (not 2025-03-18, the click test; never val / eval days), CPU only.

    PYTHONPATH=. .venv/bin/python -m model_fine_tuning.parts_weak.build [--workers 6] [--limit N]
        -> CACHE/maps/<day>/<stem>.png   3-channel uint8 at the 2560-px frame:
             ch0 weak label: 0 certain background, 1 fluke (clicked discs), 2 dorsal fin (clicked triangle), 255 unknown
             ch1 instance map: k+1 = region of old instance <stem>~k (k < 8), 254 = two regions overlap, 0 none
        -> CACHE/index.json              per photo the instances with their constraints and the not-a-whale blobs

Certain background (ch0 = 0):
  * farther than 64 px (2560 frame) from every pixel of whale evidence: the live segmentation's blobs
    (DAY/instances_v2/blobs, seg_r04_ft4 at P >= 0.5, >= 100 px), the old instance masks (DAY/cse, as parts_hr / predict
    make them) and, where cached, seg_r04_ft4's P >= 0.25 (seg_splash/pred/r04); and farther than 0.1 x the instance's
    long side from each instance region;
  * blobs labelled by eye as not a whale (experiments/seg_splash/blob_labels.json, role background), with the strict
    reading of seg_real (an UNSURE splash / wake blob only on its white pixels; unsure water / other not used);
    blobs labelled ignore (whale under splash, unclear) are unknown, grown by 4 px.
  The black padding outside the photo is background too (set when cropping, data.py).
Instance region = the old instance mask grown by ceil(1 / s) + 2 px (s = cse crop scale), as predict.py's "near".

Constraints per instance (by-eye code first where it exists, else the view classifier DAY/safe/view_class.json):
  fluke yes  = code F / FD / Fd  or  P(fluke) >= 0.9          fluke no = code D / O (no f)  or  P(fluke) <= 0.1
  fin yes    = code D / FD / Df  or  P(dorsal) >= 0.9         fin no   = code F / O (no d)  or  P(dorsal) <= 0.1
  forbid fluke where fluke is no (fin views AND other views: no fluke can be there);
  forbid fin only where fin is no AND fluke is yes (fluke views); other views keep the fin allowed (a breach or a roll
  can show a fin the classifier does not call a dorsal-fin view);
  require some fluke where fluke is yes, some fin where fin is yes (top-k terms, losses.py).
  Skipped: code X, instances the pipeline drops as not a whale (DAY/safe/not_whale.json), instances that overlap a
  blob labelled not-a-whale by more than 30 %.
Clicked points (2025-01-02 only, Whale-Review-Web pair_marks.json read only):
  dorsal fin = the tip part of the clicked fin only: the triangle tip, tip + 0.3 x (front base - tip), tip + 0.3 x
  (back base - tip), shrunk by 10 % towards its centre. The researcher's bases span the whole hump, while the render
  labels (and the cut-outs' fin-base ratio 0.28) call only the upper blade fin: the whole triangle would redefine the
  class (preview sheet check_clicks_baseline). A hooked fin's concave back edge is avoided by the shrink;
  fluke = discs of radius 2 % of the span at the two tips and the notch, each moved inwards: a tip 8 % of the way
  towards the centre of the three points, the notch 4 % of the span away from the tip-to-tip line (into the stock).
"""
from __future__ import annotations

import argparse
import json
import time
from multiprocessing import get_context

import cv2
import numpy as np

from . import (CACHE, CLICK_TRAIN_DAY, DAYS, FIN, FLUKE, IGNORE, INDEX, LAB, MAPS, MAX_INST, OVERLAP, ROOT, TRAIN_DAYS,
               check_train_day)

MARGIN_PX = 64
INST_MARGIN = 0.10
R04 = CACHE.parent / "seg_splash" / "pred" / "r04"
BLOB_LABELS = ROOT / "experiments" / "seg_splash" / "blob_labels.json"
VIEW_LABELS = ROOT / "experiments" / "view_class" / "view_labels.json"
TIP_PART = 0.30                                           # the part of the clicked fin next to the tip
SHRINK_FIN = 0.10


def disk(r: int) -> np.ndarray:
    r = max(int(r), 1)
    return cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * r + 1, 2 * r + 1))


def white_pixels(img_bgr: np.ndarray) -> np.ndarray:
    hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
    return cv2.dilate(((hsv[..., 2] > 190) & (hsv[..., 1] < 45)).astype(np.uint8), disk(2)) > 0


def view_answers(code: str | None, vc) -> tuple[str | None, str | None, str]:
    """(fluke, fin) each 'yes' / 'no' / None, and the source."""
    if code:
        if code == "X":
            return None, None, "eye_X"
        fl = "yes" if "F" in code else (None if "f" in code else "no")
        fi = "yes" if "D" in code else (None if "d" in code else "no")
        return fl, fi, "eye"
    if vc:
        pf, pd = vc
        fl = "yes" if pf >= 0.9 else "no" if pf <= 0.1 else None
        fi = "yes" if pd >= 0.9 else "no" if pd <= 0.1 else None
        return fl, fi, "vc"
    return None, None, "none"


def constraints(fl, fi) -> tuple[list, list]:
    forbid, present = [], []
    if fl == "no":
        forbid.append(FLUKE)
    if fi == "no" and fl == "yes":
        forbid.append(FIN)
    if fl == "yes":
        present.append(FLUKE)
    if fi == "yes":
        present.append(FIN)
    return forbid, present


def click_shapes(kp: dict, W: int, H: int) -> list:
    """[(class, polygon or (cx, cy, r))] in the 2560 frame from one photo's keypoints."""
    P = {n: np.array([p["x"] * W, p["y"] * H]) for n, p in kp.items()}
    out = []
    if all(n in P for n in ("dorsal_fin_tip", "dorsal_fin_front", "dorsal_fin_back")):
        t = P["dorsal_fin_tip"]
        tri = np.stack([t, t + TIP_PART * (P["dorsal_fin_front"] - t), t + TIP_PART * (P["dorsal_fin_back"] - t)])
        c = tri.mean(0)
        tri = c + (1 - SHRINK_FIN) * (tri - c)
        out.append((FIN, "poly", tri))
    if "fluke_tip_left" in P and "fluke_tip_right" in P:
        a, b = P["fluke_tip_left"], P["fluke_tip_right"]
        span = float(np.hypot(*(a - b)))
        if span >= 20:
            r = max(3.0, 0.02 * span)
            pts = [a, b] + ([P["fluke_notch"]] if "fluke_notch" in P else [])
            c = np.mean(pts, 0)
            for t in (a, b):
                out.append((FLUKE, "disc", (*(t + 0.08 * (c - t)), r)))
            if "fluke_notch" in P:
                m = (a + b) / 2
                d = P["fluke_notch"] - m
                n = float(np.hypot(*d))
                if n >= 0.03 * span:                         # the notch lies clearly off the tip line: its inward side is known
                    out.append((FLUKE, "disc", (*(P["fluke_notch"] + 0.04 * span * d / n), r)))
    return out


def load_clicks() -> dict:
    labs = json.loads((LAB / CLICK_TRAIN_DAY / "pair_marks.json").read_text())
    labs = labs["labels"] if isinstance(labs, dict) else labs
    photos = {}
    for l in labs:
        if l.get("verdict") == "skip":
            continue
        for sd in ("a", "b"):
            kp = (l.get("keypoints") or {}).get(sd) or {}
            if kp and l.get(f"key_{sd}"):
                photos.setdefault(l[f"key_{sd}"], {}).update(kp)
    return photos


# ------------------------------------------------------------------------------------------------------ per photo
def work(job):
    from PIL import Image
    from ..cluster.instances_v2 import old_instances
    from ..common import geometry, paths
    cv2.setNumThreads(1)
    day, stem, inst_meta, nw, clicks = job
    try:
        check_train_day(day)
        d = DAYS / day
        W, H = Image.open(d / "images" / f"{stem}.jpg").size
        table = geometry.load_folded_table(paths.FOLDED_TABLE)
        keys, masks = old_instances(d, stem, W, H, table)
        z = np.load(d / "cse" / f"{stem}_canon.npz") if (d / "cse" / f"{stem}_canon.npz").exists() else None
        s = float(z["x0y0s"][2]) if z is not None else 1.0
        blobs = cv2.imread(str(d / "instances_v2" / "blobs" / f"{stem}.png"), cv2.IMREAD_UNCHANGED)
        if blobs is None:
            blobs = np.zeros((H, W), np.uint8)
        ev = blobs > 0
        pr = R04 / day / f"{stem}.png"
        if pr.exists():
            p = cv2.imread(str(pr), cv2.IMREAD_UNCHANGED)
            ev |= cv2.resize(p, (W, H), interpolation=cv2.INTER_LINEAR) >= round(0.25 * 255)
        for m in masks:
            ev |= m
        ev_whale = ev.copy()
        nw_masks = []
        img = None
        for b in nw:
            bm = blobs == b["b"]
            if b["role"] == "background":
                if not b["sure"] and b["label"] in ("splash", "wake"):
                    img = cv2.imread(str(d / "images" / f"{stem}.jpg")) if img is None else img
                    bm = bm & white_pixels(img)
                elif not b["sure"] and b["label"] in ("water", "other"):
                    bm = None
            nw_masks.append((b, bm))
        nw_union = np.zeros((H, W), bool)
        for b, bm in nw_masks:
            if bm is not None and b["role"] == "background":
                nw_union |= bm
        ev &= ~nw_union | np.any(masks, 0) if masks else ~nw_union        # labelled not-a-whale blobs are no whale evidence
        dist = cv2.distanceTransform((~ev).astype(np.uint8), cv2.DIST_L2, 3)
        lab = np.full((H, W), IGNORE, np.uint8)
        lab[dist > MARGIN_PX] = 0
        r_near = int(np.ceil(1.0 / s)) + 2
        inst = np.zeros((H, W), np.uint8)
        cover = np.zeros((H, W), np.uint8)
        nears = []
        for k, m in enumerate(masks):
            dm = cv2.distanceTransform((~m).astype(np.uint8), cv2.DIST_L2, 3)          # distance to the instance (fast "dilate")
            near = dm <= r_near
            ys, xs = np.nonzero(m)
            L = float(max(np.ptp(xs), np.ptp(ys)) + 1) if len(xs) else 0.0
            lab[dm <= r_near + max(4.0, INST_MARGIN * L)] = IGNORE
            cover += near.astype(np.uint8)
            nears.append((near, L))
        for k, (near, L) in enumerate(nears):
            inst[near] = k + 1 if k < MAX_INST else OVERLAP
        inst[cover >= 2] = OVERLAP
        for b, bm in nw_masks:
            if bm is None:
                continue
            if b["role"] == "background":
                lab[bm] = 0
        for b, bm in nw_masks:
            if b["role"] == "ignore":
                lab[cv2.dilate((blobs == b["b"]).astype(np.uint8), disk(4)) > 0] = IGNORE
        # clicked points (2025-01-02)
        # a shape counts only on whale evidence (blobs / instance masks / P >= 0.25, grown 2 px): a tip the researcher
        # clicked where it lies under water would otherwise teach fluke on open water (preview sheet, tile 7)
        n_pts = 0
        if clicks:
            whale_ev = cv2.dilate(ev_whale.astype(np.uint8), disk(2)) > 0
            for key, kp in clicks.items():
                for cls, kind, g in click_shapes(kp, W, H):
                    sh = np.zeros((H, W), np.uint8)
                    if kind == "poly":
                        cv2.fillPoly(sh, [np.round(g).astype(np.int32)], 1)
                        cx, cy = g.mean(0)
                    else:
                        cv2.circle(sh, (int(round(g[0])), int(round(g[1]))), int(round(g[2])), 1, -1)
                        cx, cy = g[0], g[1]
                    ix, iy = int(round(cx)), int(round(cy))
                    if not (0 <= ix < W and 0 <= iy < H and whale_ev[iy, ix]):
                        continue
                    lab[(sh > 0) & whale_ev] = int(cls)
                    n_pts += 1
        # instances and their constraints
        out_inst = []
        for k, (near, L) in enumerate(nears):
            key = keys[k]
            meta = inst_meta.get(key)
            if meta is None or k >= MAX_INST:
                continue
            ys, xs = np.nonzero(near)
            rec = {"id": k + 1, "key": key, "box": [int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1], "L": round(L, 1),
                   "area": int(masks[k].sum()), "fluke": meta["fluke"], "fin": meta["fin"], "src": meta["src"]}
            ov = float((masks[k] & nw_union).sum()) / max(masks[k].sum(), 1)
            if meta["skip"] or ov > 0.3:
                rec["forbid"], rec["present"], rec["skipped"] = [], [], meta["skip"] or f"overlaps not-a-whale blob {ov:.2f}"
            else:
                rec["forbid"], rec["present"] = constraints(meta["fluke"], meta["fin"])
            rec["clicked"] = key in clicks
            out_inst.append(rec)
        nw_rec = [{"b": b["b"], "label": b["label"], "role": b["role"], "sure": b["sure"], "box": [int(v) for v in b["bbox"]],
                   "px": int(bm.sum()) if bm is not None else 0} for b, bm in nw_masks]
        out_dir = MAPS / day
        out_dir.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(out_dir / f"{stem}.png"), np.dstack([lab, inst, np.zeros_like(lab)]))
        stats = {"bg_frac": round(float((lab == 0).mean()), 4), "pts_px": int(((lab == 1) | (lab == 2)).sum())}
        return {"day": day, "stem": stem, "W": W, "H": H, "inst": out_inst, "nw": nw_rec, "points": n_pts, **stats}, None
    except Exception as e:  # noqa: BLE001
        import traceback
        return {"day": day, "stem": stem}, f"{type(e).__name__}: {e} {traceback.format_exc()[-600:]}"


def jobs(limit: int = 0) -> list:
    eye = json.loads(VIEW_LABELS.read_text())
    blabs = json.loads(BLOB_LABELS.read_text())["blobs"]
    clicks = load_clicks()
    out = []
    for day in TRAIN_DAYS:
        check_train_day(day)
        d = DAYS / day
        vc = json.loads((d / "safe" / "view_class.json").read_text())["views"]
        drop = set(json.loads((d / "safe" / "not_whale.json").read_text()).get("drop", [])) if (d / "safe" / "not_whale.json").exists() else set()
        nw_by = {}
        for b in blabs:
            if b["day"] == day and b["split"] == "train" and b["role"] in ("background", "ignore"):
                nw_by.setdefault(b["stem"], []).append(b)
        recs = [json.loads(l) for l in open(d / "parts_hr" / "index.jsonl") if l.strip()]
        n = 0
        for rec in recs:
            stem = rec["stem"]
            meta = {}
            for i in rec.get("instances", []):
                if not i.get("bbox_2560"):
                    continue
                key = i["key"]
                e = eye.get(key)
                code = e["code"] if (e and e["day"] == day) else None
                fl, fi, src = view_answers(code, vc.get(key))
                meta[key] = {"fluke": fl, "fin": fi, "src": src, "skip": ("not_whale_drop" if key in drop else ("eye_X" if src == "eye_X" else False))}
            ck = {k: v for k, v in clicks.items() if k.rsplit("~", 1)[0] == stem} if day == CLICK_TRAIN_DAY else {}
            useful = any(m["fluke"] or m["fin"] for m in meta.values()) or nw_by.get(stem) or ck
            if useful:
                out.append((day, stem, meta, nw_by.get(stem, []), ck))
                n += 1
            if limit and n >= limit:
                break
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--limit", type=int, default=0, help="debug: photos per day")
    a = ap.parse_args(argv)
    J = jobs(a.limit)
    print(f"{len(J)} photos on {len(TRAIN_DAYS)} training days", flush=True)
    t0 = time.time()
    rows, errs = [], 0
    with get_context("fork").Pool(a.workers) as pool:
        for i, (r, err) in enumerate(pool.imap_unordered(work, J, chunksize=4)):
            if err:
                errs += 1
                print("error", r["day"], r["stem"][-40:], err, flush=True)
            else:
                rows.append(r)
            if i % 500 == 499:
                print(f"  {i + 1}/{len(J)} {time.time() - t0:.0f}s", flush=True)
    rows.sort(key=lambda r: (r["day"], r["stem"]))
    CACHE.mkdir(parents=True, exist_ok=True)
    out = INDEX if not a.limit else INDEX.with_suffix(".limit.json")
    out.write_text(json.dumps({"days": TRAIN_DAYS, "photos": rows}))
    ins = [i for r in rows for i in r["inst"]]
    summ = {"photos": len(rows), "errors": errs, "instances": len(ins),
            "forbid_fluke": sum(1 in i["forbid"] for i in ins), "forbid_fin": sum(2 in i["forbid"] for i in ins),
            "present_fluke": sum(1 in i["present"] for i in ins), "present_fin": sum(2 in i["present"] for i in ins),
            "from_eye": sum(i["src"] == "eye" and bool(i["forbid"] or i["present"]) for i in ins),
            "from_vc": sum(i["src"] == "vc" and bool(i["forbid"] or i["present"]) for i in ins),
            "skipped": sum(bool(i.get("skipped")) for i in ins), "clicked_instances": sum(i["clicked"] for i in ins),
            "click_shapes": sum(r["points"] for r in rows), "nw_blobs_background": sum(b["role"] == "background" and b["px"] > 0 for r in rows for b in r["nw"]),
            "nw_blobs_ignore": sum(b["role"] == "ignore" for r in rows for b in r["nw"]),
            "mean_bg_frac": round(float(np.mean([r["bg_frac"] for r in rows])), 3), "seconds": round(time.time() - t0)}
    print(json.dumps(summ, indent=1))
    (out.with_name(out.stem + "_summary.json")).write_text(json.dumps(summ, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
