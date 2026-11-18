"""Blob pool, stratified sample and blind labelling sheets for the real-photo splash / gear labels.

    PYTHONPATH=. .venv/bin/python -m model_fine_tuning.seg_real.sample pool
        -> CACHE/pool.json: every blob >= 300 px (2560-px frame) of the 21 training and 3 validation days with its
           existing scores (not-a-whale filter nw, view P(fluke) / P(dorsal), blob gate; instances_v2.load_scores)
           and the reusable labels attached to it (AI blob labels, nw_labels.json, view_labels.json).
    PYTHONPATH=. .venv/bin/python -m model_fine_tuning.seg_real.sample draw [--train 1150 --val 330 --seed 0]
        -> experiments/seg_splash/label_sample.json: the blobs to label by eye (tile numbers), stratified over days,
           sizes and the score range with the hard strata over-sampled (scores choose WHICH blobs are drawn only; the
           tiles never show them); plus the reusable labels that need no look.
    PYTHONPATH=. .venv/bin/python -m model_fine_tuning.seg_real.sample sheets [--per 28 --cols 4]
        -> SHEETS/lab_NNN.jpg (+ lab_legend.json): tile = tight crop with the blob outlined (left) and the photo with
           a yellow box on the crop window (right); NO score is printed.

Strata (blob b of a photo; secondary = not the photo's largest blob):
  main_ok    largest blob, gate >= 0 and nw <= 0          mostly clean whales (easy, under-sampled)
  main_white as main_ok but >= 15 % of the blob is bright and unsaturated (HSV V > 190, S < 45; white_probe in
             `pool --white`): whales in or behind white water, white fluke undersides
  main_low   largest blob, gate < 0 or nw > 0             junk as the largest blob, whales in white water
  sec_nw     secondary, nw > 2 (the filter rejects)        people, boats, drones
  sec_neg    secondary, nw <= 2, gate < -1                 white water, wake, glitter, shore
  sec_mid    secondary, nw <= 2, -1 <= gate < 1            the uncertain ones: whales under splash, splash near whales
  sec_pos    secondary, nw <= 2, gate >= 1                 mostly second whales and whale pieces
  small      secondary, 300 <= area < 1000 px              tiny pieces (any score)
"""
from __future__ import annotations

import argparse
import json
import random
from collections import Counter, defaultdict
from pathlib import Path

import cv2
import numpy as np

from . import common as C

POOL = C.CACHE / "pool.json"
SAMPLE = C.EXP / "label_sample.json"
IV2 = C.REPO / "experiments" / "instances_v2"
AI_MAP = {"whale_part": "whale", "second_whale": "whale", "person": "person", "drone": "drone", "boat": "boat", "shore": "shore"}
AI_LOOK = {"splash", "other", "unclear"}          # AI categories too coarse for this scheme: drawn again on the sheets
WEIGHTS = {"main_ok": 0.07, "main_white": 0.12, "main_low": 0.12, "sec_nw": 0.07, "sec_neg": 0.20, "sec_mid": 0.17, "sec_pos": 0.13, "small": 0.12}


def stratum(b: dict) -> str:
    if b["b"] == 1:
        if b["gate"] < 0 or b["nw"] > 0:
            return "main_low"
        return "main_white" if b.get("white", 0) >= 0.15 else "main_ok"
    if b["area"] < 1000:
        return "small"
    if b["nw"] > 2.0:
        return "sec_nw"
    return "sec_neg" if b["gate"] < -1 else "sec_mid" if b["gate"] < 1 else "sec_pos"


def size_bin(a: int) -> str:
    return "300-1k" if a < 1000 else "1k-3k" if a < 3000 else "3k-10k" if a < 10000 else "10k-50k" if a < 50000 else "50k+"


def _white(x):
    """Share of a main blob's pixels that are bright and unsaturated (white water, white skin), half resolution."""
    i, day, stem = x
    img = cv2.imread(str(C.photo_path(day, stem)), cv2.IMREAD_REDUCED_COLOR_2)
    m = (C.blob_image(day, stem)[::2, ::2][:img.shape[0], :img.shape[1]] == 1)
    if not m.any():
        return i, 0.0
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    return i, float(((hsv[..., 2] > 190) & (hsv[..., 1] < 45))[m].mean())


def cmd_pool(a) -> None:
    from ..cluster import instances_v2 as V
    # reusable labels -> (day, stem, b)
    reuse = defaultdict(list)
    for smp_f, lab_f in (("blob_sample.json", "blob_labels_ai.json"), ("blob_sample_test.json", "blob_labels_ai_test.json")):
        smp = json.loads((IV2 / smp_f).read_text()); L = json.loads((IV2 / lab_f).read_text())["labels"]
        for s in smp:
            lab = L.get(str(s["tile"]))
            if lab is None or s["day"] in C.EVAL:
                continue
            reuse[(s["day"], s["stem"], s["b"])].append({"src": f"instances_v2/{lab_f}#{s['tile']}", "ai": lab})
    nwl = json.loads((C.REPO / "experiments" / "stage2_linker" / "nw_labels.json").read_text())
    view = json.loads((C.REPO / "experiments" / "view_class" / "view_labels.json").read_text())
    oldlab = defaultdict(dict)                  # day -> old key -> list of labels
    for day, d in nwl.items():
        if day in C.EVAL or day not in C.TRAIN + C.VAL:
            continue
        # nw_labels.json: "pos" = NOT a whale (the not-a-whale filter's positive class), "neg" = whale (nw_common.py)
        for k in d.get("pos", []):
            oldlab[day].setdefault(k, []).append({"src": "stage2_linker/nw_labels.json:pos", "nw": "not_whale"})
        for k in d.get("neg", []):
            oldlab[day].setdefault(k, []).append({"src": "stage2_linker/nw_labels.json:neg", "nw": "whale"})
    for k, v in view.items():
        if v["day"] in C.EVAL or v["day"] not in C.TRAIN + C.VAL:
            continue
        oldlab[v["day"]].setdefault(k, []).append({"src": f"view_class/view_labels.json:{v['round']}", "view": v["code"]})
    pool = []
    for day in C.TRAIN + C.VAL:
        S = V.load_scores(C.DAYS / day); rows = C.blob_rows(day)
        # old key -> blob ids of the new instance it maps to (only when that instance is ONE blob and the overlap is most of the old mask)
        mp = json.loads((C.DAYS / day / "instances_v2" / "mapping.json").read_text())["old_to_new"]
        idx = {json.loads(l)["stem"]: json.loads(l) for l in open(C.DAYS / day / "instances_v2" / "index.jsonl")}
        for ok, labs in oldlab[day].items():
            m = mp.get(ok)
            if not m or not m["new"]:
                continue
            stem = ok.rsplit("~", 1)[0]; inst = [i for i in idx[stem]["instances"] if i["key"] == m["new"]]
            if not inst or len(inst[0]["blobs"]) != 1 or m["overlap_px"] < 0.5 * m["old_px"]:
                continue
            for l in labs:
                reuse[(day, stem, inst[0]["blobs"][0])].append({**l, "old_key": ok})
        for stem, r in rows.items():
            for b in r["blobs"]:
                if b["area"] < 300:
                    continue
                nw, pf, pd, gt = S[f"{stem}|{b['b']}"]
                rec = {"day": day, "stem": stem, "b": b["b"], "area": b["area"], "bbox": b["bbox"], "n_blobs": len(r["blobs"]),
                       "nw": round(nw, 3), "pf": round(pf, 3), "pd": round(pd, 3), "gate": round(float(gt), 3), "split": C.split_of(day)}
                rec["stratum"] = stratum(rec); rec["size"] = size_bin(rec["area"])
                if (day, stem, b["b"]) in reuse:
                    rec["reuse"] = reuse[(day, stem, b["b"])]
                pool.append(rec)
        print(day, len(rows), "photos", sum(1 for p in pool if p["day"] == day), "blobs >= 300 px", flush=True)
    C.CACHE.mkdir(parents=True, exist_ok=True)
    from multiprocessing import Pool
    items = [(i, p["day"], p["stem"]) for i, p in enumerate(pool) if p["b"] == 1]
    with Pool(8) as P:
        for i, v in P.imap_unordered(_white, items, chunksize=32):
            pool[i]["white"] = round(v, 3)
    for p in pool:
        p["stratum"] = stratum(p)
    POOL.write_text(json.dumps(pool))
    for sp in ("train", "val"):
        P = [p for p in pool if p["split"] == sp]
        print(sp, len(P), "blobs; strata", dict(Counter(p["stratum"] for p in P)), "; with reusable labels", sum("reuse" in p for p in P))


def known_label(p: dict):
    """A reusable label that needs no look: (category, source) or None. Whales from the view / nw labels are whale
    instances; the AI blob labels' person / drone / boat / shore / whale map directly."""
    for r in p.get("reuse", []):
        if r.get("ai") in AI_MAP:
            return AI_MAP[r["ai"]], r["src"]
    for r in p.get("reuse", []):
        if r.get("view") and r["view"][0] in "FDO" and r["view"] != "X":
            return "whale", r["src"] + f" code {r['view']} ({r['old_key']})"
        if r.get("nw") == "whale":
            return "whale", r["src"] + f" ({r['old_key']})"
    return None


def cmd_draw(a) -> None:
    pool = json.loads(POOL.read_text()); rng = random.Random(a.seed)
    for p in pool:
        p["stratum"] = stratum(p)
    known, look = [], []
    # reusable labels: the ones that need no look go straight in; AI splash / other / unclear and nw "not whale" are looked at again
    for p in pool:
        kl = known_label(p)
        if kl:
            known.append({**{k: p[k] for k in ("day", "stem", "b", "area", "bbox", "split", "stratum", "size")}, "label": kl[0], "source": kl[1]})
        elif "reuse" in p:
            look.append({**p, "forced": True})
    # whales from view / nw labels are plentiful and easy: keep at most a share of them (random)
    by_split = defaultdict(list)
    for k in known:
        by_split[k["split"]].append(k)
    known_kept = []
    for sp, v in by_split.items():
        w = [k for k in v if k["label"] == "whale" and "view_class" in k["source"] or "nw_labels" in k["source"]]
        o = [k for k in v if k not in w]
        rng.shuffle(w); cap = a.cap_view_train if sp == "train" else a.cap_view_val
        known_kept += o + w[:cap]
    chosen = {(k["day"], k["stem"], k["b"]) for k in known_kept} | {(p["day"], p["stem"], p["b"]) for p in look}
    draw = []
    for sp, n in (("train", a.train), ("val", a.val)):
        forced = [p for p in look if p["split"] == sp]
        n_new = max(0, n - len(forced))
        P = [p for p in pool if p["split"] == sp and (p["day"], p["stem"], p["b"]) not in chosen]
        by = defaultdict(list)
        for p in P:
            by[p["stratum"]].append(p)
        for st, w in WEIGHTS.items():
            v = by[st]; k = min(len(v), int(round(w * n_new)))
            # spread over days: round-robin over shuffled per-day lists
            per_day = defaultdict(list)
            for p in v:
                per_day[p["day"]].append(p)
            for d in per_day.values():
                rng.shuffle(d)
            days = sorted(per_day); rng.shuffle(days); got = []
            while len(got) < k:
                for d in days:
                    if per_day[d] and len(got) < k:
                        got.append(per_day[d].pop())
            draw += got
            print(f"{sp} {st}: {len(v)} in pool, {len(got)} drawn")
        draw += forced
        print(f"{sp}: forced re-looks {len(forced)}")
    rng.shuffle(draw)
    tiles = []
    for i, p in enumerate(draw):
        t = {k: p[k] for k in ("day", "stem", "b", "area", "bbox", "split", "stratum", "size")}
        t["tile"] = i + 1
        if p.get("reuse"):
            t["reuse"] = p["reuse"]
        tiles.append(t)
    SAMPLE.write_text(json.dumps({"note": "Blobs drawn for by-eye labelling (tile numbers = sheets lab_NNN.jpg in the scratchpad) and the reusable labels "
                                          "kept without a look. Scores chose which blobs were drawn (strata); the sheets never show a score.",
                                  "weights": WEIGHTS, "seed": a.seed, "tiles": tiles, "known": known_kept}, indent=0))
    print("tiles to label", len(tiles), Counter(t["split"] for t in tiles), "known", len(known_kept), Counter((k["split"], k["label"]) for k in known_kept))


# ---------------------------------------------------------------------------------------------------------- tiles
def outline(img, mask, colour, th):
    cs, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    cv2.drawContours(img, cs, -1, colour, th, cv2.LINE_AA)


def put(img, text, org, scale=0.55):
    cv2.putText(img, text, org, cv2.FONT_HERSHEY_SIMPLEX, scale, (0, 0, 0), 3, cv2.LINE_AA)
    cv2.putText(img, text, org, cv2.FONT_HERSHEY_SIMPLEX, scale, (255, 255, 255), 1, cv2.LINE_AA)


def blob_tile(t: dict, zoom: int = 276, ctx_w: int = 180, img=None, lab=None) -> np.ndarray:
    """Left: the blob's neighbourhood (3 x its box, at least 200 px) with the blob outlined in magenta just outside
    its edge; right: the photo (yellow = the zoom window), with the blob filled faintly. Tile = zoom x (zoom + ctx_w)."""
    img = C.photo(t["day"], t["stem"]) if img is None else img
    lab = C.blob_image(t["day"], t["stem"]) if lab is None else lab
    H, W = img.shape[:2]; m = lab == t["b"]
    x0, y0, x1, y1 = t["bbox"]; side = max(2.6 * max(x1 - x0, y1 - y0), 200.0); cx, cy = (x0 + x1) / 2, (y0 + y1) / 2
    side = min(side, max(W, H))
    wx0 = int(min(max(cx - side / 2, 0), max(W - side, 0))); wy0 = int(min(max(cy - side / 2, 0), max(H - side, 0)))
    win = (wx0, wy0, int(min(wx0 + side, W)), int(min(wy0 + side, H)))
    sc = zoom / max(win[2] - win[0], win[3] - win[1])
    crop = img[win[1]:win[3], win[0]:win[2]]; mc = m[win[1]:win[3], win[0]:win[2]]
    zz = cv2.resize(crop, (max(1, int(crop.shape[1] * sc)), max(1, int(crop.shape[0] * sc))), interpolation=cv2.INTER_AREA if sc < 1 else cv2.INTER_CUBIC)
    mz = cv2.resize(mc.astype(np.uint8), (zz.shape[1], zz.shape[0]), interpolation=cv2.INTER_NEAREST)
    ring = cv2.dilate(mz, np.ones((3, 3), np.uint8), iterations=2)
    outline(zz, ring, (255, 0, 255), 1)
    zt = np.full((zoom, zoom, 3), 25, np.uint8); zt[:zz.shape[0], :zz.shape[1]] = zz
    c = img.copy(); c[m] = (0.6 * c[m] + 0.4 * np.array([255, 0, 255])).astype(np.uint8)
    cv2.rectangle(c, win[:2], win[2:], (0, 255, 255), max(4, int(W / 300)))
    k = ctx_w / W; c = cv2.resize(c, (ctx_w, max(1, int(H * k))), interpolation=cv2.INTER_AREA)
    ct = np.full((zoom, ctx_w, 3), 25, np.uint8); ct[:min(zoom, c.shape[0])] = c[:zoom]
    put(zt, str(t["tile"]), (4, 20), 0.6)
    put(ct, f"{t['area']}px", (4, min(zoom - 8, c.shape[0] + 20)), 0.45)
    return np.hstack([zt, np.full((zoom, 2, 3), 255, np.uint8), ct])


def write_sheet(tiles: list, path: Path, cols: int) -> None:
    h = max(t.shape[0] for t in tiles); w = max(t.shape[1] for t in tiles)
    tiles = [cv2.copyMakeBorder(t, 0, h - t.shape[0], 0, w - t.shape[1], cv2.BORDER_CONSTANT, value=(40, 40, 40)) for t in tiles]
    tiles = [cv2.copyMakeBorder(t, 3, 3, 3, 3, cv2.BORDER_CONSTANT, value=(255, 255, 255)) for t in tiles]
    while len(tiles) % cols:
        tiles.append(np.zeros_like(tiles[0]))
    sheet = np.vstack([np.hstack(tiles[i:i + cols]) for i in range(0, len(tiles), cols)])
    path.parent.mkdir(parents=True, exist_ok=True); cv2.imwrite(str(path), sheet, [cv2.IMWRITE_JPEG_QUALITY, 88])


def cmd_sheets(a) -> None:
    S = json.loads(SAMPLE.read_text())["tiles"]
    if a.only:
        lo, hi = (int(x) for x in a.only.split("-")); S = [t for t in S if lo <= t["tile"] <= hi]
    out = C.SHEETS / "label"; paths = []
    for s in range(0, len(S), a.per):
        part = S[s:s + a.per]; ims = [blob_tile(t) for t in part]
        p = out / f"lab_{(part[0]['tile'] - 1) // a.per + 1:03d}.jpg"; write_sheet(ims, p, a.cols); paths.append(str(p))
    (out / "lab_legend.json").write_text(json.dumps([{k: t[k] for k in ("tile", "day", "stem", "b", "area")} for t in S], indent=0))
    print(len(paths), "sheets in", out)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter); sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("pool").set_defaults(fn=cmd_pool)
    p = sub.add_parser("draw"); p.add_argument("--train", type=int, default=1150); p.add_argument("--val", type=int, default=330); p.add_argument("--seed", type=int, default=0)
    p.add_argument("--cap-view-train", type=int, default=350); p.add_argument("--cap-view-val", type=int, default=70); p.set_defaults(fn=cmd_draw)
    p = sub.add_parser("sheets"); p.add_argument("--per", type=int, default=28); p.add_argument("--cols", type=int, default=4); p.add_argument("--only", default=None); p.set_defaults(fn=cmd_sheets)
    a = ap.parse_args(argv); a.fn(a)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
