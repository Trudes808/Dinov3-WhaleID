"""Before / after sheets on VALIDATION-day photos (scratchpad only, never in the repo): the live mask against a variant.

    PYTHONPATH=. .venv/bin/python -m model_fine_tuning.seg_real.sheets --old r04 --new s2
        -> SHEETS/cmp_<new>_<set>_NN.jpg + cmp_<new>_<set>_legend.json, sets:
           splash   48 photos with a labelled splash / wake blob the live mask calls whale (>= 30 % of its pixels)
           whale_splash 24 whale-under-splash blobs (validation days, topped up from training days if short; marked)
           gear     24 pole / person / boat / shore / drone blobs the live mask calls whale
           easy     24 random sure whale blobs of the main_ok stratum (clean whales: nothing should change)
Tile: left the live mask, right the variant (whale = red fill + outline), the labelled blob outlined in yellow;
caption: tile number, label, share of the blob called whale old -> new.
"""
from __future__ import annotations

import argparse
import json
import random

import cv2
import numpy as np

from . import common as C
from .data import blob_1280
from .predict import load_prob, resize_1280


def panel(img, prob, blob, win, size=(320, 214)):
    x0, y0, x1, y1 = win; v = img[y0:y1, x0:x1].copy(); m = prob[y0:y1, x0:x1] >= 0.5; b = blob[y0:y1, x0:x1]
    v[m] = (0.55 * v[m] + 0.45 * np.array([0, 0, 255])).astype(np.uint8)
    cs, _ = cv2.findContours(m.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE); cv2.drawContours(v, cs, -1, (0, 0, 255), 1)
    k = min(size[0] / v.shape[1], size[1] / v.shape[0]); v = cv2.resize(v, (max(1, int(v.shape[1] * k)), max(1, int(v.shape[0] * k))), interpolation=cv2.INTER_AREA if k < 1 else cv2.INTER_CUBIC)
    bb = cv2.resize(b.astype(np.uint8), (v.shape[1], v.shape[0]), interpolation=cv2.INTER_NEAREST)
    bb = cv2.dilate(bb, np.ones((3, 3), np.uint8), iterations=2) & ~bb
    cs, _ = cv2.findContours(cv2.dilate(b.astype(np.uint8), np.ones((3, 3), np.uint8)).astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    cs = [(c * k).astype(np.int32) for c in cs]; cv2.drawContours(v, cs, -1, (0, 255, 255), 1, cv2.LINE_AA)
    out = np.full((size[1], size[0], 3), 30, np.uint8); out[:v.shape[0], :v.shape[1]] = v
    return out


def put(img, t, org):
    cv2.putText(img, t, org, cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 3, cv2.LINE_AA); cv2.putText(img, t, org, cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)


def tile(b, old, new, n):
    img = resize_1280(C.photo(b["day"], b["stem"])); po, pn = load_prob(old, b["day"], b["stem"]), load_prob(new, b["day"], b["stem"])
    blob = blob_1280(C.blob_image(b["day"], b["stem"]), b["b"], po.shape); H, W = po.shape
    x0, y0, x1, y1 = (v / 2 for v in b["bbox"]); cx, cy = (x0 + x1) / 2, (y0 + y1) / 2
    w = max((x1 - x0) * 3.0, 200); h = max((y1 - y0) * 3.0, 200 / 1.5)
    if w / h < 1.5:
        w = h * 1.5
    else:
        h = w / 1.5
    w, h = min(w, W), min(h, H); wx = int(min(max(cx - w / 2, 0), W - w)); wy = int(min(max(cy - h / 2, 0), H - h)); win = (wx, wy, int(wx + w), int(wy + h))
    so, sn = float((po[blob] >= 0.5).mean()), float((pn[blob] >= 0.5).mean())
    a, c = panel(img, po, blob, win), panel(img, pn, blob, win)
    put(a, f"{n} {old}", (4, 16)); put(c, f"{new}", (4, 16))
    put(a, f"{b['label']}{'' if b['sure'] else '?'} {so:.2f}->{sn:.2f}", (4, 206))
    t = np.hstack([a, np.full((214, 2, 3), 255, np.uint8), c])
    return cv2.copyMakeBorder(t, 2, 2, 2, 2, cv2.BORDER_CONSTANT, value=(200, 200, 200)), {"tile": n, "day": b["day"], "stem": b["stem"], "b": b["b"], "label": b["label"],
                                                                                          "sure": b["sure"], "split": b["split"], f"share_{old}": round(so, 3), f"share_{new}": round(sn, 3)}


def write(items, old, new, tag, cols=3, per=24):
    paths, legend = [], []
    for s in range(0, len(items), per):
        tiles = []
        for j, b in enumerate(items[s:s + per]):
            t, l = tile(b, old, new, s + j + 1); tiles.append(t); legend.append(l)
        while len(tiles) % cols:
            tiles.append(np.zeros_like(tiles[0]))
        sheet = np.vstack([np.hstack(tiles[i:i + cols]) for i in range(0, len(tiles), cols)])
        p = C.SHEETS / f"cmp_{new}_{tag}_{s // per + 1:02d}.jpg"; cv2.imwrite(str(p), sheet, [cv2.IMWRITE_JPEG_QUALITY, 87]); paths.append(str(p))
    (C.SHEETS / f"cmp_{new}_{tag}_legend.json").write_text(json.dumps(legend, indent=0))
    return paths


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--old", default="r04"); ap.add_argument("--new", required=True); ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--sets", default="splash,whale_splash,gear,easy,lowiou"); a = ap.parse_args(argv)
    per = json.loads((C.CACHE / "per_blob_scores.json").read_text())
    L = C.load_labels(); rng = random.Random(a.seed)
    sh = lambda b: per[a.old].get(C.key_of(b["day"], b["stem"], b["b"]), 0)
    val = [b for b in L if b["split"] == "val"]
    sets = {}
    s = [b for b in val if b["label"] in ("splash", "wake") and sh(b) >= 0.3]; rng.shuffle(s)
    one = {}
    for b in s:                                   # one blob per photo
        one.setdefault((b["day"], b["stem"]), b)
    sets["splash"] = list(one.values())[:48]
    w = [b for b in val if b["label"] == "whale_splash"]; rng.shuffle(w)
    if len(w) < 24:
        t = [b for b in L if b["split"] == "train" and b["label"] == "whale_splash"]; rng.shuffle(t); w += t[:24 - len(w)]
    sets["whale_splash"] = w[:24]
    g = [b for b in val if b["label"] in ("pole", "person", "boat", "shore", "drone") and sh(b) >= 0.3]; rng.shuffle(g); sets["gear"] = g[:24]
    e = [b for b in val if b["label"] == "whale" and b["sure"] and b["stratum"] == "main_ok"]; rng.shuffle(e)
    if len(e) < 24:
        e += [b for b in val if b["label"] == "whale" and b["sure"] and b not in e][:24 - len(e)]
    sets["easy"] = e[:24]
    # lowiou: the 30 view-labelled main whales (train + val days pooled; val alone has 77) with the lowest IoU of the
    # new main blob against the r04 blob (score.py per_photo_whales.json): what was lost, wash or skin?
    PP = json.loads((C.CACHE / "per_photo_whales.json").read_text()).get(a.new, [])
    low = sorted(PP, key=lambda r: r["iou"])[:30]; rows = {}
    items = []
    for r in low:
        if r["day"] not in rows:
            rows[r["day"]] = C.blob_rows(r["day"])
        bb = [b for b in rows[r["day"]][r["stem"]]["blobs"] if b["b"] == r["b"]][0]
        items.append({"day": r["day"], "stem": r["stem"], "b": r["b"], "bbox": bb["bbox"], "label": f"view {r['code']} IoU {r['iou']:.2f}", "sure": True, "split": r["split"]})
    sets["lowiou"] = items
    for tag, items in sets.items():
        if tag not in a.sets.split(","):
            continue
        print(tag, len(items), write(items, a.old, a.new, tag))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
