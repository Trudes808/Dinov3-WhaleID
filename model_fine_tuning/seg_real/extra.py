"""The area a candidate ADDS to a certain whale: pixels of its main component outside the r04 blob (view-labelled main
whales, score.py's set). Numbers + a sheet of the 30 photos with the largest added share (scratchpad).

    PYTHONPATH=. .venv/bin/python -m model_fine_tuning.seg_real.extra --names r05,s1
        -> experiments/seg_splash/extra_area.json; SHEETS/extra_<name>_NN.jpg (left: r04 blob outlined yellow,
           right: the candidate's main component, the ADDED pixels cyan, the kept ones red, lost r04 pixels yellow line)
"""
from __future__ import annotations

import argparse
import json

import cv2
import numpy as np

from . import common as C
from .data import blob_1280
from .predict import load_prob, resize_1280


def comp_of(p, old):
    n, lab = cv2.connectedComponents(p.astype(np.uint8), connectivity=8)
    ids, cnt = np.unique(lab[old], return_counts=True); cnt = cnt[ids > 0]; ids = ids[ids > 0]
    return (lab == ids[np.argmax(cnt)]) if len(ids) else np.zeros_like(p)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(); ap.add_argument("--names", default="r05,s1"); ap.add_argument("--n", type=int, default=30); a = ap.parse_args(argv)
    PP = json.loads((C.CACHE / "per_photo_whales.json").read_text())
    f = C.EXP / "extra_area.json"; R = json.loads(f.read_text()) if f.exists() else {}
    for name in a.names.split(","):
        rows = []
        for r in PP[name]:
            p = load_prob(name, r["day"], r["stem"]) >= 0.5; old = blob_1280(C.blob_image(r["day"], r["stem"]), r["b"], p.shape)
            comp = comp_of(p, old); add = comp & ~old
            # where the added pixels lie: within 4 px of the r04 outline (a halo) or farther
            ring = cv2.dilate(old.astype(np.uint8), np.ones((9, 9), np.uint8)) > 0
            rows.append({**r, "comp_px": int(comp.sum()), "old_px": int(old.sum()), "added_px": int(add.sum()), "added_halo_px": int((add & ring).sum()),
                         "added_share": round(float(add.sum() / max(comp.sum(), 1)), 4)})
        res = {}
        for sp in ("val", "train", "all"):
            v = [x for x in rows if sp == "all" or x["split"] == sp]
            a_ = np.array([x["added_px"] for x in v], float); c_ = np.array([x["comp_px"] for x in v], float); h_ = np.array([x["added_halo_px"] for x in v], float)
            s_ = np.array([x["added_share"] for x in v])
            res[sp] = {"photos": len(v), "added_share_pooled": round(float(a_.sum() / c_.sum()), 4), "added_share_median": round(float(np.median(s_)), 4),
                       "photos_added_gt_10pct": round(float((s_ > 0.1).mean()), 4), "photos_added_gt_25pct": round(float((s_ > 0.25).mean()), 4),
                       "added_within_4px_of_r04_outline": round(float(h_.sum() / max(a_.sum(), 1)), 4)}
        R[name] = {"summary": res, "top": sorted(rows, key=lambda x: -x["added_share"])[:a.n]}
        print(name, json.dumps(res))
        tiles = []
        for i, r in enumerate(R[name]["top"]):
            img = resize_1280(C.photo(r["day"], r["stem"])); p = load_prob(name, r["day"], r["stem"]) >= 0.5
            old = blob_1280(C.blob_image(r["day"], r["stem"]), r["b"], p.shape); comp = comp_of(p, old)
            ys, xs = np.nonzero(comp | old); H, W = p.shape
            x0, x1, y0, y1 = xs.min(), xs.max(), ys.min(), ys.max(); cx, cy = (x0 + x1) / 2, (y0 + y1) / 2
            w = max((x1 - x0) * 1.5, 160); h = max((y1 - y0) * 1.5, 107)
            w, h = (h * 1.5, h) if w / h < 1.5 else (w, w / 1.5); w, h = min(w, W), min(h, H)
            wx = int(min(max(cx - w / 2, 0), W - w)); wy = int(min(max(cy - h / 2, 0), H - h)); sl = (slice(wy, int(wy + h)), slice(wx, int(wx + w)))
            A = img[sl].copy(); B_ = img[sl].copy(); o, c = old[sl], comp[sl]
            cs, _ = cv2.findContours(o.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE); cv2.drawContours(A, cs, -1, (0, 255, 255), 1)
            B_[c & o] = (0.55 * B_[c & o] + 0.45 * np.array([0, 0, 255])).astype(np.uint8)
            B_[c & ~o] = (0.4 * B_[c & ~o] + 0.6 * np.array([255, 255, 0])).astype(np.uint8)
            cv2.drawContours(B_, cs, -1, (0, 255, 255), 1)
            def fit(v):
                k = min(320 / v.shape[1], 214 / v.shape[0]); v = cv2.resize(v, (max(1, int(v.shape[1] * k)), max(1, int(v.shape[0] * k))), interpolation=cv2.INTER_AREA if k < 1 else cv2.INTER_CUBIC)
                out = np.full((214, 320, 3), 30, np.uint8); out[:v.shape[0], :v.shape[1]] = v; return out
            A, B_ = fit(A), fit(B_)
            for im, t, org in ((A, f"{i + 1} r04 blob", (4, 16)), (B_, f"{name}: +{r['added_share']:.2f} added (cyan)", (4, 16)), (A, f"view {r['code']} {r['split']}", (4, 206))):
                cv2.putText(im, t, org, cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 3, cv2.LINE_AA); cv2.putText(im, t, org, cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
            tiles.append(cv2.copyMakeBorder(np.hstack([A, np.full((214, 2, 3), 255, np.uint8), B_]), 2, 2, 2, 2, cv2.BORDER_CONSTANT, value=(200, 200, 200)))
        for s in range(0, len(tiles), 24):
            part = tiles[s:s + 24]
            while len(part) % 3:
                part.append(np.zeros_like(tiles[0]))
            cv2.imwrite(str(C.SHEETS / f"extra_{name}_{s // 24 + 1:02d}.jpg"), np.vstack([np.hstack(part[i:i + 3]) for i in range(0, len(part), 3)]), [cv2.IMWRITE_JPEG_QUALITY, 87])
        (C.SHEETS / f"extra_{name}_legend.json").write_text(json.dumps([{k: r[k] for k in ("day", "stem", "b", "code", "split", "added_share")} for r in R[name]["top"]], indent=0))
    f.write_text(json.dumps({k: {"summary": v["summary"], "top": [{x: t[x] for x in ("day", "stem", "b", "split", "added_share")} for t in v["top"]]} for k, v in R.items()}, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
