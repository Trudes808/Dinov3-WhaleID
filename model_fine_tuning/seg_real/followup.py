"""S5 follow-up (LOG 2026-11-17): can its two costs (thin / wet whale parts dropped, flukes split) be removed without
retraining? Thresholds and hysteresis on the cached P(whale), a re-tuned fluke-join gap, and a researcher sheet.

    PYTHONPATH=. .venv/bin/python -m model_fine_tuning.seg_real.followup sweep --name s5 --split train
        every setting of GRID on the cached photos of the split (CPU): blob rows, views kept / added, new blobs per
        photo, and on 2025-01-02 (train) a component proxy for flukes / main whales / other whales
        -> experiments/seg_splash/followup_sweep_<name>_<split>.json
    flock LOCK .venv/bin/python -m model_fine_tuning.seg_real.followup e2e --name s5 --setting h0.6_0.2 --checkpoint CKPT
        instances_v2 candidates + assemble (unchanged, imported) with segment() replaced by the hysteresis mask
        -> CACHE/e2e/<name>_<setting>/<day>/instances_v2
    PYTHONPATH=. .venv/bin/python -m model_fine_tuning.seg_real.followup join --name s5_h0.6_0.2 --gaps 0.25,0.35,0.45,0.6
        instances_v2 assemble only (CPU) with other join gaps, on links to the candidates of that run
    PYTHONPATH=. .venv/bin/python -m model_fine_tuning.seg_real.followup researcher --name s5 --setting h0.6_0.2
        -> SHEETS/for_researcher_01.jpg, _02.jpg (+ legend)

Hysteresis "h<hi>_<lo>": components of P >= lo that contain a pixel with P >= hi. Threshold "t<t>": P >= t.
"""
from __future__ import annotations

import argparse
import json
import sys
from argparse import Namespace
from collections import Counter, defaultdict
from multiprocessing import Pool
from pathlib import Path

import cv2
import numpy as np

from . import common as C
from .data import blob_1280
from .e2e import E2E, shadow
from .predict import CLICK_DAYS, PRED, load_marks, load_prob, view_keys

GRID = ["t0.5", "t0.4", "t0.3", "t0.2"] + [f"h{h}_{l}" for h in (0.5, 0.6, 0.7) for l in (0.1, 0.2, 0.3, 0.4)]
GEAR = {"person", "boat", "pole", "drone", "shore"}


def mask_of(p: np.ndarray, s: str) -> np.ndarray:
    if s[0] == "t":
        return p >= float(s[1:])
    hi, lo = (float(x) for x in s[1:].split("_"))
    n, lab = cv2.connectedComponents((p >= lo).astype(np.uint8), connectivity=8)
    keep = np.zeros(n, bool); keep[np.unique(lab[p >= hi])] = True; keep[0] = False
    return keep[lab]


def comp_at(lab: np.ndarray, x: float, y: float, tol: float = 12.5):
    H, W = lab.shape; xi, yi = int(round(x)), int(round(y)); t = int(np.ceil(tol))
    if 0 <= xi < W and 0 <= yi < H and lab[yi, xi]:
        return int(lab[yi, xi])
    y0, y1, x0, x1 = max(yi - t, 0), min(yi + t + 1, H), max(xi - t, 0), min(xi + t + 1, W)
    ys, xs = np.nonzero(lab[y0:y1, x0:x1])
    if not len(ys):
        return None
    d = np.hypot(xs + x0 - x, ys + y0 - y); k = int(np.argmin(d))
    return int(lab[ys[k] + y0, xs[k] + x0]) if d[k] <= tol else None


_CTX = {}


def _photo(args):
    name, day, stem, settings = args
    ctx = _CTX
    p = load_prob(name, day, stem); H, W = p.shape; L = C.blob_image(day, stem)
    old_any = cv2.resize((L > 0).astype(np.uint8), (W, H), interpolation=cv2.INTER_NEAREST) > 0
    blobs = ctx["blobs"].get((day, stem), []); view = ctx["views"].get((day, stem)); clk = ctx["clicks"].get((day, stem))
    bm = {b["b"]: blob_1280(L, b["b"], (H, W)) for b in blobs}
    vold = blob_1280(L, view, (H, W)) if view else None
    out = {}
    for s in settings:
        m = mask_of(p, s); r = {"blobs": [], "new": 0}
        for b in blobs:
            n = int(bm[b["b"]].sum())
            if n:
                r["blobs"].append((b["label"], b["sure"], int(m[bm[b["b"]]].sum()), n))
        n, lab = cv2.connectedComponents(m.astype(np.uint8), connectivity=8)
        if n > 1:
            area = np.bincount(lab.ravel()); touch = np.unique(lab[old_any])
            r["new"] = int(sum(1 for i in range(1, n) if area[i] >= 75 and i not in touch))
        if vold is not None and vold.sum() >= 50:
            ids, cnt = np.unique(lab[vold], return_counts=True); ok = ids > 0
            if ok.any():
                comp = lab == ids[ok][np.argmax(cnt[ok])]; r["view"] = (int(m[vold].sum()), int(vold.sum()), int((comp & ~vold).sum()), int(comp.sum()))
            else:
                r["view"] = (0, int(vold.sum()), 0, 0)
        if clk:
            main, others = clk; P = {k: (q["x"] * W, q["y"] * H) for k, q in main.items()}
            cm = {k: comp_at(lab, *xy) for k, xy in P.items()}
            v = [c for c in cm.values() if c is not None]; mj = Counter(v).most_common(1)[0][0] if v else None
            r["main_one"] = None not in cm.values() and len(set(cm.values())) == 1
            if "fluke_tip_left" in cm and "fluke_tip_right" in cm:
                r["fluke"] = cm["fluke_tip_left"] is not None and cm["fluke_tip_left"] == cm["fluke_tip_right"]
            r["others"] = []
            for w in others:
                c2 = [comp_at(lab, q["x"] * W, q["y"] * H) for q in w.values()]; c2 = [c for c in c2 if c is not None]
                o = Counter(c2).most_common(1)[0][0] if c2 else None
                r["others"].append("none" if o is None else "fused" if o == mj else "own")
        out[s] = r
    return day, stem, out


def collect(name: str, split: str, settings: list, click_days=None):
    days = C.TRAIN if split == "train" else C.VAL
    _CTX["blobs"] = defaultdict(list)
    for b in C.load_labels():
        if b["split"] == split:
            _CTX["blobs"][(b["day"], b["stem"])].append(b)
    _CTX["views"] = {}
    for k, (day, code) in view_keys().items():
        if day not in days:
            continue
        mp = json.loads((C.DAYS / day / "instances_v2" / "mapping.json").read_text())["old_to_new"] if (day, "mp") not in _CTX else _CTX[(day, "mp")]
        _CTX[(day, "mp")] = mp
    for k, (day, code) in view_keys().items():
        if day not in days:
            continue
        mp = _CTX[(day, "mp")]; m = mp.get(k); stem = k.rsplit("~", 1)[0]
        if not m or not m["new"]:
            continue
        if (day, "idx") not in _CTX:
            _CTX[(day, "idx")] = {json.loads(l)["stem"]: json.loads(l) for l in open(C.DAYS / day / "instances_v2" / "index.jsonl")}
        inst = [i for i in _CTX[(day, "idx")].get(stem, {"instances": []})["instances"] if i["key"] == m["new"]]
        if inst and len(inst[0]["blobs"]) == 1:
            _CTX["views"][(day, stem)] = inst[0]["blobs"][0]
    _CTX["clicks"] = {}
    for day in (click_days or []):
        photos, oth = load_marks(day)
        for key, kp in photos.items():
            _CTX["clicks"][(day, key.rsplit("~", 1)[0])] = (kp, [w for w in (oth.get(key) or []) if w])
    todo = sorted({(d, p.stem) for d in days for p in (PRED / name / d).glob("*.png")} if True else [])
    with Pool(8) as P:
        res = list(P.imap_unordered(_photo, [(name, d, s, settings) for d, s in todo], chunksize=8))
    return res


def summarise(res, settings):
    S = {}
    for s in settings:
        cat = defaultdict(lambda: [0, 0, 0, 0]); vk = [0, 0, 0, 0]; new = []; fl = []; mo = []; ot = Counter()
        for day, stem, out in res:
            r = out[s]; new.append(r["new"])
            for lab, sure, c, n in r["blobs"]:
                for g in ([lab] + (["whale (sure)"] if lab == "whale" and sure else []) + (["gear"] if lab in GEAR else []) + (["ALL non-whale"] if lab in C.NEG else [])):
                    cat[g][0] += c; cat[g][1] += n; cat[g][2] += c / n >= 0.3; cat[g][3] += 1
            if "view" in r:
                vk = [a + b for a, b in zip(vk, r["view"])]
            if "fluke" in r:
                fl.append(r["fluke"])
            if "main_one" in r:
                mo.append(r["main_one"]); ot.update(r["others"])
        S[s] = {"rows": {g: {"n": v[3], "pixel_share": round(v[0] / v[1], 4), "ge30_n": v[2]} for g, v in cat.items()},
                "view_kept": round(vk[0] / max(vk[1], 1), 4), "view_added": round(vk[2] / max(vk[3], 1), 4),
                "new_per_photo": round(float(np.mean(new)), 4), "photos": len(new),
                "proxy_flukes_one": f"{sum(fl)}/{len(fl)}", "proxy_main_one": f"{sum(mo)}/{len(mo)}", "proxy_others": dict(ot)}
    return S


def cmd_sweep(a) -> None:
    settings = a.settings.split(",") if a.settings else GRID
    res = collect(a.name, a.split, settings, click_days=["2025-01-02"] if a.split == "train" else [])
    S = summarise(res, settings)
    (C.EXP / f"followup_sweep_{a.name}_{a.split}.json").write_text(json.dumps(S, indent=1))
    print(f"{'setting':10s} {'kept':>6s} {'added':>6s} {'new/ph':>7s} {'splash':>7s} {'wake':>7s} {'water':>7s} {'gear':>7s} {'nonwh':>7s} {'whale_s':>7s} {'fluke':>6s} {'main':>8s} others")
    for s in settings:
        R = S[s]["rows"]; g = lambda k: R.get(k, {}).get("pixel_share", float("nan"))
        print(f"{s:10s} {S[s]['view_kept']:6.3f} {S[s]['view_added']:6.3f} {S[s]['new_per_photo']:7.3f} {g('splash'):7.3f} {g('wake'):7.3f} {g('water'):7.3f} {g('gear'):7.3f} "
              f"{g('ALL non-whale'):7.3f} {g('whale (sure)'):7.3f} {S[s]['proxy_flukes_one']:>6s} {S[s]['proxy_main_one']:>8s} {S[s]['proxy_others']}")


def cmd_e2e(a) -> None:
    """instances_v2 unchanged, with segment() returning the setting's mask (monkeypatched in segmentation.predict)."""
    import torch
    from ..segmentation import predict as SP
    from ..cluster import instances_v2 as V
    orig = SP.segment

    def seg(model, image, device, max_side=1280, threshold=0.5, flip_tta=False):
        small, _ = SP.resize_long_side(image, max_side)
        prob, _ = orig(model, small, device, max_side, threshold, flip_tta)          # P at 1280 (no rescale: already 1280)
        m = mask_of(prob, a.setting).astype(np.uint8)
        m = cv2.resize(m, (image.shape[1], image.shape[0]), interpolation=cv2.INTER_NEAREST) > 0
        return None, m
    SP.segment = seg
    name = f"{a.name}_{a.setting}"
    for day in a.days.split(","):
        d = shadow(name, day)
        if not (d / "instances_v2" / "feat.npz").exists():
            V.cmd_candidates(Namespace(day_dir=str(d), collect=100, workers=4, seg_checkpoint=str(Path(a.checkpoint).resolve())))
        V.cmd_assemble(Namespace(day_dir=str(d), **V.DEFAULTS))


def cmd_join(a) -> None:
    from ..cluster import instances_v2 as V
    for g in a.gaps.split(","):
        name = f"{a.name}_g{g}"
        for day in a.days.split(","):
            src = E2E / a.name / day; d = shadow(name, day); iv = d / "instances_v2"
            for f in ("blobs", "blobs.jsonl", "feat.npz", "candidates.json"):
                if not (iv / f).exists():
                    (iv / f).symlink_to(src / "instances_v2" / f)
            V.cmd_assemble(Namespace(day_dir=str(d), **{**V.DEFAULTS, "join_gap": float(g)}))


def cmd_researcher(a) -> None:
    from .sheets import panel, put
    from .predict import resize_1280
    per = json.loads((C.CACHE / "per_blob_scores.json").read_text()); L = C.load_labels(); rng = np.random.default_rng(0)
    val = [b for b in L if b["split"] == "val"]; sh = lambda b: per["r04"].get(C.key_of(b["day"], b["stem"], b["b"]), 0)
    def pick(items, n):
        seen, out = set(), []
        for i in rng.permutation(len(items)):
            b = items[i]
            if (b["day"], b["stem"]) not in seen:
                seen.add((b["day"], b["stem"])); out.append(b)
            if len(out) == n:
                break
        return out
    sets = [("splash / wake", pick([b for b in val if b["label"] in ("splash", "wake") and sh(b) >= 0.3], 12)),
            ("gear", pick([b for b in val if b["label"] in GEAR | {"bird"} and sh(b) >= 0.3], 12)),
            ("whale in wash", pick([b for b in val if b["label"] == "whale_splash"] + [b for b in val if b["label"] == "whale" and b["stratum"] == "main_white"], 12))]
    PP = json.loads((C.CACHE / "per_photo_whales.json").read_text())[a.name]
    low = sorted([r for r in PP if r["split"] == "val"], key=lambda r: r["kept"])[:12]; rows = C.blob_rows(low[0]["day"]) if low else {}
    lw = []
    for r in low:
        bb = [b for b in C.blob_rows(r["day"])[r["stem"]]["blobs"] if b["b"] == r["b"]][0]
        lw.append({"day": r["day"], "stem": r["stem"], "b": r["b"], "bbox": bb["bbox"], "label": f"view {r['code']}", "sure": True, "split": "val"})
    sets.append(("whales S5 keeps least", lw))
    tiles, legend = [], []
    for group, items in sets:
        for b in items:
            img = resize_1280(C.photo(b["day"], b["stem"])); po = load_prob("r04", b["day"], b["stem"])
            pn = mask_of(load_prob(a.name, b["day"], b["stem"]), a.setting).astype(np.float32)
            blob = blob_1280(C.blob_image(b["day"], b["stem"]), b["b"], po.shape); H, W = po.shape
            x0, y0, x1, y1 = (v / 2 for v in b["bbox"]); cx, cy = (x0 + x1) / 2, (y0 + y1) / 2
            w = max((x1 - x0) * 3.0, 200); h = max((y1 - y0) * 3.0, 133); w, h = (h * 1.5, h) if w / h < 1.5 else (w, w / 1.5); w, h = min(w, W), min(h, H)
            wx = int(min(max(cx - w / 2, 0), W - w)); wy = int(min(max(cy - h / 2, 0), H - h)); win = (wx, wy, int(wx + w), int(wy + h))
            A, B = panel(img, po, blob, win), panel(img, pn, blob, win); n = len(tiles) + 1
            put(A, f"{n} now (r04)", (4, 16)); put(B, "new (S5)", (4, 16)); put(A, group, (4, 206))
            tiles.append(cv2.copyMakeBorder(np.hstack([A, np.full((214, 2, 3), 255, np.uint8), B]), 2, 2, 2, 2, cv2.BORDER_CONSTANT, value=(200, 200, 200)))
            legend.append({"tile": n, "group": group, "day": b["day"], "stem": b["stem"], "blob": b["b"], "label": b["label"]})
    for s in range(0, len(tiles), 24):
        part = tiles[s:s + 24]
        while len(part) % 3:
            part.append(np.zeros_like(tiles[0]))
        cv2.imwrite(str(C.SHEETS / f"for_researcher_{s // 24 + 1:02d}.jpg"), np.vstack([np.hstack(part[i:i + 3]) for i in range(0, len(part), 3)]), [cv2.IMWRITE_JPEG_QUALITY, 88])
    (C.SHEETS / "for_researcher_legend.json").write_text(json.dumps({"setting": a.setting, "model": a.name, "tiles": legend}, indent=0))
    print(len(tiles), "tiles", [(g, len(i)) for g, i in sets])


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter); sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("sweep"); p.add_argument("--name", default="s5"); p.add_argument("--split", default="train"); p.add_argument("--settings", default=None); p.set_defaults(fn=cmd_sweep)
    p = sub.add_parser("e2e"); p.add_argument("--name", default="s5"); p.add_argument("--setting", required=True); p.add_argument("--checkpoint", required=True)
    p.add_argument("--days", default=",".join(C.VAL + CLICK_DAYS)); p.set_defaults(fn=cmd_e2e)
    p = sub.add_parser("join"); p.add_argument("--name", required=True); p.add_argument("--gaps", default="0.25,0.35,0.45,0.6"); p.add_argument("--days", default=",".join(C.VAL + CLICK_DAYS)); p.set_defaults(fn=cmd_join)
    p = sub.add_parser("researcher"); p.add_argument("--name", default="s5"); p.add_argument("--setting", required=True); p.set_defaults(fn=cmd_researcher)
    a = ap.parse_args(argv); a.fn(a)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
