"""W3e (rule A on real AND synthetic dorsal labels) against W3c / W3d on renders: the render-side measures.

    flock LOCK .venv/bin/python experiments/parts_tri_expand/w3e_scores.py pred  --model NAME --checkpoint CKPT
        -> CACHE/render_parts/<model>/r06r_check21/<sid>.png      full frame at 1280 (as pred_fill.py render-pred)
           CACHE/render_parts_hr/<model>/r06r_check21/<sid>~k.png  per-whale window, predict.py's hr_window (as real "hr")
           CACHE/render_parts_zoom/<model>/r06r_check21/<sid>~k.png fin window (4 x the fin's size, >= 128 px, at 2 x)
    flock LOCK .venv/bin/python experiments/parts_tri_expand/w3e_scores.py synth --model NAME --checkpoint CKPT
        -> CACHE/synth_val/<model>.json  run05 val IoU per class against the OLD label cache and the rule-A cache
    .venv/bin/python experiments/parts_tri_expand/w3e_scores.py score --models ...  -> w3e_renders.json

Render fins: r06r_check21, every whale with all three fin keypoints visible (131). Truth in the 2560 frame: OLD =
parts/ class 2 of the whale's own pixels; NEW = the factory layer dorsal_triA == k (rule A, round 21d). The predicted
region R = predicted dorsal pixels on the own whale (grown 4 px) connected to the triangle grown 14 px (pred_fill.py's
near_triangle at 2560). Also the helper's protocol (1280 frame, old label, RF windows) for the full maps as a check.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
from model_fine_tuning.parts_weak import tri_expand as TE  # noqa: E402

FAC = REPO.parent / "Whale-Image-Data-Factory" / "data" / "renders"
CACHE = Path("/home/sat3737/NOAA_whales/moonshot_cache/parts_tri_expand")
OUT = Path(__file__).resolve().parent
ST = "r06r_check21"
FP = ("dorsal_fin_tip", "dorsal_fin_front", "dorsal_fin_back")
FIN = 2


def rd(p):
    a = np.asarray(Image.open(p))
    return a[..., 0] if a.ndim == 3 else a


def fins(st=ST):
    ds = FAC / st / "dataset"
    out = []
    for mp in sorted((ds / "meta").glob("*.json")):
        m = json.loads(mp.read_text())
        for a in m.get("animals") or []:
            if all((a["keypoints"].get(n) or {}).get("state") == "visible" for n in FP):
                T, F, B = (np.array(a["keypoints"][n]["uv"], float) for n in FP)
                out.append({"sid": m["id"], "k": a["instance_id"], "T": T, "F": F, "B": B,
                            "sight_deg": ((a.get("part_planes") or {}).get("fin") or {}).get("sight_deg"), "n_animals": m.get("n_animals", 1)})
    return out


def zoom_window(f):
    fr = TE.frame(f["T"], f["F"], f["B"])
    size = max(fr["base"], fr["h_eff"]) if fr else 32.0
    pts = np.stack([f["T"], f["F"], f["B"]])
    c = (pts.min(0) + pts.max(0)) / 2
    side = max(4.0 * size, 128.0)
    osz = int(min(768, max(256, round(side * 2 / 16) * 16)))
    return float(c[0] - side / 2), float(c[1] - side / 2), side, osz


def cmd_pred(a) -> None:
    import torch
    from model_fine_tuning.common.data import pad_to_multiple, to_tensor
    from model_fine_tuning.parts_seg.model import PartsSegModel
    from model_fine_tuning.parts_seg.predict import hr_window, run_model
    dev = torch.device("cuda")
    model, _ = PartsSegModel.from_checkpoint(Path(a.checkpoint), dev)
    model.eval()
    F = fins()
    ds = FAC / ST / "dataset"
    o_full, o_hr, o_zoom = (CACHE / d / a.model / ST for d in ("render_parts", "render_parts_hr", "render_parts_zoom"))
    for d in (o_full, o_hr, o_zoom):
        d.mkdir(parents=True, exist_ok=True)
    wins = {}
    for sid in sorted({f["sid"] for f in F}):
        im = cv2.cvtColor(cv2.imread(str(ds / "images" / f"{sid}.png")), cv2.COLOR_BGR2RGB)
        H, W = im.shape[:2]
        if not (o_full / f"{sid}.png").exists():
            s = 1280 / max(H, W)
            sm = cv2.resize(im, (round(W * s), round(H * s)), interpolation=cv2.INTER_AREA)
            padded, (h, w) = pad_to_multiple(sm)
            cv2.imwrite(str(o_full / f"{sid}.png"), run_model(model, to_tensor(padded), dev)[:h, :w])
        inst = rd(ds / "instances" / f"{sid}.png")
        for f in [f for f in F if f["sid"] == sid]:
            ys, xs = np.nonzero(inst == f["k"])
            hr = hr_window([float(xs.min()), float(ys.min()), float(xs.max() + 1), float(ys.max() + 1)], W, H)
            for kind, (hx, hy, side, osz), od in (("hr", hr, o_hr), ("zoom", zoom_window(f), o_zoom)):
                M = np.array([[osz / side, 0, -hx * osz / side], [0, osz / side, -hy * osz / side]])
                crop = cv2.warpAffine(im, M, (osz, osz), flags=cv2.INTER_AREA if side > osz else cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=0)
                cv2.imwrite(str(od / f"{sid}~{f['k']}.png"), run_model(model, to_tensor(crop), dev))
                wins[f"{kind}|{sid}~{f['k']}"] = [hx, hy, side, osz]
    (CACHE / "render_parts_hr" / a.model / f"{ST}_windows.json").write_text(json.dumps(wins))
    print(a.model, len(F), "fins")


def cmd_synth(a) -> None:
    """run05 val: per-class IoU against the old cache and the rule-A cache, same predictions."""
    import torch
    from model_fine_tuning.common.data import IGNORE, WhaleRun, pad_to_multiple, to_tensor
    from model_fine_tuning.parts_seg.labels import CLASSES
    from model_fine_tuning.parts_seg.model import PartsSegModel
    dev = torch.device("cuda")
    model, _ = PartsSegModel.from_checkpoint(Path(a.checkpoint), dev)
    model.eval()
    run = WhaleRun(FAC / "run05", REPO / "model_fine_tuning" / "splits" / "run05.json")
    K = len(CLASSES)
    caches = {"old": Path("/var/tmp/sat3737-parts-cache/run05/labels"), "triA": Path("/var/tmp/sat3737-parts-cache/run05/labels_triA")}
    tot = {c: np.zeros((K, K), np.int64) for c in caches}
    for sid in run.ids("val"):
        img, (h, w) = pad_to_multiple(run.image(sid))
        x = to_tensor(img)[None].to(dev)
        with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
            p = model(x).float().argmax(1)[0, :h, :w].cpu().numpy().ravel()
        for c, d in caches.items():
            t = np.asarray(Image.open(d / f"{sid}.png")).ravel()
            ok = t != IGNORE
            tot[c] += np.bincount(t[ok].astype(np.int64) * K + p[ok], minlength=K * K).reshape(K, K)
    res = {}
    for c, m in tot.items():
        tp = np.diag(m).astype(float)
        den = m.sum(0) + m.sum(1) - np.diag(m)
        res[c] = {CLASSES[i]: (round(float(tp[i] / den[i]), 4) if den[i] else None) for i in range(K)}
    (CACHE / "synth_val").mkdir(parents=True, exist_ok=True)
    (CACHE / "synth_val" / f"{a.model}.json").write_text(json.dumps({"model": a.model, "n_val": len(run.ids("val")), "iou": res}, indent=1))
    print(a.model, json.dumps(res))


def near_triangle(cls_fin, tri, grow):
    n, lab = cv2.connectedComponents(cls_fin.astype(np.uint8), connectivity=8)
    touch = cv2.dilate(tri.astype(np.uint8), np.ones((2 * grow + 1, 2 * grow + 1), np.uint8)) > 0
    ids = np.unique(lab[touch & cls_fin])
    ids = ids[ids > 0]
    return np.isin(lab, ids) if len(ids) else np.zeros_like(cls_fin)


def iou(a, b):
    u = (a | b).sum()
    return float((a & b).sum() / u) if u else np.nan


def boot(x, seed=0):
    x = np.asarray([v for v in x if np.isfinite(v)])
    if not len(x):
        return [None, None]
    b = np.random.default_rng(seed).choice(x, (2000, len(x))).mean(1)
    return [round(float(np.percentile(b, 2.5)), 3), round(float(np.percentile(b, 97.5)), 3)]


def to2560(cls, win, shape, x0, y0, x1, y1):
    """class map of a window (hx, hy, side, osz) -> the 2560-frame box [x0:x1, y0:y1], nearest."""
    hx, hy, side, osz = win
    a = osz / side
    off = 0.5 * a - 0.5                                                   # pixel centres
    M = np.array([[a, 0, (x0 - hx) * a + off], [0, a, (y0 - hy) * a + off]])   # box px -> window px
    return cv2.warpAffine(cls, M, (x1 - x0, y1 - y0), flags=cv2.INTER_NEAREST | cv2.WARP_INVERSE_MAP, borderMode=cv2.BORDER_CONSTANT, borderValue=0)


def cmd_score(a) -> None:
    F = fins()
    ds = FAC / ST / "dataset"
    res, per_fin = {}, {}
    cache_img = {}
    for m in a.models:
        wins = json.loads((CACHE / "render_parts_hr" / m / f"{ST}_windows.json").read_text())
        rows = {mode: {"old": [], "new": [], "px": [], "new_area_ratio": []} for mode in ("full", "hr", "zoom")}
        for f in F:
            sid, k = f["sid"], f["k"]
            if sid not in cache_img:
                cache_img[sid] = (rd(ds / "instances" / f"{sid}.png"), rd(ds / "parts" / f"{sid}.png"), rd(ds / "dorsal_triA" / f"{sid}.png"))
            inst, parts, did = cache_img[sid]
            H, W = inst.shape
            fr = TE.frame(f["T"], f["F"], f["B"])
            reach = 3.0 * max(fr["base"], fr["h_eff"]) + 16
            pts = np.stack([f["T"], f["F"], f["B"]])
            x0, y0 = int(max(0, pts[:, 0].min() - reach)), int(max(0, pts[:, 1].min() - reach))
            x1, y1 = int(min(W, pts[:, 0].max() + reach + 1)), int(min(H, pts[:, 1].max() + reach + 1))
            own = inst[y0:y1, x0:x1] == k
            old = (parts[y0:y1, x0:x1] == FIN) & own
            new = did[y0:y1, x0:x1] == k
            ownd = cv2.dilate(own.astype(np.uint8), np.ones((9, 9), np.uint8)) > 0
            tri = TE.triangle(f["T"] - [x0, y0], f["F"] - [x0, y0], f["B"] - [x0, y0], own.shape)
            for mode in ("full", "hr", "zoom"):
                if mode == "full":
                    full = cv2.imread(str(CACHE / "render_parts" / m / ST / f"{sid}.png"), cv2.IMREAD_UNCHANGED)
                    cls = to2560(full, [0.0, 0.0, float(W), full.shape[1]], None, x0, y0, x1, y1)
                else:
                    cls = cv2.imread(str(CACHE / f"render_parts_{mode}" / m / ST / f"{sid}~{k}.png"), cv2.IMREAD_UNCHANGED)
                    cls = to2560(cls, wins[f"{mode}|{sid}~{k}"], None, x0, y0, x1, y1)
                R = near_triangle((cls == FIN) & ownd, tri, 14)
                rows[mode]["old"].append(iou(R, old))
                rows[mode]["new"].append(iou(R, new))
                rows[mode]["px"].append(int(R.sum()))
                rows[mode]["new_area_ratio"].append(float(R.sum() / max(new.sum(), 1)))
        res[m] = {mode: {"n": len(v["old"]), "iou_old": {"mean": round(float(np.nanmean(v["old"])), 3), "ci": boot(v["old"])},
                         "iou_new": {"mean": round(float(np.nanmean(v["new"])), 3), "ci": boot(v["new"])},
                         "found": round(float(np.mean([p > 0 for p in v["px"]])), 3),
                         "area_over_new_median": round(float(np.median(v["new_area_ratio"])), 3)} for mode, v in rows.items()}
        per_fin[m] = {mode: {"old": v["old"], "new": v["new"]} for mode, v in rows.items()}
    # label-only ceiling: the OLD label scored against the NEW one (what a perfect old-label model would get)
    ceil = []
    for f in F:
        inst, parts, did = cache_img[f["sid"]]
        ceil.append(iou((parts == FIN) & (inst == f["k"]), did == f["k"]))
    out = {"set": ST, "n_fins": len(F), "models": res, "old_label_vs_new_label_iou": {"mean": round(float(np.nanmean(ceil)), 3), "ci": boot(ceil)},
           "views": [f["sight_deg"] for f in F]}
    (OUT / "w3e_renders.json").write_text(json.dumps(out, indent=1))
    (CACHE / "w3e_renders_per_fin.json").write_text(json.dumps(per_fin))
    for m, v in res.items():
        print(m, {mode: (x["iou_old"]["mean"], x["iou_new"]["mean"], x["found"]) for mode, x in v.items()})
    print("old label vs new label", out["old_label_vs_new_label_iou"])


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["pred", "synth", "score"])
    ap.add_argument("--model")
    ap.add_argument("--checkpoint")
    ap.add_argument("--models", nargs="+", default=[])
    a = ap.parse_args()
    {"pred": cmd_pred, "synth": cmd_synth, "score": cmd_score}[a.cmd](a)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
