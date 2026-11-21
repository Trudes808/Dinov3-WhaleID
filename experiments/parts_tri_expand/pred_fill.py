"""(a) The expanded triangle at prediction time, no retraining: from a part model's predicted dorsal region R derive a
triangle (the smallest triangle around R; its base = the side with the most whale-mask pixels just outside it, tip =
the opposite corner), then fill with the whale mask S6h by the same frozen rule (rule.json, A):
    filled = R  OR  expand(triangle, S6h)
Plus the two new measures for every arm (W3c, W3c filled, W3d): IoU against the expanded-triangle truth on the clicked
fins, and IoU against the exact labels on renders.

    .venv/bin/python experiments/parts_tri_expand/pred_fill.py shadow --model parts_seg_r05_w3c   (CPU)
        -> SHADOW/<day>/<model>_fill/{hr,full,instances.json}  (scored by the unchanged W3c scripts)
    flock LOCK .venv/bin/python experiments/parts_tri_expand/pred_fill.py render-pred --model NAME --checkpoint CKPT
        -> CACHE/render_parts/<model>/<set>/<sid>.png (class map, 1280 px, whole frame)
    .venv/bin/python experiments/parts_tri_expand/pred_fill.py score --models ...   -> new_measures.json
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from multiprocessing import get_context
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "experiments" / "parts_weak"))
from model_fine_tuning.parts_weak import tri_expand as TE  # noqa: E402

OUT = Path(__file__).resolve().parent
SHADOW = Path("/var/tmp/sat3737-parts-cache/parts_weak/days_shadow")
CACHE = Path("/home/sat3737/NOAA_whales/moonshot_cache/parts_tri_expand")
S6_CACHED = Path("/var/tmp/sat3737-parts-cache/seg_splash/pred/s6h")
S6_MINE = CACHE / "s6h"
FIN = 2
MIN_PX = 12


def rule():
    return json.loads((OUT / "rule.json").read_text())["params"]


def s6h(day: str, stem: str):
    for p in (S6_CACHED / day / f"{stem}.png", S6_MINE / day / f"{stem}.png"):
        if p.exists():
            return cv2.imread(str(p), cv2.IMREAD_UNCHANGED) > 0
    return None


# ----------------------------------------------------------------------------------------------- the derived triangle
def pseudo_triangle(R: np.ndarray, M: np.ndarray):
    """(T, F, B) from region R (bool) and whale mask M (bool, same frame), or None."""
    ys, xs = np.nonzero(R)
    if len(xs) < MIN_PX:
        return None
    pts = np.stack([xs, ys], 1).astype(np.float32)
    hull = cv2.convexHull(pts)
    try:
        _, tri = cv2.minEnclosingTriangle(hull)
    except cv2.error:
        return None
    tri = tri.reshape(3, 2).astype(np.float64)
    H, W = M.shape
    c = tri.mean(0)
    best, best_score = None, -1.0
    for i in range(3):
        a, b = tri[(i + 1) % 3], tri[(i + 2) % 3]
        e = b - a
        L = float(np.hypot(*e))
        if L < 2:
            continue
        nrm = np.array([-e[1], e[0]]) / L
        if (c - a) @ nrm > 0:
            nrm = -nrm                                    # outward
        h = abs(float((tri[i] - a) @ nrm))
        ts = np.linspace(0.1, 0.9, 9)
        ds = np.linspace(1.0, max(2.0, 0.3 * h), 5)
        P = a[None, None] + ts[:, None, None] * e[None, None] + ds[None, :, None] * nrm[None, None]
        P = np.round(P.reshape(-1, 2)).astype(int)
        ok = (P[:, 0] >= 0) & (P[:, 0] < W) & (P[:, 1] >= 0) & (P[:, 1] < H)
        sc = float(M[P[ok, 1], P[ok, 0]].mean()) if ok.any() else 0.0
        if sc > best_score:
            best, best_score = i, sc
    if best is None:
        return None
    T = tri[best]
    F, B = tri[(best + 1) % 3], tri[(best + 2) % 3]
    return T, F, B


def fill_map(cls: np.ndarray, M: np.ndarray, p: dict, only_largest: bool) -> tuple[np.ndarray, int]:
    out = cls.copy()
    fin = (cls == FIN).astype(np.uint8)
    n, lab, st, _ = cv2.connectedComponentsWithStats(fin, connectivity=8)
    if n <= 1:
        return out, 0
    ids = [1 + int(np.argmax(st[1:, cv2.CC_STAT_AREA]))] if only_largest else [k for k in range(1, n) if st[k, cv2.CC_STAT_AREA] >= MIN_PX]
    k_done = 0
    for k in ids:
        R = lab == k
        t = pseudo_triangle(R, M)
        if t is None:
            continue
        e = TE.expand(*t, M, **p)
        out[e & (cls != 255)] = FIN
        k_done += 1
    return out, k_done


def save_palette(a: np.ndarray, path: Path) -> None:
    from model_fine_tuning.parts_seg.predict import save_palette as sp
    sp(a, path)


def shadow_day(args):
    day, model, p = args
    src, dst = SHADOW / day / model, SHADOW / day / f"{model}_fill"
    (dst / "hr").mkdir(parents=True, exist_ok=True)
    (dst / "full").mkdir(parents=True, exist_ok=True)
    ins = json.loads((src / "instances.json").read_text())["instances"]
    n_hr = n_full = n_nomask = 0
    for f in sorted((src / "full").glob("*.png")):
        stem = f.stem
        cls = np.asarray(Image.open(f))
        M = s6h(day, stem)
        if M is None or not (cls == FIN).any():
            if (cls == FIN).any():
                n_nomask += 1
            save_palette(cls, dst / "full" / f.name)
            continue
        if M.shape != cls.shape:
            M = cv2.resize(M.astype(np.uint8), (cls.shape[1], cls.shape[0]), interpolation=cv2.INTER_NEAREST) > 0
        o, k = fill_map(cls, M, p, only_largest=False)
        n_full += k
        save_palette(o, dst / "full" / f.name)
    for f in sorted((src / "hr").glob("*.png")):
        key = f.stem
        stem = key.rsplit("~", 1)[0]
        cls = np.asarray(Image.open(f))
        M = s6h(day, stem) if (cls == FIN).any() else None
        if M is None or key not in ins or "hr_window" not in ins[key]:
            save_palette(cls, dst / "hr" / f.name)
            continue
        hx, hy, side, osz = ins[key]["hr_window"]
        a = osz / side                                                  # hr map px per 2560 px
        s = 2560.0 / max(M.shape)                                       # 2560 px per mask px
        A = np.array([[a * s, 0, -hx * a], [0, a * s, -hy * a]])       # 1280 mask px -> hr map px
        Mw = cv2.warpAffine(M.astype(np.uint8), A, (cls.shape[1], cls.shape[0]), flags=cv2.INTER_NEAREST) > 0
        o, k = fill_map(cls, Mw, p, only_largest=True)
        n_hr += k
        save_palette(o, dst / "hr" / f.name)
    return day, n_full, n_hr, n_nomask


def cmd_shadow(a) -> None:
    from model_fine_tuning.common import geometry, paths
    from model_fine_tuning.parts_seg import predict
    p = rule()
    days = sorted(d for d in os.listdir(SHADOW) if (SHADOW / d / a.model / "instances.json").exists())
    with get_context("fork").Pool(a.workers) as pool:
        for day, nf, nh, nm in pool.imap_unordered(shadow_day, [(d, a.model, p) for d in days]):
            print(day, "filled full", nf, "hr", nh, "photos with fin but no S6h", nm, flush=True)
    predict.DAYS = SHADOW
    t = geometry.load_folded_table(paths.FOLDED_TABLE)
    table = {"x_min_m": float(t["x_min_m"]), "length_m": float(t["length_m"])}
    for day in days:
        predict.stats_from_pngs(day, f"{a.model}_fill", table)
    print("instances.json rebuilt for", len(days), "days")


# -------------------------------------------------------------------------------------------------- renders, predict
def cmd_render_pred(a) -> None:
    import torch
    from model_fine_tuning.common.data import pad_to_multiple, to_tensor
    from model_fine_tuning.parts_seg.model import PartsSegModel
    from model_fine_tuning.parts_seg.predict import run_model
    dev = torch.device("cuda")
    model, _ = PartsSegModel.from_checkpoint(Path(a.checkpoint), dev)
    model.eval()
    fac = REPO.parent / "Whale-Image-Data-Factory" / "data" / "renders"
    for st in a.sets:
        sids = sorted({r["sid"] for r in json.loads((CACHE / f"render_fins_{st}.json").read_text())})
        out = CACHE / "render_parts" / a.model / st
        out.mkdir(parents=True, exist_ok=True)
        for sid in sids:
            im = cv2.cvtColor(cv2.imread(str(fac / st / "dataset" / "images" / f"{sid}.png")), cv2.COLOR_BGR2RGB)
            H, W = im.shape[:2]
            s = 1280 / max(H, W)
            sm = cv2.resize(im, (round(W * s), round(H * s)), interpolation=cv2.INTER_AREA)
            padded, (h, w) = pad_to_multiple(sm)
            cls = run_model(model, to_tensor(padded), dev)[:h, :w]
            cv2.imwrite(str(out / f"{sid}.png"), cls)
        print(a.model, st, len(sids))


# ------------------------------------------------------------------------------------------------------- scoring
def iou(a, b):
    u = (a | b).sum()
    return float((a & b).sum() / u) if u else np.nan


def boot(x, seed=0):
    x = np.asarray([v for v in x if np.isfinite(v)])
    if not len(x):
        return [np.nan, np.nan]
    b = np.random.default_rng(seed).choice(x, (2000, len(x))).mean(1)
    return [round(float(np.percentile(b, 2.5)), 3), round(float(np.percentile(b, 97.5)), 3)]


def near_triangle(cls_fin: np.ndarray, tri: np.ndarray, grow: int = 7) -> np.ndarray:
    """The predicted dorsal pixels connected to the (grown) clicked triangle."""
    n, lab = cv2.connectedComponents(cls_fin.astype(np.uint8), connectivity=8)
    touch = cv2.dilate(tri.astype(np.uint8), np.ones((2 * grow + 1, 2 * grow + 1), np.uint8)) > 0
    ids = np.unique(lab[touch & cls_fin])
    ids = ids[ids > 0]
    return np.isin(lab, ids) if len(ids) else np.zeros_like(cls_fin)


def score_clicks(models: list[str]) -> dict:
    """IoU with the expanded-triangle truth (rule A on S6h, made in the 1280 frame and warped into each hr window)
    and with the bare triangle, per clicked fin, per model. 2025-03-18 = held out (no model trained on its clicks);
    2025-01-02 = the clicks W3c / W3d trained on (reference only)."""
    from fin_definition import clicks
    p = rule()
    res = {}
    for day in ("2025-03-18", "2025-01-02"):
        ck = clicks(day)
        ins = {m: json.loads((SHADOW / day / m / "instances.json").read_text())["instances"] for m in models}
        per = {m: {"exp": [], "tri": [], "cover_exp": [], "area_over_exp": []} for m in models}
        n = 0
        for key, kp in sorted(ck.items()):
            stem = key.rsplit("~", 1)[0]
            if not all(key in ins[m] and "hr_window" in ins[m][key] for m in models):
                continue
            M = s6h(day, stem)
            if M is None:
                continue
            H, W = M.shape
            T, F, B = (np.array([kp[x]["x"] * W, kp[x]["y"] * H]) for x in ("dorsal_fin_tip", "dorsal_fin_front", "dorsal_fin_back"))
            E = TE.expand(T, F, B, M, **p)
            n += 1
            for m in models:
                hx, hy, side, osz = ins[m][key]["hr_window"]
                cls = np.asarray(Image.open(SHADOW / day / m / "hr" / f"{key}.png"))
                a = osz / side
                s = 2560.0 / max(H, W)
                A = np.array([[a * s, 0, -hx * a], [0, a * s, -hy * a]])
                Ew = cv2.warpAffine(E.astype(np.uint8), A, (cls.shape[1], cls.shape[0]), flags=cv2.INTER_NEAREST) > 0
                tri = TE.triangle(*(np.r_[q * s - [hx, hy]] * a for q in (T, F, B)), cls.shape)
                R = near_triangle(cls == FIN, tri)
                per[m]["exp"].append(iou(R, Ew))
                per[m]["tri"].append(iou(R, tri))
                per[m]["cover_exp"].append(float((R & Ew).sum() / max(Ew.sum(), 1)))
                per[m]["area_over_exp"].append(float(R.sum() / max(Ew.sum(), 1)))
        res[day] = {"n_fins": n, **{m: {"iou_expanded": {"mean": round(float(np.nanmean(v["exp"])), 3), "ci": boot(v["exp"]), "median": round(float(np.nanmedian(v["exp"])), 3)},
                                        "iou_triangle": {"mean": round(float(np.nanmean(v["tri"])), 3), "ci": boot(v["tri"])},
                                        "expanded_covered": round(float(np.nanmedian(v["cover_exp"])), 3),
                                        "area_over_expanded": round(float(np.nanmedian(v["area_over_exp"])), 3),
                                        "_per_fin_iou_expanded": [round(x, 4) for x in v["exp"]]} for m, v in per.items()}}
    return res


def score_renders(models: list[str], fill_of: dict, st: str = "r06r_check21") -> dict:
    """IoU of each arm's dorsal region near each fin against the exact labels (fin + hump, fin only), check21."""
    import render_fins as RF
    p = rule()
    fins = [r for r in RF.load_set(st, ("gt", "s6h")) if r["all_visible"]]
    out = {}
    for m in models:
        base = fill_of.get(m, m)
        per = {"fin_hump": [], "fin": [], "found": []}
        for r in fins:
            x0, y0, x1, y1 = r["win"]
            cls = cv2.imread(str(CACHE / "render_parts" / base / st / f"{r['sid']}.png"), cv2.IMREAD_UNCHANGED)
            if cls is None:
                break
            cls = cls[y0:y1, x0:x1]
            if m in fill_of:
                cls, _ = fill_map(cls, r["mask"]["s6h"], p, only_largest=False)
            own = cv2.dilate(r["own"].astype(np.uint8), np.ones((5, 5), np.uint8)) > 0
            tri = TE.triangle(r["Tw"], r["Fw"], r["Bw"], own.shape)
            R = near_triangle((cls == FIN) & own, tri)
            per["fin_hump"].append(iou(R, r["dorsal"]))
            per["fin"].append(iou(R, r["fin"]))
            per["found"].append(bool(R.any()))
        if per["found"]:
            out[m] = {"n": len(per["found"]), "found": round(float(np.mean(per["found"])), 3),
                      **{k: {"mean": round(float(np.nanmean(per[k])), 3), "ci": boot(per[k])} for k in ("fin_hump", "fin")},
                      "_per_fin_fin_hump": [round(x, 4) for x in per["fin_hump"]]}
    return out


def cmd_score(a) -> None:
    fill_of = {f"{m}_fill": m for m in a.models if not m.endswith("_fill")}
    arms = list(a.models) + list(fill_of)
    arms = [m for m in arms if (SHADOW / "2025-03-18" / m / "instances.json").exists()]
    res = {"rule": rule(), "clicks": score_clicks(arms)}
    rarms = list(a.models) + [k for k in fill_of if (CACHE / "render_parts" / fill_of[k]).exists()]
    res["renders"] = score_renders(rarms, fill_of)
    (OUT / a.out).write_text(json.dumps(res, indent=1))
    for day in res["clicks"]:
        print(day, res["clicks"][day]["n_fins"], {m: (v["iou_expanded"], v["iou_triangle"]["mean"]) for m, v in res["clicks"][day].items() if m != "n_fins"})
    print({m: (v["fin_hump"], v["fin"], v["found"]) for m, v in res["renders"].items()})


# ------------------------------------------------------------------------------------------------------- sheets
def cmd_sheet(a) -> None:
    """triexp_model_examples.jpg: 30 held-out clicked fins of 2025-03-18 at random (seed 0): photo | W3c region |
    W3c filled | W3d region | expanded-triangle truth, clicked triangle outlined; seed 0 models. Scratchpad only."""
    from fin_definition import clicks
    from real_fins import SHEETS, grid_sheet
    p = rule()
    day = "2025-03-18"
    arms = ["parts_seg_r05_w3c", "parts_seg_r05_w3c_fill", "parts_seg_r05_w3d"]
    ins = {m: json.loads((SHADOW / day / m / "instances.json").read_text())["instances"] for m in arms}
    keys = [k for k in sorted(clicks(day)) if all(k in ins[m] and "hr_window" in ins[m][k] for m in arms)]
    ck = clicks(day)
    pick = sorted(np.random.default_rng(0).choice(len(keys), 30, replace=False).tolist())
    tiles = []
    for i in pick:
        key = keys[i]
        stem = key.rsplit("~", 1)[0]
        kp = ck[key]
        M = s6h(day, stem)
        H, W = M.shape
        T, F, B = (np.array([kp[x]["x"] * W, kp[x]["y"] * H]) for x in ("dorsal_fin_tip", "dorsal_fin_front", "dorsal_fin_back"))
        E = TE.expand(T, F, B, M, **p)
        hx, hy, side, osz = ins[arms[0]][key]["hr_window"]
        aa, s = osz / side, 2560.0 / max(H, W)
        A = np.array([[aa * s, 0, -hx * aa], [0, aa * s, -hy * aa]])
        img = cv2.warpAffine(cv2.imread(str(REPO / "experiments" / "days_all" / day / "images" / f"{stem}.jpg")),
                             np.array([[aa, 0, -hx * aa], [0, aa, -hy * aa]]), (int(osz), int(osz)), flags=cv2.INTER_AREA)
        q = np.round(np.stack([(x * s - [hx, hy]) * aa for x in (T, F, B)])).astype(np.int32)
        fr = TE.frame(*q.astype(float))
        c = q.mean(0)
        L = max(np.ptp(q[:, 0]), np.ptp(q[:, 1]), fr["h_eff"]) * 2.4 + 16
        Z = np.array([[200 / L, 0, -(c[0] - L / 2) * 200 / L], [0, 200 / L, -(c[1] - L / 2) * 200 / L]])
        layers = [None] + [np.asarray(Image.open(SHADOW / day / m / "hr" / f"{key}.png")) == FIN for m in arms] + \
                 [cv2.warpAffine(E.astype(np.uint8), A, (int(osz), int(osz)), flags=cv2.INTER_NEAREST) > 0]
        row = []
        for j, Lr in enumerate(layers):
            t = cv2.warpAffine(img, Z, (200, 200), flags=cv2.INTER_LINEAR)
            if Lr is not None:
                m = cv2.warpAffine(Lr.astype(np.uint8), Z, (200, 200), flags=cv2.INTER_NEAREST) > 0
                ov = t.copy()
                ov[m] = (40, 40, 255) if j < 4 else (60, 200, 60)
                t = cv2.addWeighted(t, 0.55, ov, 0.45, 0)
                qq = np.round(np.c_[q, np.ones(3)] @ Z.T).astype(np.int32)
                cv2.polylines(t, [qq], True, (0, 255, 255), 1, cv2.LINE_AA)
            row.append(np.pad(t, ((2, 2), (2, 2), (0, 0)), constant_values=255))
        r = np.hstack(row)
        lab = np.full((18, r.shape[1], 3), 255, np.uint8)
        cv2.putText(lab, f"{key[-34:]}", (4, 13), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 0), 1, cv2.LINE_AA)
        tiles.append(np.vstack([lab, r]))
    head = ["Held-out clicked fins 2025-03-18 (no model trained on these clicks), 30 at random (seed 0); seed-0 models.",
            "photo | W3c dorsal region | W3c filled at prediction time (a) | W3d trained on the expanded triangle (b) | expanded-triangle truth (green); yellow = clicked triangle"]
    cv2.imwrite(str(SHEETS / "triexp_model_examples.jpg"), grid_sheet(tiles, 2, head), [cv2.IMWRITE_JPEG_QUALITY, 88])
    print(SHEETS / "triexp_model_examples.jpg")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["shadow", "render-pred", "score", "sheet"])
    ap.add_argument("--model")
    ap.add_argument("--models", nargs="+")
    ap.add_argument("--checkpoint")
    ap.add_argument("--sets", nargs="+", default=["r06r_check21"])
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--out", default="new_measures.json")
    a = ap.parse_args()
    {"shadow": cmd_shadow, "render-pred": cmd_render_pred, "score": cmd_score, "sheet": cmd_sheet}[a.cmd](a)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
