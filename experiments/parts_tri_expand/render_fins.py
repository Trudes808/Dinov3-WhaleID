"""Renders with exact labels: per fin (all three fin keypoints visible) the window arrays the rule needs, at 1280 px
(the real masks' frame), saved once; then the grid and the scores of the bare and the expanded triangle.

    .venv/bin/python experiments/parts_tri_expand/render_fins.py collect      -> CACHE/render_fins_<set>.npz (+ .json)
    .venv/bin/python experiments/parts_tri_expand/render_fins.py grid  --fit r06r_check20
    .venv/bin/python experiments/parts_tri_expand/render_fins.py score --sets r06r_check20 r06r_check21 --mask gt s6h r04

Truth per fin (animal k): "fin+hump" = run06 part label 2 (blade and hump) on k's visible pixels; "fin" = the blade
sub-label (animals/<id>.npz a{k}_blade) on k's visible pixels. Whale masks: gt = any visible whale (instances > 0),
s6h / r04 = the candidate / live mask run on the render (render_masks.py). Read-only on the factory folders.
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

FACTORY = REPO.parent / "Whale-Image-Data-Factory" / "data" / "renders"
CACHE = Path("/home/sat3737/NOAA_whales/moonshot_cache/parts_tri_expand")
OUT = Path(__file__).resolve().parent
NEED = ("dorsal_fin_tip", "dorsal_fin_front", "dorsal_fin_back")
SCALE = 0.5                                                    # 2560 -> 1280
GRID = {"m_along": [0.0, 0.05, 0.1, 0.2, 0.3], "m_below": [0.0, 0.05, 0.1, 0.15, 0.2, 0.3, 0.45],
        "d_near": [0.1, 0.2, 0.3, 0.4, 0.5, 0.7, 1.0, 9.0]}


def load_shot(ds: Path, sid: str):
    inst = np.asarray(Image.open(ds / "instances" / f"{sid}.png"))
    parts = np.asarray(Image.open(ds / "parts" / f"{sid}.png"))
    z = np.load(ds / "animals" / f"{sid}.npz")
    return inst, parts, z


def small(a: np.ndarray, H: int, W: int) -> np.ndarray:
    return cv2.resize(a.astype(np.uint8), (W, H), interpolation=cv2.INTER_NEAREST)


def collect(set_name: str) -> None:
    ds = FACTORY / set_name / "dataset"
    fins, arrays = [], {}
    for mp in sorted((ds / "meta").glob("*.json")):
        m = json.loads(mp.read_text())
        if m.get("negative") or m.get("no_whale_in_frame"):
            continue
        sid = m["id"]
        # all three visible (the fit set), or the tip visible and both bases in frame under the water / a wave with some
        # blade showing (the "how much of the fin is above water" axis; the points are the true 3D points projected)
        todo = [a for a in m.get("animals", []) if a["keypoints"].get("dorsal_fin_tip", {}).get("state") == "visible"
                and all(a["keypoints"].get(x, {}).get("state") in ("visible", "under_water", "under_wave") for x in NEED[1:])
                and (a.get("dorsal_blade_pixels") or {}).get("visible", 0) > 0]
        if not todo:
            continue
        inst, parts, z = load_shot(ds, sid)
        H0, W0 = inst.shape
        H, W = int(round(H0 * SCALE)), int(round(W0 * SCALE))
        inst_s, parts_s = small(inst, H, W), small(parts, H, W)
        for a in todo:
            k = a["instance_id"]
            own = inst_s == k
            blade = np.zeros((H0, W0), bool)
            if f"a{k}_blade" in z.files:
                x0, y0, x1, y1 = [int(v) for v in z[f"a{k}_box"]]
                bz = z[f"a{k}_blade"]
                hh, ww = min(bz.shape[0], H0 - y0), min(bz.shape[1], W0 - x0)
                blade[y0:y0 + hh, x0:x0 + ww] = bz[:hh, :ww]
            blade_s = small(blade, H, W) > 0
            T, F, B = (np.array(a["keypoints"][x]["uv"], float) * SCALE for x in NEED)
            fr = TE.frame(T, F, B)
            if fr is None:
                continue
            key = f"{sid}:{k}"
            fs = a.get("fin_spec") or {}
            bp = a.get("dorsal_blade_pixels") or {}
            pp = a.get("part_planes", {}).get("fin", {})
            rec = {"all_visible": all(a["keypoints"][x]["state"] == "visible" for x in NEED), "set": set_name, "key": key, "sid": sid, "k": k, "T": T.tolist(), "F": F.tolist(), "B": B.tolist(),
                   "base": fr["base"], "h": fr["h"], "sight_deg": pp.get("sight_deg"), "category": a.get("category"),
                   "n_animals": m.get("n_animals", 1), "blade_vis_share": (bp.get("visible", 0) / bp["amodal"]) if bp.get("amodal") else None,
                   "splash": bool(m.get("splash")) or (m.get("whitewater") or {}).get("amount", 0) > 0.8,
                   "whitewater": (m.get("whitewater") or {}).get("amount", 0.0), "sea_state": m.get("sea_state")}
            # window: generous (3 x the larger fin size), kept as arrays
            reach = 3.0 * max(fr["base"], fr["h_eff"]) + 8
            pts = np.stack([T, F, B])
            x0, y0 = int(max(0, pts[:, 0].min() - reach)), int(max(0, pts[:, 1].min() - reach))
            x1, y1 = int(min(W, pts[:, 0].max() + reach + 1)), int(min(H, pts[:, 1].max() + reach + 1))
            rec["win"] = [x0, y0, x1, y1]
            sl = (slice(y0, y1), slice(x0, x1))
            arrays[key + "|inst"] = inst_s[sl]
            arrays[key + "|parts"] = parts_s[sl]
            arrays[key + "|blade"] = (blade_s & own)[sl]
            fins.append(rec)
    CACHE.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(CACHE / f"render_fins_{set_name}.npz", **arrays)
    (CACHE / f"render_fins_{set_name}.json").write_text(json.dumps(fins))
    print(set_name, len(fins), "fins")


def load_set(set_name: str, masks=("gt",)):
    fins = json.loads((CACHE / f"render_fins_{set_name}.json").read_text())
    z = np.load(CACHE / f"render_fins_{set_name}.npz")
    for r in fins:
        key = r["key"]
        inst, parts = z[key + "|inst"], z[key + "|parts"]
        r["own"] = inst == r["k"]
        r["dorsal"] = (parts == 2) & r["own"]
        r["fin"] = z[key + "|blade"]
        r["inst"] = inst
        r["parts"] = parts
        r["mask"] = {"gt": inst > 0}
        for mk in masks:
            if mk == "gt":
                continue
            p = CACHE / "render_masks" / set_name / f"{r['sid']}_{mk}.png"
            if p.exists():
                full = cv2.imread(str(p), cv2.IMREAD_UNCHANGED) > 0
                x0, y0, x1, y1 = r["win"]
                r["mask"][mk] = full[y0:y1, x0:x1]
        x0, y0 = r["win"][:2]
        r["Tw"], r["Fw"], r["Bw"] = (np.array(r[x]) - [x0, y0] for x in ("T", "F", "B"))
    return fins


def iou_pr(pred: np.ndarray, truth: np.ndarray) -> tuple:
    i = float((pred & truth).sum())
    u = float((pred | truth).sum())
    return (i / u if u else np.nan, i / truth.sum() if truth.sum() else np.nan, i / pred.sum() if pred.sum() else np.nan)


def view_bin(s):
    if s is None:
        return "?"
    return "side-on (<30)" if s < 30 else ("oblique (30-60)" if s < 60 else "end-on (>60)")


def water_bin(s):
    if s is None:
        return "?"
    return "blade >= 90 % visible" if s >= 0.9 else ("blade 50-90 % visible" if s >= 0.5 else "blade < 50 % visible")


def boot_ci(x, n=2000, seed=0):
    x = np.asarray([v for v in x if np.isfinite(v)])
    if len(x) == 0:
        return [np.nan, np.nan]
    rng = np.random.default_rng(seed)
    b = rng.choice(x, (n, len(x))).mean(1)
    return [float(np.percentile(b, 2.5)), float(np.percentile(b, 97.5))]


def cmd_grid(a) -> None:
    fins = [r for r in load_set(a.fit) if r["all_visible"]]
    res = []
    for ma in GRID["m_along"]:
        for mb in GRID["m_below"]:
            for dn in GRID["d_near"]:
                v_dh, v_f = [], []
                for r in fins:
                    e = TE.expand(r["Tw"], r["Fw"], r["Bw"], r["mask"]["gt"], ma, mb, dn)
                    v_dh.append(iou_pr(e, r["dorsal"])[0])
                    v_f.append(iou_pr(e, r["fin"])[0])
                res.append({"m_along": ma, "m_below": mb, "d_near": dn, "iou_fin_hump": float(np.nanmean(v_dh)), "iou_fin": float(np.nanmean(v_f)),
                            "median_fin_hump": float(np.nanmedian(v_dh))})
                print(res[-1], flush=True)
    res.sort(key=lambda r: -r["iou_fin_hump"])
    (OUT / f"grid_{a.fit}.json").write_text(json.dumps({"fit_set": a.fit, "n_fins": len(fins), "objective": "mean IoU, fin + hump label, gt whale mask",
                                                        "grid": GRID, "results": res}, indent=1))
    print("best", res[:5])


def score_rows(fins, masks, params, params_b=None):
    rows = []
    for r in fins:
        tri = TE.triangle(r["Tw"], r["Fw"], r["Bw"], r["own"].shape)
        row = {k: r[k] for k in ("all_visible", "set", "key", "sid", "k", "base", "h", "sight_deg", "category", "n_animals", "blade_vis_share", "splash", "whitewater")}
        row["base_over_h"] = r["base"] / max(r["h"], 1e-6)
        row["truth_px"] = {"fin_hump": int(r["dorsal"].sum()), "fin": int(r["fin"].sum())}
        row["tri"] = {}
        for tname, t in (("fin_hump", r["dorsal"]), ("fin", r["fin"])):
            row["tri"][tname] = iou_pr(tri, t)
        row["exp"] = {}
        runs = [(mk, mk, params) for mk in masks] + ([(mk + "_B", mk, params_b) for mk in masks if mk != "r04"] if params_b else [])
        for name, mk, pp in runs:
            if mk not in r["mask"]:
                continue
            e = TE.expand(r["Tw"], r["Fw"], r["Bw"], r["mask"][mk], **pp)
            d = {t: iou_pr(e, tt) for t, tt in (("fin_hump", r["dorsal"]), ("fin", r["fin"]))}
            # where the extra pixels go: other whale, own fluke, not whale (mask error: splash / water)
            extra = e & ~r["dorsal"]
            d["leak_px"] = {"other_whale": int((extra & (r["inst"] > 0) & ~r["own"]).sum()),
                            "own_fluke": int((extra & r["own"] & (r["parts"] == 1)).sum()),
                            "own_body_head_pec": int((extra & r["own"] & np.isin(r["parts"], (3, 4, 5))).sum()),
                            "not_whale": int((extra & (r["inst"] == 0)).sum())}
            d["px"] = int(e.sum())
            d["tri_and_mask_in_truth"] = iou_pr(tri & r["mask"][mk], r["dorsal"])
            row["exp"][name] = d
        rows.append(row)
    return rows


def summarise(rows, masks):
    def block(sel):
        out = {"n": len(sel)}
        if not sel:
            return out
        for tname in ("fin_hump", "fin"):
            t = [r["tri"][tname] for r in sel]
            out[f"tri_{tname}"] = {"iou": float(np.nanmean([x[0] for x in t])), "iou_ci": boot_ci([x[0] for x in t]),
                                   "recall": float(np.nanmean([x[1] for x in t])), "precision": float(np.nanmean([x[2] for x in t]))}
            for mk in masks:
                e = [r["exp"][mk][tname] for r in sel if mk in r["exp"]]
                if e:
                    out[f"exp_{mk}_{tname}"] = {"iou": float(np.nanmean([x[0] for x in e])), "iou_ci": boot_ci([x[0] for x in e]),
                                                "recall": float(np.nanmean([x[1] for x in e])), "precision": float(np.nanmean([x[2] for x in e])),
                                                "iou_median": float(np.nanmedian([x[0] for x in e])), "n": len(e)}
        return out
    s = {"all": block(rows)}
    for name, f in (("view", lambda r: view_bin(r["sight_deg"])), ("water", lambda r: water_bin(r["blade_vis_share"])),
                    ("flat_triangle", lambda r: "base/h > 6 (points near a line)" if r["base_over_h"] > 6 else "base/h <= 6"),
                    ("animals", lambda r: "one whale" if r["n_animals"] == 1 else "several whales"),
                    ("whitewater", lambda r: "whitewater > 0.8" if r["whitewater"] > 0.8 else "whitewater <= 0.8")):
        groups = {}
        for r in rows:
            groups.setdefault(f(r), []).append(r)
        s[name] = {g: block(v) for g, v in sorted(groups.items())}
    return s


def cmd_score(a) -> None:
    rule = json.loads((OUT / "rule.json").read_text())
    params, params_b = rule["params"], rule.get("params_B")
    allres = {"params": params, "params_B": params_b, "sets": {}}
    for st in a.sets:
        fins = load_set(st, a.mask)
        rows = score_rows(fins, a.mask, params, params_b)
        allres["sets"][st] = summarise(rows, list(a.mask) + [m + "_B" for m in a.mask if m != "r04"])
        (CACHE / f"render_rows_{st}.json").write_text(json.dumps(rows))
        print(st, json.dumps(allres["sets"][st]["all"], indent=1))
    (OUT / "render_scores.json").write_text(json.dumps(allres, indent=1))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["collect", "grid", "score"])
    ap.add_argument("--sets", nargs="+", default=["r06r_check20", "r06r_check21"])
    ap.add_argument("--fit", default="r06r_check20")
    ap.add_argument("--mask", nargs="+", default=["gt", "s6h", "r04"])
    a = ap.parse_args()
    if a.cmd == "collect":
        for s in a.sets:
            collect(s)
    elif a.cmd == "grid":
        cmd_grid(a)
    else:
        cmd_score(a)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
