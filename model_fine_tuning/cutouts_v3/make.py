"""Cut-outs v3 from the full-resolution RAW, offline (nothing of a day folder is written).

    PYTHONPATH=. .venv/bin/python -m model_fine_tuning.cutouts_v3.make train,val [--workers 8] [--limit N]

Per whale instance of the pipeline (the old keys <stem>~k of DAY/parts_hr / parts_up, rebuilt from DAY/cse as
cluster/parts.py split_instances makes them), into CACHE/<day>/<variant>/<key>.jpg (768 px) + <key>.npz:

  fluke   for instances the view classifier (DAY/safe/view_class.json, P >= 0.5) calls fluke: "turned, never
          flipped". Region: the instance; if instances_v2 joins it with another blob as the two blades of one fluke,
          the joined instance (one cut-out, written under every old key it holds). Fluke pixels: the part map's
          (DAY/parts_seg_r05_a/hr) fluke class inside the region when that is >= 25 % of it, else the whole region
          (blades the old clean-up deleted have no part map and count as fluke). The span = principal axis of the fluke
          pixels is laid level; of the two level turns (0 / 180 deg) the one with the tail stock on top is taken. Stock
          direction: the part map's body / head pixels attached to the fluke pixels (connected components touching
          them, within 0.6 span of the fluke centre, at least 30 px and 2 % of the fluke, and lying across the span
          rather than along it); else the photo: a fluke rises out of the water, the stock goes down into it. Never
          mirrored: the transform is a rotation and a scale (det > 0). Square = the span-aligned box (0.5 / 99.5
          percentiles; along the span of the whole region, across it of the fluke pixels) plus a 0.15 margin on each side, black outside the photo (as parts.py).
  finB / finC   for instances the view classifier calls dorsal fin and that have a parts_up fin cut-out: the turn and
          mirror of that cut-out (its npz M and "mirrored"), re-centred on the part map's largest dorsal-fin region
          (instances.json hr.fin centroid), side 4 (B) or 2 (C) fin bases, fin base = region extent (longer box side)
          / 0.28 (the median ratio of region to the researcher's fin base, LOG 2026-11-17), capped at the whale's
          visible length (principal-axis extent of the instance). No fin region: the part map's largest head patch if
          it sits on the upper outline of the instance (its top edge on the instance's top edge over >= 60 % of its
          columns, within 3 px + 4 % of the instance height) - the knob the model calls head - with its extent as the
          fin base (the patch covers hump and fin, not the blade only; at least 0.2 of the whale's visible length); else nothing is written
          and the fin keeps today's cut-out (A = parts_up). Black outside the photo.

npz: M (2x3, RAW -> chart before any mirroring), mirrored, whale (bool chart mask, for the networks' token mask),
size, plus per variant the method fields; CACHE/<day>/meta.jsonl one row per cut-out.
"""
from __future__ import annotations

import argparse
import json
import time
from multiprocessing import get_context
from pathlib import Path

import cv2
import numpy as np

from . import CACHE, DAYS, day_list

SIZE = 768
MARGIN_FLUKE = 0.15
RATIO_BLADE_BASE = 0.28
CLS = {"background": 0, "fluke": 1, "dorsal_fin": 2, "pectoral_fin": 3, "head": 4, "body": 5}


def to_raw_M(R: np.ndarray, c2560: np.ndarray, side2560: float, scale_full: float, size: int = SIZE) -> np.ndarray:
    """Chart = sc * R (q - c_raw) + size / 2 for RAW pixel q; R rows = the chart's x / y axes in image coordinates."""
    sc = size / (side2560 * scale_full)
    A = sc * R
    t = np.array([size / 2, size / 2]) - A @ (np.asarray(c2560, float) * scale_full)
    return np.hstack([A, t[:, None]])


def warp(full: np.ndarray, M: np.ndarray, size: int = SIZE) -> np.ndarray:
    sc = float(np.hypot(M[0, 0], M[0, 1]))
    return cv2.warpAffine(full, M, (size, size), flags=cv2.INTER_AREA if sc < 1 else cv2.INTER_CUBIC, borderMode=cv2.BORDER_CONSTANT, borderValue=0)


def warp_mask2560(mask: np.ndarray, M: np.ndarray, scale_full: float, size: int = SIZE) -> np.ndarray:
    M3 = M.copy(); M3[:, :2] = M[:, :2] * scale_full
    return cv2.warpAffine(mask.astype(np.uint8), M3, (size, size), flags=cv2.INTER_NEAREST) > 0


def class_map_2560(day_dir: Path, key: str, rec: dict | None, W: int, H: int) -> np.ndarray | None:
    """The part map of one instance (hr window) in the 2560-px frame; 255 = not covered."""
    if not rec or "hr_window" not in rec:
        return None
    p = day_dir / "parts_seg_r05_a" / "hr" / f"{key}.png"
    if not p.exists():
        return None
    from PIL import Image
    m = np.asarray(Image.open(p))
    hx, hy, side, osz = rec["hr_window"]
    a = osz / side                                                         # map px per 2560 px
    Minv = np.array([[a, 0, -hx * a], [0, a, -hy * a]], np.float64)          # 2560 -> map
    return cv2.warpAffine(m, Minv, (W, H), flags=cv2.INTER_NEAREST | cv2.WARP_INVERSE_MAP, borderMode=cv2.BORDER_CONSTANT, borderValue=255)


def principal(mask: np.ndarray):
    ys, xs = np.nonzero(mask)
    P = np.stack([xs, ys], 1).astype(float); c = P.mean(0)
    if len(P) < 3:
        return c, np.array([1.0, 0.0]), P
    _, _, vt = np.linalg.svd(P - c, full_matrices=False)
    return c, vt[0], P


# ------------------------------------------------------------------ fluke
def fluke_cut(full, region: np.ndarray, cls: np.ndarray | None, old_union: np.ndarray, scale_full: float):
    """region: bool 2560-frame mask of the fluke instance (possibly joined); cls: its part map in the same frame (or
    None); old_union: pixels the old instances cover (outside them, no part map exists)."""
    H, W = region.shape
    k3 = np.ones((5, 5), np.uint8)
    near = cv2.dilate(region.astype(np.uint8), k3) > 0
    info = {"area": int(region.sum())}
    if cls is not None:
        fl = near & (cls == CLS["fluke"])
        fl = fl | (region & ~old_union)                                   # restored blades: no map, count as fluke
        info["partmap_fluke_share"] = round(float(fl.sum()) / max(region.sum(), 1), 3)
    else:
        fl = None
    if fl is not None and fl.sum() >= 0.25 * region.sum() and fl.sum() >= 50:
        F = fl; info["pixels"] = "partmap"
    else:
        F = region; info["pixels"] = "mask"
    c, v, P = principal(F)
    if v[0] < 0:
        v = -v                                                                 # span pointing right in the photo (any choice; the stock decides)
    n = np.array([-v[1], v[0]])                                                # the chart's y axis for ex = v (rotation, det +1)
    span_len = float(np.ptp(P @ v)) + 1.0
    # stock direction from the part map: body / head components touching the fluke pixels
    stock_pm = None
    if cls is not None:
        body = (near & ((cls == CLS["body"]) | (cls == CLS["head"]))).astype(np.uint8)
        if body.any():
            nb, lab = cv2.connectedComponents(body, connectivity=8)
            touch = cv2.dilate(F.astype(np.uint8), np.ones((7, 7), np.uint8)) > 0
            ids = np.unique(lab[touch & (body > 0)]); ids = ids[ids > 0]
            if len(ids):
                B = np.isin(lab, ids)
                by, bx = np.nonzero(B)
                Q = np.stack([bx, by], 1).astype(float)
                Q = Q[np.hypot(*(Q - c).T) <= 0.6 * span_len]
                if len(Q) >= max(30, 0.02 * F.sum()):
                    d = Q.mean(0) - c
                    across, along = float(d @ n), float(d @ v)
                    info["stock_pm_px"] = int(len(Q)); info["stock_pm_across"] = round(across / span_len, 3); info["stock_pm_along"] = round(along / span_len, 3)
                    if abs(across) > 0.3 * np.hypot(across, along) and abs(across) > 0.02 * span_len:
                        stock_pm = np.sign(across)                            # + : the stock lies on the +n side
    stock_photo = np.sign(np.array([0.0, 1.0]) @ n) or 1.0                     # the photo's down, on the n axis
    if stock_pm is not None:
        s = stock_pm; info["stock"] = "partmap"; info["stock_agree_photo"] = bool(stock_pm == stock_photo)
    else:
        s = stock_photo; info["stock"] = "photo"
    ex, ey = (v, n) if s < 0 else (-v, -n)                                    # the stock (on the s side of n) must come out at chart -y (top)
    R = np.stack([ex, ey])
    assert np.linalg.det(R) > 0
    # along the span the whole region counts (the part map often covers only part of a blade; the stock lies across
    # the span and barely widens this range), across it the fluke pixels only (the stock would push the square off)
    ry, rx = np.nonzero(region | F); Pr = np.stack([rx, ry], 1).astype(float)
    lo_x, hi_x = np.percentile(Pr @ ex, [0.5, 99.5]); lo_y, hi_y = np.percentile(P @ ey, [0.5, 99.5])
    ext = max(hi_x - lo_x, hi_y - lo_y) + 1.0
    cc = R.T @ np.array([(lo_x + hi_x) / 2, (lo_y + hi_y) / 2])              # centre back in image coordinates
    M = to_raw_M(R, cc, ext * (1 + 2 * MARGIN_FLUKE), scale_full)
    chart = warp(full, M)
    whale = warp_mask2560(region, M, scale_full)
    info.update(angle=round(float(np.degrees(np.arctan2(ex[1], ex[0]))), 1), span_px2560=round(span_len, 1), det=float(np.linalg.det(M[:, :2])))
    return chart, M, whale, warp_mask2560(F, M, scale_full), info


# ------------------------------------------------------------------ fin
def upper_outline_head(cls: np.ndarray, inst: np.ndarray):
    """Largest head patch of the instance whose top edge lies on the instance's top edge (photo frame)."""
    near = cv2.dilate(inst.astype(np.uint8), np.ones((5, 5), np.uint8)) > 0
    hd = (near & (cls == CLS["head"])).astype(np.uint8)
    if hd.sum() < 10:
        return None
    n, lab, st, _ = cv2.connectedComponentsWithStats(hd, connectivity=8)
    k = 1 + int(np.argmax(st[1:, cv2.CC_STAT_AREA])); comp = lab == k
    ys, xs = np.nonzero(comp)
    both = inst | comp
    iy = np.nonzero(both.any(1))[0]; ih = float(iy.max() - iy.min()) + 1.0; tol = 3.0 + 0.04 * ih
    cols = np.unique(xs)
    top_inst = np.argmax(both[:, cols], 0); top_h = np.argmax(comp[:, cols], 0)       # first row from the top per column
    on = float(np.mean(top_h - top_inst <= tol))
    if on < 0.6:
        return {"ok": False, "on_top": round(float(on), 2)}
    return {"ok": True, "on_top": round(float(on), 2), "centroid": [float(xs.mean()), float(ys.mean())],
            "box": [float(xs.min()), float(ys.min()), float(xs.max() + 1), float(ys.max() + 1)], "px": int(len(xs))}


def fin_cuts(full, key: str, up_npz: Path, inst: np.ndarray, rec: dict | None, cls: np.ndarray | None, scale_full: float):
    z = np.load(up_npz)
    M_up = z["M"].astype(float); mirrored = bool(z["mirrored"])
    sc_up = float(np.hypot(M_up[0, 0], M_up[0, 1])); R = M_up[:, :2] / sc_up             # the turn parts_up gave this fin
    c_i, v_i, P_i = principal(inst)
    whale_len = float(np.ptp(P_i @ v_i)) + 1.0
    fin = (rec or {}).get("hr", {}).get("fin")
    info = {"whale_len": round(whale_len, 1), "mirrored": mirrored}
    if fin:
        c = np.array(fin["centroid"]); b = fin["box"]; blade = max(b[2] - b[0], b[3] - b[1]); info["src"] = "fin"; info["fin_px"] = fin["px"]
    else:
        hd = upper_outline_head(cls, inst) if cls is not None else None
        info["head_check"] = None if hd is None else hd.get("on_top")
        if not hd or not hd.get("ok"):
            info["src"] = "A"
            return {}, info
        c = np.array(hd["centroid"]); b = hd["box"]; blade = max(b[2] - b[0], b[3] - b[1]); info["src"] = "head"
    # a fin region is the blade only (about 0.28 of the base); the head patch the model paints on a fin is the whole
    # knob, hump and fin (first sheet: blade / 0.28 made squares several times the photo), so it is taken as the base
    base = blade / RATIO_BLADE_BASE if info["src"] == "fin" else max(blade, 0.2 * whale_len)    # a speck of head: not a 10x zoom (second sheet)
    info["capped"] = bool(base > whale_len); base = min(base, whale_len)
    info.update(blade=round(float(blade), 1), base=round(float(base), 1), centre=[round(float(x), 1) for x in c])
    out = {}
    for name, k in (("finB", 4.0), ("finC", 2.0)):
        M = to_raw_M(R, c, k * base, scale_full)
        chart = warp(full, M); whale = warp_mask2560(inst, M, scale_full)
        if mirrored:
            chart, whale = chart[:, ::-1].copy(), whale[:, ::-1].copy()
        out[name] = (chart, M, whale)
    return out, info


# ------------------------------------------------------------------ per photo
def work(job):
    day, stem, source, want_fluke, want_fin = job
    from PIL import Image
    from ..cluster.chart import decode_full
    from ..cluster.instances_v2 import old_instances
    from ..common import geometry, paths
    cv2.setNumThreads(1)
    d = DAYS / day; out_root = CACHE / day; rows = []
    try:
        W, H = Image.open(d / "images" / f"{stem}.jpg").size
        table = geometry.load_folded_table(paths.FOLDED_TABLE)
        okeys, omasks = old_instances(d, stem, W, H, table)
        old = dict(zip(okeys, omasks)); union = np.any(omasks, 0) if omasks else np.zeros((H, W), bool)
        full = decode_full(source); scale_full = full.shape[1] / W
        segrec = SEG_CACHE.get(day) or {}
        v2 = V2_CACHE.get(day, {}).get(stem)
        done = set()
        for key in want_fluke:
            if key in done or key not in old:
                continue
            region = old[key]; group = [key]; joined = 0
            if v2:
                for inst in v2["instances"]:
                    if inst.get("joined") and key in (inst.get("old") or []):
                        jm = np.asarray(Image.open(d / "instances_v2" / "masks" / f"{stem}.png")) == inst["id"]
                        region = jm | region; group = [k for k in inst["old"] if k in old] or [key]; joined = len(inst["old"])
                        break
            cls = None
            for k in group:
                cm = class_map_2560(d, k, segrec.get(k), W, H)
                if cm is not None:
                    cls = cm if cls is None else np.where(cls == 255, cm, cls)
            chart, M, whale, flm, info = fluke_cut(full, region, cls, union, scale_full)
            info.update(joined=joined, group=group)
            for k in group:                                                    # one cut-out for every old key the fluke holds
                if k not in done:
                    save(out_root / "fluke", k, chart, dict(M=M, mirrored=False, whale=whale, fluke=flm, size=SIZE))
                    rows.append(dict(key=k, variant="fluke", **info)); done.add(k)
        for key in want_fin:
            if key not in old:
                continue
            up = d / "parts_up" / f"{key}__fin.npz"
            if not up.exists():
                continue
            cls = class_map_2560(d, key, segrec.get(key), W, H)
            cuts, info = fin_cuts(full, key, up, old[key], segrec.get(key), cls, scale_full)
            for name, (chart, M, whale) in cuts.items():
                save(out_root / name, key, chart, dict(M=M, mirrored=info["mirrored"], whale=whale, size=SIZE))
            rows.append(dict(key=key, variant="fin", **info))
        return stem, rows, None
    except Exception as e:  # noqa: BLE001
        import traceback
        return stem, rows, f"{type(e).__name__}: {e} {traceback.format_exc()[-400:]}"


def save(folder: Path, key: str, chart: np.ndarray, arrays: dict):
    folder.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(folder / f"{key}.jpg"), cv2.cvtColor(chart, cv2.COLOR_RGB2BGR), [cv2.IMWRITE_JPEG_QUALITY, 92])
    np.savez_compressed(folder / f"{key}.npz", **arrays)


SEG_CACHE: dict = {}
V2_CACHE: dict = {}


def _init(days):
    for day in days:
        d = DAYS / day
        SEG_CACHE[day] = json.loads((d / "parts_seg_r05_a" / "instances.json").read_text())["instances"]
        V2_CACHE[day] = {json.loads(l)["stem"]: json.loads(l) for l in open(d / "instances_v2" / "index.jsonl")}


def jobs_of(day: str, limit: int = 0, only_keys: set | None = None) -> list:
    d = DAYS / day
    vc = json.loads((d / "safe" / "view_class.json").read_text()); thr = float(vc.get("threshold", 0.5))
    src = {Path(json.loads(l)["image"]).stem: json.loads(l)["source"] for l in open(d / "images" / "index.jsonl")}
    by = {}
    for l in open(d / "parts_up" / "index.jsonl"):
        r = json.loads(l)
        for inst in r.get("instances", []):
            k = inst["key"]
            if only_keys is not None and k not in only_keys:
                continue
            pf, pd = vc["views"].get(k, (0.0, 0.0))
            f = pf >= thr; g = pd >= thr and "fin" in inst["parts"]
            fk, gk = by.setdefault(r["stem"], ([], []))
            if f:
                fk.append(k)
            if g:
                gk.append(k)
    jobs = [(day, s, src[s], fk, gk) for s, (fk, gk) in sorted(by.items()) if s in src]
    return jobs[:limit] if limit else jobs


def main(argv=None):
    ap = argparse.ArgumentParser(); ap.add_argument("days"); ap.add_argument("--workers", type=int, default=8); ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--redo", action="store_true")
    a = ap.parse_args(argv)
    days = day_list(a.days)
    for day in days:
        meta = CACHE / day / "meta.jsonl"
        if meta.exists() and not a.redo and not a.limit:
            print(day, "done, skipped", flush=True); continue
        t0 = time.time(); jobs = jobs_of(day, a.limit); n_err = 0; rows = []
        with get_context("spawn").Pool(a.workers, initializer=_init, initargs=([day],)) as pool:     # spawn: rawpy's OpenMP deadlocks under fork
            for i, (stem, rr, err) in enumerate(pool.imap_unordered(work, jobs, chunksize=1)):
                rows += rr
                if err:
                    n_err += 1; print("  error", stem[-40:], err, flush=True)
                if i % 100 == 99:
                    print(f"  {day}: {i + 1}/{len(jobs)} photos, {time.time() - t0:.0f} s", flush=True)
        (CACHE / day).mkdir(parents=True, exist_ok=True)
        tmp = meta.with_suffix(".tmp")
        tmp.write_text("".join(json.dumps(r) + "\n" for r in rows)); tmp.replace(meta) if not a.limit else tmp.rename(meta.with_suffix(".limit.jsonl"))
        nf = sum(r["variant"] == "fluke" for r in rows); nfin = [r for r in rows if r["variant"] == "fin"]
        print(f"{day}: {len(jobs)} photos, {nf} flukes, fins {len(nfin)} (fin {sum(r['src'] == 'fin' for r in nfin)}, head {sum(r['src'] == 'head' for r in nfin)}, A {sum(r['src'] == 'A' for r in nfin)}), "
              f"{n_err} errors, {time.time() - t0:.0f} s", flush=True)


if __name__ == "__main__":
    main()
