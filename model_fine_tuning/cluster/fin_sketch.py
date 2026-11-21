"""Fin sketch (2026-11-18, v2 2026-11-19): the researcher's sketch-pad drawing of a dorsal fin, made from the photos, one per safe
cluster and side, as an ASIDE for the Labeling Assistant (step 2 shows it beside the photos). It feeds nothing: no
ranking, no hint, no link, no saved state reads it. No label is read, so it runs on any day (eval days included).

The field researchers identify whales with a small pad: the dorsal fin's outline plus its scars and marks, left side
and right side. This draws the same thing from a burst of photos:

  frame     every instance of the cluster that shows a dorsal fin (DAY/safe/view_class.json P >= 0.5) is warped from
            its RAW fin cut-out (DAY/parts_up/charts/<key>__fin.jpg + .npz, else the 2560-px photo + canon mask) into
            one keypoint frame: back base -> (0, 0), tip -> (-0.25, -1) in fin heights H, head LEFT (a right-side view
            is mirrored for the work and mirrored back for the picture). Keypoints: the pose-side point model B_both
            (model_fine_tuning/pose_side, frozen DINOv3 + heatmap head), kept per day in DAY/sketch/points.json.
  left out  a frame with wrong keypoints (no fin under the tip, fin above the tip, tip not visible, absurd base),
            a fin under MIN_H px in its source, and a frame that does not register onto the best-seen frame
  per frame contour = the whale mask's top edge and trailing edge, snapped to the strongest brightness step;
            marks = skin markedly brighter than its local skin (scars, rakes, pale pigment); clipped highlights (glare)
            are "not seen"; water is never skin (mask, and in the band along the outline a colour test against the
            water around the fin). Unlike the prototype (experiments/moonshot_sizing/sketch_fin, which dropped a 5-px
            band along the outline for the matcher) the band IS drawn: the researcher's marks often sit on the pale
            trailing edge and the tip. Rim light there is told from pigment by depth and brightness: a bright component that
            hugs the outermost pixels (or is near-clipped) is light on the rim, not the animal; what is left of the band is
            drawn in a lighter ink (pale edge, or light on the edge).
  per side  frames registered on the best-seen of the most side-on frames (base ratio at or above the stack's median)
            (scale + shift); a mark is kept where at least half of the frames that saw the place drew it (and at least
            two did) and when it is LARGE (v2: >= MARK_D fin heights across and >= SRC_MIN_PX source px; the
            researchers ignore very small marks); contour = median per column / row over the most side-on frames;
            nicks = cuts of the trailing edge (and near the tip) against its own smoothed curve. One frame = a
            single-frame drawing, drawn in grey and flagged: less trustworthy than a consensus.
  picture   (v2) the back and hump in front of the fin and the fin as one outline, in ONE frame for every drawing
            (fin height, tip and back base at the same place, a fixed turn PHI0 so the typical back lies level), with
            an estimate of the view (view_label: "side" / "angled" / unknown; never a corrected shape: straightening
            was measured and not adopted, see experiments/sketch_v2). The picture-frame arrays are kept beside the PNG
            (<cluster>__<L|R>.npz) so the web app can compose an ANIMAL-LEVEL drawing from the stacks the researcher
            put together (compose: outline of the most side-on stack, marks of all stacks lined up, solid when found
            in two or more stacks, light when in one).

    python -m model_fine_tuning.cluster.fin_sketch points --days DAY[,DAY...]   keypoints (GPU only for the instances
                                                       the pose-side caches lack; run under the GPU lock)
    python -m model_fine_tuning.cluster.fin_sketch draw --days DAY[,DAY...] [--workers 6]
    python -m model_fine_tuning.cluster.fin_sketch sheet --day DAY --out X.jpg [--n 12]   fin cut-out | drawing
    --days all = every day folder with safe clusters; --root = the days folder (default experiments/days_all)

    draw --out-name sketch/v2 writes the set into DAY/sketch/v2/ (beside the drawings in use, to check them first)

Writes DAY/sketch/points.json, DAY/sketch/<cluster id>__<L|R>.png (white paper, outline black, marks dark, marks on the
outline mid grey (pale edge or light on the edge), nicks red, skin not seen hatched grey, thin blue guides at the tip
and the base; head left on a left side, right on a right side), <cluster id>__<L|R>.npz (the picture-frame arrays) and
DAY/sketch/index.json: {day, made, version, clusters_sha1, picture, clusters: {id: {L|R: {png, npz, n, n_side,
n_outline, keys, H, base, bperp, profile, level_deg, view, ncc, cover, score, grade, single, marks, small_left_out,
nicks}}}, summary}. n = frames drawn, n_side = frames of that side (n_side - n were left out),
grade good | fair | poor = how well the fin was seen (frames, size in the source, side-on view, coverage).
"""
from __future__ import annotations

import argparse
import collections
import datetime
import hashlib
import json
import os
import sys
import time
from multiprocessing import Pool
from pathlib import Path

import cv2
import numpy as np
from scipy.ndimage import gaussian_filter1d, median_filter

ROOT = Path(__file__).resolve().parents[2]
DAYS = ROOT / "experiments" / "days_all"
POSE_CACHE = Path(os.environ.get("FIN_SKETCH_POSE_CACHE", "/var/tmp/sat3737-parts-cache/pose_side"))
CACHED_POINTS = ("pred_Bboth_train", "pred2_Bboth_val")      # B_both on every instance of the 21 train + 3 val days
HEADS = [Path(os.environ["FIN_SKETCH_HEAD"])] if os.environ.get("FIN_SKETCH_HEAD") else []
HEADS += [Path("/home/sat3737/NOAA_whales/moonshot_cache/sketch_aside/point_model/head.pt"), POSE_CACHE / "B_both" / "head.pt"]
VERSION = 2
P_DORSAL = 0.5

# work tile (the prototype's frame, larger since v2 to hold the hump and the back in front of the fin): x in [X0, X1],
# y in [Y0, Y1] fin heights, PXH px per fin height; back base (0, 0), tip TIP_C, head LEFT
TIP_C = np.array([-0.25, -1.0])
X0, X1, Y0, Y1 = -4.6, 1.0, -2.0, 1.3
PXH = 96
TW, TH = int(round((X1 - X0) * PXH)), int(round((Y1 - Y0) * PXH))

# the picture (v2): the sketch-pad view, the same for every drawing so two can be compared by eye: back base and tip
# where the frame put them (one fin height = DPX px on every drawing, the tip at the same place), the whole tile turned
# by the fixed PHI0 so that the back of the typical fin lies level (median turn of the base line front base -> back
# base over the stacks of the 21 train days: 10.3 degrees). u along the body (head toward -u), tip at (0, -1).
# Measured 2026-11-19 (experiments/sketch_v2/straighten.py): levelling EACH drawing on its own base line made the
# outlines of the same animal less alike than this fixed frame (outline AUC same vs different 0.525 against 0.585; the
# front-base point is the noisiest of the three), so the turn is one and the same for all.
PHI0 = 10.0
DU0, DU1, DV0, DV1 = -3.3, 0.9, -1.14, 0.28
DPX = 150
DW, DH = int(round((DU1 - DU0) * DPX)), int(round((DV1 - DV0) * DPX))

MIN_H = 24                              # fin height in its source (RAW cut-out px) below which a frame is not drawn
BASE_RANGE = (0.6, 16.0)                # |front base - back base| / H: below = seen from behind or in front, not side-on; above = points wrong
TIP_VISIBLE = 0.5                       # the point model's P(tip visible) below this: left out
NICK_D, NICK_SM = 0.025, 4.0
NICK_TRAIL, NICK_SINGLE = 0.035, 0.05   # nick depth (H) on a consensus trailing edge / on a single frame
K_MARK, MIN_AREA = 2.8, 12
K_EDGE, EDGE_DEPTH = 3.0, 4             # the band along the outline: stricter threshold; half a component >= 3 px deep (else rim light)
RIM = 5
CLIP = 245
CLIP_EDGE = 220                         # in the band along the outline a near-clipped pixel is glare or rim light, not seen
MIN_NCC = 0.35
TOL, PERSIST = 5, 0.5
# marks drawn (v2, the researchers: "large marks no matter what they are, very small ones ignored"): a mark is drawn when
# it is at least MARK_D fin heights across (equivalent diameter, so relative to the fin, not to the photo) AND at least
# SRC_MIN_PX source pixels in the frames it was seen in (a mark of 3 px in a small fin is noise however it is scaled).
# Measured 2026-11-19 (experiments/sketch_v2/mark_size.py, 21 train + 3 val days): two stacks of the same animal lined up,
# is the first one's mark found again in the second, against two different animals: the excess (lift) is flat for the
# marks under 0.06 H (0.080 / 0.076) and grows from there (0.06-0.08: 0.098, 0.08-0.10: 0.112, 0.10-0.13: 0.130,
# 0.13-0.18: 0.178), so marks under 0.06 H are left out; the source-pixel floor removes next to nothing beyond that.
MARK_D = 0.06
SRC_MIN_PX = 12
DRAW_K = 12                             # at most the 12 largest
EDGE_FROM = -1.9                        # marks in the band along the outline are drawn only from here back (fin heights)

COLS_TOP = np.arange(0, int(round((-0.25 - X0) * PXH)) + 1)          # the top edge, from the tile's front end to the tip
ROWS_TR = np.arange(int(round((-0.97 - Y0) * PXH)), int(round((-0.08 - Y0) * PXH)) + 1)
TRSEL = (ROWS_TR >= (-0.85 - Y0) * PXH) & (ROWS_TR <= (-0.12 - Y0) * PXH)
def px(x, y):
    return (x - X0) * PXH, (y - Y0) * PXH


def to_tile(q):
    return np.array([(q[0] - X0) * PXH, (q[1] - Y0) * PXH])


# ---- the day -----------------------------------------------------------------------------------------------------
def day_dir(day: str, root: Path = DAYS) -> Path:
    return Path(root) / day


def clusters_of(d: Path) -> list[dict]:
    return json.loads((d / "safe" / "clusters.json").read_text())["clusters"]


def dorsal_keys(d: Path) -> dict:
    """{instance key: box} for every clustered instance the view classifier says shows a dorsal fin."""
    v = json.loads((d / "safe" / "view_class.json").read_text())
    j = v.get("order", ["fluke", "dorsal"]).index("dorsal")
    inc = {k for c in clusters_of(d) for k in c["keys"]}
    return {k: v["box"][k] for k, p in v["views"].items() if p[j] >= P_DORSAL and k in inc and k in v.get("box", {})}


# ---- keypoints ---------------------------------------------------------------------------------------------------
def _cached_points() -> dict:
    P, META = {}, {}
    for f in CACHED_POINTS:
        if (POSE_CACHE / f"{f}.json").exists():
            P.update(json.loads((POSE_CACHE / f"{f}.json").read_text()))
            META.update(json.loads((POSE_CACHE / f"{f}_meta.json").read_text())["rows"])
    return P, META


def _row(p, meta, src):
    st = meta.get("state", {}) if meta else {}
    vis = lambda n: (st.get(n) or [None])[0]
    return {"back": p["dorsal_fin_back"], "tip": p["dorsal_fin_tip"], "front": p["dorsal_fin_front"],
            "vis": [vis("dorsal_fin_back"), vis("dorsal_fin_tip"), vis("dorsal_fin_front")], "src": src}


def predict(d: Path, boxes: dict, head_path: Path) -> dict:
    """The pose-side point model on instances of a day: the same window (the part model's hr window: the box centre,
    side 1.5 x its long side, at least 192 px), the same mirror test-time augmentation and decoding as
    model_fine_tuning/pose_side/predict.py (which refuses the eval days; this reads no label)."""
    import torch
    from ..pose_side.data import CROP, FLIP, KP, _affine, normalise
    from ..pose_side.model import Backbone, Head, decode
    dev = torch.device("cuda")
    bb = Backbone().to(dev); head = Head().to(dev); head.load_state_dict(torch.load(head_path, map_location=dev)); head.eval()
    out, last = {}, (None, None)
    with torch.no_grad():
        for key in sorted(boxes):
            b = boxes[key]; cx, cy = (b[0] + b[2]) / 2, (b[1] + b[3]) / 2; side = max(192.0, 1.5 * max(b[2] - b[0], b[3] - b[1]))
            stem = key.rsplit("~", 1)[0]
            if last[0] != stem:
                im = cv2.imread(str(d / "images" / f"{stem}.jpg"))
                if im is None:
                    continue
                last = (stem, cv2.cvtColor(im, cv2.COLOR_BGR2RGB))
            frame = last[1]; M = _affine(cx, cy, side, 0.0, False)
            crop = cv2.warpAffine(frame, M, (CROP, CROP), flags=cv2.INTER_AREA if side > CROP else cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)
            x = torch.stack([normalise(crop), normalise(crop[:, ::-1].copy())]).to(dev)
            hm, st = head(bb(x)); hm = hm.float(); st = st.float().softmax(-1)
            hm_f = hm[1].flip(-1)[FLIP]; hm = (hm[0] + hm_f)[None] / 2; stp = ((st[0] + st[1][FLIP]) / 2).cpu().numpy()
            pts, conf = decode(hm); pts = pts[0].cpu().numpy()
            Mi = np.linalg.inv(np.vstack([M, [0, 0, 1]]))[:2]
            p = {n: [float(v) for v in Mi[:, :2] @ pts[k] + Mi[:, 2]] for k, n in enumerate(KP)}
            meta = {"state": {n: [round(float(v), 4) for v in stp[k]] for k, n in enumerate(KP)}}
            out[key] = _row(p, meta, "predicted")
    return out


def cmd_points(days, root, force_predict=False, head=None):
    P, META = ({}, {}) if force_predict else _cached_points()
    for day in days:
        d = day_dir(day, root); boxes = dorsal_keys(d); t0 = time.time()
        rows = {k: _row(P[k], META.get(k), "pose_side cache") for k in boxes if k in P}
        miss = {k: b for k, b in boxes.items() if k not in rows}
        if miss:
            hp = Path(head) if head else next((h for h in HEADS if h.exists()), None)
            if hp is None:
                raise SystemExit(f"{day}: {len(miss)} instances need the point model and no head.pt was found ({', '.join(map(str, HEADS))})")
            rows.update(predict(d, miss, hp))
        out = d / "sketch" / "points.json"; out.parent.mkdir(parents=True, exist_ok=True)
        tmp = out.with_suffix(".tmp"); tmp.write_text(json.dumps({"day": day, "model": "pose_side B_both", "made": _now(), "points": rows})); tmp.replace(out)
        print(f"{day}: {len(rows)} dorsal instances ({len(miss)} predicted now, {time.time() - t0:.0f} s) -> {out}", flush=True)


# ---- one frame ---------------------------------------------------------------------------------------------------
def _sim(src, dst):
    s0, s1 = complex(*src[0]), complex(*src[1]); d0, d1 = complex(*dst[0]), complex(*dst[1])
    a = (d1 - d0) / (s1 - s0 + 1e-9); b = d0 - a * s0
    return np.array([[a.real, -a.imag, b.real], [a.imag, a.real, b.imag]])


def frame(B, T, F):
    """similarity (2x3, with a reflection when the head is on the right) source px -> tile px; mirrored; F canonical."""
    M = _sim([B, T], [to_tile([0, 0]), to_tile(TIP_C)])
    Fc = (M[:, :2] @ F + M[:, 2]) / PXH + np.array([X0, Y0])
    mirrored = bool(Fc[0] > 0)
    if mirrored:
        R = np.array([[-1.0, 0, 0], [0, 1.0, 0]])
        Ms = _sim([R[:, :2] @ B, R[:, :2] @ T], [to_tile([0, 0]), to_tile(TIP_C)])
        M = Ms @ np.vstack([R, [0, 0, 1]])
        Fc = (M[:, :2] @ F + M[:, 2]) / PXH + np.array([X0, Y0])
    return M, mirrored, Fc


def source(d: Path, key: str):
    """(rgb, mask bool, map 2560-px point -> source px, name) of an instance: the RAW fin cut-out, else the photo."""
    zp, jp = d / "parts_up" / f"{key}__fin.npz", d / "parts_up" / "charts" / f"{key}__fin.jpg"
    if zp.exists() and jp.exists():
        z = np.load(zp, allow_pickle=True); im = cv2.imread(str(jp))
        if im is not None:
            img = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
            Mc, sf, mir = z["M"], float(z["scale_full"]), bool(z["mirrored"])
            fv = bool(z["flipped_v"]) if "flipped_v" in z.files else False
            n = img.shape[0] - 1

            def f(p):
                c = Mc[:, :2] @ (np.asarray(p, float) * sf) + Mc[:, 2]
                c = c * img.shape[0] / 1536.0
                if mir:
                    c[0] = n - c[0]
                if fv:
                    c[1] = n - c[1]
                return c
            m = z["whale"]
            if m.shape != img.shape[:2]:
                m = cv2.resize(m.astype(np.uint8), img.shape[1::-1], interpolation=cv2.INTER_NEAREST) > 0
            return img, m.astype(bool), f, "raw"
    stem = key.rsplit("~", 1)[0]
    im = cv2.imread(str(d / "images" / f"{stem}.jpg"))
    if im is None:
        return None
    img = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
    m = cv2.imread(str(d / "canon" / f"{stem}_mask.png"), cv2.IMREAD_GRAYSCALE)
    m = (m > 0) if m is not None else np.zeros(img.shape[:2], bool)
    return img, m, (lambda p: np.asarray(p, float)), "photo"


def side_of(p) -> str:
    """'R' when the head points right in the photo (the whale's right side faces the camera), else 'L': the sign of
    (tip - back base) x (front base - back base), which a rotation of the photo does not change."""
    B, T, F = (np.asarray(p[k], float) for k in ("back", "tip", "front"))
    a, b = T - B, F - B
    return "R" if a[0] * b[1] - a[1] * b[0] > 0 else "L"


def box_frac(m, x, y, r):
    c, rr = px(x, y), r * PXH
    x0, x1, y0, y1 = int(max(c[0] - rr, 0)), int(min(c[0] + rr, TW)), int(max(c[1] - rr, 0)), int(min(c[1] + rr, TH))
    sub = m[y0:y1, x0:x1]
    return float(sub.mean()) if sub.size else 0.0


def one(d: Path, key: str, p: dict):
    """(tile rgb, mask, inside, info) or (None, reason)."""
    if p.get("vis") and p["vis"][1] is not None and p["vis"][1] < TIP_VISIBLE:
        return None, "tip not visible"
    s = source(d, key)
    if s is None:
        return None, "no image"
    img, m, f, src = s
    B, T, F = f(p["back"]), f(p["tip"]), f(p["front"])
    H = float(np.linalg.norm(T - B))
    if H < MIN_H:
        return None, "too small"
    base = float(np.linalg.norm(F - B) / H)
    if base < BASE_RANGE[0]:
        return None, "not side-on"
    if base > BASE_RANGE[1]:
        return None, "points wrong"
    M, mir, Fc = frame(B, T, F)
    down = abs(np.linalg.det(M[:, :2])) < 1
    t = cv2.warpAffine(img, M, (TW, TH), flags=cv2.INTER_AREA if down else cv2.INTER_CUBIC, borderValue=(0, 0, 0))
    mt = cv2.warpAffine(m.astype(np.uint8) * 255, M, (TW, TH), flags=cv2.INTER_LINEAR, borderValue=0)
    ins = cv2.warpAffine(np.full(img.shape[:2], 255, np.uint8), M, (TW, TH), flags=cv2.INTER_NEAREST, borderValue=0)
    wm = (mt > 127) & (ins > 0)
    if box_frac(wm, -0.6, -0.45, 0.08) < 0.6 or box_frac(wm, -0.25, -1.18, 0.07) > 0.5:      # a fin under the tip, open water above it
        return None, "points wrong"
    return (t, mt, ins, dict(H=H, base=base, src=src, side=side_of(p), Fc=[float(Fc[0]), float(Fc[1])], base_p=base_perp(Fc))), None


def base_perp(Fc) -> float:
    """|front base - back base| over the fin height measured square to the base line (canonical units)."""
    F = np.asarray(Fc, float); L = float(np.linalg.norm(F))
    if L < 1e-6:
        return 0.0
    a = -F / L
    hp = abs(a[0] * TIP_C[1] - a[1] * TIP_C[0])
    return L / max(hp, 1e-6)


# ---- drawing of one frame -------------------------------------------------------------------------------------------
def whale_of(mask, inside):
    m = (mask > 127) & (inside > 0)
    m = cv2.morphologyEx(m.astype(np.uint8), cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    n, lab, st, _ = cv2.connectedComponentsWithStats(m, connectivity=8)
    if n <= 1:
        return m > 0
    cx, cy = (int(v) for v in px(-0.45, -0.35))
    k = lab[cy, cx] if lab[cy, cx] > 0 else 1 + int(np.argmax(st[1:, cv2.CC_STAT_AREA]))
    return lab == k


def contour(whale, inside, L):
    g = cv2.GaussianBlur(L, (0, 0), 1.2)
    gy, gx = np.diff(g, axis=0, prepend=g[:1]), np.diff(g, axis=1, prepend=g[:, :1])
    top = np.full(len(COLS_TOP), np.nan, np.float32)
    for j, c in enumerate(COLS_TOP):
        col = whale[:, c]
        if not col.any():
            continue
        y = int(np.argmax(col))
        if y <= 3 or not inside[y - 3, c]:
            continue
        lo, hi = max(y - 4, 1), min(y + 5, TH)
        top[j] = lo + int(np.argmax(np.abs(gy[lo:hi, c])))
    trail = np.full(len(ROWS_TR), np.nan, np.float32)
    for i, r in enumerate(ROWS_TR):
        yr = r / PXH + Y0
        xref = int(round(px(-0.25 - 0.3 * (yr + 1.0), 0)[0]))
        row = whale[r]
        if not row[xref]:
            near = np.nonzero(row[max(xref - 12, 0):xref + 12])[0]
            if not len(near):
                continue
            xref = max(xref - 12, 0) + int(near[0])
        off = np.nonzero(~row[xref:])[0]
        if not len(off):
            continue
        x = xref + int(off[0])
        if x >= TW - 3 or not inside[r, min(x + 2, TW - 1)] or x / PXH + X0 > 0.35:
            continue
        lo, hi = max(x - 4, 1), min(x + 5, TW)
        trail[i] = lo + int(np.argmax(np.abs(gx[r, lo:hi])))
    return top, trail


def nicks(prof, sign, sm=NICK_SM, depth=NICK_D):
    ok = ~np.isnan(prof)
    if ok.sum() < 10:
        return [], np.zeros_like(prof)
    idx = np.arange(len(prof))
    f = np.interp(idx, idx[ok], prof[ok])
    s = gaussian_filter1d(f, sm, mode="nearest")
    r = sign * (f - s) / PXH
    r[~ok] = 0
    out = []
    for i in range(1, len(r) - 1):
        if r[i] >= depth and r[i] >= r[i - 1] and r[i] >= r[i + 1] and ok[max(i - 3, 0):i + 4].all():
            if not out or i - out[-1] > 4:
                out.append(i)
    return out, r


def marks(rgb, whale, inside):
    """(marks, seen, contrast) of one tile. Skin = the whale mask without its boundary pixel. The core (RIM px or more
    inside the outline) as in the prototype; the band along the outline with a stricter threshold, only pixels that
    are not water-coloured, and only components that fill the band (at least half their pixels EDGE_DEPTH - 1 px or more
    inside) or run on into a mark of the core: rim light hugs the outermost pixels."""
    lab = cv2.cvtColor(rgb, cv2.COLOR_RGB2LAB).astype(np.float32)
    L, Bc = lab[..., 0], lab[..., 2]
    ins = inside > 0
    clip = cv2.dilate((rgb.max(2) >= CLIP).astype(np.uint8), np.ones((3, 3), np.uint8)).astype(bool)
    w8 = whale.astype(np.uint8)
    skin = cv2.erode(w8, np.ones((3, 3), np.uint8)) > 0
    core = cv2.erode(w8, np.ones((2 * RIM + 1, 2 * RIM + 1), np.uint8)) > 0
    seen_core = core & ~clip & ins
    if seen_core.sum() < 200:
        z = np.zeros_like(seen_core)
        return z, seen_core, np.zeros_like(L)
    ring = (cv2.dilate(w8, np.ones((11, 11), np.uint8)) > 0) & ~(cv2.dilate(w8, np.ones((5, 5), np.uint8)) > 0) & ins & ~clip
    b_whale = float(np.median(Bc[seen_core]))
    not_water = np.ones_like(seen_core)
    if ring.sum() >= 50:
        b_water = float(np.median(Bc[ring]))
        if b_whale - b_water > 4:                         # the water is bluer than the skin: a pixel bluer than half-way is water
            not_water = Bc > (b_whale + b_water) / 2
    glare_edge = cv2.dilate((rgb.max(2) >= CLIP_EDGE).astype(np.uint8), np.ones((3, 3), np.uint8)).astype(bool)
    seen_edge = skin & ~core & ~clip & ~glare_edge & ins & not_water
    fill = L.copy()
    fill[~whale] = np.median(L[seen_core])
    bg = median_filter(fill, size=17)
    dd = L - bg
    v = dd[seen_core]
    sd = max(1.4826 * np.median(np.abs(v - np.median(v))), 3.0)
    c = dd / sd
    def comps(m):
        n, lab_, st, _ = cv2.connectedComponentsWithStats(m.astype(np.uint8), connectivity=8)
        return n, lab_, st[:, cv2.CC_STAT_AREA]
    n, lab_, area = comps((c >= K_MARK) & seen_core)
    keep = area >= MIN_AREA; keep[0] = False
    m_core = keep[lab_]
    # the band along the outline: a bright component there is pale pigment when it fills the band (or runs on into a
    # mark of the core), rim light when it hugs the outermost pixels
    depth = cv2.distanceTransform(w8, cv2.DIST_L2, 3)
    n, lab_, area = comps((c >= K_EDGE) & seen_edge)
    if n > 1:
        deep = np.bincount(lab_.ravel(), weights=(depth >= EDGE_DEPTH - 1).ravel().astype(float), minlength=n) / np.maximum(area, 1)
        touch = np.zeros(n, bool)
        touch[np.unique(lab_[cv2.dilate(m_core.astype(np.uint8), np.ones((3, 3), np.uint8)) > 0])] = True
        keep = (area >= MIN_AREA // 2) & ((deep >= 0.5) | touch); keep[0] = False
        m_edge = keep[lab_]
    else:
        m_edge = np.zeros_like(m_core)
    m = m_core | m_edge
    return m, seen_core | seen_edge, c


def frame_drawing(rgb, mask, inside):
    whale = whale_of(mask, inside)
    L = cv2.cvtColor(rgb, cv2.COLOR_RGB2LAB)[..., 0].astype(np.float32)
    top, trail = contour(whale, inside > 0, L)
    m, seen, c = marks(rgb, whale, inside)
    return dict(top=top, trail=trail, marks=m, seen=seen, whale=whale, c=c)


SCALES = (0.9, 0.95, 1.0, 1.05, 1.1)
SHIFT = 24


def feat(fr):
    c = np.clip(fr["c"], 0, 6) * fr["seen"]
    w = fr["whale"].astype(np.uint8)
    band = (w - cv2.erode(w, np.ones((5, 5), np.uint8))).astype(np.float32)
    return cv2.GaussianBlur(c.astype(np.float32), (0, 0), 1.5) + 2.0 * cv2.GaussianBlur(band, (0, 0), 2.0)


def register(fa, fb):
    A, Bf = feat(fa), feat(fb)
    Ap = cv2.copyMakeBorder(A, SHIFT, SHIFT, SHIFT, SHIFT, cv2.BORDER_CONSTANT, value=0)
    cx, cy = px(-0.5, -0.4)
    best = (-2, None)
    for s in SCALES:
        M = np.array([[s, 0, cx - s * cx], [0, s, cy - s * cy]], np.float32)
        r = cv2.matchTemplate(Ap, cv2.warpAffine(Bf, M, (TW, TH)), cv2.TM_CCORR_NORMED)
        _, v, _, loc = cv2.minMaxLoc(r)
        if v > best[0]:
            M2 = M.copy(); M2[0, 2] += loc[0] - SHIFT; M2[1, 2] += loc[1] - SHIFT
            best = (v, M2)
    return best[1], float(best[0])


def side_drawing(tiles, infos):
    """tiles [(rgb, mask, inside)] of one side -> (consensus dict, frame indices used, ncc per frame, F of the frames in the
    reference frame's coordinates). v2: the reference is the best-seen of the most side-on frames (base ratio square to
    the base line at or above the stack's median), and the outline is the median of those frames only; the marks come
    from every frame that lines up."""
    fr = [frame_drawing(*t) for t in tiles]
    bp = np.array([infos[j]["base_p"] for j in range(len(fr))])
    side_on = [j for j in range(len(fr)) if len(fr) < 3 or bp[j] >= np.median(bp) - 1e-9]
    ref = max(side_on, key=lambda j: fr[j]["seen"].sum() * min(infos[j]["H"], 150))
    ncc = [1.0] * len(fr)
    Fr = [np.asarray(infos[j]["Fc"], float) for j in range(len(fr))]
    for j in range(len(fr)):
        if j == ref:
            continue
        M, v = register(fr[ref], fr[j])
        rgb, mask, ins = tiles[j]
        fr[j] = frame_drawing(cv2.warpAffine(rgb, M, (TW, TH), flags=cv2.INTER_LINEAR), cv2.warpAffine(mask, M, (TW, TH), flags=cv2.INTER_LINEAR),
                              cv2.warpAffine(ins, M, (TW, TH), flags=cv2.INTER_NEAREST))
        ncc[j] = v
        q = M[:, :2] @ to_tile(Fr[j]) + M[:, 2]                    # its front base, in the reference frame
        Fr[j] = np.array([q[0] / PXH + X0, q[1] / PXH + Y0])
    used = [j for j in range(len(fr)) if ncc[j] >= MIN_NCC]
    if not used:
        return None, [], ncc, []
    outline = [j for j in used if j in side_on] or used
    H = float(np.median([infos[j]["H"] for j in used]))
    cons = consensus([fr[j] for j in used], [fr[j] for j in outline], H)
    cons["n_outline"] = len(outline)
    return cons, used, ncc, [Fr[j] for j in used]


def mark_min_area(H: float) -> float:
    """smallest mark drawn, in tile px: MARK_D fin heights across, and SRC_MIN_PX pixels in a source whose fin is H px high."""
    return max(np.pi / 4 * (MARK_D * PXH) ** 2, SRC_MIN_PX * (PXH / max(H, 1.0)) ** 2)


def consensus(frames, outline_frames=None, H=100.0):
    n = len(frames)
    of = outline_frames or frames
    top = np.nanmedian(np.stack([f["top"] for f in of]), 0)
    trail = np.nanmedian(np.stack([f["trail"] for f in of]), 0)
    seen_n = np.sum([f["seen"] for f in frames], 0)
    k = np.ones((2 * TOL + 1, 2 * TOL + 1), np.uint8)
    near = np.sum([cv2.dilate(f["marks"].astype(np.uint8), k).astype(bool) & f["seen"] for f in frames], 0)
    raw = np.sum([f["marks"] for f in frames], 0)
    if n >= 2:
        cm = (seen_n >= 2) & (near >= PERSIST * np.maximum(seen_n, 1)) & (raw >= 1) & (near >= 2)
    else:
        cm = raw >= 1
    # marks in the band along the outline ahead of the fin (the back and the hump, drawn since v2) are left out: there
    # the band is mostly sun on the rim of the back; v1 drew the band only from x = -1.9 H, and so does v2
    whale0 = np.mean([f["whale"] for f in frames], 0) >= 0.5
    rim = cv2.distanceTransform(whale0.astype(np.uint8), cv2.DIST_L2, 3) <= RIM + 1
    rim[:, int(round((EDGE_FROM - X0) * PXH)):] = False
    cm &= ~rim
    cm, small = clean_marks(cm, mark_min_area(H))
    tipc = int(round((-0.7 - X0) * PXH))
    ntop = [i for i in nicks(top, +1, depth=NICK_D if n >= 2 else NICK_SINGLE)[0] if COLS_TOP[i] >= tipc]
    ntr = [i for i in nicks(trail, -1, depth=NICK_TRAIL if n >= 2 else NICK_SINGLE)[0] if TRSEL[i]]
    whale = np.mean([f["whale"] for f in frames], 0) >= 0.5
    return dict(top=top, trail=trail, marks=cm, small=small, seen_n=seen_n, n=n, nick_top=ntop, nick_trail=ntr, whale=whale)


def clean_marks(m, min_area):
    """(marks drawn, marks left out as too small): components >= min_area tile px, the DRAW_K largest."""
    n, lab, st, _ = cv2.connectedComponentsWithStats(m.astype(np.uint8), connectivity=8)
    area = st[:, cv2.CC_STAT_AREA]
    comps = sorted(((area[i], i) for i in range(1, n) if area[i] >= min_area), reverse=True)[:DRAW_K]
    keep = np.zeros(n, bool)
    for _, i in comps:
        keep[i] = True
    small = np.zeros(n, bool); small[1:] = ~keep[1:]
    return keep[lab], small[lab]


# ---- the sketch-pad view -------------------------------------------------------------------------------------------
def view_frame(F=None):
    """(2x3 tile px -> picture px, turn of this drawing's base line in degrees, base ratio square to the base line).
    The picture transform is the same for every drawing (PHI0, DPX, tip at (0, -1)); F (the front base, canonical
    units, median over the frames) only gives the two view cues."""
    r = np.radians(-PHI0)
    R = np.array([[np.cos(r), -np.sin(r)], [np.sin(r), np.cos(r)]])
    # canonical p -> (u, v) = R (p - T) + (0, -1); picture px = ((u - DU0) DPX, (v - DV0) DPX); tile px q -> p = q / PXH + (X0, Y0)
    off = R @ (np.array([X0, Y0]) - TIP_C) + np.array([0.0, -1.0])
    M = np.zeros((2, 3))
    M[:, :2] = R / PXH * DPX
    M[:, 2] = (off - np.array([DU0, DV0])) * DPX
    phi, bperp = float("nan"), float("nan")
    if F is not None:
        F = np.asarray(F, float); L = float(np.linalg.norm(F))
        if L > 1e-6:
            a = -F / L
            phi = float(np.degrees(np.arctan2(a[1], a[0])))
            bperp = L / max(abs(a[0] * TIP_C[1] - a[1] * TIP_C[0]), 1e-6)
    return M, phi, bperp


def to_view(dr, M):
    """the consensus drawing (tile) in the picture frame: rasters (whale, seen_n, marks, small marks, mark on the
    outline), outline polylines (top edge, trailing edge; NaN rows = gaps) and nicks, all in picture px."""
    warp = lambda a, interp=cv2.INTER_LINEAR: cv2.warpAffine(a.astype(np.float32), M, (DW, DH), flags=interp, borderValue=0)
    depth = cv2.distanceTransform(dr["whale"].astype(np.uint8), cv2.DIST_L2, 3)
    edge = dr["marks"] & (depth <= RIM + 1)
    pt = lambda xs, ys: np.stack([M[0, 0] * xs + M[0, 1] * ys + M[0, 2], M[1, 0] * xs + M[1, 1] * ys + M[1, 2]], 1)
    top = pt(COLS_TOP.astype(float), smooth_nan(dr["top"], 3))
    trail = pt(smooth_nan(dr["trail"], 1.5), ROWS_TR.astype(float))
    nk = [pt(np.array([float(COLS_TOP[i])]), np.array([float(dr["top"][i])]))[0].tolist() + [0] for i in dr["nick_top"] if not np.isnan(dr["top"][i])]
    nk += [pt(np.array([float(dr["trail"][i])]), np.array([float(ROWS_TR[i])]))[0].tolist() + [1] for i in dr["nick_trail"] if not np.isnan(dr["trail"][i])]
    single = dr["n"] < 2
    return dict(whale=warp(dr["whale"]) >= 0.5, seen=warp(dr["seen_n"] >= (1 if single else 2)) >= 0.5,
                marks=warp(dr["marks"]) >= 0.45, edge=warp(edge) >= 0.45, small=warp(dr["small"]) >= 0.45,
                top=top.astype(np.float32), trail=trail.astype(np.float32), nicks=np.array(nk, np.float32).reshape(-1, 3), n=int(dr["n"]))


def save_view(path: Path, v: dict, meta: dict):
    """the picture-frame drawing as arrays (for the animal-level drawing composed on request by the web app)."""
    tmp = path.with_suffix(".tmp.npz")
    np.savez_compressed(tmp, whale=v["whale"], seen=v["seen"], marks=v["marks"], edge=v["edge"], small=v["small"], top=v["top"], trail=v["trail"],
                        nicks=v["nicks"], n=v["n"], meta=json.dumps(meta))
    tmp.replace(path)


# ---- the picture ---------------------------------------------------------------------------------------------------
def smooth_nan(p, s):
    ok = ~np.isnan(p)
    if ok.sum() < 3:
        return p
    i = np.arange(len(p))
    f = gaussian_filter1d(np.interp(i, i[ok], p[ok]), s, mode="nearest")
    return np.where(ok, f, np.nan)


INK = dict(paper=(255, 255, 255), skin=(244, 242, 234), unseen=(205, 205, 205), line=(0, 0, 0), mark=(30, 30, 30), edge=(120, 120, 120),
           light=(150, 150, 150), light_edge=(185, 185, 185), small=(255, 150, 40), nick=(215, 25, 25), guide=(170, 200, 230))


def render_view(v: dict, side: str, single: bool = False, label: str = "", show_small: bool = False, light=None, contour_ink=None) -> np.ndarray:
    """RGB picture of a picture-frame drawing (DW x DH), head left on a left side, right on a right side. single: drawn
    in grey and marked "1 frame". light (bool raster): marks drawn in a light ink (the animal-level drawing: seen in
    one stack only). show_small: the marks left out as too small, in orange (sheets only). A thin blue guide marks the
    base line (v = 0) and the tip height (v = -1), the same on every drawing."""
    H, W = DH, DW
    img = np.full((H, W, 3), 255, np.uint8)
    for vv in (0.0, -1.0):
        y = int(round((vv - DV0) * DPX)); img[y, ::3] = INK["guide"]
    x = int(round((0 - DU0) * DPX)); img[::3, x] = INK["guide"]
    hatch = (np.add.outer(np.arange(H), np.arange(W)) // 5) % 6 == 0
    wh, seen = v["whale"], v["seen"]
    img[wh & ~seen & hatch] = INK["unseen"]
    img[wh & seen] = INK["skin"]
    mk, edge = v["marks"], v["edge"] & v["marks"]
    if show_small:
        img[v["small"] & ~mk] = INK["small"]
    lt = light if light is not None else np.zeros_like(mk)
    img[mk & ~lt] = (110, 110, 110) if single else INK["mark"]
    img[edge & ~lt] = (165, 165, 165) if single else INK["edge"]
    img[mk & lt] = INK["light"]
    img[edge & lt] = INK["light_edge"]
    line = contour_ink or ((90, 90, 90) if single else INK["line"])
    for pts in (v["top"], v["trail"]):
        ok = ~np.isnan(pts).any(1)
        for i in range(len(pts) - 1):
            if ok[i] and ok[i + 1]:
                a, b = pts[i], pts[i + 1]
                if abs(a[0] - b[0]) + abs(a[1] - b[1]) < 0.15 * DPX:
                    cv2.line(img, (int(round(a[0] * 4)), int(round(a[1] * 4))), (int(round(b[0] * 4)), int(round(b[1] * 4))), line, 3, cv2.LINE_AA, shift=2)
    s = DPX / 96.0
    for x_, y_, kind in v["nicks"]:
        x_, y_ = int(round(x_)), int(round(y_)); d = int(round(6 * s))
        if kind == 0:
            cv2.line(img, (x_, y_ - d), (x_, y_ + d), INK["nick"], 3)
        else:
            cv2.line(img, (x_ - d, y_), (x_ + d, y_), INK["nick"], 3)
    if side == "R":
        img = np.ascontiguousarray(img[:, ::-1])
    txt = ([label] if label else []) + (["1 frame"] if single else [])
    if txt:
        t = " - ".join(txt); sc = 0.55
        (tw, th), _ = cv2.getTextSize(t, cv2.FONT_HERSHEY_SIMPLEX, sc, 1)
        cv2.putText(img, t, (8, th + 8), cv2.FONT_HERSHEY_SIMPLEX, sc, (110, 110, 110), 1, cv2.LINE_AA)      # top left: paper above the hump (left side) or behind the fin (right side)
    return img


# ---- the animal-level drawing (composed on request by the web app: the animal's stacks come from the sorting) ------
TIP_PX = ((0.0 - DU0) * DPX, (-1.0 - DV0) * DPX)
V_STRETCH = tuple(np.exp(np.linspace(np.log(0.7), np.log(1.0 / 0.7), 9)))      # along the body: another viewing angle
V_SCALE = (0.95, 1.0, 1.05)
V_SHIFT = 24                            # picture px (0.16 fin height)
V_MIN_SCORE = 0.30                      # a stack whose drawing lines up worse than this is not added: below 0.3 the same
                                        # animal's marks agree hardly more than two animals' (mark_size.py by score: lift
                                        # 0.06 under 0.3, 0.12-0.13 at 0.3-0.5, 0.19-0.28 above)
V_TOL = 8                               # picture px: the same mark in two stacks (after lining up)


def load_view(path) -> dict:
    z = np.load(path)
    v = {k: z[k] for k in ("whale", "seen", "marks", "edge", "small", "top", "trail", "nicks")}
    v["n"] = int(z["n"]); v["meta"] = json.loads(str(z["meta"]))
    return v


def outline_raster(v, width=3):
    img = np.zeros((DH, DW), np.uint8)
    for pts in (v["top"], v["trail"]):
        ok = ~np.isnan(pts).any(1)
        for i in range(len(pts) - 1):
            if ok[i] and ok[i + 1] and abs(pts[i][0] - pts[i + 1][0]) + abs(pts[i][1] - pts[i + 1][1]) < 0.15 * DPX:
                cv2.line(img, tuple(int(round(c)) for c in pts[i]), tuple(int(round(c)) for c in pts[i + 1]), 1, width)
    return img


def width50(v) -> float:
    """the fin's width (leading edge to trailing edge) halfway up, in fin heights: the drawing's own cue of how
    side-on it is (a fin turned away is narrower); NaN when either edge is not seen there."""
    y = (-0.5 - DV0) * DPX
    def at(pts):
        ok = ~np.isnan(pts).any(1); p = pts[ok]
        if len(p) < 2:
            return []
        xs = []
        for a, b in zip(p[:-1], p[1:]):
            if (a[1] - y) * (b[1] - y) <= 0 and a[1] != b[1] and abs(a[0] - b[0]) < 0.15 * DPX:
                xs.append(a[0] + (y - a[1]) * (b[0] - a[0]) / (b[1] - a[1]))
        return xs
    lead = [x for x in at(v["top"]) if x < TIP_PX[0] + 0.05 * DPX]
    trail = [x for x in at(v["trail"]) if x > TIP_PX[0] - 0.4 * DPX]
    if not lead or not trail:
        return float("nan")
    return float((max(trail) - max(lead)) / DPX)


def _vfeat(v):
    """the feature lined up (at half the picture resolution): the outline (mostly) and the marks, blurred."""
    if "_feat" not in v:
        o = cv2.GaussianBlur(outline_raster(v).astype(np.float32), (0, 0), 2.5)
        m = cv2.GaussianBlur((v["marks"] & v["seen"]).astype(np.float32), (0, 0), 2.0)
        v["_feat"] = cv2.resize(2.0 * o + m, (DW // 2, DH // 2), interpolation=cv2.INTER_AREA)
    return v["_feat"]


def _about_tip(sx, s, dx, dy, k=1.0):
    tx, ty = TIP_PX[0] / k, TIP_PX[1] / k
    return np.array([[sx * s, 0, tx - sx * s * tx + dx], [0, s, ty - s * ty + dy]], np.float32)


def register_view(ref: dict, v: dict):
    """(2x3 picture px of v -> picture px of ref, score): a stretch along the body (another viewing angle), a small
    scale and a shift, about the tip, on the outline (mostly) and the marks; searched at half resolution."""
    sh = V_SHIFT // 2
    A = cv2.copyMakeBorder(_vfeat(ref), sh, sh, sh, sh, cv2.BORDER_CONSTANT, value=0)
    Bf = _vfeat(v); w, hh = Bf.shape[1], Bf.shape[0]
    best = (-2.0, None)
    for sx in V_STRETCH:
        for s in V_SCALE:
            r = cv2.matchTemplate(A, cv2.warpAffine(Bf, _about_tip(sx, s, 0, 0, 2.0), (w, hh)), cv2.TM_CCORR_NORMED)
            _, val, _, loc = cv2.minMaxLoc(r)
            if val > best[0]:
                best = (float(val), _about_tip(sx, s, 2.0 * (loc[0] - sh), 2.0 * (loc[1] - sh)))
    return best[1], best[0]


def reference_of(views: list[dict]) -> int:
    """the animal's most side-on stack: the widest fin halfway up (a fin turned away from the camera is narrower), among
    the drawings of at least two frames when there are any; ties -> the better-seen one."""
    multi = [i for i, v in enumerate(views) if v["n"] >= 2] or list(range(len(views)))
    def key(i):
        w = width50(views[i]); sc = views[i]["meta"].get("score", 0)
        return (0 if np.isnan(w) else 1, round(w, 1) if not np.isnan(w) else 0, sc)
    return max(multi, key=key)


def compose(views: list[dict], side: str):
    """The animal-level drawing of one side from the drawings of several of its stacks (picture frame): the outline of
    the most side-on stack; every stack's marks lined up onto it; a mark found in two or more stacks (different moments,
    different light) drawn solid, a mark of one stack drawn light. Returns (rgb, info)."""
    r = reference_of(views); ref = views[r]
    lined, info_st = [], []
    for i, v in enumerate(views):
        if i == r:
            lined.append((i, v["marks"] & v["seen"], v["edge"], v["seen"])); info_st.append({"i": i, "ref": True, "score": 1.0, "stretch": 1.0}); continue
        M, sc = register_view(ref, v)
        ok = sc >= V_MIN_SCORE
        info_st.append({"i": i, "ref": False, "score": round(sc, 3), "stretch": round(float(M[0, 0] / M[1, 1]), 3), "used": bool(ok)})
        if not ok:
            continue
        w = lambda a: cv2.warpAffine(a.astype(np.uint8), M, (DW, DH), flags=cv2.INTER_NEAREST) > 0
        lined.append((i, w(v["marks"] & v["seen"]), w(v["edge"]), w(v["seen"])))
    info_st[r]["used"] = True
    wh = ref["whale"]
    union = np.zeros((DH, DW), bool); seen = np.zeros((DH, DW), bool); edge = np.zeros((DH, DW), bool)
    for _, m, e, sn in lined:
        union |= m & wh; seen |= sn & wh; edge |= e & wh
    k = np.ones((2 * V_TOL + 1, 2 * V_TOL + 1), np.uint8)
    near = [cv2.dilate(m.astype(np.uint8), k) > 0 for _, m, _, _ in lined]
    n, lab = cv2.connectedComponents(union.astype(np.uint8), connectivity=8)
    light = np.zeros((DH, DW), bool); solid_n = light_n = 0
    for c in range(1, n):
        comp = lab == c
        cnt = sum(bool((nr & comp).any()) for nr in near)
        if cnt >= 2:
            solid_n += 1
        else:
            light |= comp; light_n += 1
    v = dict(ref, marks=union, edge=edge & union, seen=seen | ref["seen"], small=np.zeros_like(union))
    used = sum(1 for x in info_st if x.get("used"))
    img = render_view(v, side, single=False, label=f"animal - {used} stacks", light=light)
    return img, {"ref": r, "stacks": info_st, "used": used, "marks_solid": solid_n, "marks_light": light_n}


# ---- the pad drawing (2026-11-19): the NOTED shape's standard outline, the marks carried onto it -----------------------
# The researchers draw a whale on their pad as one of a small set of standard dorsal shapes (side profile, hump then fin)
# and add the large marks. The trip log notes each whale's shape as "height form" (e.g. "moderate hook"). A TEMPLATE is
# the standard outline of one noted shape, built from the side-on drawings of the animals noted with it on the 21 train +
# 3 val days (experiments/sketch_template/templates.py; file: PAD_TEMPLATES). A drawing's marks are carried onto a
# template (pad_transfer, kept: "stretch_v"): one stretch along the body about the tip by the square root of the fin's
# width ratio (template / observed, over 0.2-0.7 of the height: only part of a width difference is the view, the rest is
# the animal's own shape), and on the back and hump a vertical shift that puts the observed back on the template's.
# Measured in experiments/sketch_template/transfer.py (21 train + 3 val days): marks of two stacks compared place by place
# agree as well on the template as in the fixed frame (extra hit rate 0.147 vs 0.148 on the pairs that line up; across
# clearly different angles 0.083 vs 0.078), worse than lining the pair up (register_view, 0.163 / 0.110); the
# outline-relative mapping (pad_maps: height kept, place between the leading and the trailing edge kept) measured worse
# (0.121) and is not drawn. The template is the NOTED shape, never the photographed one; it is drawn in its own ink.
PAD_TEMPLATES = Path(os.environ.get("FIN_SKETCH_TEMPLATES", "/home/sat3737/NOAA_whales/moonshot_cache/sketch_template/templates.npz"))
PAD_UG = np.round(np.arange(-3.0, 0.0 + 1e-9, 0.02), 3)          # the top curve v(u): the back and hump ahead of the fin, the leading edge
PAD_VG = np.round(np.arange(-0.95, -0.08 + 1e-9, 0.01), 3)       # the edges u(v): leading and trailing
PAD_BG = np.round(np.arange(0.06, 0.86 + 1e-9, 0.02), 3)          # the back behind the fin v(u)
PAD_VB = -0.12                          # below this height the edges are held at their value here (the fin's foot)
PAD_W_BAND = (-0.7, -0.2)               # heights over which the fin's width ratio (template / observed) is taken
PAD_MIN_EDGE = 0.6                      # an edge seen on less than this share of the fin proper: no transfer
PAD_COVER_BAND = (-0.9, -0.5)           # the fin proper (heights) where the edges must be seen
PAD_LEAD_GAP = 0.5                      # the leading edge is followed forward from the tip across unseen stretches up to this (fin heights)
PAD_MIN_W50 = 0.45                      # fin narrower than this halfway up (fin heights): seen nearly end-on, no transfer
PAD_K_RANGE = (0.5, 2.2)                # width ratio outside this: no transfer (points or outline wrong)
PAD_K_POW = 0.5                         # the kept transfer stretches by ratio ** 0.5: only part of a width difference is the view (the
                                        # rest is the animal's own shape against the template); measured best of the transfers tried
PAD_DV_MAX = 0.25                       # largest vertical correction ahead of / behind the fin (fin heights)


def _h(p):
    return np.asarray(p, float) / DPX + np.array([DU0, DV0])


def pad_sample(poly, along, grid, gap=0.1):
    """poly (N, 2) [u, v] fin heights; along 0 = v(u) on a u grid, 1 = u(v) on a v grid; NaN where not seen."""
    ok = ~np.isnan(poly).any(1)
    p = poly[ok]
    out = np.full(len(grid), np.nan)
    if len(p) < 5:
        return out
    x, y = p[:, along], p[:, 1 - along]
    o = np.argsort(x, kind="stable"); x, y = x[o], y[o]
    v = np.interp(grid, x, y, left=np.nan, right=np.nan)
    j = np.clip(np.searchsorted(x, grid), 1, len(x) - 1)
    v[(x[j] - x[j - 1]) > gap] = np.nan
    return v


def pad_back(whale):
    """the top of the back behind the fin, v(u) on PAD_BG, from a whale raster (first whale row per column)."""
    out = np.full(len(PAD_BG), np.nan); r0 = int(round((-0.7 - DV0) * DPX))
    for i, u in enumerate(PAD_BG):
        c = int(round((u - DU0) * DPX))
        if 0 <= c < DW:
            rows = np.nonzero(whale[r0:, c])[0]
            if len(rows):
                out[i] = (r0 + rows[0]) / DPX + DV0
    return out


def pad_curves(v) -> dict:
    """a drawing's outline as sampled curves (fin heights): top v(u), trail u(v), back v(u)."""
    return dict(top=pad_sample(_h(v["top"]), 0, PAD_UG), trail=pad_sample(_h(v["trail"]), 1, PAD_VG), back=pad_back(v["whale"]))


def pad_lead(top):
    """the leading edge u(v) on PAD_VG from the top curve v(u): from the tip forward, the first u where the outline is at
    or below height v (made monotone going forward from the tip; contiguous from the tip only)."""
    ok = ~np.isnan(top)
    u, t = PAD_UG[ok][::-1], top[ok][::-1]
    out = np.full(len(PAD_VG), np.nan)
    if len(u) < 5 or u[0] < -0.12:
        return out
    near = np.nonzero(u >= -0.5)[0]                # start at the outline's highest point near the tip (the top line can run on
    a = int(near[np.argmin(t[near])]) if len(near) else 0      # past the tip down the trailing side)
    u, t = u[a:], t[a:]
    t = np.maximum.accumulate(t)
    gaps = np.nonzero(np.abs(np.diff(u)) > PAD_LEAD_GAP)[0]       # an unseen stretch of the outline longer than this ends the edge
    if len(gaps):
        u, t = u[:gaps[0] + 1], t[:gaps[0] + 1]
    for i, vv in enumerate(PAD_VG):
        j = np.nonzero(t >= vv)[0]
        if len(j):
            k = j[0]
            out[i] = u[0] if k == 0 else u[k - 1] + (vv - t[k - 1]) * (u[k] - u[k - 1]) / max(t[k] - t[k - 1], 1e-6)
    return out


def _filled(x):
    ok = ~np.isnan(x)
    if ok.sum() < 2:
        return None
    return np.interp(np.arange(len(x)), np.nonzero(ok)[0], x[ok])


def _extended(x, low_slope=True):
    """gaps filled; toward the base (larger v) continued along the line of the last 10 seen points (the leading edge
    often runs out of the picture down the hump), toward the tip held."""
    f = _filled(x)
    if f is None:
        return None
    ok = np.nonzero(~np.isnan(x))[0]; last = ok[-1]
    if low_slope and last < len(x) - 1 and len(ok) >= 10:
        k = ok[-10:]; a, b0 = np.polyfit(PAD_VG[k], x[k], 1)
        f[last + 1:] = np.maximum(a * PAD_VG[last + 1:] + b0, -4.5)
    return f


def pad_edges(cur) -> dict:
    """{lead, trail (u on PAD_VG, gaps filled, ends extended), top, back (NaN kept), cover_lead, cover_trail (share of the
    fin proper, heights PAD_COVER_BAND, where the edge was seen), w50, foot, ok, why}."""
    L, T = pad_lead(cur["top"]), cur["trail"]
    band = (PAD_VG >= PAD_COVER_BAND[0]) & (PAD_VG <= PAD_COVER_BAND[1])
    cl, ct = float(np.mean(~np.isnan(L[band]))), float(np.mean(~np.isnan(T[band])))
    e = dict(top=cur["top"], back=cur["back"], cover_lead=round(cl, 3), cover_trail=round(ct, 3), ok=True, why="", w50=float("nan"), foot=float("nan"))
    Lf, Tf = _extended(L), _extended(T)
    sel = (PAD_UG >= -1.4) & (PAD_UG <= -1.0) & ~np.isnan(cur["top"])
    e["foot"] = float(np.median(cur["top"][sel])) if sel.sum() >= 5 else float("nan")
    if Lf is None or Tf is None or cl < PAD_MIN_EDGE or ct < PAD_MIN_EDGE:
        near = cur["top"][PAD_UG >= -1.0]
        above = bool(np.isfinite(near).any() and np.nanmin(near) < -1.15)      # the outline runs well above the "tip": points wrong
        e.update(ok=False, why="outline above the tip" if above else "outline not seen", lead=Lf, trail=Tf)
        return e
    Tf = np.maximum(Tf, Lf + 0.01)                 # just under the tip the two edges meet
    e.update(lead=Lf, trail=Tf)
    i50 = int(np.argmin(np.abs(PAD_VG + 0.5)))
    e["w50"] = float(Tf[i50] - Lf[i50])
    low = PAD_VG >= -0.8
    if (T[low] - L[low] <= 0.05).any():           # seen edges cross below the tip
        e.update(ok=False, why="edges cross")
    elif e["w50"] < PAD_MIN_W50:
        e.update(ok=False, why="seen nearly end-on")
    return e


def _wband(e):
    s = (PAD_VG >= PAD_W_BAND[0]) & (PAD_VG <= PAD_W_BAND[1])
    return float(np.median((e["trail"] - e["lead"])[s]))


def pad_ratio(eo, et) -> float:
    """the fin's width ratio template / observed over the middle heights (> 1: the observed fin looks narrower)."""
    return _wband(et) / max(_wband(eo), 1e-3)


def pad_maps(eo, et, vcorr: bool = True):
    """(map_x, map_y) for cv2.remap: for every pixel of the TEMPLATE picture, where it lies in the OBSERVED picture
    (eo, et = pad_edges of the observed drawing and of the template). The outline-relative transfer, built in the
    template-to-observed direction (the same construction with the roles swapped, so a raster is pulled, not pushed)."""
    uu, vv = np.meshgrid(np.arange(DW, dtype=np.float64) / DPX + DU0, np.arange(DH, dtype=np.float64) / DPX + DV0)
    vc = np.clip(vv, PAD_VG[0], PAD_VB)
    LT, RT = np.interp(vc, PAD_VG, et["lead"]), np.interp(vc, PAD_VG, et["trail"])
    LO, RO = np.interp(vc, PAD_VG, eo["lead"]), np.interp(vc, PAD_VG, eo["trail"])
    k = 1.0 / pad_ratio(eo, et)                               # observed / template, along the body
    u = np.where(uu < LT, LO + k * (uu - LT), np.where(uu > RT, RO + k * (uu - RT), LO + (uu - LT) / np.maximum(RT - LT, 1e-3) * (RO - LO)))
    dv = np.zeros_like(vv)

    def corr(cur_o, cur_t, grid, sel):
        if not sel.any():
            return
        ct, co = _filled(cur_t), _filled(cur_o)
        if ct is None or co is None:
            return
        a = np.interp(u[sel], grid, co, left=np.nan, right=np.nan) - np.interp(uu[sel], grid, ct, left=np.nan, right=np.nan)
        dv[sel] = np.clip(np.nan_to_num(a), -PAD_DV_MAX, PAD_DV_MAX)
    if vcorr:
        corr(eo["top"], et["top"], PAD_UG, uu < LT)
        corr(eo["back"], et["back"], PAD_BG, uu > RT)
    v = vv + dv
    return ((u - DU0) * DPX).astype(np.float32), ((v - DV0) * DPX).astype(np.float32)


def pad_pull(a, maps, interp=cv2.INTER_NEAREST):
    """a raster of the observed drawing, in the template's picture."""
    return cv2.remap(a.astype(np.uint8), maps[0], maps[1], interp, borderMode=cv2.BORDER_CONSTANT, borderValue=0) > 0


@__import__("functools").lru_cache(maxsize=4)
def _templates_file(path: str, mtime: float) -> dict:
    z = np.load(path)
    shapes = [str(s) for s in z["shapes"]]
    out = {}
    for s in shapes:
        cur = {p: z[f"{s}|{p}"] for p in ("top", "trail", "back")}
        out[s] = dict(cur=cur, how=None, n=None)
    meta = Path(path).with_suffix(".json")
    if meta.exists():
        for s, m in json.loads(meta.read_text()).items():
            if s in out:
                out[s].update(how=m.get("how"), n=m.get("animals"))
    return out


def pad_templates(path: Path | None = None) -> dict:
    """{shape: {cur, how ('direct' | 'parts' | 'all'), n}} (an empty dict when the file is missing)."""
    p = Path(path or PAD_TEMPLATES)
    if not p.exists():
        return {}
    return _templates_file(str(p), p.stat().st_mtime)


def pad_template_view(cur) -> dict:
    """the template as a picture-frame drawing (whale raster, top / trail polylines in picture px): the hump and the
    fin's leading edge (top curve), the trailing edge, the back behind the fin; below them the body."""
    top_ok = ~np.isnan(cur["top"]); tr_ok = ~np.isnan(cur["trail"]); bk_ok = ~np.isnan(cur["back"])
    P = lambda u, v: np.stack([(np.asarray(u) - DU0) * DPX, (np.asarray(v) - DV0) * DPX], 1)
    top = P(PAD_UG[top_ok], cur["top"][top_ok]); trail = P(cur["trail"][tr_ok], PAD_VG[tr_ok]); back = P(PAD_BG[bk_ok], cur["back"][bk_ok])
    poly = [top]
    if len(trail):
        poly.append(trail)
        bt = trail[-1]
    else:
        bt = top[-1]
    b = back[(back[:, 0] > bt[0] + 0.08 * DPX) & (back[:, 1] >= bt[1] - 0.02 * DPX)] if len(back) else back      # the back behind the fin, from a little behind its foot
    if len(b):
        poly.append(b)
        last = b[-1]
    else:
        last = bt
    poly.append(np.array([[DW + 5, last[1]], [DW + 5, DH + 5], [top[0][0], DH + 5]]))
    pts = np.concatenate(poly)
    whale = np.zeros((DH, DW), np.uint8)
    cv2.fillPoly(whale, [np.round(pts * 4).astype(np.int32)], 1, cv2.LINE_8, shift=2)
    # the back behind the fin joins the trailing edge's foot: one polyline (trail, then back)
    tr_line = np.concatenate([trail, b]) if len(b) and len(trail) else trail
    return dict(whale=whale > 0, seen=whale > 0, marks=np.zeros((DH, DW), bool), edge=np.zeros((DH, DW), bool), small=np.zeros((DH, DW), bool),
                top=top.astype(np.float32), trail=tr_line.astype(np.float32), nicks=np.zeros((0, 3), np.float32), n=0)


def pad_stretch_m(k: float):
    """2x3 picture px -> picture px: the stretch along the body about the tip by k ** PAD_K_POW (the kept transfer)."""
    ke = float(k) ** PAD_K_POW; tx = TIP_PX[0]
    return np.array([[ke, 0.0, tx - ke * tx], [0.0, 1.0, 0.0]], np.float32)


PAD_HUMP = (-1.2, -0.6)                 # the back and hump: full vertical correction ahead of u = -1.2, none behind -0.6 (the fin)


def pad_stretch_maps(k: float, eo: dict, et: dict):
    """remap maps (template picture -> observed picture): the kept stretch along the body, and on the back and hump ahead
    of the fin (PAD_HUMP) a vertical shift that puts the observed back's outline on the template's (at most PAD_DV_MAX)."""
    ke = float(k) ** PAD_K_POW
    uu, vv = np.meshgrid(np.arange(DW, dtype=np.float64) / DPX + DU0, np.arange(DH, dtype=np.float64) / DPX + DV0)
    us = uu / ke                                                    # the tip is at u = 0
    u1 = np.arange(DW) / DPX + DU0
    to, tt = _filled(eo["top"]), _filled(et["top"])
    dv = np.zeros(DW)
    if to is not None and tt is not None:
        a = np.interp(u1, PAD_UG, tt, left=np.nan, right=np.nan) - np.interp(u1 / ke, PAD_UG, to, left=np.nan, right=np.nan)
        w = np.clip((PAD_HUMP[1] - u1) / (PAD_HUMP[1] - PAD_HUMP[0]), 0, 1)
        dv = np.clip(np.nan_to_num(a), -PAD_DV_MAX, PAD_DV_MAX) * w
    vs = vv - dv[None, :]
    return ((us - DU0) * DPX).astype(np.float32), ((vs - DV0) * DPX).astype(np.float32)


def pad_transfer(v: dict, tcur: dict, edges_t=None, method: str = "stretch_v"):
    """one drawing's marks on a template: (dict(marks, edge, seen, small) rasters in the template picture, info) or
    (None, info) when the transfer is refused (info['why']: outline not seen / outline above the tip / edges cross / seen
    nearly end-on / width ratio out of range). method "stretch_v" (kept): one stretch along the body about the tip by
    (template width / observed width) ** PAD_K_POW, and on the back and hump a vertical shift onto the template's back
    (pad_stretch_maps); "stretch": the stretch alone; "outline": the outline-relative maps (pad_maps). All three measured
    in experiments/sketch_template/transfer.py; only stretch_v is drawn."""
    eo = pad_edges(pad_curves(v)); et = edges_t or pad_edges(tcur)
    info = dict(cover_lead=eo["cover_lead"], cover_trail=eo["cover_trail"], w50=None if not np.isfinite(eo.get("w50", np.nan)) else round(eo["w50"], 3))
    if not eo["ok"]:
        return None, dict(info, why=eo["why"])
    k = pad_ratio(eo, et); info["ratio"] = round(k, 3)
    if not PAD_K_RANGE[0] <= k <= PAD_K_RANGE[1]:
        return None, dict(info, why="width ratio out of range")
    if method == "outline":
        maps = pad_maps(eo, et)
        out = {n: pad_pull(v[n], maps) for n in ("marks", "edge", "seen", "small")}
    elif method == "stretch_v":
        maps = pad_stretch_maps(k, eo, et); info["stretch"] = round(float(k) ** PAD_K_POW, 3)
        out = {n: pad_pull(v[n], maps) for n in ("marks", "edge", "seen", "small")}
    else:
        M = pad_stretch_m(k); info["stretch"] = round(float(M[0, 0]), 3)
        out = {n: cv2.warpAffine(v[n].astype(np.uint8), M, (DW, DH), flags=cv2.INTER_NEAREST) > 0 for n in ("marks", "edge", "seen", "small")}
    return out, dict(info, why="")


PAD_INK = (40, 75, 150)                 # the template outline: the NOTED shape, drawn in its own ink (never the photographed outline's black)


def _lined_up(views: list, r: int):
    """compose's evidence without its picture: every stack's marks lined up onto views[r] (register_view, score >=
    V_MIN_SCORE), in views[r]'s picture frame. Returns (union, light, edge, seen, info per stack, solid n, light n)."""
    ref = views[r]; lined, info_st = [], []
    for i, v in enumerate(views):
        if i == r:
            lined.append((v["marks"] & v["seen"], v["edge"], v["seen"])); info_st.append({"i": i, "ref": True, "score": 1.0, "used": True}); continue
        M, sc = register_view(ref, v)
        ok = sc >= V_MIN_SCORE
        info_st.append({"i": i, "ref": False, "score": round(sc, 3), "used": bool(ok)})
        if ok:
            w = lambda a: cv2.warpAffine(a.astype(np.uint8), M, (DW, DH), flags=cv2.INTER_NEAREST) > 0
            lined.append((w(v["marks"] & v["seen"]), w(v["edge"]), w(v["seen"])))
    wh = ref["whale"]
    union = np.zeros((DH, DW), bool); seen = np.zeros((DH, DW), bool); edge = np.zeros((DH, DW), bool)
    for m, e, sn in lined:
        union |= m & wh; seen |= sn & wh; edge |= e & wh
    k = np.ones((2 * V_TOL + 1, 2 * V_TOL + 1), np.uint8)
    near = [cv2.dilate(m.astype(np.uint8), k) > 0 for m, _, _ in lined]
    n, lab = cv2.connectedComponents(union.astype(np.uint8), connectivity=8)
    light = np.zeros((DH, DW), bool); solid_n = light_n = 0
    for c in range(1, n):
        comp = lab == c
        if sum(bool((nr & comp).any()) for nr in near) >= 2:
            solid_n += 1
        else:
            light |= comp; light_n += 1
    return union, light, edge & union, seen | ref["seen"], info_st, solid_n, light_n


def pad_compose(views: list, side: str, shape: str, path: Path | None = None, label: str | None = None):
    """The pad drawing of one side: the template of the noted `shape` (head left on a left side, right on a right side)
    with the marks of the drawings `views` (one per stack) carried onto it. One stack: its marks, carried over by its own
    transfer (pad_transfer). Several stacks: the marks are first put together as the animal-level drawing does (compose:
    lined up onto one stack, found in two or more stacks solid, in one light; the reference = the most side-on stack
    whose transfer is not refused), then carried onto the template by that stack's transfer, so the pad drawing holds the
    same evidence as the animal-level drawing, only on the noted outline. No stack: the empty template (the outline the
    researcher would start from). Skin of the template no stack saw is hatched (not without a stack). Marks are kept
    inside the template's skin. Returns (rgb, info) or (None, info) when there is no template for the shape."""
    T = pad_templates(path)
    if shape not in T:
        return None, {"why": "no template for this shape"}
    tcur = T[shape]["cur"]; tv = pad_template_view(tcur); et = pad_edges(tcur); wh = tv["whale"]
    tr = [pad_transfer(v, tcur, et) for v in views]
    info_st = [dict(inf, i=i, used=False) for i, (_, inf) in enumerate(tr)]
    ok = [i for i, (x, _) in enumerate(tr) if x is not None]
    union = light = edge = np.zeros((DH, DW), bool); seen = wh if not views else np.zeros((DH, DW), bool); solid_n = light_n = 0
    if len(views) == 1 and ok:
        x = tr[ok[0]][0]; info_st[ok[0]]["used"] = True
        union = x["marks"] & x["seen"] & wh; edge = x["edge"] & union; seen = x["seen"] & wh
        solid_n = int(cv2.connectedComponents(union.astype(np.uint8), connectivity=8)[0] - 1)
    elif len(views) >= 2 and ok:
        sub = [views[i] for i in ok]
        r_sub = reference_of(sub); r = ok[r_sub]
        others = [i for i in range(len(views)) if i != r]               # every stack is lined up onto the reference (a refused one too:
        order = [r] + others                                            # only its outline cannot be carried, its marks can, once lined up)
        u, lt, e, sn, inf, solid_n, light_n = _lined_up([views[i] for i in order], 0)
        for j, x in enumerate(inf):
            info_st[order[j]].update(used=x["used"], score=x["score"], ref=x["ref"])
        maps = pad_stretch_maps(tr[r][1]["ratio"], pad_edges(pad_curves(views[r])), et)      # the reference's own transfer
        w = lambda a: pad_pull(a, maps)
        union, light, edge, seen = w(u) & wh, w(lt) & wh, w(e) & wh, w(sn) & wh
    for x in info_st:
        if x["used"]:
            x["why"] = ""                                               # carried over (by its own transfer or lined up onto the reference)
        elif not x.get("why"):
            x["why"] = "did not line up"
    used = sum(1 for x in info_st if x["used"])
    v = dict(tv, marks=union, edge=edge & union, seen=seen, small=np.zeros_like(union))
    text = label if label is not None else f"pad shape: {shape} (trip log)"
    img = render_view(v, side, single=False, label=text, light=light if len(views) >= 2 else None, contour_ink=PAD_INK)
    return img, {"shape": shape, "how": T[shape].get("how"), "template_animals": T[shape].get("n"), "stacks": info_st, "used": used,
                 "refused": len(views) - used, "marks_solid": solid_n, "marks_light": light_n,
                 "carried_by": None if not ok else (r if len(views) >= 2 else ok[0])}


# ---- the view: how far from side-on ------------------------------------------------------------------------------
# Measured 2026-11-19 on the 21 train + 3 val days (experiments/sketch_v2/view_label.py): no angle is known, so the check
# is the animal itself: of two stacks of the same animal and side, the one seen more side-on shows the wider fin halfway
# up. With this rule, of the same-animal pairs where one stack is labelled side-on and the other angled, the side-on one
# has the wider fin in 0.90 of 324 pairs on the train days (where the thresholds were picked) and 0.69 of 13 on the val
# days. An estimate, said as one. Straightening (un-foreshortening) is NOT drawn: straighten.py found it brings the
# same animal's outlines closer but different animals' as much (AUC 0.584 -> 0.592 all, val days 0.577 -> 0.573).
VIEW_SIDE_PROFILE, VIEW_SIDE_BASE = 0.8, 2.0
VIEW_ANGLED_PROFILE, VIEW_ANGLED_BASE = 0.5, 1.6


def view_label(profile: float, bperp: float) -> str:
    """'side' (estimated near the direct side profile) | 'angled' (the fin turned toward or away from the camera, drawn
    as seen) | '' (cannot tell). profile = the photo pose model's |nose right| (parts_up npz, median over the frames),
    bperp = the keypoint base ratio square to the base line."""
    p = profile if profile is not None and np.isfinite(profile) else None
    if p is not None and p >= VIEW_SIDE_PROFILE and bperp >= VIEW_SIDE_BASE:
        return "side"
    if (p is not None and p < VIEW_ANGLED_PROFILE) or bperp < VIEW_ANGLED_BASE:
        return "angled"
    return ""


VIEW_TEXT = {"side": "side-on (est.)", "angled": "angled view", "": ""}


# ---- a day ---------------------------------------------------------------------------------------------------------
def grade(n, H, base, cover):
    detail = min(H / 90.0, 1.0); side_on = float(np.clip((base - 0.8) / 0.6, 0, 1))
    score = detail * (0.5 + 0.5 * side_on) * (1 - 0.5 ** n) * (0.4 + 0.6 * cover)
    if n >= 3 and H >= 60 and base >= 1.1 and cover >= 0.5:
        g = "good"
    elif (n >= 2 and H >= 40 and cover >= 0.3) or (n == 1 and H >= 80 and base >= 1.1 and cover >= 0.5):
        g = "fair"
    else:
        g = "poor"
    return round(float(score), 4), g


def fin_cover(dr):
    """share of the fin (tip to base, the part above y = -0.1 H right of x = -1.2 H) seen in enough frames."""
    c0, c1 = int(round((-1.2 - X0) * PXH)), int(round((0.4 - X0) * PXH)); r0, r1 = int(round((-1.0 - Y0) * PXH)), int(round((-0.1 - Y0) * PXH))
    w = dr["whale"][r0:r1, c0:c1]; s = dr["seen_n"][r0:r1, c0:c1] >= (2 if dr["n"] >= 2 else 1)
    return float((w & s).sum() / max(w.sum(), 1))


def profile_of(d: Path, key: str) -> float:
    zp = d / "parts_up" / f"{key}__fin.npz"
    try:
        return float(np.load(zp, allow_pickle=True)["profile"])
    except Exception:
        return float("nan")


def draw_day(day: str, root: Path = DAYS, out_name: str = "sketch") -> dict:
    d = day_dir(day, root)
    pts_f = d / "sketch" / "points.json"
    if not pts_f.exists():
        raise SystemExit(f"{day}: no {pts_f}; run `fin_sketch points --days {day}` first")
    P = json.loads(pts_f.read_text())["points"]
    boxes = dorsal_keys(d)
    cl_bytes = (d / "safe" / "clusters.json").read_bytes()
    out_dir = d / out_name; out_dir.mkdir(parents=True, exist_ok=True); t0 = time.time()
    index, why = {}, collections.Counter()
    n_fin = 0
    for c in clusters_of(d):
        ks = [k for k in c["keys"] if k in boxes]
        if not ks:
            continue
        n_fin += 1
        by = collections.defaultdict(list)
        for k in ks:
            if k not in P:
                why["no points"] += 1; continue
            r, bad = one(d, k, P[k])
            if r is None:
                why[bad] += 1; by[side_of(P[k])].append((k, None)); continue
            by[r[3]["side"]].append((k, r))
        ent = {}
        for side, rows in by.items():
            good = [(k, r) for k, r in rows if r is not None]
            if not good:
                continue
            tiles = [r[:3] for _, r in good]; infos = [r[3] for _, r in good]
            dr, used, ncc, Fr = side_drawing(tiles, infos)
            why["not registered"] += len(good) - len(used)
            if not used:
                continue
            keys = [good[j][0] for j in used]
            Hm = float(np.median([infos[j]["H"] for j in used])); bm = float(np.median([infos[j]["base"] for j in used]))
            prof = float(np.nanmedian([profile_of(d, k) for k in keys])) if any(np.isfinite(profile_of(d, k)) for k in keys) else float("nan")
            Fm = np.median(np.stack(Fr), 0)
            M, phi, bperp = view_frame(Fm)
            label = view_label(prof, bperp)
            v = to_view(dr, M)
            cov = fin_cover(dr); score, g = grade(len(used), Hm, bm, cov)
            name = f"{c['id']}__{side}"
            meta = {"side": side, "n": len(used), "label": label, "profile": None if not np.isfinite(prof) else round(prof, 3), "bperp": round(bperp, 2),
                    "level_deg": round(phi, 1), "H": round(Hm, 1), "grade": g, "score": score}
            save_view(out_dir / f"{name}.npz", v, meta)
            img = render_view(v, side, single=len(used) < 2, label=VIEW_TEXT[label])
            tmp = out_dir / (name + ".tmp.png")
            cv2.imwrite(str(tmp), cv2.cvtColor(img, cv2.COLOR_RGB2BGR), [cv2.IMWRITE_PNG_COMPRESSION, 9]); tmp.replace(out_dir / f"{name}.png")
            ent[side] = {"png": f"{name}.png", "npz": f"{name}.npz", "n": len(used), "n_side": len(rows), "n_outline": dr["n_outline"], "keys": keys,
                         "H": round(Hm, 1), "base": round(bm, 2), "bperp": round(bperp, 2), "profile": meta["profile"], "level_deg": meta["level_deg"],
                         "view": label, "ncc": round(float(np.median([ncc[j] for j in used])), 3), "cover": round(cov, 3), "score": score, "grade": g,
                         "single": len(used) < 2, "marks": int(cv2.connectedComponents(dr["marks"].astype(np.uint8))[0] - 1),
                         "small_left_out": int(cv2.connectedComponents(dr["small"].astype(np.uint8))[0] - 1),
                         "nicks": len(dr["nick_trail"]) + len(dr["nick_top"])}
        if ent:
            index[str(c["id"])] = ent
    keep = {e[f] for v in index.values() for e in v.values() for f in ("png", "npz")}
    for f in list(out_dir.glob("*__[LR].png")) + list(out_dir.glob("*__[LR].npz")):       # drawings of an earlier clustering
        if f.name not in keep:
            f.unlink()
    sides = [e for v in index.values() for e in v.values()]
    summary = {"clusters": len(clusters_of(d)), "fin_clusters": n_fin, "drawn_clusters": len(index), "drawings": len(sides),
               "single": sum(e["single"] for e in sides), "grades": dict(collections.Counter(e["grade"] for e in sides)),
               "views": dict(collections.Counter(e["view"] or "unknown" for e in sides)),
               "left_out": dict(why), "seconds": round(time.time() - t0, 1)}
    doc = {"day": day, "made": _now(), "version": VERSION, "points": "pose_side B_both", "clusters_sha1": hashlib.sha1(cl_bytes).hexdigest(),
           "picture": {"PHI0": PHI0, "DU0": DU0, "DU1": DU1, "DV0": DV0, "DV1": DV1, "DPX": DPX, "W": DW, "H": DH},
           "clusters": index, "summary": summary}
    tmp = out_dir / "index.tmp.json"; tmp.write_text(json.dumps(doc)); tmp.replace(out_dir / "index.json")
    return summary


def _now():
    return datetime.datetime.now().isoformat(timespec="seconds")


def _draw_job(a):
    cv2.setNumThreads(1)
    day, root, out_name = a
    try:
        return day, draw_day(day, root, out_name)
    except SystemExit as e:
        return day, {"error": str(e)}


def cmd_draw(days, root, workers, out_name="sketch"):
    with Pool(max(1, min(workers, len(days)))) as pool:
        for day, s in pool.imap_unordered(_draw_job, [(d, root, out_name) for d in days]):
            print(day, json.dumps(s), flush=True)


# ---- sheet (to look at; images go outside the repo) ----------------------------------------------------------------
def cutout_view(d: Path, key: str, p: dict):
    """the fin cut-out of one frame in the picture frame, head left; None if not drawable."""
    r, _ = one(d, key, p)
    if r is None:
        return None
    t, mt, ins, info = r
    M = view_frame()[0]
    return cv2.warpAffine(t, M, (DW, DH), flags=cv2.INTER_CUBIC, borderValue=(0, 0, 0)), info


def cmd_sheet(day, root, out, n=12, seed=0, ids=None, sub="sketch"):
    """rows of: the best fin cut-out of the cluster side (in the picture frame) | the drawing. Fin cut-outs only."""
    d = day_dir(day, root); idx = json.loads((d / sub / "index.json").read_text())
    P = json.loads((d / "sketch" / "points.json").read_text())["points"]
    items = [(cid, s, e) for cid, v in idx["clusters"].items() for s, e in v.items()]
    if ids:
        items = [x for x in items if x[0] in ids]
    else:
        rng = np.random.default_rng(seed); items = [items[i] for i in rng.permutation(len(items))[:n]]
    rows = []
    for cid, s, e in items:
        cands = [(cutout_view(d, k, P[k]), k) for k in e["keys"]]
        cands = [(c, k) for c, k in cands if c is not None]
        if not cands:
            continue
        t = max(cands, key=lambda x: x[0][1]["H"])[0][0]
        if s == "R":
            t = t[:, ::-1]
        dr = cv2.cvtColor(cv2.imread(str(d / sub / e["png"])), cv2.COLOR_BGR2RGB)
        row = np.hstack([t, np.full((t.shape[0], 6, 3), 255, np.uint8), dr])
        lab = f"{day} cluster {cid} {'left' if s == 'L' else 'right'} side - {e['n']} of {e['n_side']} frames - {e['grade']} - H {e['H']:.0f}px - view {e.get('view') or '?'}"
        head = np.full((22, row.shape[1], 3), 255, np.uint8)
        cv2.putText(head, lab, (5, 16), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)
        rows += [head, np.ascontiguousarray(row), np.full((6, row.shape[1], 3), 255, np.uint8)]
    img = np.vstack(rows)
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out), cv2.cvtColor(img, cv2.COLOR_RGB2BGR), [cv2.IMWRITE_JPEG_QUALITY, 88])
    print(out)


def all_days(root):
    return sorted(p.name for p in Path(root).iterdir() if (p / "safe" / "clusters.json").exists() and (p / "safe" / "view_class.json").exists())


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("cmd", choices=["points", "draw", "sheet"])
    ap.add_argument("--days", default=None); ap.add_argument("--day", default=None); ap.add_argument("--root", default=str(DAYS))
    ap.add_argument("--workers", type=int, default=6); ap.add_argument("--predict-all", action="store_true", help="predict every instance (ignore the pose-side caches)")
    ap.add_argument("--head", default=None); ap.add_argument("--out", default=None); ap.add_argument("--n", type=int, default=12); ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--ids", default=None); ap.add_argument("--out-name", default="sketch", help="folder of DAY for the drawings (default sketch)")
    a = ap.parse_args(argv)
    root = Path(a.root)
    days = all_days(root) if a.days == "all" else (a.days or a.day or "").split(",")
    days = [x for x in days if x]
    if a.cmd == "points":
        cmd_points(days, root, a.predict_all, a.head)
    elif a.cmd == "draw":
        cmd_draw(days, root, a.workers, a.out_name)
    else:
        cmd_sheet(days[0], root, a.out, a.n, a.seed, a.ids.split(",") if a.ids else None, a.out_name)


if __name__ == "__main__":
    sys.exit(main())
