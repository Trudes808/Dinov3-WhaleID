"""Hand-made image statistics, one value per sample and statistic, the same code for both worlds.

Views: `crop` = the whale-centred square (512 px), `frame` = the full frame (long side 1280).
Regions: whale and band (water within one whale-height) are measured on the crop, far water / sky
on the frame (see common.py for the region ids). Units: L*, a*, b*, C* are CIELAB (L* 0..100);
"share" = fraction 0..1; pixel lengths are in the named view's pixels.
"""

from __future__ import annotations

import math

import cv2
import numpy as np

MIN_PX = 200
BANDS = {"fine": (0.8, 1.6), "medium": (2.0, 4.0), "coarse": (5.0, 10.0)}

# name -> (what it measures, unit)
INFO: dict[str, tuple[str, str]] = {}


def _info(name, text, unit):
    INFO[name] = (text, unit)


for reg, where in (("whale", "whale (crop)"), ("band", "water band around the whale (crop)"), ("far", "far water (frame)"),
                   ("above", "sky / land above the horizon (frame)")):
    _info(f"{reg}.L_mean", f"brightness, {where}", "L*")
    _info(f"{reg}.L_std", f"contrast (sd of L*), {where}", "L*")
    _info(f"{reg}.a_mean", f"green(-)/red(+) cast, {where}", "a*")
    _info(f"{reg}.b_mean", f"blue(-)/yellow(+) cast, {where}", "b*")
    _info(f"{reg}.C_mean", f"colourfulness (chroma), {where}", "C*")
    _info(f"{reg}.S_hsv", f"HSV saturation, {where}", "0..1")
    _info(f"{reg}.hue_deg", f"hue angle of the mean colour, {where}", "deg (Lab)")
    _info(f"{reg}.clip_hi", f"clipped highlights (any channel >= 250), {where}", "share")
    _info(f"{reg}.crush_lo", f"crushed shadows (all channels <= 8), {where}", "share")
    for b in BANDS:
        _info(f"{reg}.tex_{b}", f"texture energy, {b} scale (RMS band-pass L*), {where}", "L*")
    _info(f"{reg}.psd_slope", f"power-spectrum slope of L* (64 px patches; steeper = smoother), {where}", "log-log")
    _info(f"{reg}.vgrad_share", f"anisotropy: vertical-gradient energy share (0.5 isotropic, 1 = horizontal stripes), {where}", "share")
    _info(f"{reg}.coherence", f"anisotropy: structure-tensor coherence (0 isotropic, 1 = one direction), {where}", "0..1")
for reg, where in (("band", "band (crop)"), ("far", "far water (frame)")):
    _info(f"{reg}.foam", f"whitecap / foam: bright (L*>=max(70, median+20)) low-chroma (C*<=12) water, {where}", "share")
_info("whale.dL_vs_band", "whale darker than the water around it: median L* whale - median L* band (silhouette)", "L*")
_info("whale.spec_share", "specular highlights on the whale: L* >= whale median + 25", "share")
_info("whale.hi_spread", "highlight spread on the whale: L* p95 - p50", "L*")
_info("whale.edge_width", "outline softness: L* step across the outline / peak gradient at the outline", "crop px")
_info("whale.lapvar", "sharpness on the whale: variance of the Laplacian of L* (crop)", "L*^2")
_info("crop.lapvar", "sharpness of the whole crop: variance of the Laplacian of L*", "L*^2")
_info("crop.native_per_px", "native pixels per crop pixel (<1 = the crop was upscaled)", "px/px")
_info("frame.lapvar", "sharpness of the frame: variance of the Laplacian of L* (frame at 1280)", "L*^2")
_info("frame.noise", "noise level: Immerkaer sigma on flat areas (frame at 1280, grey 0-255)", "grey levels")
_info("frame.L_mean", "frame brightness", "L*")
_info("frame.L_std", "frame contrast (sd of L*)", "L*")
_info("frame.a_mean", "frame green/red cast", "a*")
_info("frame.b_mean", "frame blue/yellow cast", "b*")
_info("frame.clip_hi", "frame clipped highlights", "share")
_info("frame.crush_lo", "frame crushed shadows", "share")
_info("frame.L_p99", "frame 99th percentile L* (how bright the brightest parts are)", "L*")
_info("frame.L_p01", "frame 1st percentile L* (how dark the darkest parts are)", "L*")
_info("frame.has_horizon", "horizon in frame (renders: camera geometry; real: line finder)", "share")
_info("frame.above_share", "share of the frame above the horizon", "share")
_info("frame.aspect", "frame aspect ratio W/H", "W/H")
_info("frame.width_px", "native frame width", "px")
_info("whale.len_share", "whale length: mask box width / frame width", "share")
_info("whale.height_share", "whale mask box height / frame height", "share")
_info("whale.area_share", "whale visible area / frame area", "share")
_info("whale.cx", "whale centre x / frame width", "0..1")
_info("whale.cy", "whale centre y / frame height", "0..1")
_info("whale.native_width_px", "whale box width in native pixels (detail available)", "px")


def to_lab(rgb: np.ndarray):
    f = rgb.astype(np.float32) / 255.0
    lab = cv2.cvtColor(f, cv2.COLOR_RGB2LAB)
    hsv = cv2.cvtColor(f, cv2.COLOR_RGB2HSV)
    return lab, hsv[..., 1]


def _nblur(x, m, s):
    """Normalised (masked) Gaussian blur and the blurred mask."""
    mb = cv2.GaussianBlur(m, (0, 0), s)
    xb = cv2.GaussianBlur(x * m, (0, 0), s)
    return xb / np.maximum(mb, 1e-4), mb


def basic(out, reg, rgb, lab, S, m):
    n = int(m.sum())
    if n < MIN_PX:
        return
    L, a, b = lab[..., 0][m], lab[..., 1][m], lab[..., 2][m]
    out[f"{reg}.L_mean"] = float(L.mean()); out[f"{reg}.L_std"] = float(L.std())
    out[f"{reg}.a_mean"] = float(a.mean()); out[f"{reg}.b_mean"] = float(b.mean())
    out[f"{reg}.C_mean"] = float(np.hypot(a, b).mean()); out[f"{reg}.S_hsv"] = float(S[m].mean())
    out[f"{reg}.hue_deg"] = float(math.degrees(math.atan2(b.mean(), a.mean())) % 360.0)
    px = rgb[m]
    out[f"{reg}.clip_hi"] = float((px.max(1) >= 250).mean()); out[f"{reg}.crush_lo"] = float((px.max(1) <= 8).mean())


def texture(out, reg, L, m):
    mf = m.astype(np.float32)
    if m.sum() < MIN_PX:
        return
    for name, (s1, s2) in BANDS.items():
        b1, w1 = _nblur(L, mf, s1)
        b2, w2 = _nblur(L, mf, s2)
        keep = m & (w2 > 0.7)
        if keep.sum() >= MIN_PX:
            out[f"{reg}.tex_{name}"] = float(np.sqrt(np.mean((b1 - b2)[keep] ** 2)))
    # anisotropy at the medium scale
    Lb, _ = _nblur(L, mf, 1.5)
    gx = cv2.Sobel(Lb, cv2.CV_32F, 1, 0, ksize=3) / 8.0
    gy = cv2.Sobel(Lb, cv2.CV_32F, 0, 1, ksize=3) / 8.0
    inner = cv2.erode(m.astype(np.uint8), np.ones((7, 7), np.uint8)) > 0
    if inner.sum() >= MIN_PX:
        Ex, Ey, Exy = float(np.mean(gx[inner] ** 2)), float(np.mean(gy[inner] ** 2)), float(np.mean((gx * gy)[inner]))
        out[f"{reg}.vgrad_share"] = Ey / max(Ex + Ey, 1e-9)
        tr = Ex + Ey; det = Ex * Ey - Exy * Exy
        disc = math.sqrt(max(tr * tr / 4 - det, 0.0))
        l1, l2 = tr / 2 + disc, tr / 2 - disc
        out[f"{reg}.coherence"] = (l1 - l2) / max(l1 + l2, 1e-9)
    # power-spectrum slope on up to 6 patches of 64 px fully inside the region
    P = 64
    H, W = L.shape
    ok = cv2.erode(m.astype(np.uint8), np.ones((3, 3), np.uint8))
    ii = cv2.integral(ok)
    cands = []
    for y in range(0, H - P + 1, P // 2):
        for x in range(0, W - P + 1, P // 2):
            if ii[y + P, x + P] - ii[y, x + P] - ii[y + P, x] + ii[y, x] >= P * P - 1:
                cands.append((y, x))
    if cands:
        rng = np.random.default_rng(len(cands))
        pick = [cands[i] for i in rng.choice(len(cands), size=min(6, len(cands)), replace=False)]
        win = np.outer(np.hanning(P), np.hanning(P)).astype(np.float32)
        fy = np.fft.fftfreq(P)[:, None]; fx = np.fft.fftfreq(P)[None, :]
        fr = np.sqrt(fx ** 2 + fy ** 2) * P
        sl = []
        for (y, x) in pick:
            p = L[y:y + P, x:x + P]
            p = (p - p.mean()) * win
            ps = np.abs(np.fft.fft2(p)) ** 2
            ks = np.arange(3, 21)
            rad = np.array([ps[(fr >= k - 0.5) & (fr < k + 0.5)].mean() for k in ks])
            if (rad > 0).all():
                sl.append(np.polyfit(np.log(ks), np.log(rad), 1)[0])
        if sl:
            out[f"{reg}.psd_slope"] = float(np.mean(sl))


def foam(out, reg, lab, m):
    if m.sum() < MIN_PX:
        return
    L = lab[..., 0][m]; C = np.hypot(lab[..., 1][m], lab[..., 2][m])
    thr = max(70.0, float(np.median(L)) + 20.0)
    out[f"{reg}.foam"] = float(((L >= thr) & (C <= 12)).mean())


def lapvar(L, m=None):
    lap = cv2.Laplacian(L, cv2.CV_32F, ksize=1)
    if m is None:
        return float(lap[1:-1, 1:-1].var())
    inner = cv2.erode(m.astype(np.uint8), np.ones((3, 3), np.uint8)) > 0
    return float(lap[inner].var()) if inner.sum() >= MIN_PX // 2 else float("nan")


def noise_sigma(gray: np.ndarray, valid: np.ndarray | None = None) -> float:
    """Immerkaer (1996) fast noise estimate, on the flattest 30 % of the frame."""
    g = gray.astype(np.float32)
    k = np.array([[1, -2, 1], [-2, 4, -2], [1, -2, 1]], np.float32)
    r = np.abs(cv2.filter2D(g, -1, k))[1:-1, 1:-1]
    gm = np.hypot(cv2.Sobel(g, cv2.CV_32F, 1, 0), cv2.Sobel(g, cv2.CV_32F, 0, 1))[1:-1, 1:-1]
    sel = np.ones_like(r, bool) if valid is None else valid[1:-1, 1:-1]
    if sel.sum() < 1000:
        return float("nan")
    thr = np.percentile(gm[sel], 30)
    flat = sel & (gm <= thr)
    return float(math.sqrt(math.pi / 2) / 6.0 * r[flat].mean())


def edge_width(L, whale, valid):
    """Step of L* across the outline / the peak gradient near the outline (crop px)."""
    w8 = whale.astype(np.uint8)
    if w8.sum() < MIN_PX:
        return float("nan")
    din = cv2.distanceTransform(w8, cv2.DIST_L2, 5)
    dout = cv2.distanceTransform(1 - w8, cv2.DIST_L2, 5)
    inner = (din >= 3) & (din <= 6)
    outer = (dout >= 3) & (dout <= 6) & valid
    if inner.sum() < 30 or outer.sum() < 30:
        return float("nan")
    step = abs(float(np.median(L[inner])) - float(np.median(L[outer])))
    if step < 5:
        return float("nan")
    g = np.hypot(cv2.Sobel(L, cv2.CV_32F, 1, 0, ksize=3), cv2.Sobel(L, cv2.CV_32F, 0, 1, ksize=3)) / 8.0
    gmax = cv2.dilate(g, np.ones((5, 5), np.uint8))
    boundary = (dout > 0) & (dout <= 1.5) & valid
    if boundary.sum() < 20:
        return float("nan")
    return step / max(float(np.median(gmax[boundary])), 1e-3)


def crop_stats(rgb, lab, S, lab_map, valid):
    out = {}
    L = lab[..., 0]
    whale, band = lab_map == 1, lab_map == 2
    basic(out, "whale", rgb, lab, S, whale)
    basic(out, "band", rgb, lab, S, band)
    texture(out, "whale", L, whale)
    texture(out, "band", L, band)
    foam(out, "band", lab, band)
    if whale.sum() >= MIN_PX and band.sum() >= MIN_PX:
        out["whale.dL_vs_band"] = float(np.median(L[whale]) - np.median(L[band]))
    if whale.sum() >= MIN_PX:
        Lw = L[whale]; med = float(np.median(Lw))
        out["whale.spec_share"] = float((Lw >= med + 25).mean())
        out["whale.hi_spread"] = float(np.percentile(Lw, 95) - med)
        out["whale.edge_width"] = edge_width(L, whale, valid & (lab_map != 0))
        out["whale.lapvar"] = lapvar(L, whale)
    out["crop.lapvar"] = lapvar(L, valid) if valid.sum() > MIN_PX else float("nan")
    return out


def frame_stats(rgb, lab, S, lab_map, valid):
    out = {}
    L = lab[..., 0]
    far, above = lab_map == 3, lab_map == 4
    basic(out, "far", rgb, lab, S, far)
    texture(out, "far", L, far)
    foam(out, "far", lab, far)
    basic(out, "above", rgb, lab, S, above)
    texture(out, "above", L, above)
    v = valid
    out["frame.L_mean"] = float(L[v].mean()); out["frame.L_std"] = float(L[v].std())
    out["frame.a_mean"] = float(lab[..., 1][v].mean()); out["frame.b_mean"] = float(lab[..., 2][v].mean())
    out["frame.L_p99"] = float(np.percentile(L[v], 99)); out["frame.L_p01"] = float(np.percentile(L[v], 1))
    px = rgb[v]
    out["frame.clip_hi"] = float((px.max(1) >= 250).mean()); out["frame.crush_lo"] = float((px.max(1) <= 8).mean())
    out["frame.lapvar"] = lapvar(L)
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    out["frame.noise"] = noise_sigma(gray)
    out["frame.above_share"] = float(above.mean())
    return out
