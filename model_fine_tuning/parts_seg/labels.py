"""Part labels for a Whale-Image-Data-Factory render (run05), derived on the fly from the existing label files.

Nothing in the dataset is changed; labels can be cached as palette PNGs in a separate folder (``build_cache``).

Classes per pixel (``CLASSES``):
    0 background   masks == 0
    1 fluke        visible whale behind the narrowest point of the tail stock: canonical x < x(spine_end)
                   (the tail stock is narrowest at canonical x -565 .. -545, spine_end sits at -559)
    2 dorsal fin   blade + hump: a 3D test in the posed whale frame (canon_deformed, the frame of the keypoints'
                   xyz_whale_cm). The fin's outline (dorsal_fin_front, the leading-edge chain fin_le_11..00 =
                   root .. tip, the trailing-edge chain fin_te_00..11 = tip .. back base = dorsal_fin_back) is
                   laid into its own plane; a pixel is fin when its surface point lies within a band of that
                   plane and inside the outline, or in the hump zone (between the blade root and the front base,
                   above the chord back base -> front base). The canonical frame's fin is NOT used (run05 canon
                   carries a fabricated peak ~45 cm behind the real blade).
    3 pectoral fin within the fin's radius of the segment pectoral_base -> pectoral_tip (posed frame), outside
                   the body's own radius around the spine (meta "spine", posed frame)
    4 head         canonical x >= 354 cm (the factory's amodal "head" band: 0.72 of the length from the tail)
    5 body         every other visible whale pixel
    255 ignore     masks == 2 (whale just under water); the calf ("companion", 15 % of shots: whale, part unknown):
                   its whole projection rasterised from the factory's rig model posed with the shot's companion
                   record, grown by 3 px, over every pixel that is background or instance value 2 (some calves are
                   missing from the label pass altogether: mask 0 over a visible calf); visible whale pixels without a
                   surface point farther than 3 px from a labelled one

The instance PNG carries a rim of value 2 around the main whale in shots without a calf (supersampled label
values between the two thresholds of the label pass), so value 2 means "calf" only inside the calf's projection;
elsewhere it is the main whale's anti-aliased edge and is labelled like its nearest labelled neighbour.
Order of precedence: dorsal fin > fluke > pectoral fin > head > body. The fluke test also requires the posed surface
point to lie near the posed tail (within 0.85 x the fluke span of the mean of spine_end, notch and tips), because
run05's registered canonical frame gives a few other points a canonical x behind the tail stock.
"""

from __future__ import annotations

import json
from pathlib import Path

import cv2
import numpy as np
from PIL import Image
from scipy import ndimage

IGNORE = 255
CLASSES = ["background", "fluke", "dorsal_fin", "pectoral_fin", "head", "body"]
COARSE = {0: 0, 1: 1, 2: 2, 3: 3, 4: 3, 5: 3}            # background / fluke / dorsal fin / other whale
PALETTE = [0, 0, 0, 230, 60, 60, 60, 200, 60, 240, 200, 40, 70, 120, 255, 170, 170, 170]   # bg, fluke red, fin green, pec yellow, head blue, body grey
OFFSET = 32768                                           # canon npz value = mm + offset
X_HEAD_CM = -804.8 + 0.72 * 2 * 804.8                    # 354.1: amodal head band (factory.project.BODY_PARTS)
FIN_BAND = 0.22                                          # half-thickness of the fin slab, x the fin's base chord
PEC_RADIUS = 0.20                                        # pectoral fin radius about its axis, x the fin's length


def _read(path: Path) -> np.ndarray:
    a = np.asarray(Image.open(path))
    return a[..., 0] if a.ndim == 3 else a


def _xyz(path: Path) -> tuple[np.ndarray, np.ndarray]:
    with np.load(path) as z:
        v = z["xyz_mm"]
    defined = v[..., 0] > 0
    cm = (v.astype(np.float32) - OFFSET) / 10.0
    return cm, defined


def _kp(meta: dict, name: str, field: str = "xyz_whale_cm"):
    k = (meta.get("keypoints") or {}).get(name)
    return None if not k or field not in k else np.asarray(k[field], np.float64)


def fin_outline(meta: dict) -> np.ndarray | None:
    """(N, 3) fin outline in the posed whale frame: front base, root .. tip along the leading edge, tip .. back base."""
    pts = [_kp(meta, "dorsal_fin_front")]
    pts += [_kp(meta, f"fin_le_{i:02d}") for i in range(11, -1, -1)]
    pts += [_kp(meta, f"fin_te_{i:02d}") for i in range(1, 12)]
    pts += [_kp(meta, "dorsal_fin_back")]
    if any(p is None for p in pts):
        return None
    return np.stack(pts)


def dorsal_mask(D: np.ndarray, defined: np.ndarray, meta: dict) -> np.ndarray:
    poly = fin_outline(meta)
    out = np.zeros(defined.shape, bool)
    if poly is None:
        return out
    front, root, back = poly[0], poly[1], poly[-1]
    tip = _kp(meta, "dorsal_fin_tip")
    # plane of the fin: spanned by the base chord and the tip
    e1 = front - back
    chord = float(np.linalg.norm(e1))
    if chord < 1e-3:
        return out
    e1 /= chord
    up = tip - back - np.dot(tip - back, e1) * e1
    height = float(np.linalg.norm(up))
    if height < 1e-3:
        return out
    e2 = up / height
    n = np.cross(e1, e2)
    ys, xs = np.nonzero(defined)
    P = D[ys, xs].astype(np.float64) - back
    a, b, d = P @ e1, P @ e2, P @ n
    band = FIN_BAND * chord
    cand = np.abs(d) < band
    # outline in plane coordinates (cm), rasterised at 1 cm
    q = np.stack([(poly - back) @ e1, (poly - back) @ e2], 1)
    r_root = float((root - back) @ e1)
    hump = np.array([[r_root, -0.02 * height], [chord, -0.02 * height], [chord, 1.2 * height], [r_root, 1.2 * height]])
    lo = np.minimum(q.min(0), hump.min(0)) - 5
    hi = np.maximum(q.max(0), hump.max(0)) + 5
    res = 1.0 if max(hi - lo) < 3000 else max(hi - lo) / 3000
    W, H = int((hi[0] - lo[0]) / res) + 1, int((hi[1] - lo[1]) / res) + 1
    canvas = np.zeros((H, W), np.uint8)
    cv2.fillPoly(canvas, [np.round((q - lo) / res).astype(np.int32)], 1)
    # hump zone: between the blade root and the front base, above the chord (the back line) -- the rig has no hump,
    # so this is the ridge strip the researcher counts as the fin's front slope
    cv2.fillPoly(canvas, [np.round((hump - lo) / res).astype(np.int32)], 1)
    ia = np.clip(np.round((a - lo[0]) / res).astype(int), 0, W - 1)
    ib = np.clip(np.round((b - lo[1]) / res).astype(int), 0, H - 1)
    inside = (canvas[ib, ia] > 0) & (a >= lo[0]) & (a <= hi[0]) & (b >= lo[1]) & (b <= hi[1])
    out[ys[cand & inside], xs[cand & inside]] = True
    return out


def _seg_dist(P: np.ndarray, A: np.ndarray, B: np.ndarray):
    v = B - A
    L2 = float(v @ v)
    t = np.clip(((P - A) @ v) / max(L2, 1e-9), 0, 1)
    return np.linalg.norm(P - (A + t[:, None] * v), axis=1), (P - A) @ v / max(L2, 1e-9)


def pectoral_mask(D: np.ndarray, defined: np.ndarray, meta: dict) -> np.ndarray:
    out = np.zeros(defined.shape, bool)
    spine = np.asarray((meta.get("spine") or {}).get("points_cm") or [], np.float64)
    ys, xs = np.nonzero(defined)
    if len(ys) == 0:
        return out
    P = D[ys, xs].astype(np.float64)
    # distance to the spine polyline (the body's own radius around it)
    if len(spine) >= 2:
        ds = np.full(len(P), np.inf)
        for A, B in zip(spine[:-1], spine[1:]):
            ds = np.minimum(ds, _seg_dist(P, A, B)[0])
    else:
        ds = np.full(len(P), np.inf)
    for side in ("left", "right"):
        A, B = _kp(meta, f"pectoral_base_{side}"), _kp(meta, f"pectoral_tip_{side}")
        if A is None or B is None:
            continue
        L = float(np.linalg.norm(B - A))
        dist, t = _seg_dist(P, A, B)
        # the base keypoint sits on the body surface; the spine distance there is the body radius
        r_body = float(min(_seg_dist(A[None], s0, s1)[0][0] for s0, s1 in zip(spine[:-1], spine[1:]))) if len(spine) >= 2 else 0.0
        sel = (dist < PEC_RADIUS * L) & (t > -0.05) & (t < 1.08) & (ds > 1.03 * r_body)
        out[ys[sel], xs[sel]] = True
    return out


def fluke_mask(D: np.ndarray, cand: np.ndarray, meta: dict) -> np.ndarray:
    """Canonical-x candidates that are also near the posed tail (a far pectoral fin can carry a canonical x behind
    the tail stock in run05's registered canonical frame)."""
    names = ["spine_end", "fluke_notch", "fluke_tip_left", "fluke_tip_right"]
    pts = [_kp(meta, n) for n in names]
    if any(p is None for p in pts):
        return cand
    span = float(np.linalg.norm(pts[2] - pts[3]))
    centre = np.mean(pts, 0)
    ys, xs = np.nonzero(cand)
    ok = np.linalg.norm(D[ys, xs].astype(np.float64) - centre, axis=1) < 0.85 * span
    out = np.zeros(cand.shape, bool)
    out[ys[ok], xs[ok]] = True
    return out


FACTORY = Path.home() / "NOAA_whales" / "Whale-Image-Data-Factory"
_MODELS: dict = {}


def calf_projection(run_dir: Path, meta: dict) -> np.ndarray | None:
    """The calf's whole projection (amodal, also under water), rasterised from the factory's rig model posed with the
    shot's companion record (factory.whalemodel.WhaleModel.posed + factory.project.amodal_and_parts, as the label pass
    does for the shot's own whale). Some calves are missing from the rendered label pass altogether (mask 0 over a
    visible calf), so the instance PNG alone cannot find them."""
    comp = meta.get("companion")
    if not comp:
        return None
    import sys
    if str(FACTORY) not in sys.path:
        sys.path.insert(0, str(FACTORY))
    from factory import project
    from factory.whalemodel import WhaleModel

    key = str(run_dir)
    if key not in _MODELS:
        cfg = json.loads((Path(run_dir) / "run_config.json").read_text())
        _MODELS[key] = WhaleModel.from_config(cfg.get("config", {}))
    model = _MODELS[key]
    cam = meta["camera_intrinsics"]
    camera = project.Camera(meta["camera"]["location"], meta["camera"]["rotation"], meta["camera"]["focal_mm"], int(cam["width"]), int(cam["height"]))
    img = project.amodal_and_parts(model.posed(comp), model.faces, comp["whale"], camera)
    return img > 0


def calf_region(run_dir: Path, inst: np.ndarray | None, meta: dict) -> np.ndarray | None:
    """Pixels that are (or may be) the calf: its projection grown by 3 px. None without a companion."""
    proj = calf_projection(run_dir, meta)
    if proj is None or not proj.any():
        return None
    return ndimage.binary_dilation(proj, iterations=3)


def derive(dataset_dir: Path, shot_id: str, return_parts: bool = False):
    """uint8 (H, W) part label for one render (see module doc)."""
    ds = Path(dataset_dir)
    meta = json.loads((ds / "meta" / f"{shot_id}.json").read_text())
    m = _read(ds / "masks" / f"{shot_id}.png")
    if m.max(initial=0) > 2:
        m = np.where(m >= 200, 1, np.where(m > 0, 2, 0))
    label = np.zeros(m.shape, np.uint8)
    label[m == 2] = IGNORE
    vis = m == 1
    if not vis.any():
        return (label, {"calf_px": 0, "fin_px": 0, "pec_px": 0}) if return_parts else label
    inst_path = ds / "instances" / f"{shot_id}.png"
    inst = _read(inst_path) if inst_path.exists() else None
    calf_zone = calf_region(ds.parent, inst, meta)
    # the calf = visible whale pixels of instance value 2 inside its projection; the other value-2 pixels are the
    # main whale's anti-aliased rim
    calf = None
    if calf_zone is not None:
        calf = calf_zone & vis & ((inst == 2) if inst is not None else np.ones_like(vis))
    C, cdef = _xyz(ds / "canon" / f"{shot_id}.npz")
    D, ddef = _xyz(ds / "canon_deformed" / f"{shot_id}.npz")
    defined = cdef & ddef & vis
    if calf is not None:
        defined &= ~calf
    part = np.zeros(m.shape, np.uint8)
    x = C[..., 0]
    spine_end = _kp(meta, "spine_end", "xyz_canonical_cm")
    x_fluke = float(spine_end[0]) if spine_end is not None else -559.0
    part[defined] = 5
    part[defined & (x >= X_HEAD_CM)] = 4
    pec = pectoral_mask(D, defined, meta)
    part[pec] = 3
    part[fluke_mask(D, defined & (x < x_fluke), meta)] = 1
    fin = dorsal_mask(D, defined, meta)
    part[fin] = 2
    # visible whale pixels without a surface point (anti-aliased edge, the rim of instance value 2): nearest label
    todo = vis & ~defined
    if calf is not None:
        todo &= ~calf
    if todo.any() and defined.any():
        dist, (iy, ix) = ndimage.distance_transform_edt(~defined, return_indices=True)
        near = todo & (dist <= 3.0)
        part[near] = part[iy[near], ix[near]]
        part[todo & ~near] = IGNORE
    label[vis] = part[vis]
    label[vis & (part == 0)] = IGNORE
    if calf is not None:
        label[ndimage.binary_dilation(calf, iterations=2) & vis] = IGNORE
    if calf_zone is not None:
        label[calf_zone & (label == 0)] = IGNORE          # a calf the label pass missed, or under water: not "background"
    if return_parts:
        return label, {"calf_px": int(calf.sum()) if calf is not None else 0, "fin_px": int((label == 2).sum()), "pec_px": int((label == 3).sum())}
    return label


def save_palette(label: np.ndarray, path: Path) -> None:
    im = Image.fromarray(label.astype(np.uint8), mode="P")
    pal = PALETTE + [0, 0, 0] * (256 - len(PALETTE) // 3)
    pal[255 * 3: 256 * 3] = [255, 0, 255]
    im.putpalette(pal)
    im.save(path, optimize=True)


def colorize(label: np.ndarray) -> np.ndarray:
    """RGB image of a label map (ignore = magenta)."""
    pal = np.zeros((256, 3), np.uint8)
    pal[: len(PALETTE) // 3] = np.asarray(PALETTE, np.uint8).reshape(-1, 3)
    pal[255] = (255, 0, 255)
    return pal[label]


def _cache_one(job):
    ds, out, shot_id = job
    path = Path(out) / f"{shot_id}.png"
    if path.exists():
        return shot_id, None
    try:
        label = derive(Path(ds), shot_id)
    except Exception as exc:  # noqa: BLE001
        return shot_id, str(exc)[:200]
    save_palette(label, path)
    return shot_id, None


def build_cache(dataset_dir: Path, out_dir: Path, ids: list[str], workers: int = 6) -> list:
    from multiprocessing import Pool

    out_dir.mkdir(parents=True, exist_ok=True)
    errors = []
    with Pool(workers) as pool:
        for k, (sid, err) in enumerate(pool.imap_unordered(_cache_one, [(str(dataset_dir), str(out_dir), i) for i in ids], chunksize=8)):
            if err:
                errors.append((sid, err))
            if (k + 1) % 500 == 0:
                print(f"{k + 1}/{len(ids)} labels", flush=True)
    return errors


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser(description="cache part labels of a factory run as palette PNGs")
    ap.add_argument("--run-dir", type=Path, default=Path.home() / "NOAA_whales/Whale-Image-Data-Factory/data/renders/run05")
    ap.add_argument("--out", type=Path, default=Path("/var/tmp/sat3737-parts-cache/run05/labels"))
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--limit", type=int, default=None)
    a = ap.parse_args()
    ds = a.run_dir / "dataset"
    ids = [json.loads(l)["id"] for l in open(ds / "manifest.jsonl") if l.strip()][: a.limit]
    errs = build_cache(ds, a.out, ids, a.workers)
    print("done", len(ids), "errors", len(errs), errs[:5])
