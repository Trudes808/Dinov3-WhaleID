"""Shared helpers of the instances_v2 evaluation scripts (read-only on everything but the scratchpad sheets):
the researcher's marks, old and new instance masks of a photo, tiles and sheets."""
import json
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
DAYS = ROOT / "experiments" / "days_all"
LAB = ROOT.parent / "Whale-Review-Web" / "data"
SPLIT = json.loads((ROOT / "model_fine_tuning/splits/days.json").read_text())
TRAINVAL = SPLIT["train"] + SPLIT["val"]
SHEETS = Path("/tmp/claude-1072903897/-home-sat3737-NOAA-whales/4fbba7ff-3ba5-4b50-9cb7-970c81250409/scratchpad/instances_v2")
assert not set(TRAINVAL) & set(SPLIT["eval"])


def check_day(day: str) -> None:
    assert day not in SPLIT["eval"], "evaluation days are not read"


def load_marks(day: str):
    """bodymap_clicks.py's reading: per old instance key the main whale's keypoints (records merged) and the other whales
    (the latest record of the photo wins). Points as photo fractions."""
    check_day(day)
    labs = json.loads((LAB / day / "pair_marks.json").read_text()); labs = labs["labels"] if isinstance(labs, dict) else labs
    photos, others = {}, {}
    for l in labs:
        if l.get("verdict") == "skip":
            continue
        for sd in ("a", "b"):
            key = l.get(f"key_{sd}")
            if not key:
                continue
            kp = (l.get("keypoints") or {}).get(sd) or {}
            if kp:
                photos.setdefault(key, {}).update(kp)
            ow = (l.get("other_whales") or {}).get(sd)
            if ow is not None and (kp or ow):
                others[key] = ow
    return photos, others


_TABLE = None


def old_masks(day: str, stem: str, W: int, H: int):
    """Old instance keys and masks (2560-px frame) and the old not-a-whale drop list of the day."""
    global _TABLE
    from model_fine_tuning.cluster.instances_v2 import old_instances
    from model_fine_tuning.common import geometry, paths
    if _TABLE is None:
        _TABLE = geometry.load_folded_table(paths.FOLDED_TABLE)
    return old_instances(DAYS / day, stem, W, H, _TABLE)


_DROP, _IDX = {}, {}


def old_drop(day: str) -> set:
    if day not in _DROP:
        f = DAYS / day / "safe" / "not_whale.json"
        _DROP[day] = set(json.loads(f.read_text()).get("drop", [])) if f.exists() else set()
    return _DROP[day]


def new_index(day: str) -> dict:
    check_day(day)
    if day not in _IDX:
        _IDX[day] = {json.loads(l)["stem"]: json.loads(l) for l in open(DAYS / day / "instances_v2" / "index.jsonl")}
    return _IDX[day]


def blob_rows(day: str) -> dict:
    check_day(day)
    return {json.loads(l)["stem"]: json.loads(l) for l in open(DAYS / day / "instances_v2" / "blobs.jsonl")}


def new_masks(day: str, stem: str):
    """Kept new instances of a photo: keys, masks (2560-px frame), records."""
    r = new_index(day)[stem]; m = cv2.imread(str(DAYS / day / "instances_v2" / "masks" / f"{stem}.png"), cv2.IMREAD_UNCHANGED)
    return [i["key"] for i in r["instances"]], [m == i["id"] for i in r["instances"]], r["instances"]


def blob_image(day: str, stem: str):
    return cv2.imread(str(DAYS / day / "instances_v2" / "blobs" / f"{stem}.png"), cv2.IMREAD_UNCHANGED)


def photo(day: str, stem: str):
    check_day(day)
    return cv2.imread(str(DAYS / day / "images" / f"{stem}.jpg"))


def which(masks: list, p, tol: float = 25.0):
    """Index of the mask that holds point p (x, y; px), else the nearest within tol px, else None."""
    x, y = int(round(p[0])), int(round(p[1])); best = (tol + 1e-6, None); t = int(np.ceil(tol))
    for i, m in enumerate(masks):
        H, W = m.shape
        if 0 <= x < W and 0 <= y < H and m[y, x]:
            return i
        y0, y1, x0, x1 = max(y - t, 0), min(y + t + 1, H), max(x - t, 0), min(x + t + 1, W)
        ys, xs = np.nonzero(m[y0:y1, x0:x1])
        if len(ys):
            d = np.hypot(xs + x0 - x, ys + y0 - y).min()
            if d < best[0]:
                best = (d, i)
    return best[1]


COLOURS = [(0, 0, 255), (0, 200, 0), (255, 128, 0), (255, 0, 255), (0, 255, 255), (128, 0, 255), (255, 255, 0), (0, 128, 255)]   # BGR


def outline(img, mask, colour, th=3):
    cs, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(img, cs, -1, colour, th, cv2.LINE_AA)


def region(box, W, H, pad=0.3, min_side=300):
    """A crop window (x0, y0, x1, y1) around a box, padded, at least min_side, the photo's aspect kept where possible."""
    x0, y0, x1, y1 = box; cx, cy = (x0 + x1) / 2, (y0 + y1) / 2
    w = max((x1 - x0) * (1 + 2 * pad), min_side); h = max((y1 - y0) * (1 + 2 * pad), min_side)
    a = W / H
    if w / h < a:
        w = h * a
    else:
        h = w / a
    w, h = min(w, W), min(h, H)
    x0 = min(max(cx - w / 2, 0), W - w); y0 = min(max(cy - h / 2, 0), H - h)
    return int(x0), int(y0), int(x0 + w), int(y0 + h)


def cut(img, win, size):
    x0, y0, x1, y1 = win
    return cv2.resize(img[y0:y1, x0:x1], size, interpolation=cv2.INTER_AREA if (x1 - x0) > size[0] else cv2.INTER_CUBIC)


def fit(img, size=(320, 214)):
    """The image letterboxed into a fixed box (portrait and landscape photos give tiles of one size)."""
    W, H = size; h, w = img.shape[:2]; k = min(W / w, H / h); r = cv2.resize(img, (max(1, int(w * k)), max(1, int(h * k))), interpolation=cv2.INTER_AREA)
    out = np.full((H, W, 3), 30, np.uint8); y0, x0 = (H - r.shape[0]) // 2, (W - r.shape[1]) // 2; out[y0:y0 + r.shape[0], x0:x0 + r.shape[1]] = r
    return out


def label(img, text, org=(5, 18), scale=0.5):
    cv2.putText(img, text, org, cv2.FONT_HERSHEY_SIMPLEX, scale, (0, 0, 0), 3, cv2.LINE_AA)
    cv2.putText(img, text, org, cv2.FONT_HERSHEY_SIMPLEX, scale, (255, 255, 255), 1, cv2.LINE_AA)


def write_sheets(tiles: list, legend: list, tag: str, cols: int, per: int = 24) -> list:
    """Sheets of `per` tiles (cols per row) -> SHEETS/<tag>_NN.jpg and <tag>_legend.json (tile number -> day, stem, ...)."""
    SHEETS.mkdir(parents=True, exist_ok=True); paths = []
    if not tiles:
        return paths
    h = max(t.shape[0] for t in tiles); w = max(t.shape[1] for t in tiles)
    tiles = [cv2.copyMakeBorder(t, 0, h - t.shape[0], 0, w - t.shape[1], cv2.BORDER_CONSTANT, value=(40, 40, 40)) for t in tiles]
    tiles = [cv2.copyMakeBorder(t, 2, 2, 2, 2, cv2.BORDER_CONSTANT, value=(255, 255, 255)) for t in tiles]
    for s in range(0, len(tiles), per):
        part = tiles[s:s + per]
        while len(part) % cols:
            part.append(np.zeros_like(tiles[0]))
        sheet = np.vstack([np.hstack(part[i:i + cols]) for i in range(0, len(part), cols)])
        p = SHEETS / f"{tag}_{s // per + 1:02d}.jpg"; cv2.imwrite(str(p), sheet, [cv2.IMWRITE_JPEG_QUALITY, 85]); paths.append(str(p))
    (SHEETS / f"{tag}_legend.json").write_text(json.dumps(legend, indent=0))
    return paths


def blob_tile(s, zoom=320, ctx=(320, 213), text=None, extra_masks=()):
    img = photo(s["day"], s["stem"]); H, W = img.shape[:2]; lab = blob_image(s["day"], s["stem"]); m = np.isin(lab, s.get("blobs") or [s["b"]])
    x0, y0, x1, y1 = s["bbox"]; side = max(3.0 * max(x1 - x0, y1 - y0), 160.0); cx, cy = (x0 + x1) / 2, (y0 + y1) / 2
    win = (int(max(cx - side / 2, 0)), int(max(cy - side / 2, 0)), int(min(cx + side / 2, W)), int(min(cy + side / 2, H)))
    z = img.copy(); ring = cv2.dilate(m.astype(np.uint8), np.ones((5, 5), np.uint8), iterations=max(1, int(side / 160)))
    outline(z, ring, (255, 0, 255), max(1, int(side / 200)))
    zw, zh = win[2] - win[0], win[3] - win[1]; sc = zoom / max(zw, zh)
    zt = np.zeros((zoom, zoom, 3), np.uint8); zz = cv2.resize(z[win[1]:win[3], win[0]:win[2]], (max(1, int(zw * sc)), max(1, int(zh * sc))), interpolation=cv2.INTER_CUBIC if sc > 1 else cv2.INTER_AREA)
    zt[:zz.shape[0], :zz.shape[1]] = zz
    c = img.copy(); outline(c, lab == 1, (0, 0, 255), 6)
    for em, col in extra_masks:
        outline(c, em, col, 6)
    cv2.rectangle(c, win[:2], win[2:], (0, 255, 255), 8)
    c = cv2.resize(c, ctx, interpolation=cv2.INTER_AREA); ct = np.zeros((zoom, ctx[0], 3), np.uint8); ct[:ctx[1]] = c
    label(zt, text or str(s["tile"]), (5, 22), 0.7)
    label(ct, f"{s['area']} px", (5, ctx[1] + 22), 0.55)
    for k, line in enumerate(s.get("lines", [])):
        label(ct, line, (5, ctx[1] + 44 + 20 * k), 0.5)
    return np.hstack([zt, ct])
