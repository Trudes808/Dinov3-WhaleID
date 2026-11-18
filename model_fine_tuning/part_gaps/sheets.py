"""Sheets for the gap table (real photos: scratchpad only, never the repository).

    PYTHONPATH=. .venv/bin/python -m model_fine_tuning.part_gaps.sheets failures [--model parts_seg_r05_a --variant hr]
        per category, the wrongly called real instances of the labelled sample: photo | the model's part map
    PYTHONPATH=. .venv/bin/python -m model_fine_tuning.part_gaps.sheets nearest [--model ...]
        per failing real instance: photo | model's part map | three nearest renders (frozen DINOv3 whale tokens, nearest.py)
"""

from __future__ import annotations

import argparse
import json
import sys

import cv2
import numpy as np
from PIL import Image

from ..parts_seg import evaluate_real as ER
from ..parts_seg.labels import PALETTE
from . import common as C

LEG = "fluke red, fin green, pectoral yellow, head blue, body grey"


def real_tile(row: dict, name: str, variant: str, size: int = 220) -> np.ndarray:
    old = ER.DAYS
    ER.DAYS = C.arm_root(name)
    try:
        t = ER.tile(row, name, variant, size)
    finally:
        ER.DAYS = old
    return t


def put(t, text, y=18, scale=0.5):
    cv2.putText(t, text, (4, y), cv2.FONT_HERSHEY_SIMPLEX, scale, (0, 0, 0), 3)
    cv2.putText(t, text, (4, y), cv2.FONT_HERSHEY_SIMPLEX, scale, (255, 255, 255), 1)


def grid(tiles, cols, title, path):
    h, w = tiles[0].shape[:2]
    nrows = (len(tiles) + cols - 1) // cols
    out = np.full((28 + nrows * (h + 4), cols * (w + 4), 3), 255, np.uint8)
    cv2.putText(out, title, (6, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 0), 1)
    for i, t in enumerate(tiles):
        r, c = divmod(i, cols)
        out[28 + r * (h + 4): 28 + r * (h + 4) + h, c * (w + 4): c * (w + 4) + w] = t
    cv2.imwrite(str(path), out, [cv2.IMWRITE_JPEG_QUALITY, 88])


def overlay_parts(rgb, cls):
    pal = np.zeros((256, 3), np.uint8)
    pal[: len(PALETTE) // 3] = np.asarray(PALETTE, np.uint8).reshape(-1, 3)
    m = (cls > 0) & (cls < 255)
    o = rgb.copy()
    o[m] = (0.4 * rgb[m] + 0.6 * pal[cls][m]).astype(np.uint8)
    return o


def render_tile(ds_root, shot, box, W, H, size=220, parts=True):
    """render crop around a whale box (native px) | its truth part map."""
    img = cv2.imread(str(ds_root / "images" / f"{shot}.png"))
    x1, y1, x2, y2 = box
    side = max(x2 - x1, y2 - y1) * 1.3 + 20
    cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
    M = np.array([[size / side, 0, -(cx - side / 2) * size / side], [0, size / side, -(cy - side / 2) * size / side]])
    t = cv2.warpAffine(img, M, (size, size), flags=cv2.INTER_AREA, borderMode=cv2.BORDER_CONSTANT, borderValue=0)
    if not parts:
        return t
    p = np.asarray(Image.open(ds_root / "parts" / f"{shot}.png")) if (ds_root / "parts" / f"{shot}.png").exists() else \
        np.asarray(Image.open(f"/var/tmp/sat3737-parts-cache/run05/labels/{shot}.png"))
    if p.shape[:2] != img.shape[:2]:
        p = cv2.resize(p, (img.shape[1], img.shape[0]), interpolation=cv2.INTER_NEAREST)
    pc = cv2.warpAffine(p, M, (size, size), flags=cv2.INTER_NEAREST, borderMode=cv2.BORDER_CONSTANT, borderValue=0)
    return overlay_parts(t[..., ::-1], pc)[..., ::-1].copy()


def failures(a):
    rep = json.loads((C.OUT / f"report_{a.model}_{a.variant}.json").read_text())
    rows = {r["key"]: r for r in C.arm_rows(a.model, C.TRAIN_DAYS + C.VAL_DAYS)}
    sd = C.SCRATCH / "sheets"
    sd.mkdir(parents=True, exist_ok=True)
    inst = rep["instances"]
    made = []
    for cat in C.CATS:
        ks = [k for k, v in inst.items() if v["cat"] == cat and not v["ok"]]
        if not ks:
            continue
        ks.sort(key=lambda k: (inst[k]["split"], k))
        tiles, legend = [], []
        for k in ks[:24]:
            t = np.ascontiguousarray(real_tile(rows[k], a.model, a.variant))
            put(t, f"{len(tiles) + 1} {inst[k]['call']}")
            put(t, " ".join(inst[k]["attrs"]) + f" {inst[k]['size']}", y=212, scale=0.45)
            tiles.append(t)
            legend.append({"tile": len(tiles), "key": k, **{x: inst[k][x] for x in ("cat", "attrs", "call", "conf", "split")}})
        p = sd / f"fail_{cat}_{a.model}_{a.variant}.jpg"
        grid(tiles, 4, f"{cat}: wrongly called real instances ({len(ks)}), {a.model} {a.variant}; left photo, right parts ({LEG})", p)
        p.with_suffix(".json").write_text(json.dumps(legend, indent=1))
        made.append(str(p))
    print("\n".join(made))


def nearest(a):
    from .nearest import load_neighbours
    nb = load_neighbours(a.model, a.variant)
    rows = {r["key"]: r for r in C.arm_rows(a.model, C.TRAIN_DAYS + C.VAL_DAYS)}
    sd = C.SCRATCH / "sheets"
    sd.mkdir(parents=True, exist_ok=True)
    by_cat = {}
    for k, v in nb["real"].items():
        if not v["ok"]:
            by_cat.setdefault(v["cat"], []).append(k)
    made = []
    for cat, ks in sorted(by_cat.items(), key=lambda x: -len(x[1])):
        if a.cats and cat not in a.cats:
            continue
        # failing cases whose nearest renders include the marked set first, then the rest
        ks = sorted(ks, key=lambda k: (-sum(n["tag"] in a.mark for n in nb["real"][k]["nn_dino"][:3]), k))[: a.per]
        lines = []
        legend = []
        for k in ks:
            v = nb["real"][k]
            t = np.ascontiguousarray(real_tile(rows[k], a.model, a.variant))
            put(t, f"{v['cat']} -> {v['call']}")
            parts_ = [t]
            for j, n in enumerate(v["nn_dino"][:3]):
                ds = C.FACTORY / n["tag"] / "dataset"
                rt = np.ascontiguousarray(render_tile(ds, n["shot"], n["box"], None, None))
                put(rt, f"{n['tag'].replace('r06r_test_', 'r06 ')} {n['cat']} {n['sim']:.2f}")
                if n["tag"] in a.mark:                        # the factory's newest check set: orange frame
                    cv2.rectangle(rt, (0, 0), (rt.shape[1] - 1, rt.shape[0] - 1), (0, 140, 255), 6)
                parts_.append(rt)
            lines.append(np.concatenate(parts_, 1))
            legend.append({"key": k, "cat": v["cat"], "call": v["call"], "nn": v["nn_dino"][:3], "nn_shape": v.get("nn_shape", [])[:3]})
        p = sd / f"nearest_{cat}_{a.model}.jpg"
        grid(lines, 1, f"{cat}: failing real (photo | model parts) | 3 nearest renders, DINOv3 whale tokens (photo + TRUTH parts; orange = {','.join(a.mark)})", p)
        p.with_suffix(".json").write_text(json.dumps(legend, indent=1))
        made.append(str(p))
    print("\n".join(made))


def eye(a):
    """blind by-eye sheets of every whale of a render set (render + outline only; no labels, no model output), for
    labels_ai_<tag>.txt (<shot> <iid> <CATEGORY> [attrs] [#S|#R])."""
    import json as _j
    tag = a.tag
    ds = C.FACTORY / tag / "dataset"
    from .coverage import scan
    rs = [_j.loads(l) for l in open(scan(ds))]
    out = C.SCRATCH / "eye" / tag
    out.mkdir(parents=True, exist_ok=True)
    tiles, idx = [], {}
    for r in rs:
        img = cv2.imread(str(ds / "images" / f"{r['shot']}.png"))
        inst = np.asarray(Image.open(ds / "instances" / f"{r['shot']}.png"))
        inst = inst[..., 0] if inst.ndim == 3 else inst
        x1, y1, x2, y2 = r["box"]
        side, S = max(x2 - x1, y2 - y1) * 1.4 + 30, 260
        cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
        M = np.array([[S / side, 0, -(cx - side / 2) * S / side], [0, S / side, -(cy - side / 2) * S / side]])
        t = cv2.warpAffine(img, M, (S, S), flags=cv2.INTER_AREA if side > S else cv2.INTER_LINEAR, borderValue=(40, 0, 40))
        m = cv2.warpAffine((inst == r["iid"]).astype(np.uint8), M, (S, S), flags=cv2.INTER_NEAREST)
        cn, _ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(t, cn, -1, (255, 255, 0), 1)
        t = np.ascontiguousarray(t)
        put(t, str(len(tiles) + 1), y=20, scale=0.6)
        tiles.append(t)
        idx[len(tiles)] = [r["shot"], r["iid"]]
    for i in range(0, len(tiles), 25):
        grid(tiles[i:i + 25], 5, f"{tag} whales {i + 1}-{min(i + 25, len(tiles))}: blind by-eye sheet", out / f"eye_{i // 25}.jpg")
    (out / "index.json").write_text(_j.dumps(idx))
    print(out, len(tiles), "whales")


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("what", choices=["failures", "nearest", "eye"])
    ap.add_argument("--tag", default=None, help="eye: render set tag (folder name under the factory's data/renders)")
    ap.add_argument("--model", default="parts_seg_r05_a")
    ap.add_argument("--variant", default="hr")
    ap.add_argument("--per", type=int, default=8)
    ap.add_argument("--mark", nargs="*", default=["r06r_test_r6b", "r06r_test_r7"], help="render sets framed in orange")
    ap.add_argument("--cats", nargs="*", default=None, help="only these real categories")
    a = ap.parse_args(argv)
    {"failures": failures, "nearest": nearest, "eye": eye}[a.what](a)
    return 0


if __name__ == "__main__":
    sys.exit(main())
