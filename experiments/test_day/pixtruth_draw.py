"""Drawing the pixel truth (rules: pixtruth_rules.md). The prompts are mine (boxes, points, polygons); SAM 2.1 (public
weights, facebook/sam2.1-hiera-large) turns box / point prompts into masks on a crop around the prompt. Nothing else.

    S=SCRATCH/pixtruth; P=$R/pixel_truth
    .venv/bin/python experiments/test_day/pixtruth_draw.py grid --half $S/half --out $S/view/a.jpg SEQ ...   # photos with a coordinate grid
    .venv/bin/python experiments/test_day/pixtruth_draw.py zoom --half $S/half --out $S/view/z.jpg SEQ X0 Y0 X1 Y1
    flock LOCK .venv/bin/python experiments/test_day/pixtruth_draw.py run --half $S/half --prompts $P/prompts.json --labels $P/labels --cache $S/samcache [SEQ ...]
    .venv/bin/python experiments/test_day/pixtruth_draw.py check --half $S/half --labels $P/labels --out $S/check/a.jpg SEQ ...

prompts.json: {seq: [op, ...]} painted in order (later ops overwrite earlier ones). An op is one of
  {"c": cls, "box": [x0, y0, x1, y1], "p": [[x, y], ...], "n": [[x, y], ...], "mm": "best" | 0 | 1 | 2, "fill": true}   SAM
  {"c": cls, "p": [...], "n": [...]}                                                                                       SAM, points only
  {"c": cls, "poly": [[x, y], ...]} | {"c": cls, "rect": [x0, y0, x1, y1]} | {"c": cls, "line": [[x, y], ...], "w": px}     painted directly
  {"c": cls, "wand": {"line": [...], "w": px, "t": thr, "dark": bool}}   brightness wand inside my own corridor (see wand())
  "crop": [x0, y0, x1, y1] on a SAM op sets the crop SAM sees ("full": true = the whole frame).
optional on any op: "over": [classes] (paint only where the label is one of these now), "blur": r (pixels within r of the
op's own boundary become 255), "clip": true (SAM mask cut to the box), "within": {"line": [...], "w": px} | {"poly": [...]}
(mask cut to my own corridor; added 2026-11-21 for the rail re-draw). Coordinates on the half-resolution grid (2748 x 1835).
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

W, H = 2748, 1835
PAL = {0: (25, 50, 110), 1: (255, 140, 0), 2: (255, 0, 0), 3: (0, 255, 255), 4: (255, 255, 0), 5: (0, 200, 0), 6: (150, 75, 40),
       7: (255, 0, 255), 8: (130, 130, 0), 9: (170, 170, 255), 255: (128, 128, 128)}
NAMES = {0: "water", 1: "whale", 2: "person", 3: "rail", 4: "tape", 5: "deck", 6: "pontoon", 7: "vessel", 8: "land", 9: "gear", 255: "unsure"}


def font(sz):
    try:
        return ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", sz)
    except OSError:
        return ImageFont.load_default()


def gridded(im: Image.Image, step=250, scale=1.0, origin=(0, 0)) -> Image.Image:
    d = ImageDraw.Draw(im); f = font(max(12, int(14 / scale * 1.6)))
    w, h = im.size; ox, oy = origin
    for gx in range((ox // step + 1) * step if ox % step else ox, ox + int(w * scale) + 1, step):
        x = (gx - ox) / scale; d.line([(x, 0), (x, h)], fill=(255, 255, 0), width=1); d.text((x + 2, 2), str(gx), fill=(255, 255, 0), font=f)
    for gy in range((oy // step + 1) * step if oy % step else oy, oy + int(h * scale) + 1, step):
        y = (gy - oy) / scale; d.line([(0, y), (w, y)], fill=(255, 255, 0), width=1); d.text((2, y + 2), str(gy), fill=(255, 255, 0), font=f)
    return im


def tile(ims, labels, cols=2, tw=1100):
    th = int(tw * H / W); out = Image.new("RGB", (cols * tw, ((len(ims) + cols - 1) // cols) * (th + 26)), (0, 0, 0)); d = ImageDraw.Draw(out)
    for i, (im, lab) in enumerate(zip(ims, labels)):
        x, y = (i % cols) * tw, (i // cols) * (th + 26); out.paste(im.resize((tw, th), Image.BILINEAR), (x, y + 26)); d.text((x + 4, y + 3), lab, fill=(255, 255, 255), font=font(18))
    return out


def overlay(img: np.ndarray, lab: np.ndarray, alpha=0.4) -> np.ndarray:
    o = img.astype(np.float32).copy()
    for c, col in PAL.items():
        if c == 0:
            continue
        m = lab == c
        if m.any():
            o[m] = o[m] * (1 - alpha) + np.array(col) * alpha
            cs, _ = cv2.findContours(m.astype(np.uint8), cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
            cv2.drawContours(o, cs, -1, col, 3)
    return o.clip(0, 255).astype(np.uint8)


class Sam:
    def __init__(self, cache: Path):
        self.cache = cache; cache.mkdir(parents=True, exist_ok=True); self.m = None   # model loaded on the first uncached prompt

    def _load(self):
        import torch
        from transformers import Sam2Model, Sam2Processor
        self.torch = torch; self.dev = "cuda" if torch.cuda.is_available() else "cpu"
        self.m = Sam2Model.from_pretrained("facebook/sam2.1-hiera-large").to(self.dev).eval()
        self.p = Sam2Processor.from_pretrained("facebook/sam2.1-hiera-large")

    def mask(self, img: np.ndarray, seq: str, op: dict) -> np.ndarray:
        key = hashlib.sha1(json.dumps([seq, {k: op[k] for k in ("box", "p", "n", "mm", "crop", "full") if k in op}], sort_keys=True).encode()).hexdigest()[:16]
        f = self.cache / f"{seq}_{key}.npz"
        if f.exists():
            return np.unpackbits(np.load(f)["m"])[: H * W].reshape(H, W).astype(bool)
        if self.m is None:
            self._load()
        pts = (op.get("p") or []) + (op.get("n") or []); lbl = [1] * len(op.get("p") or []) + [0] * len(op.get("n") or [])
        if op.get("full"):
            x0, y0, x1, y1 = 0, 0, W, H
        elif "crop" in op:
            x0, y0, x1, y1 = op["crop"]
        else:
            xs = [b for b in ([op["box"][0], op["box"][2]] if "box" in op else [])] + [q[0] for q in pts]
            ys = [b for b in ([op["box"][1], op["box"][3]] if "box" in op else [])] + [q[1] for q in pts]
            x0, x1, y0, y1 = min(xs), max(xs), min(ys), max(ys); s = max(x1 - x0, y1 - y0)
            mg = max(48, 0.2 * s) if "box" in op else max(150, 0.4 * s)
            half = max(s / 2 + mg, 160); cx, cy = (x0 + x1) / 2, (y0 + y1) / 2
            x0, x1, y0, y1 = cx - max(half, (x1 - x0) / 2 + mg), cx + max(half, (x1 - x0) / 2 + mg), cy - max(half, (y1 - y0) / 2 + mg), cy + max(half, (y1 - y0) / 2 + mg)
        x0, y0, x1, y1 = int(max(0, x0)), int(max(0, y0)), int(min(W, x1)), int(min(H, y1))
        crop = Image.fromarray(img[y0:y1, x0:x1])
        kw = {}
        if "box" in op:
            b = op["box"]; kw["input_boxes"] = [[[b[0] - x0, b[1] - y0, b[2] - x0, b[3] - y0]]]
        if pts:
            kw["input_points"] = [[[[q[0] - x0, q[1] - y0] for q in pts]]]; kw["input_labels"] = [[lbl]]
        inp = self.p(images=crop, return_tensors="pt", **kw).to(self.dev)
        mm = op.get("mm", 0 if "box" in op else "best")
        # image embeddings are kept per (photo, crop): several prompts on one crop cost one encoder pass
        ef = self.cache / f"emb_{seq}_{x0}_{y0}_{x1}_{y1}.pt"
        with self.torch.no_grad():
            if ef.exists():
                emb = [e.to(self.dev) for e in self.torch.load(ef)]
            else:
                emb = self.m.get_image_embeddings(inp["pixel_values"]); self.torch.save([e.cpu() for e in emb], ef)
            kw2 = {k: v for k, v in inp.items() if k in ("input_points", "input_labels", "input_boxes")}
            out = self.m(image_embeddings=emb, **kw2, multimask_output=(mm != 0 or "box" not in op))
        ms = self.p.post_process_masks(out.pred_masks.cpu(), inp["original_sizes"])[0][0]   # K x h x w
        sc = out.iou_scores.cpu()[0, 0]
        k = int(sc.argmax()) if mm == "best" else int(mm)
        k = min(k, ms.shape[0] - 1)
        full = np.zeros((H, W), bool); full[y0:y1, x0:x1] = ms[k].numpy().astype(bool)
        np.savez_compressed(f, m=np.packbits(full.ravel()))
        return full


def wand(img: np.ndarray, wd: dict) -> np.ndarray:
    """A brightness wand inside my own corridor (no learned model): {"line": [[x, y], ...], "w": corridor px,
    "t": threshold on max(R,G,B), or on max - min (colour) with "feat": "s" (default Otsu inside the corridor), "dark": true = take the dark side (a pole),
    "frac": f (cut at background + f x (object - background) instead of Otsu), "close" / "open": px (default 5 / 5; close 25 bridges a tape band across the tube)}. Keeps the components that touch the drawn centre line; holes filled across the corridor."""
    cm = np.zeros((H, W), np.uint8); cl = np.zeros((H, W), np.uint8); pts = np.array(wd["line"], np.int32)
    cv2.polylines(cm, [pts], False, 1, int(wd.get("w", 80))); cv2.polylines(cl, [pts], False, 1, 3)
    i16 = img.astype(np.int16)
    v = (i16.max(axis=2) - i16.min(axis=2)).astype(np.uint8) if wd.get("feat") == "s" else img.max(axis=2)
    inside = cm.astype(bool)
    t = wd.get("t")
    if t is None:
        t, _ = cv2.threshold(v[inside].reshape(-1, 1), 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        if "frac" in wd:   # blurred tube edges: move the cut from Otsu toward the background level
            vi = v[inside].astype(float); lo, hi = np.median(vi[vi <= t]), np.median(vi[vi > t])
            t = lo + wd["frac"] * (hi - lo) if not wd.get("dark") else hi - wd["frac"] * (hi - lo)
    m = ((v <= t) if wd.get("dark") else (v > t)) & inside
    m = m.astype(np.uint8)
    for op_, k in ((cv2.MORPH_CLOSE, int(wd.get("close", 5))), (cv2.MORPH_OPEN, int(wd.get("open", 5)))):
        if k > 1:
            m = cv2.morphologyEx(m, op_, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k)))
    m = m.astype(bool) & inside
    n, cc = cv2.connectedComponents(m.astype(np.uint8))
    keep = np.unique(cc[(cl > 0) & m]); keep = keep[keep > 0]
    m = np.isin(cc, keep)
    u = m.astype(np.uint8); cs, _ = cv2.findContours(u, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE); cv2.drawContours(u, cs, -1, 1, -1)
    return u.astype(bool) & inside


def paint(seq: str, ops: list, img: np.ndarray, sam) -> np.ndarray:
    lab = np.zeros((H, W), np.uint8)
    for op in ops:
        c = op["c"]
        if "wand" in op:
            m = wand(img, op["wand"])
        elif "poly" in op or "rect" in op or "line" in op:
            m = np.zeros((H, W), np.uint8)
            if "line" in op:
                cv2.polylines(m, [np.array(op["line"], np.int32)], False, 1, int(op.get("w", 10)))
            elif "poly" in op:
                cv2.fillPoly(m, [np.array(op["poly"], np.int32)], 1)
            else:
                x0, y0, x1, y1 = op["rect"]; m[y0:y1, x0:x1] = 1
            m = m.astype(bool)
        else:
            m = sam.mask(img, seq, op)
            if op.get("clip") and "box" in op:
                b = op["box"]; cm = np.zeros_like(m); cm[b[1]:b[3], b[0]:b[2]] = True; m &= cm
            if op.get("fill"):
                u = m.astype(np.uint8); cs, _ = cv2.findContours(u, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE); cv2.drawContours(u, cs, -1, 1, -1); m = u.astype(bool)
        if "within" in op:   # my own corridor: {"line": [[x, y], ...], "w": px} or {"poly": [...]}; the mask is cut to it
            wi = op["within"]; cm = np.zeros((H, W), np.uint8)
            if "line" in wi:
                cv2.polylines(cm, [np.array(wi["line"], np.int32)], False, 1, int(wi.get("w", 40)))
            else:
                cv2.fillPoly(cm, [np.array(wi["poly"], np.int32)], 1)
            m &= cm.astype(bool)
        if "over" in op:
            m &= np.isin(lab, op["over"])
        lab[m] = c
        if op.get("blur"):
            r = int(op["blur"]); k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * r + 1, 2 * r + 1)); u = m.astype(np.uint8)
            edge = (cv2.dilate(u, k) - cv2.erode(u, k)).astype(bool)
            if "over" in op:
                edge &= np.isin(lab, op["over"] + [c])
            lab[edge] = 255
    return lab


def main() -> None:
    ap = argparse.ArgumentParser(); ap.add_argument("cmd", choices=["grid", "zoom", "run", "check"]); ap.add_argument("args", nargs="*")
    ap.add_argument("--half", type=Path, required=True); ap.add_argument("--out", type=Path); ap.add_argument("--prompts", type=Path)
    ap.add_argument("--labels", type=Path); ap.add_argument("--cache", type=Path); ap.add_argument("--cols", type=int, default=2); ap.add_argument("--tw", type=int, default=1100)
    a = ap.parse_intermixed_args()
    load = lambda s: np.asarray(Image.open(a.half / f"{s}.jpg").convert("RGB"))
    if a.cmd == "grid":
        ims = [gridded(Image.fromarray(load(s)).resize((a.tw, int(a.tw * H / W))), scale=W / a.tw) for s in a.args]
        a.out.parent.mkdir(parents=True, exist_ok=True); tile(ims, a.args, a.cols, a.tw).save(a.out, quality=88)
    elif a.cmd == "zoom":
        s, x0, y0, x1, y1 = a.args[0], *map(int, a.args[1:5]); step = int(a.args[5]) if len(a.args) > 5 else 50
        im = Image.fromarray(load(s)[y0:y1, x0:x1]); sc = max(1.0, 1400 / max(im.size)); im = im.resize((int(im.size[0] * sc), int(im.size[1] * sc)), Image.LANCZOS)
        gridded(im, step=step, scale=1 / sc, origin=(x0, y0)).save(a.out, quality=90)
    elif a.cmd == "run":
        pr = json.loads(a.prompts.read_text()); seqs = a.args or sorted(pr); sam = Sam(a.cache); a.labels.mkdir(parents=True, exist_ok=True)
        for s in seqs:
            lab = paint(s, pr[s], load(s), sam); Image.fromarray(lab).save(a.labels / f"{s}.png")
            u, n = np.unique(lab, return_counts=True); print(s, {NAMES[int(k)]: int(v) for k, v in zip(u, n)}, flush=True)
    else:
        ims, labs = [], []
        for s in a.args:
            if ":" in s:   # seq:x0,y0,x1,y1 = a zoomed check
                s, bb = s.split(":"); x0, y0, x1, y1 = map(int, bb.split(","))
            else:
                x0, y0, x1, y1 = 0, 0, W, H
            lab = np.asarray(Image.open(a.labels / f"{s}.png")); o = overlay(load(s), lab)[y0:y1, x0:x1]
            im = Image.fromarray(o)
            if (x1 - x0, y1 - y0) != (W, H):
                im = im.resize((a.tw, int(a.tw * H / W)))
                gridded(im, step=100, scale=(x1 - x0) / a.tw, origin=(x0, y0))
            else:
                im = gridded(im.resize((a.tw, int(a.tw * H / W))), scale=W / a.tw)
            ims.append(im); labs.append(s + " " + " ".join(sorted({NAMES[int(k)] for k in np.unique(lab) if k})))
        a.out.parent.mkdir(parents=True, exist_ok=True); tile(ims, labs, a.cols, a.tw).save(a.out, quality=88)


if __name__ == "__main__":
    main()
