"""Real training-day crops with weak labels (build.py), and a preview sheet.

    PYTHONPATH=. .venv/bin/python -m model_fine_tuning.parts_weak.data preview [--n 24] [--pool all|clicks|fluke|fin|other|nw]
        -> SHEETS/preview_<pool>.jpg (+ .json legend), never inside the repository

A sample is drawn by its global number g (deterministic: rng(seed, g)) from five pools:
  fluke views (fluke required, fin forbidden where the view says no fin)   0.25
  dorsal-fin views (fin required, fluke forbidden)                         0.25
  other views (fluke forbidden only)                                       0.25
  blobs labelled by eye as not a whale (centre of the crop)               0.15
  instances with the researcher's clicks on 2025-01-02                     0.10
Two crop geometries, as the two ways predict.py runs the model on real photos:
  hr   (p 0.6) a square around the target box, side = long side x U(1.25, 2.5) (log), at least 256 px of the 2560-px
       photo (so never more than 2x up-sampled), warped from the 2560-px jpg with the flags predict.py uses for its
       per-instance windows; the box stays inside the square.
  full (p 0.4) the render recipe's crop: the 2560-px photo reduced to 1280 (INTER_AREA, as the full-frame variant), a
       512 window at scale U(0.5, 1.5) (log) placed with a random target pixel at 0.1-0.9 of the window.
Outside the photo the crop is black (as predict.py pads) and labelled background. Horizontal flip p 0.5, then the
same photometric augmentation as the renders (common.data.photometric, strength 1).
Constraint table per crop: rows = instance ids 0..8, columns [forbid fluke, forbid fin, need fluke, need fin]. The
"need" columns are kept only when the instance's whole region lies inside the crop and inside the photo, and when
its region in the crop has at least 400 px (fluke) / 1500 px (fin): a partial or tiny view proves nothing.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset

from ..common.data import photometric, to_tensor
from . import DAYS, FIN, FLUKE, IGNORE, INDEX, MAPS, MAX_INST, SHEETS, check_train_day

POOLS = ("fluke", "fin", "other", "nw", "clicks")
WEIGHTS = (0.25, 0.25, 0.25, 0.15, 0.10)
MIN_PX = {FLUKE: 400, FIN: 1500}


class RealWeakCrops(Dataset):
    def __init__(self, index: Path = INDEX, crop: int = 512, seed: int = 0, p_hr: float = 0.6, photometric_strength: float = 1.0,
                 weights=WEIGHTS, length: int = 10 ** 9, whale_dir: Path | None = None, maps_dir: Path | None = None):
        idx = json.loads(Path(index).read_text())
        self.maps_dir = Path(maps_dir) if maps_dir else MAPS                  # W3: maps with the whole clicked triangle
        self.whale_dir = Path(whale_dir) if whale_dir else None          # W2: certain-whale maps (build_whale.py); None = W1 exactly
        self.photos = idx["photos"]
        for r in self.photos:
            check_train_day(r["day"])
        self.crop, self.seed, self.p_hr, self.ph, self.length = crop, seed, p_hr, photometric_strength, length
        self.pools = {p: [] for p in POOLS}
        for pi, r in enumerate(self.photos):
            for ii, ins in enumerate(r["inst"]):
                pres, forb = set(ins["present"]), set(ins["forbid"])
                if FLUKE in pres:
                    self.pools["fluke"].append((pi, "inst", ii))
                elif FIN in pres:
                    self.pools["fin"].append((pi, "inst", ii))
                elif FLUKE in forb:
                    self.pools["other"].append((pi, "inst", ii))
                if ins.get("clicked") and r["points"]:
                    self.pools["clicks"].append((pi, "inst", ii))
            for bi, b in enumerate(r["nw"]):
                if b["role"] == "background" and b["px"] > 0:
                    self.pools["nw"].append((pi, "nw", bi))
        names = [p for p, w in zip(POOLS, weights) if w > 0 and self.pools[p]]
        w = np.array([dict(zip(POOLS, weights))[p] for p in names], float)
        self.pool_names, self.pool_p = names, w / w.sum()

    def __len__(self):
        return self.length

    def counts(self) -> dict:
        return {p: len(v) for p, v in self.pools.items()}

    # ---------------------------------------------------------------------------------------------------- geometry
    def window(self, r: dict, box, rng, mode: str):
        """(x0, y0, side) of the square window in the 2560-px frame."""
        x1, y1, x2, y2 = box
        L = max(x2 - x1, y2 - y1, 8)
        cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
        if mode == "hr":
            side = max(256.0, L * float(np.exp(rng.uniform(np.log(1.25), np.log(2.5)))))
            slack_x, slack_y = max(0.0, side - (x2 - x1)) / 2, max(0.0, side - (y2 - y1)) / 2
            cx += rng.uniform(-1, 1) * 0.8 * slack_x
            cy += rng.uniform(-1, 1) * 0.8 * slack_y
            return cx - side / 2, cy - side / 2, side
        s = float(np.exp(rng.uniform(np.log(0.5), np.log(1.5))))
        side = 2.0 * self.crop / s                                    # 512 / s px of the 1280 frame
        px, py = rng.uniform(x1, x2), rng.uniform(y1, y2)
        return px - rng.uniform(0.1, 0.9) * side, py - rng.uniform(0.1, 0.9) * side, side

    def make(self, pi: int, kind: str, j: int, rng, mode: str | None = None):
        r = self.photos[pi]
        day, stem = r["day"], r["stem"]
        box = r["inst"][j]["box"] if kind == "inst" else r["nw"][j]["box"]
        mode = mode or ("hr" if rng.random() < self.p_hr else "full")
        x0, y0, side = self.window(r, box, rng, mode)
        C = self.crop
        a = C / side                                                  # crop px per 2560 px
        M = np.array([[a, 0, -x0 * a], [0, a, -y0 * a]], np.float64)
        bgr = cv2.imread(str(DAYS / day / "images" / f"{stem}.jpg"), cv2.IMREAD_COLOR)
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        H, W = rgb.shape[:2]
        if mode == "full":
            sc = 1280 / max(H, W)
            small = cv2.resize(rgb, (int(round(W * sc)), int(round(H * sc))), interpolation=cv2.INTER_AREA)
            Ms = M.copy(); Ms[:, :2] /= sc
            img = cv2.warpAffine(small, Ms, (C, C), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=0)
        else:                                                         # predict.py's hr window warp
            img = cv2.warpAffine(rgb, M, (C, C), flags=cv2.INTER_AREA if side > C else cv2.INTER_LINEAR,
                                 borderMode=cv2.BORDER_CONSTANT, borderValue=0)
        maps = cv2.imread(str(self.maps_dir / day / f"{stem}.png"), cv2.IMREAD_UNCHANGED)
        weak = cv2.warpAffine(np.ascontiguousarray(maps[..., 0]), M, (C, C), flags=cv2.INTER_NEAREST, borderMode=cv2.BORDER_CONSTANT, borderValue=0)
        inst = cv2.warpAffine(np.ascontiguousarray(maps[..., 1]), M, (C, C), flags=cv2.INTER_NEAREST, borderMode=cv2.BORDER_CONSTANT, borderValue=0)
        # (cv2.imwrite / imread keep the array's channel order: 0 = weak label, 1 = instance map)
        cons = np.zeros((MAX_INST + 1, 4), np.float32)
        wx0, wy0, wx1, wy1 = x0, y0, x0 + side, y0 + side
        for ins in r["inst"]:
            i = ins["id"]
            if i > MAX_INST:
                continue
            cons[i, 0] = float(FLUKE in ins["forbid"])
            cons[i, 1] = float(FIN in ins["forbid"])
            bx0, by0, bx1, by1 = ins["box"]
            inside = bx0 >= wx0 and by0 >= wy0 and bx1 <= wx1 and by1 <= wy1
            if inside and ins["present"]:
                n = int((inst == i).sum())
                cons[i, 2] = float(FLUKE in ins["present"] and n >= MIN_PX[FLUKE])
                cons[i, 3] = float(FIN in ins["present"] and n >= MIN_PX[FIN])
        whale = None
        if self.whale_dir is not None:                                # no random draw: W1's crops stay identical
            wm = cv2.imread(str(self.whale_dir / day / f"{stem}.png"), cv2.IMREAD_UNCHANGED)
            whale = cv2.warpAffine(wm, M, (C, C), flags=cv2.INTER_NEAREST, borderMode=cv2.BORDER_CONSTANT, borderValue=0)
        if rng.random() < 0.5:
            img, weak, inst = img[:, ::-1], weak[:, ::-1], inst[:, ::-1]
            if whale is not None:
                whale = whale[:, ::-1]
        img = photometric(np.ascontiguousarray(img), rng, self.ph)
        out = {"id": f"real:{day}|{stem}|{kind}{j}|{mode}", "image": to_tensor(img),
               "label": torch.from_numpy(np.ascontiguousarray(weak).astype(np.int64)),
               "inst": torch.from_numpy(np.ascontiguousarray(inst)), "cons": torch.from_numpy(cons)}
        if whale is not None:
            out["whale"] = torch.from_numpy(np.ascontiguousarray(whale))
        return out

    def __getitem__(self, g: int):
        rng = np.random.default_rng([self.seed, int(g), 4242])
        pool = self.pool_names[int(rng.choice(len(self.pool_names), p=self.pool_p))]
        pi, kind, j = self.pools[pool][int(rng.integers(len(self.pools[pool])))]
        out = self.make(pi, kind, j, rng)
        out["id"] = f"{out['id']}|{pool}"
        return out


# ------------------------------------------------------------------------------------------------------ preview
COLS = {FLUKE: (230, 60, 60), FIN: (60, 200, 60)}


def preview_tile(it: dict, size: int = 320) -> np.ndarray:
    im = (it["image"].numpy().transpose(1, 2, 0) * 255).clip(0, 255).astype(np.uint8)
    w = it["label"].numpy(); inst = it["inst"].numpy(); cons = it["cons"].numpy()
    ov = im.copy()
    ov[w == 0] = (0.55 * ov[w == 0] + [0, 0, 90]).astype(np.uint8)              # certain background: blue tint
    for c, col in COLS.items():
        ov[w == c] = col
    ov[w == 6] = (0.5 * ov[w == 6] + [127, 70, 0]).astype(np.uint8)            # W3c: whale, not dorsal (orange tint)
    txt = []
    for i in range(1, MAX_INST + 1):
        m = (inst == i).astype(np.uint8)
        if not m.any() or not cons[i].any():
            if m.any():
                cnt, _ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE); cv2.drawContours(ov, cnt, -1, (160, 160, 160), 1)
            continue
        f = cons[i]
        col = (255, 60, 60) if f[2] else (60, 255, 60) if f[3] else (255, 230, 0) if f[0] else (200, 200, 200)
        cnt, _ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE); cv2.drawContours(ov, cnt, -1, col, 2)
        txt.append(f"{i}:" + ("noFL " if f[0] else "") + ("noFIN " if f[1] else "") + ("needFL " if f[2] else "") + ("needFIN" if f[3] else ""))
    ov[inst == 254] = (0.5 * ov[inst == 254] + [70, 70, 0]).astype(np.uint8)
    t = np.hstack([cv2.resize(im, (size, size), interpolation=cv2.INTER_AREA), cv2.resize(ov, (size, size), interpolation=cv2.INTER_NEAREST)])
    t = cv2.cvtColor(t, cv2.COLOR_RGB2BGR)
    for n, s in enumerate(txt[:3]):
        cv2.putText(t, s, (size + 4, 16 + 16 * n), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 0), 3)
        cv2.putText(t, s, (size + 4, 16 + 16 * n), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1)
    return t


def cmd_preview(a) -> None:
    ds = RealWeakCrops(seed=a.seed)
    print(ds.counts())
    rng = np.random.default_rng(a.seed)
    items = []
    for n in range(a.n):
        if a.pool == "all":
            items.append(ds[int(rng.integers(10 ** 8))])
        else:
            pi, kind, j = ds.pools[a.pool][int(rng.integers(len(ds.pools[a.pool])))]
            r2 = np.random.default_rng([a.seed, n, 7])
            it = ds.make(pi, kind, j, r2, mode=a.mode)
            it["id"] += f"|{a.pool}"
            items.append(it)
    tiles = [preview_tile(it) for it in items]
    for k, t in enumerate(tiles):
        cv2.putText(t, str(k + 1), (4, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 3)
        cv2.putText(t, str(k + 1), (4, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
    cols = 3
    while len(tiles) % cols:
        tiles.append(np.full_like(tiles[0], 255))
    rows = [np.hstack([np.pad(t, ((2, 2), (2, 2), (0, 0)), constant_values=255) for t in tiles[i:i + cols]]) for i in range(0, len(tiles), cols)]
    SHEETS.mkdir(parents=True, exist_ok=True)
    out = SHEETS / f"preview_{a.pool}{'_' + a.mode if a.mode else ''}.jpg"
    cv2.imwrite(str(out), np.vstack(rows), [cv2.IMWRITE_JPEG_QUALITY, 85])
    out.with_suffix(".json").write_text(json.dumps([{"tile": k + 1, "id": it["id"],
                                                     "cons": {i: it["cons"][i].tolist() for i in range(MAX_INST + 1) if it["cons"][i].any()}}
                                                    for k, it in enumerate(items)], indent=1))
    print(out)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("preview")
    p.add_argument("--n", type=int, default=24)
    p.add_argument("--pool", default="all", choices=("all",) + POOLS)
    p.add_argument("--mode", default=None, choices=(None, "hr", "full"))
    p.add_argument("--seed", type=int, default=0)
    p.set_defaults(fn=cmd_preview)
    a = ap.parse_args(argv)
    a.fn(a)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
