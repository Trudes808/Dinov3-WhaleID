"""Local token descriptors at native resolution, tiled: every token sees 16 chart pixels of the finest chart available.

    python -m model_fine_tuning.cluster.local_feats --day-dir DAY --out DAY/local_xr/features_v2.npz \
        [--checkpoint experiments/dino_hr/checkpoints/best.pt | --local-checkpoint experiments/local_v1/checkpoints/best.pt | --frozen]

Fin and fluke charts come from parts_xr (3072 px, native resolution for most photos), body charts from parts_hr (1536 px).
A chart is covered by 1024-px tiles (stride 768, only tiles that hold whale pixels); DINOv3 ViT-B/16 runs on each tile at
native scale, so a token is 16 chart pixels (4x finer on the fin / fluke than the 1024-px pass over the 1536-px chart);
where tiles overlap, the token farther from its tile border wins. Descriptors: the token head of a cluster/train_local.py
model (128-d, trained for marks), or the last-block tokens of a cluster/dino_hr.py backbone / the frozen backbone reduced
to 64-d by a per-day PCA at signature time. Only whale tokens are kept (ragged storage):
  features_v2.npz: stems, parts, chart_size (n), tok_offsets (n + 1), tok_desc (total, D) fp16, tok_xy (total, 2) chart px
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np
import torch

from ..common.backbone import normalize_image

PATCH = 16
PARTS = ("body", "fin", "fluke")


def load_backbone(args, device):
    """Returns (token_fn, dim): token_fn(x in [0,1] (B,3,T,T)) -> (B, T/16, T/16, D) descriptors (unit norm for the trained head)."""
    if args.local_checkpoint:
        from .train_local import LocalDescModel

        model, _ = LocalDescModel.from_checkpoint(args.local_checkpoint, device)
        return (lambda x: model.tokens(x)), model.config["dim"], "local"
    from .dino_hr import DinoHRModel

    if args.frozen:
        model = DinoHRModel(1, unfreeze_blocks=0, pretrained=True).to(device).eval()
    else:
        model, _ = DinoHRModel.from_checkpoint(args.checkpoint, device)

    def fn(x):
        feats, _ = model.backbone(normalize_image(x))
        return feats[-1].permute(0, 2, 3, 1)
    return fn, model.backbone.embed_dim, "backbone"


def tiles_for(size: int, tile: int, stride: int) -> list[int]:
    if size <= tile:
        return [0]
    starts = list(range(0, size - tile + 1, stride))
    if starts[-1] != size - tile:
        starts.append(size - tile)
    return starts


@torch.no_grad()
def chart_tokens(token_fn, img: np.ndarray, mask: np.ndarray, device, tile: int = 1024, stride: int = 768, min_frac: float = 0.3, batch: int = 4):
    """(desc (n, D) fp16, xy (n, 2) chart px) of the whale tokens of a chart (img (S, S, 3) RGB uint8, mask (S, S) bool)."""
    S = img.shape[0]
    if S < tile:                                                                   # pad small charts to one tile
        pad = tile - S; img = cv2.copyMakeBorder(img, 0, pad, 0, pad, cv2.BORDER_REFLECT); mask = np.pad(mask, ((0, pad), (0, pad)))
        S = tile
    starts = tiles_for(S, tile, stride)
    g = tile // PATCH
    best = {}                                                                      # (row, col) in chart token grid -> (border distance, desc)
    jobs = []
    for y0 in starts:
        for x0 in starts:
            m = mask[y0:y0 + tile, x0:x0 + tile]
            if m.mean() < 0.002:
                continue
            jobs.append((y0, x0))
    for b0 in range(0, len(jobs), batch):
        chunk = jobs[b0:b0 + batch]
        x = torch.from_numpy(np.stack([img[y0:y0 + tile, x0:x0 + tile] for y0, x0 in chunk])).permute(0, 3, 1, 2).float().div(255.0).to(device)
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=device.type == "cuda"):
            T = token_fn(x).float().cpu().numpy()                                 # (B, g, g, D)
        for (y0, x0), t in zip(chunk, T):
            m = mask[y0:y0 + tile, x0:x0 + tile].reshape(g, PATCH, g, PATCH).mean((1, 3)) >= min_frac
            rows, cols = np.nonzero(m)
            for r, c in zip(rows, cols):
                key = (y0 // PATCH + r, x0 // PATCH + c)
                border = min(r, c, g - 1 - r, g - 1 - c)
                if key not in best or border > best[key][0]:
                    best[key] = (border, t[r, c])
    if not best:
        return np.zeros((0, 1), np.float16), np.zeros((0, 2), np.float32)
    keys = sorted(best)
    desc = np.stack([best[k][1] for k in keys]).astype(np.float16)
    xy = np.array([[(k[1] + 0.5) * PATCH, (k[0] + 0.5) * PATCH] for k in keys], np.float32)
    return desc, xy


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--day-dir", type=Path, required=True); p.add_argument("--out", type=Path, required=True)
    p.add_argument("--checkpoint", type=Path, default=None); p.add_argument("--local-checkpoint", type=Path, default=None); p.add_argument("--frozen", action="store_true")
    p.add_argument("--xr-tag", default="parts_xr"); p.add_argument("--hr-tag", default="parts_hr"); p.add_argument("--parts", default="body,fin,fluke")
    p.add_argument("--tile", type=int, default=1024); p.add_argument("--stride", type=int, default=768); p.add_argument("--max-charts", type=int, default=None)
    a = p.parse_args(argv)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    token_fn, dim, kind = load_backbone(a, device)
    parts = [x for x in a.parts.split(",") if x]
    charts = []
    for part in parts:
        src = a.day_dir / (a.hr_tag if part == "body" else a.xr_tag)
        if not (src / "index.jsonl").exists():
            src = a.day_dir / a.hr_tag
        for f in sorted((src / "charts").glob(f"*__{part}.jpg")):
            key = f.stem.rsplit("__", 1)[0]
            if (src / f"{key}__{part}.npz").exists():
                charts.append((key, part, f, src / f"{key}__{part}.npz"))
    if a.max_charts:
        charts = charts[: a.max_charts]
    stems, pts, sizes, offs, descs, xys = [], [], [], [0], [], []
    import time
    t0 = time.time()
    for n, (key, part, jpg, npz) in enumerate(charts):
        img = cv2.cvtColor(cv2.imread(str(jpg), cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB); mask = np.load(npz)["whale"]
        d, xy = chart_tokens(token_fn, img, mask, device, a.tile, a.stride)
        stems.append(key); pts.append(part); sizes.append(img.shape[0]); descs.append(d); xys.append(xy); offs.append(offs[-1] + len(d))
        if n % 200 == 0 and n:
            print(f"  {n}/{len(charts)} charts, {offs[-1]} tokens, {time.time() - t0:.0f} s", flush=True)
    a.out.parent.mkdir(parents=True, exist_ok=True)
    D = max((d.shape[1] for d in descs if len(d)), default=dim)
    np.savez(a.out, stems=np.array(stems), parts=np.array(pts), chart_size=np.array(sizes), tok_offsets=np.array(offs),
             tok_desc=np.concatenate([d for d in descs if len(d)]) if any(len(d) for d in descs) else np.zeros((0, D), np.float16),
             tok_xy=np.concatenate([x for x in xys if len(x)]) if any(len(x) for x in xys) else np.zeros((0, 2), np.float32), kind=kind, token_px=PATCH)
    print(f"{a.day_dir.name}: {len(charts)} charts, {offs[-1]} whale tokens ({kind}, {D}-d) -> {a.out}  ({time.time() - t0:.0f} s)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
