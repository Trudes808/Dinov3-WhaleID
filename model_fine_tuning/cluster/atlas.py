"""Per-photo canonical atlas: for every visible surface cell, the mean DINOv3 patch feature and
mean colour of the pixels that map to it, keyed by (cell, side).

    python -m model_fine_tuning.cluster.atlas --images DIR --canon-dir PRED_DIR --out OUT [--canon-checkpoint CK] [--k 512]

Inputs are the `<stem>_canon.npz` files written by canon.predict (signed canonical points at
crop resolution, the use mask, the resolved side map, the mirror-disagreement map and the crop
transform). Two feature sources are stored per photo: "dino" = the frozen DINOv3 ViT-B/16
(last block and block 8, patch tokens at stride 16) and, with --canon-checkpoint, "canon" =
the same backbone after the canonical-surface fine-tune. Output per photo: cells, sides,
pixel counts, mean mirror disagreement, features per source, LAB colour, plus the (H, W)
cell-id map of the crop for the explanation heatmaps.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np
import torch

from ..common import geometry, paths
from ..common.backbone import DinoFeatures, normalize_image
from ..common.data import to_tensor, warp_image, warp_params
from .cells import assign_cells, build_cells, cells_path, load_cells, save_cells


@torch.no_grad()
def patch_features(backbone: DinoFeatures, crop: np.ndarray, device) -> list[np.ndarray]:
    x = normalize_image(to_tensor(np.ascontiguousarray(crop))[None].to(device))
    with torch.autocast("cuda", dtype=torch.bfloat16, enabled=device.type == "cuda"):
        feats, _ = backbone(x)
    return [f[0].float().cpu().numpy() for f in feats]          # each (C, h, w)


def aggregate(cell_map: np.ndarray, side_map: np.ndarray, use: np.ndarray, disagree_m: np.ndarray, crop: np.ndarray,
              feats: dict[str, list[np.ndarray]], midline: np.ndarray, patch: int = 16, min_pixels: int = 12) -> dict:
    ys, xs = np.nonzero(use)
    cell = cell_map[ys, xs]
    side = np.where(midline[cell], 0, side_map[ys, xs]).astype(np.int8)
    key = cell.astype(np.int64) * 3 + (side + 1)
    uniq, inv, counts = np.unique(key, return_inverse=True, return_counts=True)
    keep = counts >= min_pixels
    lab = cv2.cvtColor(crop, cv2.COLOR_RGB2LAB).astype(np.float32)[ys, xs]
    out = {"cell": (uniq // 3).astype(np.int32)[keep], "side": (uniq % 3 - 1).astype(np.int8)[keep], "count": counts[keep].astype(np.int32)}
    def mean_by_key(values):
        acc = np.zeros((len(uniq), values.shape[1]), np.float64)
        np.add.at(acc, inv, values)
        return (acc / counts[:, None])[keep].astype(np.float32)
    out["colour_lab"] = mean_by_key(lab)
    out["disagree_m"] = mean_by_key(disagree_m[ys, xs][:, None])[:, 0]
    py, px = ys // patch, xs // patch
    for name, layers in feats.items():
        for li, f in enumerate(layers):
            out[f"feat_{name}_{li}"] = mean_by_key(f[:, py, px].T)
    return out


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--images", type=Path, required=True)
    p.add_argument("--canon-dir", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--canon-checkpoint", type=Path, default=None)
    p.add_argument("--k", type=int, default=512)
    p.add_argument("--layers", type=int, nargs="*", default=[8, 11])
    p.add_argument("--upsample", type=int, default=1, help="run the backbone on the crop enlarged by this factor (finer patch grid)")
    args = p.parse_args(argv)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    table = geometry.load_folded_table(paths.FOLDED_TABLE)
    cp = cells_path(args.k)
    if not cp.exists():
        save_cells(build_cells(table, args.k), cp)
    cells = load_cells(cp)
    backbones = {"dino": DinoFeatures("vitb16", layers=args.layers).to(device).eval()}
    if args.canon_checkpoint:
        from ..canon.model import WhaleCanonModel

        model, _ = WhaleCanonModel.from_checkpoint(args.canon_checkpoint, device)
        model.backbone.layers = list(args.layers)
        backbones["canon"] = model.backbone.eval()
    args.out.mkdir(parents=True, exist_ok=True)
    done = 0
    for npz in sorted(args.canon_dir.glob("*_canon.npz")):
        stem = npz.name[: -len("_canon.npz")]
        img_path = args.images / f"{stem}.jpg"
        if not img_path.exists():
            continue
        z = np.load(npz)
        x0, y0, s = z["x0y0s"]
        use, side, signed = z["use"], z["side"], z["signed_mm"].astype(np.float32) / 1000.0
        C = use.shape[0]
        image = cv2.cvtColor(cv2.imread(str(img_path), cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB)
        up = args.upsample
        crop = warp_image(image, warp_params(x0, y0, s * up), (C * up, C * up))
        folded = signed.copy(); folded[..., 1] = np.abs(folded[..., 1])
        cell_map = assign_cells(folded, cells).astype(np.int32)
        cell_map[~use] = -1
        if up > 1:
            nearest = lambda m: cv2.resize(m.astype(np.int32), (C * up, C * up), interpolation=cv2.INTER_NEAREST)
            cell_map, side_up, use_up = nearest(cell_map), nearest(side).astype(np.int8), nearest(use.astype(np.uint8)) > 0
            dis_up = cv2.resize(z["disagree_mm"].astype(np.float32) / 1000.0, (C * up, C * up), interpolation=cv2.INTER_LINEAR)
        else:
            side_up, use_up, dis_up = side, use, z["disagree_mm"].astype(np.float32) / 1000.0
        feats = {name: patch_features(bb, crop, device) for name, bb in backbones.items()}
        rec = aggregate(cell_map, side_up, use_up, dis_up, crop, feats, cells["midline"])
        rec.update({"cell_map": cell_map.astype(np.int16), "side_map": side_up, "x0y0s": np.array([x0, y0, s * up])})
        np.savez_compressed(args.out / f"{stem}.npz", **rec)
        done += 1
        if done % 50 == 0:
            print(f"{done} photos", flush=True)
    (args.out / "meta.json").write_text(json.dumps({"k": cells["k"], "layers": args.layers, "sources": list(backbones), "photos": done, "upsample": args.upsample}, indent=1))
    print(f"atlas: {done} photos -> {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
