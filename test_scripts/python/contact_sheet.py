"""Contact sheets of real-photo predictions: photo with the predicted mask tinted green, next
to the predictor's crop/body-coordinate overlay.

    .venv/bin/python test_scripts/python/contact_sheet.py IMAGES_DIR PRED_DIR OUT_DIR [--per-sheet 6] [--select random|largest|smallest|none]
"""
from __future__ import annotations
import argparse, json, sys
from pathlib import Path
import numpy as np, cv2

def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("images", type=Path); p.add_argument("pred", type=Path); p.add_argument("out", type=Path)
    p.add_argument("--per-sheet", type=int, default=6); p.add_argument("--select", default="random")
    p.add_argument("--count", type=int, default=18); p.add_argument("--seed", type=int, default=0); p.add_argument("--width", type=int, default=640)
    a = p.parse_args(argv)
    recs = [json.loads(l) for l in open(a.pred / "poses.jsonl")]
    if a.select == "random":
        rng = np.random.default_rng(a.seed); recs = [recs[i] for i in rng.permutation(len(recs))[: a.count]]
    elif a.select == "largest": recs = sorted(recs, key=lambda r: -r["whale_pixels"])[: a.count]
    elif a.select == "smallest": recs = sorted(recs, key=lambda r: r["whale_pixels"])[: a.count]
    elif a.select == "none": recs = [r for r in recs if r.get("status") == "no_whale"][: a.count]
    a.out.mkdir(parents=True, exist_ok=True)
    tiles = []
    for r in recs:
        stem = Path(r["image"]).stem
        img = cv2.imread(str(a.images / f"{stem}.jpg"))
        mask = cv2.imread(str(a.pred / f"{stem}_mask.png"), cv2.IMREAD_UNCHANGED)
        if mask is not None:
            mask = cv2.resize(mask, (img.shape[1], img.shape[0]), interpolation=cv2.INTER_NEAREST) > 0
            img[mask] = (img[mask] * 0.45 + np.array([0, 255, 0]) * 0.55).astype(np.uint8)
        h, w = img.shape[:2]; s = a.width / w
        img = cv2.resize(img, (a.width, round(h * s)))
        over = cv2.imread(str(a.pred / f"{stem}_overlay.jpg"))
        if over is None: over = np.zeros((img.shape[0], a.width, 3), np.uint8)
        else:
            oh, ow = over.shape[:2]; over = cv2.resize(over, (a.width, round(oh * a.width / ow)))
        H = max(img.shape[0], over.shape[0])
        pad = lambda x: np.pad(x, ((0, H - x.shape[0]), (0, 0), (0, 0)))
        tile = np.concatenate([pad(img), pad(over)], 1)
        label = f"{stem[-24:]}  px={r['whale_pixels']}  " + (f"rot_direct_vs_pnp={r.get('rot_direct_vs_pnp_deg', float('nan')):.0f}deg" if r.get("status") != "no_whale" else "NO WHALE")
        cv2.putText(tile, label, (8, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        tiles.append(tile)
    for i in range(0, len(tiles), a.per_sheet):
        chunk = tiles[i: i + a.per_sheet]; W = max(t.shape[1] for t in chunk)
        sheet = np.concatenate([np.pad(t, ((0, 0), (0, W - t.shape[1]), (0, 0))) for t in chunk], 0)
        cv2.imwrite(str(a.out / f"{a.select}_{i // a.per_sheet:02d}.jpg"), sheet, [cv2.IMWRITE_JPEG_QUALITY, 85])
    print("wrote", len(tiles), "tiles to", a.out)
    return 0

if __name__ == "__main__":
    sys.exit(main())
