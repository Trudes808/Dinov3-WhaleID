"""Time-contrastive identity head: learn, from one day's photos and their timestamps alone,
a projection of the per-cell DINOv3 features under which the same body spot looks alike on
the same whale and unlike on another. Positive = the same (cell, side) in a photo taken within
`pos_s` seconds on the same camera (almost surely the same whale); negatives = the same cell in
photos at least `neg_s` apart (a different encounter, almost surely a different whale). No
identity labels are used. The head is small (768 -> 512 -> dim), trained with InfoNCE.

    python -m model_fine_tuning.cluster.identity --atlas ATLAS --timestamps TS --out ATLAS_ID [--feature feat_dino_1] \
        [--train-cameras "Canon EOS 7D Mark II"] [--epochs 30]

Writes a copy of the atlas with an extra feature source "id" (the projected, unit-norm
features) so match.py / cluster.py / explain.py use it as any other source, plus head.pt.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch


def train_head(F: np.ndarray, P: np.ndarray, key: np.ndarray, t: np.ndarray, cam: np.ndarray, train_mask: np.ndarray,
               pos_s: float, neg_s: float, dim: int, epochs: int, seed: int, device) -> torch.nn.Module:
    rng = np.random.default_rng(seed); torch.manual_seed(seed)
    dt = np.abs(t[:, None] - t[None, :]); same_cam = cam[:, None] == cam[None, :]
    by_key = {}
    for idx in np.nonzero(train_mask[P])[0]:
        by_key.setdefault(int(key[idx]), []).append(idx)
    triplets = []
    for ids in by_key.values():
        ids = np.array(ids)
        for a in ids:
            pos = ids[(dt[P[a], P[ids]] <= pos_s) & same_cam[P[a], P[ids]] & (P[ids] != P[a])]
            neg = ids[dt[P[a], P[ids]] >= neg_s]
            if len(pos) and len(neg) >= 4:
                triplets.append((a, pos, neg))
    print(f"identity head: {len(triplets)} anchors (positives <= {pos_s:.0f} s, negatives >= {neg_s / 60:.0f} min) from {int(train_mask.sum())} photos", flush=True)
    head = torch.nn.Sequential(torch.nn.Linear(F.shape[1], 512), torch.nn.GELU(), torch.nn.Linear(512, dim)).to(device)
    opt = torch.optim.AdamW(head.parameters(), lr=1e-3, weight_decay=1e-4)
    Ft = torch.from_numpy(F).to(device)
    for epoch in range(epochs):
        rng.shuffle(triplets); losses = []
        for b in range(0, len(triplets), 256):
            batch = triplets[b: b + 256]
            anc = torch.tensor([x[0] for x in batch]); pos = torch.tensor([rng.choice(x[1]) for x in batch])
            neg = torch.tensor(np.stack([rng.choice(x[2], 16, replace=len(x[2]) < 16) for x in batch]))
            za = torch.nn.functional.normalize(head(Ft[anc]), dim=1); zp = torch.nn.functional.normalize(head(Ft[pos]), dim=1)
            zn = torch.nn.functional.normalize(head(Ft[neg.reshape(-1)]), dim=1).reshape(len(batch), 16, -1)
            logits = torch.cat([(za * zp).sum(1, keepdim=True), torch.einsum("bd,bkd->bk", za, zn)], 1) / 0.1
            loss = torch.nn.functional.cross_entropy(logits, torch.zeros(len(batch), dtype=torch.long, device=device))
            opt.zero_grad(); loss.backward(); opt.step(); losses.append(loss.item())
        if epoch % 10 == 9:
            print(f"  epoch {epoch + 1}: loss {np.mean(losses):.4f}", flush=True)
    return head.eval()


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--atlas", type=Path, required=True)
    p.add_argument("--timestamps", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--feature", default="feat_dino_1")
    p.add_argument("--train-cameras", nargs="*", default=None, help="restrict training to these camera models (generalisation check)")
    p.add_argument("--epochs", type=int, default=30)
    p.add_argument("--dim", type=int, default=128)
    p.add_argument("--pos-s", type=float, default=10.0)
    p.add_argument("--neg-s", type=float, default=3600.0)
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args(argv)
    files = sorted(f for f in args.atlas.glob("*.npz")); stems = [f.stem for f in files]
    ts = json.loads(args.timestamps.read_text())
    t = np.array([ts[s]["time"] for s in stems]); cam = np.array([ts[s]["camera"] for s in stems])
    P, Cc, S, F = [], [], [], []
    for i, f in enumerate(files):
        z = np.load(f); P.append(np.full(len(z["cell"]), i)); Cc.append(z["cell"]); S.append(z["side"]); F.append(z[args.feature])
    P, Cc, S, F = np.concatenate(P), np.concatenate(Cc), np.concatenate(S), np.concatenate(F).astype(np.float32)
    F /= np.maximum(np.linalg.norm(F, axis=1, keepdims=True), 1e-6)
    key = Cc.astype(np.int64) * 3 + (S.astype(np.int64) + 1)
    train_mask = np.ones(len(stems), bool) if not args.train_cameras else np.isin(cam, args.train_cameras)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    head = train_head(F, P, key, t, cam, train_mask, args.pos_s, args.neg_s, args.dim, args.epochs, args.seed, device)
    with torch.no_grad():
        Z = torch.nn.functional.normalize(head(torch.from_numpy(F).to(device)), dim=1).cpu().numpy().astype(np.float32)
    args.out.mkdir(parents=True, exist_ok=True)
    start = 0
    for i, f in enumerate(files):
        z = dict(np.load(f)); k = len(z["cell"])
        z["feat_id"] = Z[start: start + k]; start += k
        np.savez_compressed(args.out / f.name, **z)
    meta = json.loads((args.atlas / "meta.json").read_text()); meta.update({"identity_head": {"feature": args.feature, "train_cameras": args.train_cameras, "dim": args.dim, "pos_s": args.pos_s, "neg_s": args.neg_s}})
    (args.out / "meta.json").write_text(json.dumps(meta, indent=1))
    torch.save({"state": head.state_dict(), "feature": args.feature, "dim": args.dim}, args.out / "head.pt")
    print("wrote", args.out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
