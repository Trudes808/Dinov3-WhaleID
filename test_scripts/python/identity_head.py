"""Can an identity-specific feature be learned from time alone? Per surface cell, the DINOv3
feature of a photo is projected by a small head trained with InfoNCE where the positive is the
same cell in a photo taken within `pos_s` seconds on the same camera (almost surely the same
whale) and the negatives are the same cell in photos more than `neg_s` apart (almost surely a
different whale, or at least a different encounter). No labels are used for training. The
labels score the result as pair AUC, with the honest subset = pairs 10 s to 1 h apart within
the same group (never used as positives or negatives).

    .venv/bin/python test_scripts/python/identity_head.py ATLAS_DIR TIMESTAMPS LABELS [--feature feat_dino_1] [--epochs 30]
"""
from __future__ import annotations
import argparse, json, sys
from pathlib import Path
import numpy as np, torch
from sklearn.metrics import roc_auc_score

p = argparse.ArgumentParser(); p.add_argument("atlas", type=Path); p.add_argument("timestamps", type=Path); p.add_argument("labels", type=Path)
p.add_argument("--feature", default="feat_dino_1"); p.add_argument("--epochs", type=int, default=30); p.add_argument("--dim", type=int, default=128)
p.add_argument("--pos-s", type=float, default=10.0); p.add_argument("--neg-s", type=float, default=3600.0); p.add_argument("--seed", type=int, default=0)
args = p.parse_args()
torch.manual_seed(args.seed); rng = np.random.default_rng(args.seed)
files = sorted(args.atlas.glob("*.npz")); stems = [f.stem for f in files]
ts = json.loads(args.timestamps.read_text()); lab = json.loads(args.labels.read_text())["labels"]
t = np.array([ts[s]["time"] for s in stems]); cam = np.array([ts[s]["camera"] for s in stems]); gt = np.array([lab[s][0] for s in stems]); grp = np.array([g[:3] for g in gt])
# table of (photo, cell, side, feature)
P, Cc, S, F = [], [], [], []
for i, f in enumerate(files):
    z = np.load(f); P.append(np.full(len(z["cell"]), i)); Cc.append(z["cell"]); S.append(z["side"]); F.append(z[args.feature])
P, Cc, S, F = np.concatenate(P), np.concatenate(Cc), np.concatenate(S), np.concatenate(F).astype(np.float32)
F /= np.maximum(np.linalg.norm(F, axis=1, keepdims=True), 1e-6)
n = len(stems); dt = np.abs(t[:, None] - t[None, :]); same_cam = cam[:, None] == cam[None, :]
print(f"{n} photos, {len(F)} (photo, cell, side) entries, {F.shape[1]}-d")

def pair_scores(Z: np.ndarray) -> np.ndarray:
    """Mean cosine over co-visible (cell, side-compatible) entries, per photo pair."""
    key = Cc.astype(np.int64) * 3 + (S.astype(np.int64) + 1)
    out = np.full((n, n), np.nan); cnt = np.zeros((n, n))
    acc = np.zeros((n, n))
    by_cell = {}
    for idx in range(len(key)):
        by_cell.setdefault(int(Cc[idx]), []).append(idx)
    for c, ids in by_cell.items():
        ids = np.array(ids)
        if len(ids) < 2: continue
        sim = Z[ids] @ Z[ids].T; side = S[ids]
        ok = (side[:, None] == side[None, :]) | (side[:, None] == 0) | (side[None, :] == 0)
        ii, jj = np.nonzero(ok & ~np.eye(len(ids), dtype=bool))
        np.add.at(acc, (P[ids[ii]], P[ids[jj]]), sim[ii, jj]); np.add.at(cnt, (P[ids[ii]], P[ids[jj]]), 1)
    m = cnt >= 8
    out[m] = acc[m] / cnt[m]
    return out

def report(name: str, score: np.ndarray):
    iu = np.triu_indices(n, 1); a = score[iu]; ok = np.isfinite(a)
    same = (gt[:, None] == gt[None, :])[iu]; sg = (grp[:, None] == grp[None, :])[iu]; d = dt[iu]
    def auc(mask):
        m = ok & mask; return roc_auc_score(same[m], a[m]) if m.sum() > 20 and same[m].any() and (~same[m]).any() else float("nan")
    honest = sg & (d > 10) & (d < 3600)
    print(f"  {name:28s} AUC all {auc(np.ones_like(ok)):.3f} | same group {auc(sg):.3f} | honest (same group, 10s<dt<1h) {auc(honest):.3f} (n={int((ok & honest).sum())}) | G02 {auc(((grp == 'G02')[:, None] & (grp == 'G02')[None, :])[iu]):.3f}")

print("== baselines"); report("raw cosine", pair_scores(F))
# unsupervised whitening (PCA to dim, unit variance)
mu = F.mean(0); U, s_, Vt = np.linalg.svd(F - mu, full_matrices=False)
W = Vt[: args.dim].T / (s_[: args.dim] / np.sqrt(len(F)))[None, :]
Zw = (F - mu) @ W; Zw /= np.maximum(np.linalg.norm(Zw, axis=1, keepdims=True), 1e-6)
report(f"PCA-whitened {args.dim}d", pair_scores(Zw))
# time-contrastive head
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
head = torch.nn.Sequential(torch.nn.Linear(F.shape[1], 512), torch.nn.GELU(), torch.nn.Linear(512, args.dim)).to(device)
opt = torch.optim.AdamW(head.parameters(), lr=1e-3, weight_decay=1e-4)
Ft = torch.from_numpy(F).to(device)
key = Cc.astype(np.int64) * 3 + (S.astype(np.int64) + 1)
by_key = {}
for idx in range(len(key)): by_key.setdefault(int(key[idx]), []).append(idx)
# anchors with at least one positive and 4 negatives
triplets = []
for k, ids in by_key.items():
    ids = np.array(ids)
    for a in ids:
        pos = ids[(dt[P[a], P[ids]] <= args.pos_s) & same_cam[P[a], P[ids]] & (P[ids] != P[a])]
        neg = ids[dt[P[a], P[ids]] >= args.neg_s]
        if len(pos) and len(neg) >= 4: triplets.append((a, pos, neg))
print(f"== time-contrastive head: {len(triplets)} anchors with positives (<= {args.pos_s:.0f} s) and negatives (>= {args.neg_s/60:.0f} min)")
for epoch in range(args.epochs):
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
    if epoch % 10 == 9 or epoch == args.epochs - 1:
        with torch.no_grad():
            Z = torch.nn.functional.normalize(head(Ft), dim=1).cpu().numpy()
        print(f"  epoch {epoch + 1}: loss {np.mean(losses):.3f}"); report("time-contrastive head", pair_scores(Z))
