"""Weak losses on the real crops (data.py). Probabilities p = softmax of the part logits (6 classes).

  background   cross-entropy towards class 0 on the certain-background pixels (far water, labelled not-a-whale blobs,
               black padding), mean over those pixels
  points       cross-entropy towards the clicked class on the clicked fin triangles / fluke discs, mean over them
  not dorsal   (W3c; only where the label map has the value NOTFIN = 6) -log(1 - p(dorsal)) on whale pixels that are
               certainly not dorsal (below the clicked base line / beyond the clicked bases), mean over them
  forbid       per instance with a forbidden class set Z: q = sum of p over Z on the instance region; the mean of
               -log(1 - q) over the instance's k most violating pixels, k = max(32, 10 % of the region). Every pixel of
               the region must be free of Z, so this is a per-pixel truth; the top-k only keeps the many easy pixels from
               drowning the few wrong ones.
  need         per instance with a required class c whose whole region is inside the crop: -log(mean of the k largest
               p_c in the region grown by 8 px), k = max(16, 10 % of the region) for a fluke (a fluke view's fluke is
               rarely under a third of the instance; median 84 %), max(6, 0.4 % of the region) for a dorsal fin (fin
               share of correctly found fins: 25th percentile 0.6 %, median 1.0 %). Says only "some of it is here".
  whale        (W2, only when a certain-whale map is given) per crop: q = p(background) on the certain-whale pixels
               (build_whale.py); the mean of -log(1 - q) over the k most violating of them, k = max(32, 10 %), as forbid.
Total = background + points + forbid + w_need x need + w_whale x whale (each a mean over its pixels / instances / crops).
"""
from __future__ import annotations

import torch
import torch.nn.functional as F

from . import FIN, FLUKE, MAX_INST, NOTFIN

EPS = 1e-4


def weak_loss(logits: torch.Tensor, weak: torch.Tensor, inst: torch.Tensor, cons: torch.Tensor, w_need: float = 0.5,
              k_forbid: float = 0.10, kmin_forbid: int = 32, k_need=None, grow: int = 8, whale: torch.Tensor | None = None,
              w_whale: float = 0.0, need_fin: bool = True):
    k_need = k_need or {FLUKE: (0.10, 16), FIN: (0.004, 6)}
    logits = logits.float()
    logp = F.log_softmax(logits, 1)
    p = logp.exp()
    zero = logits.sum() * 0
    bg = weak == 0
    l_bg = -logp[:, 0][bg].mean() if bg.any() else zero
    pts = (weak == FLUKE) | (weak == FIN)
    if pts.any():
        l_pts = F.nll_loss(logp.permute(0, 2, 3, 1)[pts], weak[pts])
    else:
        l_pts = zero
    nf = weak == NOTFIN
    l_nf = -torch.log((1 - p[:, FIN]).clamp(min=EPS))[nf].mean() if nf.any() else None
    forb, need = [], []
    n_forb_px = 0.0
    for b in range(logits.shape[0]):
        cb = cons[b]
        for i in range(1, MAX_INST + 1):
            if not cb[i].any():
                continue
            R = inst[b] == i
            n = int(R.sum())
            if n < 20:
                continue
            Z = [c for c, col in ((FLUKE, 0), (FIN, 1)) if cb[i, col] > 0]
            if Z:
                q = p[b, Z].sum(0)[R]
                k = min(n, max(kmin_forbid, int(k_forbid * n)))
                top = q.topk(k).values
                forb.append(-torch.log((1 - top).clamp(min=EPS)).mean())
                n_forb_px += float((q > 0.5).sum())
            for c, col in ((FLUKE, 2), (FIN, 3)):
                if cb[i, col] > 0 and (need_fin or c != FIN):
                    S = F.max_pool2d(R[None, None].float(), 2 * grow + 1, 1, grow)[0, 0] > 0
                    v = p[b, c][S]
                    frac, kmin = k_need[c]
                    k = min(int(S.sum()), max(kmin, int(frac * n)))
                    need.append(-torch.log(v.topk(k).values.mean().clamp(min=EPS)))
    l_forb = torch.stack(forb).mean() if forb else zero
    l_need = torch.stack(need).mean() if need else zero
    total = l_bg + l_pts + l_forb + w_need * l_need
    if l_nf is not None:
        total = total + l_nf
    parts = {"w_notfin": float(l_nf.item()) if l_nf is not None else 0.0, "w_bg": float(l_bg.item()), "w_pts": float(l_pts.item()), "w_forbid": float(l_forb.item()), "w_need": float(l_need.item()),
             "w_n_forbid": len(forb), "w_n_need": len(need), "w_forbid_px_over_half": n_forb_px}
    if whale is not None and w_whale > 0:
        wl = []
        for b in range(logits.shape[0]):
            R = whale[b] > 0
            n = int(R.sum())
            if n < 20:
                continue
            q = p[b, 0][R]
            k = min(n, max(kmin_forbid, int(k_forbid * n)))
            wl.append(-torch.log((1 - q.topk(k).values).clamp(min=EPS)).mean())
            parts["w_whale_px_bg"] = parts.get("w_whale_px_bg", 0.0) + float((q > 0.5).sum())
        l_wh = torch.stack(wl).mean() if wl else zero
        total = total + w_whale * l_wh
        parts["w_whale"] = float(l_wh.item())
        parts["w_n_whale"] = len(wl)
    return total, parts
