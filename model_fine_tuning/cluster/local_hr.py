"""Local matching on the hi-res instance charts: marks, notches and barnacle patches at pixel level.

Two scorers, both restricted to candidate pairs (the top-K neighbours of every chart under the global embeddings) and
both aligned in chart coordinates (the charts are pose-normalised, so a mark should sit near the same place in both):
  lightglue  ALIKED keypoints (inside the eroded whale mask, 1024-px chart) matched with LightGlue; the pair value is
             log1p of the matches that land within ALIGN_R x chart size of each other (the WildFusion recipe plus the
             chart-space check).
  hrlocal    the hi-res DINO tokens (cluster/dino_hr.py extract: PCA-64 descriptors on the 64 x 64 token grid, whale
             tokens only): mutual nearest neighbours whose grid positions agree within ALIGN_R, value = mean of the
             top-k match similarities (zero-padded: few matches score low).

    python -m model_fine_tuning.cluster.local_hr --parts DAY/parts_hr --labels labels.json [--features DAY/dino_hr/features.npz] --part fin --max-charts 200
evaluates both on all pairs of a sample of one part's charts (pair AUC), for a quick check of the channels.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np
import torch

ALIGN_R = {"body": 0.5, "fin": 0.3, "fluke": 0.3}


class AlikedLightGlue:
    def __init__(self, device, size: int = 1024, max_kp: int = 1024, filter_threshold: float = 0.1, erode_frac: float = 0.01):
        import kornia.feature as KF

        self.device, self.size, self.erode = device, size, erode_frac
        self.ext = KF.ALIKED.from_pretrained("aliked-n16", max_num_keypoints=max_kp, detection_threshold=0.0, nms_radius=2).to(device).eval()
        self.lg = KF.LightGlue("aliked").to(device).eval()
        self.lg.conf.depth_confidence = -1; self.lg.conf.width_confidence = -1; self.lg.conf.filter_threshold = filter_threshold
        self.size_t = torch.tensor([[float(size), float(size)]], device=device)

    @torch.no_grad()
    def extract(self, chart: Path, npz: Path) -> dict:
        img = cv2.cvtColor(cv2.imread(str(chart), cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, (self.size, self.size), interpolation=cv2.INTER_AREA)
        mask = np.load(npz)["whale"].astype(np.uint8)
        mask = cv2.resize(mask, (self.size, self.size), interpolation=cv2.INTER_NEAREST)
        k = max(int(self.erode * self.size), 1)
        mask = cv2.erode(mask, np.ones((2 * k + 1, 2 * k + 1), np.uint8)) > 0
        x = torch.from_numpy(img).permute(2, 0, 1)[None].float().div(255.0).to(self.device)
        f = self.ext(x)[0]
        kp = f.keypoints; ij = kp.round().long().clamp(0, self.size - 1).cpu().numpy()
        keep = torch.from_numpy(mask[ij[:, 1], ij[:, 0]]).to(self.device)
        return {"kp": kp[keep], "desc": f.descriptors[keep]}

    @torch.no_grad()
    def match(self, fa: dict, fb: dict, part: str) -> tuple[int, int, float]:
        """(matches, aligned matches, sum of aligned match scores)."""
        if len(fa["kp"]) < 4 or len(fb["kp"]) < 4:
            return 0, 0, 0.0
        data = {"image0": {"keypoints": fa["kp"][None], "descriptors": fa["desc"][None], "image_size": self.size_t},
                "image1": {"keypoints": fb["kp"][None], "descriptors": fb["desc"][None], "image_size": self.size_t}}
        out = self.lg(data)
        m = out["matches"][0]; s = out["scores"][0]
        if len(m) == 0:
            return 0, 0, 0.0
        d = (fa["kp"][m[:, 0]] - fb["kp"][m[:, 1]]).norm(dim=1)
        aligned = d <= ALIGN_R[part] * self.size
        return int(len(m)), int(aligned.sum()), float(s[aligned].sum())


def lightglue_pair_values(matcher: AlikedLightGlue, charts: list[tuple[Path, Path]], pairs: np.ndarray, part: str, log_every: int = 5000, all_stats: bool = False) -> np.ndarray:
    """log1p(aligned matches) for each (i, j) in pairs (indices into charts); all_stats: (n, 3) = log1p(all), log1p(aligned), score sum."""
    feats = [matcher.extract(c, n) for c, n in charts]
    out = np.zeros((len(pairs), 3), np.float32)
    for n, (i, j) in enumerate(pairs):
        total, aligned, ssum = matcher.match(feats[i], feats[j], part)
        out[n] = (np.log1p(total), np.log1p(aligned), ssum)
        if log_every and n % log_every == 0 and n:
            print(f"    lightglue {part}: {n}/{len(pairs)} pairs", flush=True)
    return out if all_stats else out[:, 1]


@torch.no_grad()
def dino_local_pair_values(local: np.ndarray, mask: np.ndarray, grid: int, pairs: np.ndarray, part: str, device, topk: int = 20, min_sim: float = 0.0) -> np.ndarray:
    """local (n, N, 64) fp16 unit descriptors, mask (n, N) bool: mean of the top-k aligned mutual-NN similarities per pair."""
    pos = np.stack(np.divmod(np.arange(grid * grid), grid), 1).astype(np.float32)          # (N, 2) row, col
    radius = ALIGN_R[part] * grid
    cache: dict[int, tuple[torch.Tensor, torch.Tensor]] = {}

    def get(i):
        if i not in cache:
            m = mask[i]
            cache[i] = (torch.from_numpy(local[i][m].astype(np.float32)).to(device), torch.from_numpy(pos[m]).to(device))
        return cache[i]
    out = np.zeros(len(pairs), np.float32)
    for n, (i, j) in enumerate(pairs):
        la, pa = get(i); lb, pb = get(j)
        if len(la) < 4 or len(lb) < 4:
            continue
        S = la @ lb.T
        best_b = S.argmax(1); best_a = S.argmax(0)
        rows = torch.arange(len(la), device=device)
        mutual = best_a[best_b] == rows
        r = rows[mutual]; c = best_b[mutual]
        if len(r) == 0:
            continue
        d = (pa[r] - pb[c]).norm(dim=1)
        sims = S[r, c][(d <= radius) & (S[r, c] >= min_sim)]
        if len(sims) == 0:
            continue
        top = torch.topk(sims, min(topk, len(sims))).values
        out[n] = float(top.sum() / topk)
    return out


def main(argv=None) -> int:
    from sklearn.metrics import roc_auc_score

    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--parts", type=Path, required=True); p.add_argument("--labels", type=Path, required=True); p.add_argument("--features", type=Path, default=None)
    p.add_argument("--part", default="fin"); p.add_argument("--max-charts", type=int, default=200); p.add_argument("--device", default="cuda"); p.add_argument("--max-kp", type=int, default=1024); p.add_argument("--skip-lightglue", action="store_true")
    a = p.parse_args(argv)
    device = torch.device(a.device if torch.cuda.is_available() else "cpu")
    day = a.parts.parent.name; L = json.loads(a.labels.read_text())["days"][day]["photos"]
    charts = []
    for f in sorted((a.parts / "charts").glob(f"*__{a.part}.jpg")):
        key = f.stem.rsplit("__", 1)[0]; base = key.split("__")[-1]
        if "~" in base:
            base, inst = base.rsplit("~", 1)
            if inst != "0":
                continue
        ident = L.get(base)
        if ident and not ident.split("_")[1].startswith("O"):
            charts.append((key, f, a.parts / f"{key}__{a.part}.npz", ident))
    charts = charts[: a.max_charts]
    ident = np.array([c[3] for c in charts]); iu = np.triu_indices(len(charts), 1); pairs = np.stack(iu, 1); y = ident[iu[0]] == ident[iu[1]]
    print(f"{day} {a.part}: {len(charts)} charts, {len(pairs)} pairs, {int(y.sum())} positive", flush=True)
    import time
    t0 = time.time()
    if not a.skip_lightglue:
        lg = lightglue_pair_values(AlikedLightGlue(device, max_kp=a.max_kp), [(c[1], c[2]) for c in charts], pairs, a.part, all_stats=True)
        print(f"  lightglue ({a.max_kp} kp): AUC all {roc_auc_score(y, lg[:, 0]):.3f} aligned {roc_auc_score(y, lg[:, 1]):.3f} score-sum {roc_auc_score(y, lg[:, 2]):.3f}  ({time.time() - t0:.0f} s, {1000 * (time.time() - t0) / len(pairs):.1f} ms/pair)  median aligned matches same {np.median(np.expm1(lg[y, 1])):.0f} / different {np.median(np.expm1(lg[~y, 1])):.0f}")
    if a.features:
        z = np.load(a.features); stems = list(z["stems"]); parts = list(z["parts"])
        row = {(s, pt): i for i, (s, pt) in enumerate(zip(stems, parts))}
        idx = np.array([row[(c[0], a.part)] for c in charts])
        t0 = time.time()
        hl = dino_local_pair_values(z["local"][idx], z["mask"][idx], int(z["grid"]), pairs, a.part, device)
        G = z["glob"][idx]; g = (G @ G.T)[iu]
        print(f"  hrlocal: AUC {roc_auc_score(y, hl):.3f} ({time.time() - t0:.0f} s); hr global AUC {roc_auc_score(y, g):.3f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
