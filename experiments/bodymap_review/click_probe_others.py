"""Feasibility probe 2 (nothing of the pipeline is used or changed): does a dorsal-fin-tip finder trained ONLY on the main
whale's clicks also fire on the other whales the researcher marked in the same photos? Frozen DINOv3 patch features of
the whole photo (long side 1344), ridge regression onto a heat spot at the main whale's fin tip, trained on one
training day and tested on the other. For every marked other whale with a fin tip: is one of the five strongest
separate peaks of the heat map on it?

    PYTHONPATH=. .venv/bin/python experiments/bodymap_review/click_probe_others.py
"""
import json, sys
from pathlib import Path
import cv2, numpy as np, torch
ROOT = Path(__file__).resolve().parents[2]; sys.path.insert(0, str(ROOT))
DAYS = ROOT / "experiments" / "days_all"; LAB = ROOT.parent / "Whale-Review-Web" / "data"; LONG = 1344
from model_fine_tuning.common.backbone import load_backbone, IMAGENET_MEAN, IMAGENET_STD
dev = torch.device("cuda"); vit = load_backbone("vitb16").to(dev).eval()
mean = torch.tensor(IMAGENET_MEAN, device=dev).view(1, 3, 1, 1); std = torch.tensor(IMAGENET_STD, device=dev).view(1, 3, 1, 1); data = {}
for day in ("2025-01-02", "2025-03-18"):
    labs = json.loads((LAB / day / "pair_marks.json").read_text()); labs = labs["labels"] if isinstance(labs, dict) else labs; photos, others = {}, {}
    for l in labs:
        if l.get("verdict") == "skip": continue
        for sd in ("a", "b"):
            key = l.get(f"key_{sd}"); kp = (l.get("keypoints") or {}).get(sd) or {}; ow = (l.get("other_whales") or {}).get(sd)
            if key and kp: photos.setdefault(key.rsplit("~", 1)[0], {}).update(kp)
            if key and ow is not None and (kp or ow): others[key.rsplit("~", 1)[0]] = ow
    rows = []
    for stem, kp in photos.items():
        if "dorsal_fin_tip" not in kp: continue
        img = cv2.imread(str(DAYS / day / "images" / f"{stem}.jpg")); H, W = img.shape[:2]; sc = LONG / max(H, W); w, h = int(round(W * sc / 16)) * 16, int(round(H * sc / 16)) * 16
        im = cv2.cvtColor(cv2.resize(img, (w, h), interpolation=cv2.INTER_AREA), cv2.COLOR_BGR2RGB)
        with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
            x = ((torch.from_numpy(im).to(dev).permute(2, 0, 1)[None].float() / 255.0) - mean) / std; f8, f11 = vit.get_intermediate_layers(x, n=[8, 11], reshape=True, norm=True)
        F = torch.cat([f8, f11], 1)[0].float().permute(1, 2, 0).cpu().numpy().astype(np.float32); F /= np.linalg.norm(F, axis=2, keepdims=True) + 1e-6
        P = lambda p: (p["x"] * w, p["y"] * h); base = None
        if "dorsal_fin_front" in kp and "dorsal_fin_back" in kp: base = float(np.hypot(*(np.array(P(kp["dorsal_fin_front"])) - np.array(P(kp["dorsal_fin_back"])))))
        rows.append({"stem": stem, "F": F, "tip": P(kp["dorsal_fin_tip"]), "base": base, "others": [P(o["dorsal_fin_tip"]) for o in (others.get(stem) or []) if "dorsal_fin_tip" in o]})
    data[day] = rows; print(day, len(rows), "photos with a fin tip,", sum(len(r["others"]) for r in rows), "other whales with a fin tip", flush=True)
def heat(shape, p, sig=20.0):
    gy, gx = np.mgrid[0:shape[0], 0:shape[1]]; return np.exp(-(((gx + 0.5) * 16 - p[0]) ** 2 + ((gy + 0.5) * 16 - p[1]) ** 2) / (2 * sig ** 2)).astype(np.float32).ravel()
def fit(rows):
    X = np.concatenate([r["F"].reshape(-1, r["F"].shape[2]) for r in rows]); y = np.concatenate([heat(r["F"].shape[:2], r["tip"]) for r in rows]); mu = X.mean(0)
    A = torch.from_numpy(X - mu).to(dev); wv = torch.linalg.solve(A.T @ A + 30.0 * torch.eye(A.shape[1], device=dev), A.T @ torch.from_numpy(y - y.mean()).to(dev)).cpu().numpy(); return wv, mu, float(y.mean())
def peaks(hm, k=5, r=4):
    hm = hm.copy(); out = []
    for _ in range(k):
        j = int(hm.argmax()); iy, ix = divmod(j, hm.shape[1]); out.append(((ix + 0.5) * 16, (iy + 0.5) * 16, float(hm[iy, ix]))); hm[max(iy - r, 0):iy + r + 1, max(ix - r, 0):ix + r + 1] = -9
    return out
days = list(data); main_hit = []; other = []
for tr, te in ((days[0], days[1]), (days[1], days[0])):
    wv, mu, b = fit(data[tr])
    for r in data[te]:
        hm = (r["F"] - mu) @ wv + b; pk = peaks(hm); tol = max(32.0, 0.5 * (r["base"] or 64.0))
        main_hit.append(np.hypot(pk[0][0] - r["tip"][0], pk[0][1] - r["tip"][1]) <= tol)
        for o in r["others"]:
            d = [np.hypot(p[0] - o[0], p[1] - o[1]) for p in pk]; j = int(np.argmin(d)); other.append({"hit": bool(d[j] <= tol), "rank": j + 1, "rel": pk[j][2] / max(pk[0][2], 1e-6), "dist_to_main": float(np.hypot(o[0] - r["tip"][0], o[1] - r["tip"][1]) / tol)})
print(f"\nwhole photo at long side {LONG}, tested on the day not trained on")
print(f"   the strongest peak is on the main whale's fin tip: {np.mean(main_hit):.0%} (n {len(main_hit)})")
hit = [o for o in other if o["hit"]]
print(f"   other whales with a clicked fin tip: {len(other)} | one of the five strongest peaks is on it: {len(hit)} | as the 2nd peak {sum(o['rank'] == 2 for o in hit)}, 3rd {sum(o['rank'] == 3 for o in hit)}, 4th-5th {sum(o['rank'] >= 4 for o in hit)}, taken for the 1st {sum(o['rank'] == 1 for o in hit)}"
      + (f" | its height against the strongest peak, median {np.median([o['rel'] for o in hit]):.2f}" if hit else ""))
