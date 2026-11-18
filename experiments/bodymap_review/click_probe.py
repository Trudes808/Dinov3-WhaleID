"""Feasibility probe, nothing of the pipeline is used or changed: can the dorsal fin and the fluke be FOUND straight from the
photo with the researcher's clicks we already have, without a body map? Frozen DINOv3 ViT-B/16 patch features of the
same square region the body-map network sees (its crop, at 896 px instead of 512), one ridge regression per keypoint
onto a small heat spot around the click, trained on one training day and tested on the other (both ways).

    PYTHONPATH=. .venv/bin/python experiments/bodymap_review/click_probe.py OUT.json [--res 896]

Scores, on the photos of the day the probe did not see: distance of the predicted point from the click in units of the
clicked fin base (front base to back base) for fin points and of the clicked tip-to-tip span for fluke points; then
the dorsal-fin box the three fin points give against the box of the clicks, next to the body map's own cut-out.
"""
import argparse, json, sys
from pathlib import Path
import cv2, numpy as np, torch
ROOT = Path(__file__).resolve().parents[2]; sys.path.insert(0, str(ROOT))
DAYS = ROOT / "experiments" / "days_all"; LAB = ROOT.parent / "Whale-Review-Web" / "data"
ap = argparse.ArgumentParser(); ap.add_argument("out", type=Path); ap.add_argument("--res", type=int, default=896); ap.add_argument("--days", default="2025-01-02,2025-03-18"); ap.add_argument("--lam", type=float, default=30.0)
a = ap.parse_args(); R = a.res; G = R // 16
from model_fine_tuning.common.backbone import load_backbone, IMAGENET_MEAN, IMAGENET_STD
dev = torch.device("cuda"); vit = load_backbone("vitb16").to(dev).eval()
KP = ["dorsal_fin_tip", "dorsal_fin_front", "dorsal_fin_back", "fluke_tip_left", "fluke_tip_right", "fluke_notch"]
mean = torch.tensor(IMAGENET_MEAN, device=dev).view(1, 3, 1, 1); std = torch.tensor(IMAGENET_STD, device=dev).view(1, 3, 1, 1)
data = {}
for day in a.days.split(","):
    labs = json.loads((LAB / day / "pair_marks.json").read_text()); labs = labs["labels"] if isinstance(labs, dict) else labs; photos = {}
    for l in labs:
        if l.get("verdict") == "skip": continue
        for sd in ("a", "b"):
            kp = (l.get("keypoints") or {}).get(sd) or {}
            if kp and l.get(f"key_{sd}"): photos.setdefault(l[f"key_{sd}"], {}).update(kp)
    rows = []
    for key, kp in photos.items():
        stem = key.rsplit("~", 1)[0]; f = DAYS / day / "cse" / f"{stem}_canon.npz"
        if not f.exists(): continue
        x0, y0, s = np.load(f)["x0y0s"]; img = cv2.imread(str(DAYS / day / "images" / f"{stem}.jpg")); H, W = img.shape[:2]; L = 512 / s
        M = np.array([[R / L, 0, -x0 * R / L], [0, R / L, -y0 * R / L]]); crop = cv2.cvtColor(cv2.warpAffine(img, M, (R, R), flags=cv2.INTER_AREA if L > R else cv2.INTER_CUBIC), cv2.COLOR_BGR2RGB)
        with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
            x = ((torch.from_numpy(crop).to(dev).permute(2, 0, 1)[None].float() / 255.0) - mean) / std
            f8, f11 = vit.get_intermediate_layers(x, n=[8, 11], reshape=True, norm=True)
        F = torch.cat([f8, f11], 1)[0].float().permute(1, 2, 0).reshape(G * G, -1).cpu().numpy().astype(np.float32); F /= np.linalg.norm(F, axis=1, keepdims=True) + 1e-6
        pts = {n: ((p["x"] * W - x0) * R / L, (p["y"] * H - y0) * R / L) for n, p in kp.items() if n in KP}
        pts = {n: p for n, p in pts.items() if 0 <= p[0] < R and 0 <= p[1] < R}
        if pts: rows.append({"key": key, "F": F, "pts": pts, "s": float(s)})
    data[day] = rows; print(day, len(rows), "photos", flush=True)
gy, gx = np.mgrid[0:G, 0:G]; cxs = (gx.ravel() + 0.5) * 16; cys = (gy.ravel() + 0.5) * 16
def heat(p, sig=20.0): return np.exp(-((cxs - p[0]) ** 2 + (cys - p[1]) ** 2) / (2 * sig ** 2)).astype(np.float32)
def fit(rows, n):
    X = np.concatenate([r["F"] for r in rows if n in r["pts"]]); y = np.concatenate([heat(r["pts"][n]) for r in rows if n in r["pts"]]); mu = X.mean(0); Xc = X - mu
    A = torch.from_numpy(Xc).to(dev); w = torch.linalg.solve(A.T @ A + a.lam * torch.eye(A.shape[1], device=dev), A.T @ torch.from_numpy(y - y.mean()).to(dev)).cpu().numpy()
    return w, mu, float(y.mean())
def predict(model, F):
    w, mu, b = model; h = ((F - mu) @ w + b).reshape(G, G); j = int(h.argmax()); iy, ix = divmod(j, G)
    y0_, y1_, x0_, x1_ = max(iy - 1, 0), min(iy + 2, G), max(ix - 1, 0), min(ix + 2, G); win = np.clip(h[y0_:y1_, x0_:x1_], 0, None); win = win / max(win.sum(), 1e-9)
    yy, xx = np.mgrid[y0_:y1_, x0_:x1_]; return float(((xx + 0.5) * 16 * win).sum()), float(((yy + 0.5) * 16 * win).sum()), float(h.max())
days = list(data); res = []
for tr, te in ((days[0], days[1]), (days[1], days[0])):
    models = {n: fit(data[tr], n) for n in KP if sum(n in r["pts"] for r in data[tr]) >= 15}
    for r in data[te]:
        out = {"day": te, "key": r["key"], "pts": r["pts"], "pred": {n: predict(m, r["F"]) for n, m in models.items()}, "res": R}; res.append(out)
a.out.write_text(json.dumps(res))
def q(v): v = np.array(v, float); return f"{np.percentile(v, 25):.2f} / {np.median(v):.2f} / {np.percentile(v, 75):.2f} (n {len(v)})"
print(f"\ntested on the day not trained on, both ways; region = the body-map crop at {R} px")
for n in KP:
    e = []
    for r in res:
        p = r["pts"]
        if n not in p or n not in r["pred"]: continue
        if n.startswith("dorsal"):
            if not ("dorsal_fin_front" in p and "dorsal_fin_back" in p): continue
            unit = np.hypot(*(np.array(p["dorsal_fin_front"]) - np.array(p["dorsal_fin_back"])))
        else:
            if not ("fluke_tip_left" in p and "fluke_tip_right" in p): continue
            unit = np.hypot(*(np.array(p["fluke_tip_left"]) - np.array(p["fluke_tip_right"])))
        if unit < 8: continue
        e.append(np.hypot(r["pred"][n][0] - p[n][0], r["pred"][n][1] - p[n][1]) / unit)
    if e: print(f"   {n:17s} error in units of the {'fin base' if n.startswith('dorsal') else 'fluke span'}, quartiles {q(e)} | within 0.25: {np.mean(np.array(e) < 0.25):.0%} | within 0.5: {np.mean(np.array(e) < 0.5):.0%}")
# either fluke tip: the nearer of the two predictions to each click, and left / right
sw = []; et = []
for r in res:
    p, d = r["pts"], r["pred"]
    if not all(n in p and n in d for n in ("fluke_tip_left", "fluke_tip_right")): continue
    Lc, Rc = np.array(p["fluke_tip_left"]), np.array(p["fluke_tip_right"]); Lp, Rp = np.array(d["fluke_tip_left"][:2]), np.array(d["fluke_tip_right"][:2]); span = np.hypot(*(Lc - Rc))
    if span < 8: continue
    straight = np.hypot(*(Lp - Lc)) + np.hypot(*(Rp - Rc)); crossed = np.hypot(*(Lp - Rc)) + np.hypot(*(Rp - Lc)); sw.append(crossed < straight); et.append(min(straight, crossed) / 2 / span)
if sw: print(f"   fluke tips taken as a pair: mean error {q(et)} of the span | left and right the wrong way round {np.mean(sw):.0%} (n {len(sw)})")
B = []
for r in res:
    p, d = r["pts"], r["pred"]
    if not all(n in p and n in d for n in KP[:3]): continue
    T = np.array([p[n] for n in KP[:3]]); Pd = np.array([d[n][:2] for n in KP[:3]]); base = np.hypot(*(T[1] - T[2]))
    if base < 8: continue
    B.append((np.hypot(*(Pd.mean(0) - T.mean(0))) / base, max(np.ptp(Pd[:, 0]), np.ptp(Pd[:, 1])) / max(np.ptp(T[:, 0]), np.ptp(T[:, 1]), 1e-6)))
B = np.array(B); print(f"   the fin from the three predicted points: centre off by {q(B[:, 0])} fin bases, more than one fin base off {np.mean(B[:, 0] > 1):.0%} | size against the clicked fin {q(B[:, 1])}")
