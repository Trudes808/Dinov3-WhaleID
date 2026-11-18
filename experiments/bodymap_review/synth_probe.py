"""Feasibility probe 3 (nothing of the pipeline is used or changed): how far do SYNTHETIC labels alone get on real photos for
finding the dorsal fin and the fluke? The same probe as click_probe.py (frozen DINOv3 ViT-B/16 patch features, one ridge
regression per keypoint onto a heat spot, 512-px square around the whale = the body-map crop), trained three ways and
tested on the researcher's clicks of a training day:
   synthetic   run05 renders only (keypoints from the shot's meta, heat only where the point is visible)
   real        the clicks of the OTHER training day only
   both        the two together, each with the same total weight

    PYTHONPATH=. .venv/bin/python experiments/bodymap_review/synth_probe.py [--n 1500]
"""
import argparse, json, random, sys
from pathlib import Path
import cv2, numpy as np, torch
ROOT = Path(__file__).resolve().parents[2]; sys.path.insert(0, str(ROOT))
DAYS = ROOT / "experiments" / "days_all"; LAB = ROOT.parent / "Whale-Review-Web" / "data"; SYN = ROOT.parent / "Whale-Image-Data-Factory" / "data" / "renders" / "run05" / "dataset"
ap = argparse.ArgumentParser(); ap.add_argument("--n", type=int, default=1500); ap.add_argument("--lam", type=float, default=30.0); a = ap.parse_args(); R = 512; G = R // 16
from model_fine_tuning.common.backbone import load_backbone, IMAGENET_MEAN, IMAGENET_STD
dev = torch.device("cuda"); vit = load_backbone("vitb16").to(dev).eval()
KP = ["dorsal_fin_tip", "dorsal_fin_front", "dorsal_fin_back", "fluke_tip_left", "fluke_tip_right", "fluke_notch"]
mean = torch.tensor(IMAGENET_MEAN, device=dev).view(1, 3, 1, 1); std = torch.tensor(IMAGENET_STD, device=dev).view(1, 3, 1, 1)
gy, gx = np.mgrid[0:G, 0:G]; cxs = torch.tensor((gx.ravel() + 0.5) * 16, device=dev); cys = torch.tensor((gy.ravel() + 0.5) * 16, device=dev)
def feats(rgb):
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
        x = ((torch.from_numpy(rgb).to(dev).permute(2, 0, 1)[None].float() / 255.0) - mean) / std; f8, f11 = vit.get_intermediate_layers(x, n=[8, 11], reshape=True, norm=True)
    F = torch.cat([f8, f11], 1)[0].float().permute(1, 2, 0).reshape(G * G, -1); return F / (F.norm(dim=1, keepdim=True) + 1e-6)
def heat(p, sig=20.0): return torch.exp(-((cxs - p[0]) ** 2 + (cys - p[1]) ** 2) / (2 * sig ** 2)).float()
class Acc:
    def __init__(s): s.G = torch.zeros(1536, 1536, device=dev, dtype=torch.float64); s.s = torch.zeros(1536, device=dev, dtype=torch.float64); s.b = torch.zeros(1536, device=dev, dtype=torch.float64); s.sy = 0.0; s.n = 0
    def add(s, F, y): F = F.double(); s.G += F.T @ F; s.s += F.sum(0); s.b += F.T @ y.double(); s.sy += float(y.sum()); s.n += len(F)
def solve(parts):
    """parts: [(Acc, weight)] -> (w, mu, ybar)"""
    Gm = sum(w * p.G for p, w in parts); s = sum(w * p.s for p, w in parts); b = sum(w * p.b for p, w in parts); sy = sum(w * p.sy for p, w in parts); n = sum(w * p.n for p, w in parts)
    mu = s / n; wv = torch.linalg.solve(Gm - n * torch.outer(mu, mu) + a.lam * torch.eye(1536, device=dev, dtype=torch.float64), b - mu * sy); return wv.float(), mu.float(), float(sy / n)
def predict(model, F):
    w, mu, b = model; h = ((F - mu) @ w + b).reshape(G, G).cpu().numpy(); j = int(h.argmax()); iy, ix = divmod(j, G)
    y0_, y1_, x0_, x1_ = max(iy - 1, 0), min(iy + 2, G), max(ix - 1, 0), min(ix + 2, G); win = np.clip(h[y0_:y1_, x0_:x1_], 0, None); win = win / max(win.sum(), 1e-9)
    yy, xx = np.mgrid[y0_:y1_, x0_:x1_]; return float(((xx + 0.5) * 16 * win).sum()), float(((yy + 0.5) * 16 * win).sum())
# ---- synthetic
ids = json.loads((SYN / "splits.json").read_text())["train"]; random.Random(0).shuffle(ids); syn = {n: Acc() for n in KP}; used = 0; vis = {n: 0 for n in KP}
for i in ids:
    if used >= a.n: break
    m = json.loads((SYN / "meta" / f"{i}.json").read_text()); v = m.get("visible") or {}
    if m.get("negative") or v.get("visible_pixels", 0) < 800 or not v.get("bbox_xyxy"): continue
    x0, y0, x1, y1 = v["bbox_xyxy"]; L = max(max(x1 - x0, y1 - y0) * 1.6, 160.0); cx, cy = (x0 + x1) / 2, (y0 + y1) / 2; ox, oy = cx - L / 2, cy - L / 2
    img = cv2.imread(str(SYN / "images" / f"{i}.png")); M = np.array([[R / L, 0, -ox * R / L], [0, R / L, -oy * R / L]]); F = feats(cv2.cvtColor(cv2.warpAffine(img, M, (R, R), flags=cv2.INTER_LINEAR), cv2.COLOR_BGR2RGB))
    for n in KP:
        k = m["keypoints"].get(n); y = torch.zeros(G * G, device=dev)
        if k and k.get("state") == "visible":
            p = ((k["uv"][0] - ox) * R / L, (k["uv"][1] - oy) * R / L)
            if 0 <= p[0] < R and 0 <= p[1] < R: y = heat(p); vis[n] += 1
        syn[n].add(F, y)
    used += 1
print(f"synthetic: {used} run05 shots; shots with the point visible: {vis}", flush=True)
# ---- real
real = {}
for day in ("2025-01-02", "2025-03-18"):
    labs = json.loads((LAB / day / "pair_marks.json").read_text()); labs = labs["labels"] if isinstance(labs, dict) else labs; photos = {}
    for l in labs:
        if l.get("verdict") == "skip": continue
        for sd in ("a", "b"):
            kp = (l.get("keypoints") or {}).get(sd) or {}
            if kp and l.get(f"key_{sd}"): photos.setdefault(l[f"key_{sd}"], {}).update(kp)
    rows = []; acc = {n: Acc() for n in KP}
    for key, kp in photos.items():
        stem = key.rsplit("~", 1)[0]; f = DAYS / day / "cse" / f"{stem}_canon.npz"
        if not f.exists(): continue
        x0, y0, s = np.load(f)["x0y0s"]; img = cv2.imread(str(DAYS / day / "images" / f"{stem}.jpg")); H, W = img.shape[:2]; L = 512 / s
        M = np.array([[R / L, 0, -x0 * R / L], [0, R / L, -y0 * R / L]]); F = feats(cv2.cvtColor(cv2.warpAffine(img, M, (R, R), flags=cv2.INTER_AREA if L > R else cv2.INTER_CUBIC), cv2.COLOR_BGR2RGB))
        pts = {n: ((p["x"] * W - x0) * R / L, (p["y"] * H - y0) * R / L) for n, p in kp.items() if n in KP}; pts = {n: p for n, p in pts.items() if 0 <= p[0] < R and 0 <= p[1] < R}
        if not pts: continue
        for n, p in pts.items(): acc[n].add(F, heat(p))
        rows.append({"F": F.cpu(), "pts": pts})
    real[day] = (rows, acc); print(day, len(rows), "photos with clicks", flush=True)
def q(v): v = np.array(v, float); return f"{np.median(v):.2f}"
days = list(real); res = {k: [] for k in ("synthetic", "real", "both")}
for te, tr in ((days[0], days[1]), (days[1], days[0])):
    models = {"synthetic": {n: solve([(syn[n], 1.0)]) for n in KP}, "real": {n: solve([(real[tr][1][n], 1.0)]) for n in KP if real[tr][1][n].n},
              "both": {n: solve([(syn[n], 1.0), (real[tr][1][n], syn[n].n / max(real[tr][1][n].n, 1))]) for n in KP if real[tr][1][n].n}}
    for name, md in models.items():
        for r in real[te][0]:
            F = r["F"].to(dev); res[name].append({"pts": r["pts"], "pred": {n: predict(m, F) for n, m in md.items()}})
print("\ntested on the researcher's clicks (two training days; 'real' and 'both' never saw the day they are tested on); 512-px crop")
print(f"   {'trained on':10s} | fin tip: median error (fin bases), within 0.25 | fin back base | fin front base | fluke notch (spans) | fluke tips as a pair: median error, left / right wrong | fin centre off (fin bases), over one fin base")
for name, rr in res.items():
    line = f"   {name:10s} |"
    for n in ("dorsal_fin_tip", "dorsal_fin_back", "dorsal_fin_front", "fluke_notch"):
        e = []
        for r in rr:
            p, d = r["pts"], r["pred"]
            if n not in p or n not in d: continue
            ks = ("dorsal_fin_front", "dorsal_fin_back") if n.startswith("dorsal") else ("fluke_tip_left", "fluke_tip_right")
            if not all(k in p for k in ks): continue
            unit = np.hypot(*(np.array(p[ks[0]]) - np.array(p[ks[1]])))
            if unit >= 5: e.append(np.hypot(d[n][0] - p[n][0], d[n][1] - p[n][1]) / unit)
        line += f" {q(e)}, {np.mean(np.array(e) < 0.25):.0%} (n {len(e)}) |"
    sw, et = [], []
    for r in rr:
        p, d = r["pts"], r["pred"]
        if not all(k in p and k in d for k in ("fluke_tip_left", "fluke_tip_right")): continue
        Lc, Rc, Lp, Rp = (np.array(x) for x in (p["fluke_tip_left"], p["fluke_tip_right"], d["fluke_tip_left"], d["fluke_tip_right"])); span = np.hypot(*(Lc - Rc))
        if span < 5: continue
        st = np.hypot(*(Lp - Lc)) + np.hypot(*(Rp - Rc)); cr = np.hypot(*(Lp - Rc)) + np.hypot(*(Rp - Lc)); sw.append(cr < st); et.append(min(st, cr) / 2 / span)
    line += f" {q(et)}, within 0.25 {np.mean(np.array(et) < 0.25):.0%}, wrong way {np.mean(sw):.0%} (n {len(sw)}) |"
    Bx = []
    for r in rr:
        p, d = r["pts"], r["pred"]
        if not all(k in p and k in d for k in KP[:3]): continue
        T = np.array([p[k] for k in KP[:3]]); Pd = np.array([d[k] for k in KP[:3]]); base = np.hypot(*(T[1] - T[2]))
        if base >= 5: Bx.append(np.hypot(*(Pd.mean(0) - T.mean(0))) / base)
    line += f" {q(Bx)}, {np.mean(np.array(Bx) > 1):.0%}"
    print(line)
