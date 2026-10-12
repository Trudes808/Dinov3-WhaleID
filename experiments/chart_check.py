import json, numpy as np, cv2, sys
from pathlib import Path
sys.path.insert(0, ".")
from model_fine_tuning.cluster.chart import sift_score
R = Path("experiments/real_2024-12-18"); S = Path("/tmp/claude-1072903897/-home-sat3737-NOAA-whales/34554fa0-8387-4ab5-a3b5-3af7725c5bb3/scratchpad")
lab = json.loads((R / "labels_from_lightroom.json").read_text())["labels"]; ts = json.loads((R / "timestamps.json").read_text())
stems = sorted(f.stem for f in (R / "charts").glob("*.npz")); by = {}
for s in stems: by.setdefault(lab[s][0], []).append(s)
tiles = []
for l in ("G04A01", "G02A03", "G03A03", "G03A01", "G01A01"):
    row = []
    for s in sorted(by.get(l, []), key=lambda s: ts[s]["time"])[:8]:
        z = np.load(R / "charts" / f"{s}.npz"); img = cv2.resize(cv2.imread(str(R / "charts" / "charts" / f"{s}.jpg")), (256, 256))
        fin = cv2.resize(z["fin"].astype(np.uint8), (256, 256), interpolation=cv2.INTER_NEAREST) > 0
        cnt, _ = cv2.findContours(fin.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE); cv2.drawContours(img, cnt, -1, (0, 255, 255), 1)
        cv2.putText(img, f"{l} {'L' if z['view_side'] < 0 else 'R'}{'m' if z['mirrored'] else ''} kp{len(z['kp_xy'])}", (4, 16), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 0), 1, cv2.LINE_AA)
        cv2.putText(img, ts[s]["datetime"][-8:], (4, 250), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1, cv2.LINE_AA); row.append(img)
    if row: row += [np.zeros((256, 256, 3), np.uint8)] * (8 - len(row)); tiles.append(np.concatenate(row, 1))
cv2.imwrite(str(S / "chart_sheet.jpg"), np.concatenate(tiles, 0), [cv2.IMWRITE_JPEG_QUALITY, 85])
def draw(sa, sb, name):
    za, zb = dict(np.load(R / "charts" / f"{sa}.npz")), dict(np.load(R / "charts" / f"{sb}.npz"))
    ia, ib = cv2.imread(str(R / "charts" / "charts" / f"{sa}.jpg")), cv2.imread(str(R / "charts" / "charts" / f"{sb}.jpg"))
    score, inl = sift_score(za, zb)
    knn = cv2.BFMatcher(cv2.NORM_L2).knnMatch(za["kp_desc"], zb["kp_desc"], k=2)
    good = [m for m, n in (p for p in knn if len(p) == 2) if m.distance < 0.8 * n.distance]
    pa = np.float32([za["kp_xy"][m.queryIdx] for m in good]); pb = np.float32([zb["kp_xy"][m.trainIdx] for m in good])
    mask = cv2.estimateAffinePartial2D(pa, pb, method=cv2.RANSAC, ransacReprojThreshold=12.0, maxIters=2000, confidence=0.99)[1] if len(good) >= 4 else None
    canvas = np.concatenate([ia, ib], 1)
    for k, (p, q) in enumerate(zip(pa, pb)):
        ok = bool(mask[k]) if mask is not None else False
        cv2.line(canvas, (int(p[0]), int(p[1])), (int(q[0]) + ia.shape[1], int(q[1])), (0, 255, 0) if ok else (0, 0, 255), 2 if ok else 1)
    cv2.putText(canvas, f"{name}: {lab[sa][0]} vs {lab[sb][0]} dt={abs(ts[sa]['time']-ts[sb]['time']):.0f}s ratio matches={len(good)} affine inliers={inl} score={score:.2f}", (8, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2, cv2.LINE_AA)
    cv2.imwrite(str(S / f"sift_{name}.jpg"), cv2.resize(canvas, (1280, 640)), [cv2.IMWRITE_JPEG_QUALITY, 85]); print(name, sa[-8:], sb[-8:], "ratio matches", len(good), "inliers", inl, "score", round(score, 2))
side_of = lambda s: int(np.load(R / "charts" / f"{s}.npz")["view_side"])
g4 = [s for s in sorted(by["G04A01"], key=lambda s: ts[s]["time"]) if side_of(s) >= 0]
pairs = [(a, b) for a in g4 for b in g4 if 120 < abs(ts[a]["time"] - ts[b]["time"]) < 3600]
if pairs: draw(*pairs[len(pairs) // 2], "same_whale")
a = [s for s in by["G03A01"] if side_of(s) >= 0]; b = [s for s in by["G03A03"] if side_of(s) >= 0]
if a and b: draw(a[0], b[0], "different_whales")
