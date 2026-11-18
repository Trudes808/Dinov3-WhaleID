"""By-eye sheets of the body map on real photos (written to a folder OUTSIDE the repo: they show photos).

    PYTHONPATH=. .venv/bin/python experiments/bodymap_review/bodymap_sheet.py STATS.json OUT_DIR --pick multi|twofin|group|dorsal1 [--n 24] [--seed 0]

One row per photo, three panels of the region the body-map network saw (plus a margin):
  1 the photo   2 the whale instances (one colour each; hatched = dropped by the not-a-whale filter)
  3 the body map: colour = position along the body (blue = fluke end, green = mid back, red = head), white outline =
    what the cut-out code takes for the dorsal fin, magenta outline = for the fluke, arrow = where the map says the
    head is, R / L = the flank the map says is showing
"""
import argparse, json, random, sys
from pathlib import Path
import cv2, numpy as np
ROOT = Path(__file__).resolve().parents[2]; sys.path.insert(0, str(ROOT))
DAYS = ROOT / "experiments" / "days_all"
COL = [(60, 60, 255), (255, 220, 0), (0, 230, 255), (80, 255, 80), (255, 80, 255)]
ap = argparse.ArgumentParser(); ap.add_argument("stats"); ap.add_argument("out", type=Path); ap.add_argument("--pick", default="multi"); ap.add_argument("--n", type=int, default=24)
ap.add_argument("--seed", type=int, default=0); ap.add_argument("--rows", type=int, default=6); ap.add_argument("--panel", type=int, default=300)
a = ap.parse_args(); a.out.mkdir(parents=True, exist_ok=True)
from model_fine_tuning.common import geometry, paths
from model_fine_tuning.cluster.parts import split_instances
t = geometry.load_folded_table(paths.FOLDED_TABLE); X0, LEN = float(t["x_min_m"]), float(t["length_m"])
rows = [r for r in json.load(open(a.stats)) if "inst" in r and r.get("blobs", 0) > 0]
for r in rows: r["n"] = len([x for x in r["inst"] if not x.get("not_whale")])
PICK = {"multi": lambda r: r["n"] >= 2,
        "twofin": lambda r: r["n"] == 1 and any(x["fin_patches"] >= 2 for x in r["inst"] if not x.get("not_whale")),
        "group": lambda r: r["n"] == 1 and r.get("gtype") in ("CG", "MCE", "MC", "Pair", "Trio"),
        "dorsal1": lambda r: r["n"] == 1 and any(x.get("vc") and x["vc"][1] >= 0.9 and x["vc"][0] <= 0.1 for x in r["inst"])}
sel = [r for r in rows if PICK[a.pick](r)]; random.Random(a.seed).shuffle(sel); sel = sel[:a.n]
P = a.panel; legend = []
for si in range(0, len(sel), a.rows):
    strips = []
    for ri, r in enumerate(sel[si:si + a.rows]):
        d = DAYS / r["day"]; z = np.load(d / "cse" / f"{r['stem']}_canon.npz"); x0, y0, s = z["x0y0s"]; use = z["use"]; side = z["side"]; signed = z["signed_mm"].astype(np.float32) / 1000.0
        u = (signed[..., 0] - X0) / LEN; masks = split_instances(use, 0.1, u=u)
        img = cv2.imread(str(d / "images" / f"{r['stem']}.jpg")); H, W = img.shape[:2]
        side_px = 512 / s; m = 0.12 * side_px; cx0, cy0 = x0 - m, y0 - m; L = side_px + 2 * m                       # region of the photo shown (2560-px coordinates)
        M = np.array([[P / L, 0, -cx0 * P / L], [0, P / L, -cy0 * P / L]]); photo = cv2.warpAffine(img, M, (P, P), flags=cv2.INTER_AREA if L > P else cv2.INTER_CUBIC, borderValue=(40, 40, 40))
        Mc = np.array([[P / L / s, 0, (x0 - cx0) * P / L], [0, P / L / s, (y0 - cy0) * P / L]])                             # 512 crop -> panel
        warp = lambda mk: cv2.warpAffine(mk.astype(np.uint8), Mc, (P, P), flags=cv2.INTER_NEAREST) > 0
        grey = cv2.cvtColor(cv2.cvtColor(photo, cv2.COLOR_BGR2GRAY), cv2.COLOR_GRAY2BGR); p2 = (grey * 0.6).astype(np.uint8); p3 = (grey * 0.45).astype(np.uint8)
        hue = cv2.warpAffine((np.clip(u, 0, 1) * 120).astype(np.uint8), Mc, (P, P), flags=cv2.INTER_NEAREST)                # 0 red ... 120 blue (OpenCV hue): flipped below
        hsv = np.zeros((P, P, 3), np.uint8); hsv[..., 0] = 120 - hue; hsv[..., 1] = 255; hsv[..., 2] = 255; rain = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
        cap = []
        for k, mk in enumerate(masks):
            info = r["inst"][k] if k < len(r["inst"]) else {}; w = warp(mk); c = COL[k % len(COL)]
            p2[w] = (0.45 * p2[w] + 0.55 * np.array(c)).astype(np.uint8)
            if info.get("not_whale"):
                p2[w & ((np.add.outer(np.arange(P), np.arange(P)) % 6) < 2)] = (0, 0, 0)
            p3[w] = rain[w]
            fin = warp(mk & (u > 0.36) & (u < 0.54) & (signed[..., 2] > 0.9)); fl = warp(mk & (u < 0.12) & (np.abs(signed[..., 1]) > 0.5))
            for mm, cc in ((fin, (255, 255, 255)), (fl, (255, 0, 255))):
                cs, _ = cv2.findContours(mm.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE); cv2.drawContours(p3, [q for q in cs if cv2.contourArea(q) > 12], -1, cc, 1)
            ys, xs = np.nonzero(w)
            if len(xs):
                cx, cy = int(xs.mean()), int(ys.mean()); h = np.radians(info.get("head", 0.0)); ln = 0.18 * P
                cv2.arrowedLine(p3, (cx, cy), (int(cx + ln * np.cos(h)), int(cy + ln * np.sin(h))), (255, 255, 255), 2, tipLength=0.35)
                cv2.putText(p3, {1: "R", -1: "L", 0: "?"}[info.get("side", 0)], (cx + 6, cy - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
                cv2.putText(p2, str(k), (cx, cy), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            vc = info.get("vc"); cap.append(f"{k}:{'/'.join(sorted(info.get('parts', []))) or '-'}" + (f" vc f{vc[0]:.1f} d{vc[1]:.1f}" if vc else ""))
        tag = f"{si // a.rows + 1}.{ri + 1}"; legend.append({"tag": tag, "day": r["day"], "stem": r["stem"], "gtype": r.get("gtype"), "n": r["n"], "blobs": r["blobs"]})
        strip = np.hstack([photo, p2, p3]); bar = np.zeros((20, strip.shape[1], 3), np.uint8)
        cv2.putText(bar, f"{tag}  {r['day']} {r.get('gtype') or '-'}  inst {r['n']} blobs {r['blobs']}  x{s:.1f}  mirror {r.get('flip_m') or 0:.1f} m  | " + "  ".join(cap), (4, 14), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (255, 255, 255), 1)
        strips.append(np.vstack([bar, strip]))
    cv2.imwrite(str(a.out / f"{a.pick}_{si // a.rows + 1:02d}.jpg"), np.vstack(strips), [cv2.IMWRITE_JPEG_QUALITY, 85])
(a.out / f"{a.pick}_legend.json").write_text(json.dumps(legend, indent=0)); print(a.pick, len(sel), "photos of", len([r for r in rows if PICK[a.pick](r)]), "->", a.out)
