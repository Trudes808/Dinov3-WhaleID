"""By-eye comparison sheets, written OUTSIDE the repo: real photos of the situations where part finding fails today,
and run05 renders by pose category, both cut the way a model sees them (the whale's box x 1.6).

    .venv/bin/python experiments/bodymap_review/hard_sheets.py REAL_STATS.json CLICKS.json OUT_DIR
"""
import json, random, sys
from pathlib import Path
import cv2, numpy as np
ROOT = Path(__file__).resolve().parents[2]; DAYS = ROOT / "experiments" / "days_all"; SYN = Path("/home/sat3737/NOAA_whales/Whale-Image-Data-Factory/data/renders/run05/dataset")
out = Path(sys.argv[3]); out.mkdir(parents=True, exist_ok=True); P = 300; rng = random.Random(3)
R = [r for r in json.load(open(sys.argv[1])) if "inst" in r and r.get("blobs", 0) > 0]
for r in R: r["n"] = len([x for x in r["inst"] if not x.get("not_whale")])
boxes = {}
def box(day, key):
    if day not in boxes:
        boxes[day] = {i["key"]: i["bbox_2560"] for l in open(DAYS / day / "parts_hr" / "index.jsonl") for i in json.loads(l).get("instances", [])}
    return boxes[day].get(key)
def tile(img, b, label, margin=1.6):
    x0, y0, x1, y1 = b; L = max(max(x1 - x0, y1 - y0) * margin, 120); cx, cy = (x0 + x1) / 2, (y0 + y1) / 2
    M = np.array([[P / L, 0, P / 2 - cx * P / L], [0, P / L, P / 2 - cy * P / L]]); t = cv2.warpAffine(img, M, (P, P), flags=cv2.INTER_AREA if L > P else cv2.INTER_CUBIC, borderValue=(40, 40, 40))
    cv2.putText(t, label, (4, 14), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (0, 0, 0), 3); cv2.putText(t, label, (4, 14), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (255, 255, 255), 1); return t
I = [(r, x) for r in R for x in r["inst"] if not x.get("not_whale") and x.get("vc") and x.get("key")]
frame = lambda r, x: (x["long_px"] / r["s"]) / 2560.0
C = json.load(open(sys.argv[2])); other_keys = {r["key"] for r in C if any(r["others"])}
SCEN = {
    "A fluke, no fluke cut-out made": [(r, x) for r, x in I if x["vc"][0] >= 0.9 and "fluke" not in x["parts"]],
    "B one fluke cut into two whales": [(r, x) for r, x in I if r["n"] >= 2 and x["k"] == 0 and all((y.get("vc") or [0, 0])[0] >= 0.9 for y in r["inst"][:2])],
    "C neither fluke nor fin": [(r, x) for r, x in I if x["vc"][0] <= 0.1 and x["vc"][1] <= 0.1 and x["area"] > 1500],
    "D fin view, map has head and flank at odds": [(r, x) for r, x in I if x["vc"][1] >= 0.9 and x["vc"][0] <= 0.1 and abs(x["turn"]) > 120],
    "E whale fills the frame": [(r, x) for r, x in I if frame(r, x) > 0.8],
    "F small and far": [(r, x) for r, x in I if frame(r, x) < 0.08 and x["area"] > 600],
    "G no fin showing, fin cut-out made": [(r, x) for r, x in I if x["vc"][1] <= 0.1 and "fin" in x["parts"]],
    "H another whale marked by the researcher": [(r, x) for r, x in I if x["key"] in other_keys],
}
legend = []; sheets = []; rows = []
for name, S in SCEN.items():
    rng.shuffle(S); sel = S[:12]; tiles = []
    for r, x in sel:
        b = box(r["day"], x["key"]); img = cv2.imread(str(DAYS / r["day"] / "images" / f"{r['stem']}.jpg"))
        if b is None or img is None: continue
        if name.startswith(("B", "H")):                                   # show the whole group of instances / more context
            bs = [box(r["day"], y["key"]) for y in r["inst"] if y.get("key")]; bs = [q for q in bs if q]; b = [min(q[0] for q in bs), min(q[1] for q in bs), max(q[2] for q in bs), max(q[3] for q in bs)]
        tiles.append(tile(img, b, f"{name[0]}{len(tiles) + 1} {r['day'][5:]}", 2.2 if name.startswith("H") else 1.6)); legend.append({"tile": f"{name[0]}{len(tiles)}", "scenario": name, "day": r["day"], "key": x["key"]})
    while len(tiles) < 12: tiles.append(np.full((P, P, 3), 40, np.uint8))
    bar = np.zeros((22, P * 6, 3), np.uint8); cv2.putText(bar, f"REAL  {name}   ({len(S)} of {len(I)} whale instances)", (6, 16), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    rows.append(np.vstack([bar, np.hstack(tiles[:6]), np.hstack(tiles[6:12])]))
for i in range(0, len(rows), 2):
    cv2.imwrite(str(out / f"real_hard_{i // 2 + 1}.jpg"), np.vstack(rows[i:i + 2]), [cv2.IMWRITE_JPEG_QUALITY, 85])
(out / "real_hard_legend.json").write_text(json.dumps(legend, indent=0))
# renders by pose category
ids = sorted(p.stem for p in (SYN / "meta").glob("*.json")); rng.shuffle(ids); by = {}
for i in ids:
    if all(len(v) >= 6 for v in by.values()) and len(by) >= 13: break
    m = json.loads((SYN / "meta" / f"{i}.json").read_text()); v = m.get("visible") or {}
    if m.get("negative") or v.get("visible_pixels", 0) < 1500: continue
    if len(by.setdefault(m["category"], [])) < 6: by[m["category"]].append((i, v["bbox_xyxy"], m))
rows = []
for cat in ("fluke_up", "fluke_edge", "fluke_down", "surface", "back_arch", "logging", "blow", "blow_sliver", "awash", "breach", "pec_up", "roll_belly", "spyhop"):
    tiles = [tile(cv2.imread(str(SYN / "images" / f"{i}.png")), b, f"{cat} {i}" + (" +calf" if m.get("companion") else "")) for i, b, m in by.get(cat, [])]
    while len(tiles) < 6: tiles.append(np.full((P, P, 3), 40, np.uint8))
    rows.append(np.hstack(tiles))
for i in range(0, len(rows), 5):
    cv2.imwrite(str(out / f"renders_{i // 5 + 1}.jpg"), np.vstack(rows[i:i + 5]), [cv2.IMWRITE_JPEG_QUALITY, 88])
print({k: len(v) for k, v in SCEN.items()}); print(sorted(p.name for p in out.glob("*.jpg")))
