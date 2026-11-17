"""The two sets of cut-outs a day holds (parts_hr / cluster_v3 = "cur", parts_up / cluster_up = "up") in the form
chart_variants_score.py reads, so the same catalogue scores can be read off the fingerprints the pages really use.

    .venv/bin/python experiments/interpret_checks/parts_up_emb.py OUT_DIR DAY [DAY ...]      then
    .venv/bin/python experiments/interpret_checks/chart_variants_score.py OUT_DIR
"""
import json, sys, urllib.request
from pathlib import Path
import numpy as np
from PIL import Image
ROOT = Path(__file__).resolve().parents[2]; sys.path.insert(0, str(ROOT))
from model_fine_tuning.cluster.safe_clusters import load_embeddings
out = Path(sys.argv[1]); out.mkdir(parents=True, exist_ok=True); API = "http://127.0.0.1:8765"
for day in sys.argv[2:]:
    d = ROOT / "experiments" / "days_all" / day; emb = {"cur": load_embeddings(d), "up": load_embeddings(d, "cluster_up")}; rows = []; E = []; sizes = {}
    for part in ("fin", "body", "fluke"):
        r = json.load(urllib.request.urlopen(f"{API}/api/interpret/{day}/fingerprints?part={part}&layout=pca", timeout=600)); pts = {p[0]: (int(p[1]), p[4] or "") for p in r["points"]}
        for key in sorted(set(emb["cur"].get(part, {})) & set(emb["up"].get(part, {}))):
            stem = key.rsplit("~", 1)[0]
            if stem not in sizes:
                sizes[stem] = Image.open(d / "images" / f"{stem}.jpg").size
            for v, tag in (("cur", "parts_hr"), ("up", "parts_up")):
                z = np.load(d / tag / f"{key}__{part}.npz"); M = z["M"].astype(float); sf = float(z["scale_full"]); W, H = sizes[stem][0] * sf, sizes[stem][1] * sf
                cs = 1536; g = np.linspace(0, cs - 1, 48); xx, yy = np.meshgrid(g, g); P = np.linalg.inv(np.vstack([M, [0, 0, 1]])) @ np.stack([xx.ravel(), yy.ravel(), np.ones(xx.size)])
                same = "cur" if v == "up" and float(emb["cur"][part][key] @ emb["up"][part][key]) > 0.9995 else None
                rows.append(dict(key=key, part=part, variant=v, same=same, angle=float(np.degrees(np.arctan2(M[0, 1], M[0, 0]))), zoom=float(np.hypot(M[0, 0], M[0, 1])), mirrored=bool(z["mirrored"]),
                                 flipped_v=bool(z["flipped_v"]) if "flipped_v" in z else False, outside=float(((P[0] < 0) | (P[0] > W) | (P[1] < 0) | (P[1] > H)).mean()),
                                 stack=pts.get(key, (-1, ""))[0], cat=pts.get(key, (-1, ""))[1], used=key in pts))
                E.append(emb[v][part][key])
    np.savez_compressed(out / f"emb_{day}.npz", E=np.asarray(E, np.float32), rows=json.dumps(rows)); print(day, len(rows) // 2, "cut-outs", flush=True)
