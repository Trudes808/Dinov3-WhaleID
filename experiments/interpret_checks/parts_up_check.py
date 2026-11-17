"""Check a day's re-made cut-outs (DAY/parts_up, DAY/cluster_up) against the first set (parts_hr, cluster_v3):
the same instances and parts, flukes unchanged, how many fin / body fingerprints changed; with a chart_variants.py
result folder also that the fingerprints equal the tested rule's.   parts_up_check.py DAY [DAY ...] [--test DIR --rule sd30+fit]"""
import argparse, json, sys
from pathlib import Path
import numpy as np
ROOT = Path(__file__).resolve().parents[2]; sys.path.insert(0, str(ROOT))
from model_fine_tuning.cluster.safe_clusters import load_embeddings
ap = argparse.ArgumentParser(); ap.add_argument("days", nargs="+"); ap.add_argument("--test", type=Path, default=None); ap.add_argument("--rule", default="sd30+fit")
a = ap.parse_args(); bad = 0
for day in a.days:
    d = ROOT / "experiments" / "days_all" / day
    if not (d / "parts_up" / "index.jsonl").exists() or not (d / "cluster_up" / "embeddings.npz").exists():
        print(day, "| NOT MADE"); bad += 1; continue
    A = {json.loads(l)["stem"]: json.loads(l) for l in open(d / "parts_hr/index.jsonl")}; B = {json.loads(l)["stem"]: json.loads(l) for l in open(d / "parts_up/index.jsonl")}
    same = set(A) == set(B) and all(A[k].get("instances") == B[k].get("instances") and A[k]["parts"] == B[k]["parts"] for k in A)
    old, new = load_embeddings(d), load_embeddings(d, "cluster_up")
    keys_ok = all(set(old.get(p, {})) == set(new.get(p, {})) for p in ("fin", "body", "fluke"))
    fl = [float(old["fluke"][k] @ new["fluke"][k]) for k in new.get("fluke", {})]
    ch = {p: np.array([float(old[p][k] @ new[p][k]) for k in new.get(p, {})]) for p in ("fin", "body")}
    msg = f"{day} | instances identical {same} | fingerprint keys identical {keys_ok} | flukes unchanged {bool(not fl or min(fl) > 0.999)} | changed fingerprints: " + ", ".join(f"{p} {np.mean(v < 0.999):.0%} of {len(v)} (median cosine of the changed {np.median(v[v < 0.999]) if (v < 0.999).any() else 1:.2f})" for p, v in ch.items())
    if a.test and (a.test / f"emb_{day}.npz").exists():
        z = np.load(a.test / f"emb_{day}.npz"); rows = json.loads(str(z["rows"])); E = z["E"]
        c = [float(E[i] @ new[r["part"]][r["key"]]) for i, r in enumerate(rows) if r["variant"] == ("cur" if r["part"] == "fluke" else a.rule) and r["key"] in new.get(r["part"], {})]
        msg += f" | equal to the tested rule: min cosine {min(c):.4f} over {len(c)}"; bad += min(c) < 0.999
    bad += (not same) + (not keys_ok) + bool(fl and min(fl) <= 0.999); print(msg)
sys.exit(1 if bad else 0)
