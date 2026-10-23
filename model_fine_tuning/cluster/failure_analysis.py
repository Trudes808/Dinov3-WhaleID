"""Where does the day clustering fail? Error anatomy of one day from the saved v3 outputs.

    python -m model_fine_tuning.cluster.failure_analysis --day-dir DAY --tag student --labels labels.json [--montage OUT.jpg]

Reads cluster_v3/scores<tag>.npz (instance-pair scores, predictions), the catalog labels (animals, groups, group types),
parts_hr/index.jsonl (instance size, parts) and timestamps, and reports, for the labelled main instances:
  1. threshold gap: achieved vs oracle ARI (metrics json)
  2. error anatomy: wrong links inside a true group vs across groups; animals split over several clusters
  3. appearance alone (instance-pair score, cross-burst pairs): AUC overall, within true groups, across groups, by the
     pair's group type, by whale size, by the parts both photos have
  4. bounds: ARI with the predicted clusters but every within-group error fixed (= groups are the unit); ARI with the TRUE
     groups and appearance-only linkage inside each group at the best threshold (what a two-stage design could reach)
  5. the hardest photos: >= 3 cross-burst photos of the same animal exist, yet the top-3 neighbours are all other
     animals; a montage shows each with its best wrong neighbour
"""
from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

from .bursts import make_bursts
from .cluster import pair_metrics
from .linkage import constrained_average_linkage


def main(argv=None) -> int:
    from sklearn.metrics import adjusted_rand_score, roc_auc_score

    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--day-dir", type=Path, required=True); p.add_argument("--tag", default="student"); p.add_argument("--labels", type=Path, required=True)
    p.add_argument("--montage", type=Path, default=None); p.add_argument("--quiet", action="store_true")
    a = p.parse_args(argv)
    day = a.day_dir.name; out = a.day_dir / "cluster_v3"
    z = np.load(out / f"scores_{a.tag}.npz", allow_pickle=True); keys = [str(k) for k in z["keys"]]; S = z["inst_score"].astype(np.float64)
    metrics = json.loads((out / f"metrics_{a.tag}.json").read_text())
    L = json.loads(a.labels.read_text())["days"][day]; photos = L["photos"]; gtypes = L.get("groups", {})
    ts = json.loads((a.day_dir / "timestamps.json").read_text()); burst_of = {s: b["id"] for b in make_bursts(ts, 1.5) for s in b["photos"]}
    inst = {}
    for line in open(a.day_dir / "parts_hr" / "index.jsonl"):
        r = json.loads(line)
        for i in r.get("instances", []):
            inst[i["key"]] = i
    pos = {k: i for i, k in enumerate(keys)}
    # labelled main instances
    rows = []
    for k in keys:
        if "~" in k and not k.endswith("~0"):
            continue
        stem = k.rsplit("~", 1)[0]; ident = photos.get(stem.split("__")[-1], "?")
        if ident == "?" or ident.split("_")[1].startswith("O") or stem not in burst_of:
            continue
        rows.append({"key": k, "i": pos[k], "ident": ident, "group": ident.split("_")[1][:3], "burst": burst_of[stem], "area": (inst.get(k) or {}).get("area") or 0, "parts": tuple((inst.get(k) or {}).get("parts") or [])})
    n = len(rows); idx = np.array([r["i"] for r in rows]); ident = np.array([r["ident"] for r in rows]); group = np.array([r["group"] for r in rows]); burst = np.array([r["burst"] for r in rows])
    area = np.array([r["area"] for r in rows]); gtype = np.array([gtypes.get(g, "?") for g in group])
    pred = np.array([int(z["pred_instance"][i]) for i in idx])
    Sm = S[np.ix_(idx, idx)]
    rep = {"day": day, "photos": n, "animals": len(set(ident)), "groups": len(set(group))}
    # 1. threshold gap
    rep["ari"] = metrics["animals"]["ari"]; rep["oracle_ari"] = max(s["ari"] for s in metrics["oracle_sweep_animals"]); rep["threshold_gap"] = rep["oracle_ari"] - rep["ari"]
    # 2. error anatomy on photo pairs
    iu = np.triu_indices(n, 1); same_a = ident[iu[0]] == ident[iu[1]]; same_g = group[iu[0]] == group[iu[1]]; same_p = pred[iu[0]] == pred[iu[1]]; cross_b = burst[iu[0]] != burst[iu[1]]
    wrong_links = same_p & ~same_a; missed = ~same_p & same_a & cross_b
    rep["wrong_links"] = int(wrong_links.sum()); rep["wrong_links_within_group"] = float(same_g[wrong_links].mean()) if wrong_links.any() else float("nan")
    rep["missed_same_animal_pairs"] = int(missed.sum()); rep["same_animal_pairs_cross_burst"] = int((same_a & cross_b).sum())
    rep["animals_split"] = int(sum(1 for a_ in set(ident) if len(set(pred[ident == a_])) > 1))
    # 3. appearance alone (cross-burst pairs with a finite score)
    s = Sm[iu]; ok = np.isfinite(s) & cross_b
    rep["pairs_without_score"] = float((~np.isfinite(s) & cross_b).mean())
    def auc(mask):
        m = ok & mask
        return float(roc_auc_score(same_a[m], s[m])) if m.sum() > 50 and same_a[m].any() and (~same_a[m]).any() else float("nan")
    rep["auc_all"] = auc(np.ones_like(ok)); rep["auc_within_group"] = auc(same_g); rep["auc_across_group"] = auc(~same_g | same_a)
    for t in sorted(set(gtype)):
        rep[f"auc_within_{t}"] = auc(same_g & (gtype[iu[0]] == t))
    big = area >= np.percentile(area, 50)
    rep["auc_both_large"] = auc(big[iu[0]] & big[iu[1]]); rep["auc_both_small"] = auc(~big[iu[0]] & ~big[iu[1]])
    has_fin = np.array(["fin" in r["parts"] for r in rows]); has_fluke = np.array(["fluke" in r["parts"] for r in rows])
    rep["auc_both_fin"] = auc(has_fin[iu[0]] & has_fin[iu[1]]); rep["auc_both_fluke"] = auc(has_fluke[iu[0]] & has_fluke[iu[1]]); rep["auc_body_only"] = auc(~has_fin[iu[0]] & ~has_fin[iu[1]] & ~has_fluke[iu[0]] & ~has_fluke[iu[1]])
    # 4. bounds
    fixed = np.array([f"{p_}|{a_}" for p_, a_ in zip(pred, ident)])                  # predicted cluster split by true animal
    rep["ari_if_within_cluster_fixed"] = float(adjusted_rand_score(ident, fixed))
    best = -1.0
    S_app = np.where(np.isfinite(Sm), Sm, np.nanmin(Sm[np.isfinite(Sm)]))
    for thr in np.linspace(np.percentile(S_app, 60), np.percentile(S_app, 99), 20):
        lab = np.full(n, -1); base = 0
        for g in sorted(set(group)):
            sel = np.nonzero(group == g)[0]
            if len(sel) == 1:
                lab[sel] = base; base += 1; continue
            sub = S_app[np.ix_(sel, sel)].copy(); np.fill_diagonal(sub, 0.0)
            # bursts as nodes: average over member instances is implicit in the instance graph; use instance linkage
            cl = constrained_average_linkage(sub, float(thr)); lab[sel] = cl + base; base += cl.max() + 1
        best = max(best, float(adjusted_rand_score(ident, lab)))
    rep["ari_true_groups_appearance_only_oracle"] = best
    # 5. hardest photos
    hard = []
    for q in range(n):
        others = np.nonzero((burst != burst[q]))[0]
        if (ident[others] == ident[q]).sum() < 3:
            continue
        sc = Sm[q, others]; fin = np.isfinite(sc)
        if fin.sum() < 3:
            continue
        top = others[fin][np.argsort(-sc[fin])[:3]]
        if not (ident[top] == ident[q]).any():
            hard.append({"key": rows[q]["key"], "ident": ident[q], "group": group[q], "gtype": gtype[q], "area": int(area[q]), "parts": list(rows[q]["parts"]), "best_wrong": rows[top[0]]["key"], "best_wrong_ident": ident[top[0]], "score": float(Sm[q, top[0]])})
    rep["hard_photos"] = len(hard); rep["hard_fraction"] = len(hard) / max(n, 1)
    rep["hard_by_gtype"] = dict(Counter(h["gtype"] for h in hard)); rep["hard_median_area"] = float(np.median([h["area"] for h in hard])) if hard else float("nan"); rep["all_median_area"] = float(np.median(area))
    rep["hard_parts"] = dict(Counter("+".join(h["parts"]) for h in hard)); rep["hard_wrong_same_group"] = float(np.mean([h["group"] == h["best_wrong_ident"].split("_")[1][:3] for h in hard])) if hard else float("nan")
    if not a.quiet:
        print(json.dumps({k: (round(v, 3) if isinstance(v, float) else v) for k, v in rep.items()}, indent=1))
    (out / f"failure_{a.tag}.json").write_text(json.dumps({"report": rep, "hard": hard}, indent=1))
    if a.montage and hard:
        import cv2
        tiles = []
        for h in sorted(hard, key=lambda h: -h["score"])[:12]:
            pair = []
            for key, lab_ in ((h["key"], h["ident"]), (h["best_wrong"], h["best_wrong_ident"])):
                f = a.day_dir / "parts_hr" / "charts" / f"{key}__body.jpg"
                img = cv2.imread(str(f)) if f.exists() else np.zeros((1536, 1536, 3), np.uint8); img = cv2.resize(img, (384, 384))
                cv2.putText(img, f"{lab_[-6:]} {h['gtype']}", (6, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2); pair.append(img)
            cv2.putText(pair[1], f"score {h['score']:.1f}", (6, 44), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
            tiles.append(np.concatenate(pair, 1))
        rows_img = [np.concatenate(tiles[i:i + 3], 1) for i in range(0, len(tiles) - len(tiles) % 3, 3)]
        if rows_img:
            cv2.imwrite(str(a.montage), np.concatenate(rows_img, 0), [cv2.IMWRITE_JPEG_QUALITY, 82])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
