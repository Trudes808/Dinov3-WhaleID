"""Two-stage day clustering: encounters by time, animals inside each encounter by appearance alone, re-sightings across
encounters by strict appearance. Built on the saved per-day channels of cluster_day.py.

    python -m model_fine_tuning.cluster.two_stage --root experiments/days_all --labels labels.json --days days.json \
        [--source id:pair_student] [--tune] [--theta-in 1.0 --theta-out 2.5 --encounter 120]

Why: the failure analysis shows that nearly every wrong link of the single-stage linkage joins two animals of one true
group (the time term pulls animals photographed together into one cluster), while the groups themselves are found.
Stage 1 chains bursts whose capture times are within `encounter` seconds into encounters. Stage 2 clusters the bursts
of each encounter with appearance only: instance-pair score = the best over the parts both instances show of the
part-weighted channel z-score (fins and flukes weigh 1, bodies `body_weight`: inside a group they are the weakest
evidence), burst pair = mean of the top-3 instance pairs, constrained average linkage at theta_in with the
same-photo (two dorsal fins) cannot-links. Stage 3 merges the stage-2 clusters across encounters when their appearance
is at least theta_out (a re-sighting hours later), average linkage. Thresholds are on the day's z-score scale, tuned on
the val days; the eval days are printed for the record. Also reports the share of multi-instance photos among the
scored photos and the ARI over single-instance photos only (the catalog names one animal per photo; when two whales
are in frame our main instance may be the other one).
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from scipy.sparse.csgraph import connected_components

from .bursts import make_bursts
from .cluster import pair_metrics
from .cluster_bursts import evaluate, groups_from_animals
from .linkage import constrained_average_linkage
from .tracklets import cannot_link_instances, load_instances

PARTS = ("body", "fin", "fluke")


def load_day(root: Path, day: str, source: str, labels: dict, body_weight: float, parts_tag: str = "parts_hr") -> dict:
    z = np.load(root / day / "cluster_v3" / "channels.npz", allow_pickle=True); keys = [str(k) for k in z["keys"]]; n = len(keys)
    score = np.full((n, n), np.nan)
    per_part = dict(x.split("=") for x in source.split(",")) if "=" in source else {part: source for part in PARTS}
    for part in PARTS:
        name = f"{per_part.get(part, '')}|{part}"
        if name not in z.files:
            continue
        M = z[name] * (body_weight if part == "body" else 1.0)
        score = np.where(np.isnan(score), M, np.fmax(score, np.nan_to_num(M, nan=-np.inf)))
    score[np.isinf(score)] = np.nan
    ts = json.loads((root / day / "timestamps.json").read_text())
    photo = {k: (k.rsplit("~", 1)[0] if "~" in k else k) for k in keys}
    times = np.array([ts[photo[k]]["time"] for k in keys], float)
    bursts = make_bursts(ts, 1.5); pos = {k: i for i, k in enumerate(keys)}
    nodes = []
    for b in bursts:
        members = [pos[k] for k in keys if photo[k] in set(b["photos"])]
        if members:
            nodes.append({"members": np.array(members), "start": b["start"], "end": b["end"]})
    inst = load_instances(root / day, parts_tag); multi = {s: v for s, v in inst.items() if len(v) > 1}
    cl = cannot_link_instances(multi, None, rule="fin") if multi else set()
    node_of = {m: i for i, nd in enumerate(nodes) for m in nd["members"]}
    CL = np.zeros((len(nodes), len(nodes)), bool)
    for x, y in cl:
        if x in pos and y in pos and node_of.get(pos[x]) is not None and node_of.get(pos[y]) is not None and node_of[pos[x]] != node_of[pos[y]]:
            CL[node_of[pos[x]], node_of[pos[y]]] = CL[node_of[pos[y]], node_of[pos[x]]] = True
    L = labels["days"].get(day, {}).get("photos", {})
    main = [i for i, k in enumerate(keys) if "~" not in k or k.endswith("~0")]
    gt = np.array([L.get(photo[keys[i]].split("__")[-1], "?") for i in main]); multi_photo = np.array([len(inst.get(photo[keys[i]], [])) > 1 for i in main])
    return {"keys": keys, "score": score, "times": times, "nodes": nodes, "CL": CL, "main": np.array(main), "gt": gt, "gt_group": np.array([g.split("_")[-1][:3] if g != "?" else "?" for g in gt]),
            "multi_photo": multi_photo, "ptimes": np.array([times[i] for i in main])}


def node_scores(d: dict, topk: int = 3) -> tuple[np.ndarray, np.ndarray]:
    nodes = d["nodes"]; m = len(nodes); app = np.full((m, m), np.nan); dt = np.full((m, m), np.inf)
    for i in range(m):
        for j in range(i + 1, m):
            block = d["score"][np.ix_(nodes[i]["members"], nodes[j]["members"])].ravel(); block = block[np.isfinite(block)]
            if len(block):
                app[i, j] = app[j, i] = float(np.mean(np.sort(block)[-topk:]))
            dt[i, j] = dt[j, i] = max(0.0, max(nodes[i]["start"], nodes[j]["start"]) - min(nodes[i]["end"], nodes[j]["end"]))
    return app, dt


def run(d: dict, app: np.ndarray, dt: np.ndarray, theta_in: float, theta_out: float, encounter: float, fill: float = -1.0) -> np.ndarray:
    m = len(d["nodes"])
    n_enc, enc = connected_components(dt <= encounter, directed=False)
    label = np.full(m, -1); base = 0
    A = np.where(np.isfinite(app), app, fill)
    for e in range(n_enc):
        sel = np.nonzero(enc == e)[0]
        if len(sel) == 1:
            label[sel] = base; base += 1; continue
        sub = A[np.ix_(sel, sel)].copy(); np.fill_diagonal(sub, 0.0)
        cl = constrained_average_linkage(sub, theta_in, d["CL"][np.ix_(sel, sel)]); label[sel] = cl + base; base += int(cl.max()) + 1
    # stage 3: merge stage-2 clusters across encounters by appearance (mean over member-node pairs), strict threshold
    k = int(label.max()) + 1
    C = np.full((k, k), np.nan)
    for a in range(k):
        for b in range(a + 1, k):
            ia, ib = np.nonzero(label == a)[0], np.nonzero(label == b)[0]
            if enc[ia[0]] == enc[ib[0]]:
                C[a, b] = C[b, a] = -1e6                                              # same encounter: decided in stage 2
                continue
            block = A[np.ix_(ia, ib)].ravel()
            C[a, b] = C[b, a] = float(np.mean(np.sort(block)[-3:]))
    np.fill_diagonal(C, 0.0)
    merged = constrained_average_linkage(C, theta_out)
    out = merged[label]
    pred = np.full(len(d["keys"]), -1)
    for i, nd in enumerate(d["nodes"]):
        pred[nd["members"]] = out[i]
    pred[pred < 0] = pred.max() + 1 + np.arange((pred < 0).sum())
    return pred


def score_day(d: dict, pred: np.ndarray) -> dict:
    pc = pred[d["main"]]
    res = {"animals": evaluate(pc, list(d["gt"])), "groups": evaluate(groups_from_animals(pc, d["ptimes"], 120.0, 10), list(d["gt_group"]))}
    single = ~d["multi_photo"]
    res["animals_single_instance_photos"] = evaluate(pc[single], list(d["gt"][single])) if single.sum() > 10 else {}
    return res


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--root", type=Path, required=True); p.add_argument("--labels", type=Path, required=True); p.add_argument("--days", type=Path, required=True)
    p.add_argument("--source", default="id:pair_student", help="one channel source, or per part: body=id:pair_student,fin=idxr:fin_xr,fluke=idxr:fin_xr"); p.add_argument("--body-weight", type=float, default=0.8)
    p.add_argument("--tune", action="store_true"); p.add_argument("--theta-in", type=float, default=1.0); p.add_argument("--theta-out", type=float, default=2.5); p.add_argument("--encounter", type=float, default=120.0)
    p.add_argument("--tag", default="two_stage")
    a = p.parse_args(argv)
    labels = json.loads(a.labels.read_text()); days = json.loads(a.days.read_text())
    data = {}
    for d in days["val"] + days["eval"]:
        if (a.root / d / "cluster_v3" / "channels.npz").exists():
            data[d] = load_day(a.root, d, a.source, labels, a.body_weight); data[d]["app"], data[d]["dt"] = node_scores(data[d])
    print(f"days: {list(data)}; multi-instance share of scored photos: " + ", ".join(f"{d[5:]} {data[d]['multi_photo'].mean():.2f}" for d in data))
    grid = [(ti, to, enc) for ti in (0.5, 1.0, 1.5, 2.0) for to in (2.0, 2.5, 3.0, 3.5) for enc in (60.0, 120.0, 300.0)] if a.tune else [(a.theta_in, a.theta_out, a.encounter)]
    rows = []
    for ti, to, enc in grid:
        res = {d: score_day(data[d], run(data[d], data[d]["app"], data[d]["dt"], ti, to, enc)) for d in data}
        v = [res[d]["animals"]["ari"] for d in days["val"] if d in res]; e = [res[d]["animals"]["ari"] for d in days["eval"] if d in res]
        rows.append({"theta_in": ti, "theta_out": to, "encounter": enc, "val": float(np.mean(v)), "eval": float(np.mean(e)) if e else float("nan"),
                     "per_day": {d: round(res[d]["animals"]["ari"], 3) for d in res}, "single": {d: round(res[d]["animals_single_instance_photos"].get("ari", float("nan")), 3) for d in res},
                     "groups": {d: round(res[d]["groups"]["ari"], 3) for d in res}})
    rows.sort(key=lambda r: -r["val"])
    for r in rows[: (8 if a.tune else 1)]:
        print(f"theta_in {r['theta_in']:.1f} theta_out {r['theta_out']:.1f} enc {r['encounter']:.0f}s  val {r['val']:.3f}  eval {r['eval']:.3f}  per day {r['per_day']}")
    best = rows[0]
    print("single-instance photos only (best setting):", best["single"]); print("groups (best setting):", best["groups"])
    (a.root / f"tune_{a.tag}_{a.source.replace(':', '_').replace('=', '-').replace(',', '+')}.json").write_text(json.dumps(rows, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
