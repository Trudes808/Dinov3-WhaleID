"""Tracklets: one whale followed through a burst, when a photo or a burst holds several whales.

    python -m model_fine_tuning.cluster.tracklets --day-dir DAY [--parts-tag parts_hr] [--labels labels.json] [--gap 1.5]

cluster/parts.py --instances charts every mask component of a photo as an instance <stem>~k (k = 0 the largest, the one
the catalog's single label is taken to describe). Within a burst (cluster/bursts.py: one camera, gaps <= gap s) the
instances of consecutive photos are linked by continuity: centre distance in the frame, size ratio and, when an
appearance embedding is given, the cosine of their body charts (assignment by the Hungarian method, link if the cost
is under `max_cost`; 2025-02-27 with the id_v2 body embedding: cost 1.0 -> 265 tracklets, all multi-photo tracklets pure,
bursts alone 0.93 pure). A chain of linked instances is a tracklet; a burst of single-whale photos is one tracklet, a
burst where the photographer swings between two whales becomes two. Tracklets are the nodes of the day clustering.

Cannot-link: two instances of the same photo that show the same body region (their canonical body-fraction coverage
overlaps: two backs, two flukes) are two different whales, and so are their tracklets; a back and a fluke in one
photo may be one whale (a dive) and carry no constraint. The constraints go into cluster/linkage.py.

With labels: purity of the multi-photo tracklets (labelled main instances only) next to the burst purity.
"""
from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

from .bursts import make_bursts

U_BINS = np.linspace(0.0, 1.0, 11)


def load_instances(day_dir: Path, parts_tag: str = "parts_hr") -> dict:
    """stem -> [{"key", "parts", "area", "bbox_2560"}] from the parts index (instances in size order)."""
    out = {}
    for line in open(day_dir / parts_tag / "index.jsonl"):
        r = json.loads(line)
        inst = r.get("instances") or ([{"key": r["stem"], "parts": r["parts"], "area": None, "bbox_2560": None}] if r.get("parts") else [])
        out[r["stem"]] = [i for i in inst if i["parts"]]
    return out


def instance_coverage(day_dir: Path, stems: list[str], min_frac: float = 0.1) -> dict:
    """Per instance key, a (10 u bins x 2 sides) histogram of its pixels: canonical body fraction u and the side of the
    midline (left / right from the signed lateral coordinate), from the CSE map; components in the same order as
    cluster/parts.py --instances. The two blades of one fluke, or the two flanks of one back, fall in different cells."""
    from ..common import geometry, paths
    from .parts import split_instances

    table = geometry.load_folded_table(paths.FOLDED_TABLE)
    cov = {}
    for stem in stems:
        f = day_dir / "cse" / f"{stem}_canon.npz"
        if not f.exists():
            continue
        z = np.load(f); use = z["use"]
        if use.sum() < 400:
            continue
        signed = z["signed_mm"].astype(np.float32) / 1000.0
        u = (signed[..., 0] - table["x_min_m"]) / table["length_m"]
        side = (signed[..., 1] > 0).astype(np.int64)
        for k, m in enumerate(split_instances(use, min_frac, u=u)):
            ub = np.clip(np.digitize(np.clip(u[m], 0, 1), U_BINS) - 1, 0, len(U_BINS) - 2)
            cov[f"{stem}~{k}"] = np.bincount(ub * 2 + side[m], minlength=2 * (len(U_BINS) - 1)).reshape(len(U_BINS) - 1, 2)
    return cov


def cannot_link_instances(instances: dict, coverage: dict | None = None, rule: str = "fin", min_px: int = 200, min_cells: int = 2, skip_fluke_bins: int = 1) -> set[tuple[str, str]]:
    """Pairs of instance keys of one photo that are two whales for sure.
    rule "fin": both instances carry a dorsal fin chart (a whale has one dorsal fin; the safest cue). rule "coverage":
    at least min_cells shared (u bin, side) cells with >= min_px pixels in both, fluke bin skipped. On 2025-02-27 the
    coverage rule was wrong in 11 of 16 pairs (a whale's two pectoral fins during a slap, a fluke and its peduncle:
    the map's side / part labels on small blobs are not reliable enough for a hard constraint), so "fin" is the default."""
    out = set()
    for stem, inst in instances.items():
        for a in range(len(inst)):
            for b in range(a + 1, len(inst)):
                ka, kb = inst[a]["key"], inst[b]["key"]
                if rule == "fin":
                    if "fin" in inst[a]["parts"] and "fin" in inst[b]["parts"]:
                        out.add((ka, kb))
                elif rule == "coverage":
                    ha, hb = (coverage or {}).get(ka), (coverage or {}).get(kb)
                    if ha is None or hb is None:
                        continue
                    shared = (ha[skip_fluke_bins:] >= min_px) & (hb[skip_fluke_bins:] >= min_px)
                    if shared.sum() >= min_cells:
                        out.add((ka, kb))
    return out


def link_cost(a: dict, b: dict, app: dict | None, width: float = 2560.0, pos_scale: float = 0.25, size_scale: float = 1.0, app_scale: float = 0.5) -> float:
    cost = 0.0
    if a.get("bbox_2560") and b.get("bbox_2560"):
        ca = np.array([(a["bbox_2560"][0] + a["bbox_2560"][2]) / 2, (a["bbox_2560"][1] + a["bbox_2560"][3]) / 2])
        cb = np.array([(b["bbox_2560"][0] + b["bbox_2560"][2]) / 2, (b["bbox_2560"][1] + b["bbox_2560"][3]) / 2])
        cost += float(np.linalg.norm(ca - cb) / width) / pos_scale
    if a.get("area") and b.get("area"):
        cost += abs(float(np.log(a["area"] / b["area"]))) / size_scale
    if app is not None and a["key"] in app and b["key"] in app:
        cost += float(1.0 - np.dot(app[a["key"]], app[b["key"]])) / app_scale
    return cost


def make_tracklets(timestamps: dict, instances: dict, gap: float = 1.5, app: dict | None = None, max_cost: float = 1.0) -> list[dict]:
    """[{"id", "burst", "camera", "keys", "photos", "start", "end"}], one per chain of linked instances."""
    from scipy.optimize import linear_sum_assignment

    bursts = make_bursts(timestamps, gap)
    tracks = []
    for b in bursts:
        open_tracks: dict[str, int] = {}                                  # instance key of the previous photo -> track index
        prev: list[dict] = []
        for stem in b["photos"]:
            cur = instances.get(stem, [])
            t = float(timestamps[stem]["time"])
            assigned = {}
            if prev and cur:
                C = np.array([[link_cost(a, c, app) for c in cur] for a in prev])
                ri, ci = linear_sum_assignment(C)
                for i, j in zip(ri, ci):
                    if C[i, j] <= max_cost:
                        assigned[j] = open_tracks[prev[i]["key"]]
            new_open = {}
            for j, inst in enumerate(cur):
                if j in assigned:
                    k = assigned[j]
                    tracks[k]["keys"].append(inst["key"]); tracks[k]["photos"].append(stem); tracks[k]["end"] = t
                else:
                    k = len(tracks)
                    tracks.append({"id": k, "burst": b["id"], "camera": b["camera"], "keys": [inst["key"]], "photos": [stem], "start": t, "end": t})
                new_open[inst["key"]] = k
            if cur:
                open_tracks, prev = new_open, cur
    return tracks


def tracklet_report(tracks: list[dict], labels: dict | None, n_bursts: int) -> dict:
    sizes = np.array([len(t["keys"]) for t in tracks])
    rep = {"bursts": n_bursts, "tracklets": len(tracks), "instances": int(sizes.sum()), "multi_tracklets": int((sizes >= 2).sum()),
           "bursts_with_several_tracklets": int(sum(1 for b, c in _count_by(tracks, "burst").items() if c > 1))}
    if labels:
        pure = mixed = 0
        for t in tracks:
            ids = [labels.get(s.split("__")[-1]) for s, k in zip(t["photos"], t["keys"]) if k.endswith("~0") or "~" not in k]
            ids = [i for i in ids if i]
            if len(ids) >= 2:
                pure, mixed = (pure + 1, mixed) if len(set(ids)) == 1 else (pure, mixed + 1)
        rep.update({"multi_tracklets_pure": pure, "multi_tracklets_mixed": mixed, "tracklet_purity": pure / max(pure + mixed, 1)})
    return rep


def _count_by(items, key):
    c = defaultdict(int)
    for it in items:
        c[it[key]] += 1
    return c


def main(argv=None) -> int:
    from .bursts import burst_report

    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--day-dir", type=Path, required=True); p.add_argument("--parts-tag", default="parts_hr"); p.add_argument("--labels", type=Path, default=None)
    p.add_argument("--gap", type=float, default=1.5); p.add_argument("--max-cost", type=float, default=1.0); p.add_argument("--out", type=Path, default=None)
    p.add_argument("--cannot-link", default="fin", choices=["fin", "coverage", "none"])
    a = p.parse_args(argv)
    ts = json.loads((a.day_dir / "timestamps.json").read_text())
    inst = load_instances(a.day_dir, a.parts_tag)
    labels = json.loads(a.labels.read_text())["days"].get(a.day_dir.name, {}).get("photos") if a.labels else None
    bursts = make_bursts(ts, a.gap)
    tracks = make_tracklets(ts, inst, a.gap, None, a.max_cost)
    multi = {s: v for s, v in inst.items() if len(v) > 1}
    cov = instance_coverage(a.day_dir, list(multi))
    cl = cannot_link_instances(multi, cov, rule=a.cannot_link)
    rep = {"bursts": burst_report(bursts, labels), "tracklets": tracklet_report(tracks, labels, len(bursts)),
           "photos_with_several_instances": len(multi), "cannot_link_pairs": len(cl)}
    print(json.dumps(rep, indent=1))
    if a.out:
        a.out.write_text(json.dumps({"gap_s": a.gap, "max_cost": a.max_cost, "report": rep, "tracklets": tracks, "cannot_link": sorted(cl)}, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
