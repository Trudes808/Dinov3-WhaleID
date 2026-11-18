"""Score fingerprint arms with the project's own measure (experiments/interpret_checks/chart_variants_score.py).

    PYTHONPATH=. .venv/bin/python -m model_fine_tuning.cutouts_v3.score --days val --out CACHE/score_val [--r1-fin finB] [--base G]

Per part the arms below are written in chart_variants_score.py's input form (OUT/emb_<day>.npz: rows + fingerprints,
"used" = the cut-outs the Labeling Assistant counts, stack and catalogue animal from CACHE/points), then that script
is run with --base G and with --base R0 (when R0 exists). Arms (an arm is left out when its net was not embedded):
  G       pair_student_g on parts_up (production: the baseline)
  G.v3    pair_student_g on v3 flukes                     (fluke)
  G.B / G.C   pair_student_g on fins B / C, A where the fin has no v3 cut-out (fin)
  R0      the student retrained on parts_up, on parts_up
  R1      the student retrained on v3, on v3 (flukes v3, fins --r1-fin with A as fallback, bodies parts_up)
  R1.A    (fins) R1 on parts_up fins
Also OUT/queries_<part>.json: every query with its closest look-alike under each arm (for the found / lost sheet).
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

import numpy as np

from . import CACHE, ROOT, day_list
from .points import load as load_points


def load_emb(net: str, day: str) -> dict | None:
    f = CACHE / "emb" / net / f"{day}.npz"
    if not f.exists():
        return None
    z = np.load(f); out = {}
    for k in z.files:
        if k.endswith("|keys"):
            base = k[:-5]; out[base] = dict(zip([str(x) for x in z[k]], z[base + "|E"]))
    return out


FINP = {"mode": "fallback"}
RETRAIN = {"on": False}


def arms(part: str, r1_fin: str) -> dict:
    """arm -> (net, [sets in order of preference]). G.P / R0.P: the point-based fins (finP) of the pose side experiment,
    with today's cut-out where finP is missing (--finp fallback) or only where finP exists (--finp common: then every
    fin arm is scored on that common set, the per-day key intersection of score.py)."""
    if RETRAIN["on"]:                                  # pose_side retrain test: R0 / R0s1 on today's cut-outs, RP6s0 / RP6s1 with
        fin = ["finP6", "up"] if part == "fin" else ["up"]    # fins from finP6 (bodies and flukes on parts_up in every arm)
        return {"G": ("G", ["up"]), "R0": ("R0", ["up"]), "R0s1": ("R0s1", ["up"]), "RP6s0": ("RP6s0", fin), "RP6s1": ("RP6s1", fin)}
    if part == "fluke":
        return {"G": ("G", ["up"]), "G.v3": ("G", ["fluke"]), "R0": ("R0", ["up"]), "R1": ("R1", ["fluke"]), "R1.up": ("R1", ["up"])}
    if part == "fin":
        p = ["finP"] if FINP["mode"] == "common" else ["finP", "up"]
        return {"G": ("G", ["up"]), "G.B": ("G", ["finB", "up"]), "G.C": ("G", ["finC", "up"]), "R0": ("R0", ["up"]), "R1": ("R1", [r1_fin, "up"]), "R1.A": ("R1", ["up"]),
                "G.P": ("G", p), "R0.P": ("R0", p),
                # the same points re-cut at about B's scale (finP4, side 4.1 x the triangle) and A's scale (finP6, 6.0 x)
                "G.P4": ("G", [x + "4" for x in p[:1]] + p[1:]), "R0.P4": ("R0", [x + "4" for x in p[:1]] + p[1:]),
                "G.P6": ("G", [x + "6" for x in p[:1]] + p[1:]), "R0.P6": ("R0", [x + "6" for x in p[:1]] + p[1:])}
    return {"G": ("G", ["up"]), "R0": ("R0", ["up"]), "R1": ("R1", ["up"])}


def main(argv=None):
    ap = argparse.ArgumentParser(); ap.add_argument("--days", default="val"); ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--r1-fin", default="finB"); ap.add_argument("--extra-nets", default="", help="more nets as R-like arms on up, e.g. R0b")
    ap.add_argument("--finp", default="fallback", choices=["fallback", "common"])
    ap.add_argument("--retrain", action="store_true", help="the pose_side fin retrain test: arms G, R0, R0s1, RP6s0, RP6s1 only")
    a = ap.parse_args(argv); a.out.mkdir(parents=True, exist_ok=True); FINP["mode"] = a.finp; RETRAIN["on"] = a.retrain
    for f in a.out.glob("emb_*.npz"):
        f.unlink()
    queries = {p: [] for p in ("fin", "body", "fluke")}; have = set()
    for day in day_list(a.days):
        pts = load_points(day); embs = {n: load_emb(n, day) for n in (("G", "R0", "R0s1", "RP6s0", "RP6s1") if a.retrain else ("G", "R0", "R1"))}
        rows, E = [], []
        for part in ("fin", "body", "fluke"):
            A = {k: v for k, v in arms(part, a.r1_fin).items() if embs.get(v[0]) is not None and f"{part}|{v[1][-1]}" in embs[v[0]]}
            for name, (net, sets) in A.items():
                have.add(name)
                for key, (st, cat) in pts[part].items():
                    e = None; used_set = None
                    for s in sets:
                        e = embs[net].get(f"{part}|{s}", {}).get(key)
                        if e is not None:
                            used_set = s; break
                    if e is None:
                        continue
                    rows.append(dict(key=key, part=part, variant=name, same=None, angle=0.0, zoom=1.0, outside=0.0, stack=st, cat=cat, used=True, set=used_set))
                    E.append(e)
        np.savez_compressed(a.out / f"emb_{day}.npz", E=np.asarray(E, np.float32), rows=json.dumps(rows))
        # per query closest look-alike in another stack, per arm (main instances with an animal)
        for part in ("fin", "body", "fluke"):
            by = {}
            for i, r in enumerate(rows):
                if r["part"] == part and r["cat"] and r["key"].endswith("~0"):
                    by.setdefault(r["variant"], {})[r["key"]] = i
            if not by:
                continue
            keys = sorted(set.intersection(*[set(v) for v in by.values()]))
            if len(keys) < 3:
                continue
            st = np.array([rows[by["G"][k]]["stack"] for k in keys]); cat = np.array([rows[by["G"][k]]["cat"] for k in keys])
            other = st[:, None] != st[None, :]; same = cat[:, None] == cat[None, :]; can = (other & same).any(1)
            nn = {}
            for v, idx in by.items():
                X = np.asarray([E[idx[k]] for k in keys]); C = np.where(other, X @ X.T, -9.0); nn[v] = C.argmax(1)
            for i in np.nonzero(can)[0]:
                queries[part].append(dict(day=day, key=keys[i], stack=int(st[i]), cat=str(cat[i]),
                                          nn={v: keys[nn[v][i]] for v in nn}, hit={v: bool(same[i, nn[v][i]]) for v in nn},
                                          set={v: rows[by[v][keys[i]]]["set"] for v in by}))
    for p, q in queries.items():
        (a.out / f"queries_{p}.json").write_text(json.dumps(q))
    script = ROOT / "experiments" / "interpret_checks" / "chart_variants_score.py"
    for base in (("G", "R0", "R0s1") if a.retrain else ("G", "R0")):
        if base not in have:
            continue
        print(f"\n##### base {base}", flush=True)
        subprocess.run([sys.executable, str(script), str(a.out), "--base", base, "--json", str(a.out / f"score_base_{base}.json")], check=True)


if __name__ == "__main__":
    main()
