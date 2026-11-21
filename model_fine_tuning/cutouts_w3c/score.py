"""Score the W3c fin cut-outs with the project's measure (experiments/interpret_checks/chart_variants_score.py), the
same query set and stack bootstrap as cutouts_v3/score.py.

    PYTHONPATH=. .venv/bin/python -m model_fine_tuning.cutouts_w3c.score --days val --out CACHE/score_val [--extra s1_wide]

Arms per net N in (G, R0) (an arm is left out when its sets were not embedded):
  N        parts_up (fins, bodies, flukes)
  N.W      fins from s0_wide, parts_up where the fin has no W3c cut-out
  N.T      fins from s0_tight, the same fallback
  (--arms replaces W / T, e.g. "C=s0_centre,N=s0_nudge": N.C, N.N)
  N.X1     fins from the --extra set (seed 1's regions), the same fallback
--retrain SET: arms G, R0, R0s1 (parts_up) and RWs0, RWs1 (students retrained with fins from SET), bases G, R0, R0s1.
Scored with --base G and --base R0. Also OUT/queries_<part>.json (closest look-alike per query under each arm).
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

import numpy as np

from . import CACHE, ROOT, day_list
from ..cutouts_v3.points import load as load_points


def load_emb(net: str, day: str) -> dict | None:
    f = CACHE / "emb" / net / f"{day}.npz"
    if not f.exists():
        return None
    z = np.load(f); out = {}
    for k in z.files:
        if k.endswith("|keys"):
            base = k[:-5]; out[base] = dict(zip([str(x) for x in z[k]], z[base + "|E"]))
    return out


RETRAIN = {"fin": ""}
FIN_ARMS = {"W": "s0_wide", "T": "s0_tight"}
NETS = ("G", "R0", "R0s1", "RWs0", "RWs1")


def arms(part: str, extra: str) -> dict:
    if RETRAIN["fin"]:                                 # retrain test: R0 / R0s1 on parts_up, RWs0 / RWs1 with fins from the W3c set
        fin = [RETRAIN["fin"], "up"] if part == "fin" else ["up"]
        return {"G": ("G", ["up"]), "R0": ("R0", ["up"]), "R0s1": ("R0s1", ["up"]), "RWs0": ("RWs0", fin), "RWs1": ("RWs1", fin)}
    out = {}
    for n in ("G", "R0"):
        out[n] = (n, ["up"])
        if part == "fin":
            for suf, st in FIN_ARMS.items():
                out[f"{n}.{suf}"] = (n, [st, "up"])
            if extra:
                out[f"{n}.X1"] = (n, [extra, "up"])
    return out


def main(argv=None):
    ap = argparse.ArgumentParser(); ap.add_argument("--days", default="val"); ap.add_argument("--out", type=Path, required=True); ap.add_argument("--extra", default="")
    ap.add_argument("--arms", default="", help='fin arms as SUFFIX=set,..., e.g. "C=s0_centre,N=s0_nudge" (default W=s0_wide,T=s0_tight)')
    ap.add_argument("--retrain", default="", help="retrain test with fins from this set (arms G, R0, R0s1, RWs0, RWs1)")
    a = ap.parse_args(argv); RETRAIN["fin"] = a.retrain
    if a.arms:
        FIN_ARMS.clear(); FIN_ARMS.update(dict(x.split("=", 1) for x in a.arms.split(",")))
    a.out.mkdir(parents=True, exist_ok=True)
    for f in a.out.glob("emb_*.npz"):
        f.unlink()
    queries = {p: [] for p in ("fin", "body", "fluke")}; have = set()
    for day in day_list(a.days):
        pts = load_points(day); embs = {n: load_emb(n, day) for n in NETS}
        rows, E = [], []
        for part in ("fin", "body", "fluke"):
            A = {k: v for k, v in arms(part, a.extra).items() if embs.get(v[0]) is not None and all(f"{part}|{s}" in embs[v[0]] for s in v[1])}
            for name, (net, sets) in A.items():
                have.add(name)
                for key, (st, cat) in pts[part].items():
                    e = None; used = None
                    for s in sets:
                        e = embs[net].get(f"{part}|{s}", {}).get(key)
                        if e is not None:
                            used = s; break
                    if e is None:
                        continue
                    rows.append(dict(key=key, part=part, variant=name, same=None, angle=0.0, zoom=1.0, outside=0.0, stack=st, cat=cat, used=True, set=used)); E.append(e)
        np.savez_compressed(a.out / f"emb_{day}.npz", E=np.asarray(E, np.float32), rows=json.dumps(rows))
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
                queries[part].append(dict(day=day, key=keys[i], stack=int(st[i]), cat=str(cat[i]), nn={v: keys[nn[v][i]] for v in nn},
                                          hit={v: bool(same[i, nn[v][i]]) for v in nn}, set={v: rows[by[v][keys[i]]]["set"] for v in by}))
    for p, q in queries.items():
        (a.out / f"queries_{p}.json").write_text(json.dumps(q))
    script = ROOT / "experiments" / "interpret_checks" / "chart_variants_score.py"
    for base in (("G", "R0", "R0s1") if a.retrain else ("G", "R0")):
        if base in have:
            print(f"\n##### base {base}", flush=True)
            subprocess.run([sys.executable, str(script), str(a.out), "--base", base, "--json", str(a.out / f"score_base_{base}.json")], check=True)


if __name__ == "__main__":
    main()
