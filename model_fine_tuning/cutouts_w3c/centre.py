"""The centre alone (follow-up 2026-11-18 pm): the parts_up fin cut-out with its own turn, mirror AND scale, only the
centre changed by the W3c dorsal region. Two variants, fixed in RESULTS/rule.json ("centre_variants") before any
validation score:
  centre   the square's centre moved to the region's centroid, then cluster.parts.frame_inside(..., "fit") as parts_up
           (the square is shifted back inside the photo; the scale is parts_up's, which already fits).
  nudge    parts_up unchanged when its square holds the whole region (its extent along the chart's axes) with a 5 %
           margin of the side on every edge; else the smallest shift per chart axis that brings it inside (a region
           too large for the margin is centred on that axis), then frame_inside(..., "fit") as above.
No region: parts_up. Cut-outs at parts_up's 1536 px; a fin the variant leaves unchanged is not written (scoring falls
back to the parts_up file itself).

    PYTHONPATH=. .venv/bin/python -m model_fine_tuning.cutouts_w3c.centre fix          # writes the rule (no score read)
    PYTHONPATH=. .venv/bin/python -m model_fine_tuning.cutouts_w3c.centre clicks [--seed s0]   # -> RESULTS/clickcheck_centre_<seed>.json
    PYTHONPATH=. .venv/bin/python -m model_fine_tuning.cutouts_w3c.centre make val --seed s0    # -> CACHE/<day>/<seed>_centre|_nudge, meta_<seed>_cn.jsonl
    PYTHONPATH=. .venv/bin/python -m model_fine_tuning.cutouts_w3c.centre split --score DIR --arms G.C,R0.C,...
"""
from __future__ import annotations

import argparse
import json
from multiprocessing import get_context

import numpy as np

from . import CACHE, DAYS, RESULTS, day_list
from .choose import FIN_KP, summ
from ..cutouts_v3.clickcheck import CLICK_DAYS, clicks

SIZE = 1536
MARGIN = 0.05
BASE_PER_L = 1 / 1.21


def variants(M_up: np.ndarray, sf: float, FW: int, FH: int, reg: dict):
    """-> {"centre": M, "nudge": M or None (unchanged)}, info"""
    from ..cluster.parts import frame_inside
    A2 = M_up[:, :2]; sc = float(np.hypot(*A2[0])); info = {}
    c_raw = np.asarray(reg["c"], float) * sf
    Mc = np.hstack([A2, (np.array([SIZE / 2, SIZE / 2]) - A2 @ c_raw)[:, None]])
    Mc2 = frame_inside(Mc, SIZE, FW, FH, "fit")
    info["centre_clamp_px"] = round(float(np.abs(Mc2[:, 2] - Mc[:, 2]).max()), 1)
    out = {"centre": Mc2}
    m = MARGIN * SIZE; d = np.zeros(2)
    for i, nm in enumerate(("x", "y")):
        lo, hi = (sc * sf * np.asarray(reg[f"chart_{nm}"]) + M_up[i, 2])
        if hi - lo > SIZE - 2 * m:
            d[i] = SIZE / 2 - (lo + hi) / 2
        elif lo < m:
            d[i] = m - lo
        elif hi > SIZE - m:
            d[i] = SIZE - m - hi
    if np.abs(d).max() < 0.5:
        out["nudge"] = None
    else:
        Mn = M_up.copy(); Mn[:, 2] += d
        Mn2 = frame_inside(Mn, SIZE, FW, FH, "fit")
        info["nudge_px"] = round(float(np.hypot(*d)), 1); info["nudge_clamp_px"] = round(float(np.abs(Mn2[:, 2] - Mn[:, 2]).max()), 1)
        out["nudge"] = Mn2
    return out, info


def centre_of(M):
    return np.linalg.solve(M[:, :2], np.array([SIZE / 2, SIZE / 2]) - M[:, 2])            # RAW px


def cmd_fix(a):
    f = RESULTS / "rule.json"; rule = json.loads(f.read_text())
    rule["centre_variants"] = dict(size=SIZE, margin=MARGIN, note=__doc__.split("No region")[0].strip(),
                                   fixed="2026-11-18 pm, before any validation-day score of these variants")
    f.write_text(json.dumps(rule, indent=1)); print(rule["centre_variants"])


def cmd_clicks(a):
    from PIL import Image
    out = {}
    for days, tag in ((CLICK_DAYS, "both"), ((CLICK_DAYS[0],), CLICK_DAYS[0]), ((CLICK_DAYS[1],), CLICK_DAYS[1])):
        res = {"A": [], "centre": [], "nudge": []}; changed = {"centre": 0, "nudge": 0}; n = 0
        for day in days:
            R = json.loads((CACHE / "regions" / a.seed / f"{day}.json").read_text())
            for key, kp in clicks(day).items():
                if not all(k in kp for k in FIN_KP) or key not in R or not R[key]["up_fin"]:
                    continue
                W, H = Image.open(DAYS / day / "images" / f"{key.rsplit('~', 1)[0]}.jpg").size
                z = np.load(DAYS / day / "parts_up" / f"{key}__fin.npz"); M_up = z["M"].astype(float); sf = float(z["scale_full"])
                tri = np.array([[kp[k][0] * W * sf, kp[k][1] * H * sf] for k in FIN_KP]); n += 1
                Ms = {"A": M_up}
                if R[key]["reg"] is not None:
                    v, _ = variants(M_up, sf, round(W * sf), round(H * sf), R[key]["reg"])
                    Ms["centre"] = v["centre"]; changed["centre"] += 1
                    Ms["nudge"] = v["nudge"] if v["nudge"] is not None else M_up; changed["nudge"] += v["nudge"] is not None
                else:
                    Ms["centre"] = Ms["nudge"] = M_up
                base = float(np.linalg.norm(tri[0] - tri[1])); ctr = tri.mean(0)
                for nm, M in Ms.items():
                    sc = float(np.hypot(*M[0, :2])); p = tri @ M[:, :2].T + M[:, 2]; ins = (p >= 0) & (p < SIZE)
                    res[nm].append(dict(off=float(np.linalg.norm(centre_of(M) - ctr) / base), share=base * sc / SIZE, tip_in=bool(ins[2].all()), tri_in=bool(ins.all()),
                                        tri_in_m10=bool(((p >= 0.1 * SIZE) & (p <= 0.9 * SIZE)).all())))
        out[tag] = {nm: summ(rr) for nm, rr in res.items()}; out[tag]["changed"] = changed; out[tag]["n"] = n
    print(json.dumps(out, indent=1)); (RESULTS / f"clickcheck_centre_{a.seed}.json").write_text(json.dumps(out, indent=1))


def work(job):
    day, stem, source, keys, seed = job
    import cv2
    from PIL import Image
    from ..cluster.chart import decode_full
    from ..cluster.instances_v2 import old_instances
    from ..common import geometry, paths
    from ..cutouts_v3 import make as v3
    cv2.setNumThreads(1)
    d = DAYS / day; rows = []
    try:
        W, H = Image.open(d / "images" / f"{stem}.jpg").size
        okeys, omasks = old_instances(d, stem, W, H, geometry.load_folded_table(paths.FOLDED_TABLE)); old = dict(zip(okeys, omasks))
        full = None
        for key in keys:
            r = REG[day].get(key)
            if key not in old or r is None or not r["up_fin"]:
                continue
            g = r["reg"]
            if g is None:
                rows.append(dict(key=key, src="A")); continue
            if full is None:
                full = decode_full(source)
            z = np.load(d / "parts_up" / f"{key}__fin.npz"); M_up = z["M"].astype(float); sf = float(z["scale_full"]); mirrored = bool(z["mirrored"])
            v, info = variants(M_up, sf, full.shape[1], full.shape[0], g)
            L = float(g["L_pca"]); cA = centre_of(M_up) / sf
            row = dict(key=key, src="w3c", L=round(L, 1), **info)
            for nm, M in v.items():
                row[f"{nm}_moved_bases"] = 0.0 if M is None else round(float(np.linalg.norm(centre_of(M) / sf - cA) / (L * BASE_PER_L)), 3)
                if M is None:
                    continue
                sc = float(np.hypot(*M[0, :2]))
                chart = cv2.warpAffine(full, M, (SIZE, SIZE), flags=cv2.INTER_AREA if sc < 1 else cv2.INTER_CUBIC)
                whale = v3.warp_mask2560(old[key], M, sf, SIZE)
                if mirrored:
                    chart, whale = chart[:, ::-1].copy(), whale[:, ::-1].copy()
                v3.save(CACHE / day / f"{seed}_{nm}", key, chart, dict(M=M, mirrored=mirrored, whale=whale, size=SIZE))
            rows.append(row)
        return stem, rows, None
    except Exception as e:  # noqa: BLE001
        import traceback
        return stem, rows, f"{type(e).__name__}: {e} {traceback.format_exc()[-400:]}"


REG: dict = {}


def _init(day, seed):
    REG[day] = json.loads((CACHE / "regions" / seed / f"{day}.json").read_text())


def cmd_make(a):
    from ..cutouts_v3 import make as v3
    for day in day_list(a.days):
        jobs = [(day, s, src, gk, a.seed) for _, s, src, fk, gk in v3.jobs_of(day) if gk]
        rows, nerr = [], 0
        with get_context("spawn").Pool(min(a.workers, 10), initializer=_init, initargs=(day, a.seed)) as pool:
            for stem, rr, err in pool.imap_unordered(work, jobs, chunksize=1):
                rows += rr
                if err:
                    nerr += 1; print("  error", stem[-40:], err, flush=True)
        (CACHE / day / f"meta_{a.seed}_cn.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))
        w = [r for r in rows if r["src"] == "w3c"]
        print(f"{a.seed} {day}: fins {len(rows)}, region {len(w)}, nudged {sum(r['nudge_moved_bases'] > 0 for r in w)}, centre clamped {sum(r['centre_clamp_px'] > 0.5 for r in w)}, "
              f"nudge clamped {sum(r.get('nudge_clamp_px', 0) > 0.5 for r in w)}, {nerr} errors", flush=True)


def cmd_split(a):
    q = json.loads((a.score / "queries_fin.json").read_text()); M = {}
    for d in {x["day"] for x in q}:
        M.update({(d, json.loads(l)["key"]): json.loads(l) for l in open(CACHE / d / f"meta_{a.seed}_cn.jsonl")})
    bins = (("no region", None), ("moved 0 (unchanged)", (-1, 1e-9)), ("moved <= 0.5 base", (1e-9, 0.5)), ("moved 0.5-1 base", (0.5, 1.0)), ("moved > 1 base", (1.0, 1e9)))
    out = {}
    for arm in a.arms.split(","):
        base = arm.split(".")[0]; var = {"C": "centre", "N": "nudge"}[arm.split(".")[1][0]]
        for nm, rng in bins:
            sel = []
            for x in q:
                m = M.get((x["day"], x["key"]))
                if m is None or m["src"] != "w3c":
                    ok = rng is None
                else:
                    mv = m[f"{var}_moved_bases"]; ok = rng is not None and rng[0] < mv <= rng[1]
                if ok:
                    sel.append(x)
            if sel:
                b = np.array([x["hit"][base] for x in sel]); h = np.array([x["hit"][arm] for x in sel])
                out[f"{arm} {nm}"] = dict(n=len(sel), base=round(float(b.mean()), 3), arm=round(float(h.mean()), 3), found=int((h & ~b).sum()), lost=int((~h & b).sum()))
                print(arm, nm, out[f"{arm} {nm}"])
    if a.json:
        a.json.write_text(json.dumps(out, indent=1))


def main(argv=None):
    from pathlib import Path
    ap = argparse.ArgumentParser(); ap.add_argument("cmd", choices=["fix", "clicks", "make", "split"]); ap.add_argument("days", nargs="?", default="val")
    ap.add_argument("--seed", default="s0"); ap.add_argument("--workers", type=int, default=10); ap.add_argument("--score", type=Path); ap.add_argument("--arms", default="G.C,G.N,R0.C,R0.N")
    ap.add_argument("--json", type=Path)
    a = ap.parse_args(argv)
    {"fix": cmd_fix, "clicks": cmd_clicks, "make": cmd_make, "split": cmd_split}[a.cmd](a)


if __name__ == "__main__":
    main()
