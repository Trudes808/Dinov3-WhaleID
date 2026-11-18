"""Dorsal-fin outline from three points, real against renders: the researcher's clicks (dorsal_fin_front / _tip / _back,
Whale-Review-Web data/2025-01-02 and 2025-03-18 pair_marks.json, read only; training days) against the render keypoints
of the same names (meta animals[k].keypoints uv, all three visible). Same convention on both sides (front = front of the
whole fin / hump, keypoint revision 2026-11-02). Per fin, in the image:
  aspect = height of the tip above the front-back line / chord (front-back distance)
  lean   = where the tip sits along the chord, 0 above the front, 1 above the back, > 1 swept behind the back
A labels-based small-fin measure that needs no by-eye look. View-dependent on both sides (2-D projections).

    PYTHONPATH=. .venv/bin/python -m model_fine_tuning.part_gaps.fin_shape [--sets run05 r06r_test_final ... r06r_test_r7]
"""
from __future__ import annotations

import argparse
import json

import numpy as np
from PIL import Image

from . import common as C

LAB = C.ROOT.parent / "Whale-Review-Web" / "data"
CLICK_DAYS = ("2025-01-02", "2025-03-18")
MIN_CHORD = 12.0          # px in the 2560 frame


def shape(f, t, b):
    f, t, b = map(np.asarray, (f, t, b))
    c = b - f
    L = float(np.hypot(*c))
    if L < 1e-6:
        return None
    u = c / L
    v = t - f
    along = float(v @ u) / L
    h = abs(float(u[0] * v[1] - u[1] * v[0])) / L
    return L, h, along


def real():
    out = []
    for day in CLICK_DAYS:
        C.check_day(day)
        labs = json.loads((LAB / day / "pair_marks.json").read_text())
        labs = labs["labels"] if isinstance(labs, dict) else labs
        seen = set()
        for l in labs:
            if l.get("verdict") == "skip":
                continue
            for sd in ("a", "b"):
                kp = (l.get("keypoints") or {}).get(sd) or {}
                key = l.get(f"key_{sd}")
                if not key or key in seen or not all(k in kp for k in ("dorsal_fin_front", "dorsal_fin_tip", "dorsal_fin_back")):
                    continue
                seen.add(key)
                W, H = Image.open(C.DAYS_ALL / day / "images" / f"{key.rsplit('~', 1)[0]}.jpg").size
                s = 2560.0 / W
                p = {k: (kp[k]["x"] * W * s, kp[k]["y"] * H * s) for k in ("dorsal_fin_front", "dorsal_fin_tip", "dorsal_fin_back")}
                r = shape(p["dorsal_fin_front"], p["dorsal_fin_tip"], p["dorsal_fin_back"])
                if r and r[0] >= MIN_CHORD:
                    out.append(r)
    return out


def renders(tag):
    ds = C.FACTORY / tag / "dataset"
    man = [json.loads(l) for l in open(ds / "manifest.jsonl") if l.strip()]
    out = []
    for m in man:
        if m.get("negative") or m.get("rejected"):
            continue
        meta = json.loads((ds / "meta" / f"{m['id']}.json").read_text())
        W = meta.get("frame", {}).get("width") if isinstance(meta.get("frame"), dict) else None
        W = W or Image.open(ds / "images" / f"{m['id']}.png").size[0]
        s = 2560.0 / W
        animals = meta.get("animals") or [{"keypoints": meta.get("keypoints", {}), "fin_spec": None}]
        for a in animals:
            kp = a.get("keypoints") or {}
            ks = ("dorsal_fin_front", "dorsal_fin_tip", "dorsal_fin_back")
            if not all(k in kp and kp[k].get("state") == "visible" for k in ks):
                continue
            r = shape(*[np.asarray(kp[k]["uv"]) * s for k in ks])
            if r and r[0] >= MIN_CHORD:
                out.append(r + ((a.get("fin_spec") or {}).get("dorsal_height"),))
    return out


def summ(rows):
    a = np.array([r[1] for r in rows]); l = np.array([r[2] for r in rows])
    q = lambda x: [round(float(v), 2) for v in np.percentile(x, [25, 50, 75])]  # noqa: E731
    return {"n": len(rows), "aspect_q25_50_75": q(a), "lean_q25_50_75": q(l),
            "share_aspect_lt_0.35": round(float(np.mean(a < 0.35)), 2), "share_lean_gt_0.8": round(float(np.mean(l > 0.8)), 2)}


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--sets", nargs="*", default=["run05", "r06r_test_final", "r06r_test_check", "r06r_test_r3", "r06r_test_r4b",
                                                  "r06r_test_r5", "r06r_test_r6", "r06r_test_r6b", "r06r_test_r7", "r06r_test_r8"])
    a = ap.parse_args(argv)
    res = {"real_clicks": summ(real())}
    for t in a.sets:
        if (C.FACTORY / t / "dataset" / "manifest.jsonl").exists():
            rows = renders(t)
            if rows:
                res[t] = summ(rows)
    (C.OUT / "fin_shape.json").write_text(json.dumps(res, indent=1))
    print(f"{'set':18s} {'n':>5s}  aspect (height/chord) q25/50/75   lean q25/50/75   share aspect<0.35  share lean>0.8")
    for k, v in res.items():
        print(f"{k:18s} {v['n']:5d}  {v['aspect_q25_50_75']}   {v['lean_q25_50_75']}   {v['share_aspect_lt_0.35']}   {v['share_lean_gt_0.8']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


# ---------------------------------------------------------------------------------------------- blade spec route (r8 on)
TEMPLATES = C.FACTORY.parent.parent / "third_party_assets" / "humpback_4k" / "blade_templates.json"


def real_blade_metrics():
    """broad (chord at 0.75 H / H), sharp (chord at 0.9 H / chord at 0.75 H), overhang (trailing edge at 0.75 H) and base / H
    of the 196 real fin outlines the factory's blade library is built from (same formulas as factory.postprocess._fin_spec)."""
    d = json.loads(TEMPLATES.read_text())
    H = np.asarray(d["heights"])
    out = []
    for t in d["templates"]:
        tr, le = np.asarray(t["trail"]), np.asarray(t["lead"])
        c75, c90 = np.interp(0.75, H, le - tr), np.interp(0.9, H, le - tr)
        out.append((c75, c90 / max(c75, 1e-6), np.interp(0.75, H, tr), t["base_over_height"]))
    return np.array(out)


def blade_metrics(tag, eye_file=None):
    """per rendered whale with a visible dorsal fin: its fin_spec.blade numbers (+ the by-eye mark when labelled)."""
    ds = C.FACTORY / tag / "dataset"
    eye = {}
    if eye_file and eye_file.exists():
        for l in open(eye_file):
            if l.strip() and not l.startswith("#"):
                s, i, code = l.split(None, 2)
                eye[(s, int(i))] = code.strip()
    rows = []
    for m in (json.loads(l) for l in open(ds / "manifest.jsonl") if l.strip()):
        if m.get("negative") or m.get("rejected"):
            continue
        meta = json.loads((ds / "meta" / f"{m['id']}.json").read_text())
        for a in meta.get("animals") or []:
            b = (a.get("fin_spec") or {}).get("blade")
            if not b or not (a.get("parts_visible") or {}).get("dorsal"):
                continue
            rows.append({"shot": m["id"], "iid": int(a["instance_id"]), "broad": b["broad"], "sharp": b["sharp"],
                         "overhang": b["overhang"], "base_over_height": b["base_over_height"],
                         "eye": eye.get((m["id"], int(a["instance_id"])))})
    return rows


def blade_compare(tag):
    R = real_blade_metrics()
    rows = blade_metrics(tag, C.OUT / f"labels_ai_{tag}.txt")
    if not rows:
        return None
    names = ("broad", "sharp", "overhang", "base_over_height")
    X = np.array([[r[n] for n in names] for r in rows])
    lo, hi = np.percentile(R, 10, axis=0), np.percentile(R, 90, axis=0)

    def ks(a, b):
        v = np.sort(np.concatenate([a, b]))
        return float(np.max(np.abs(np.searchsorted(np.sort(a), v, "right") / len(a) - np.searchsorted(np.sort(b), v, "right") / len(b))))

    res = {"n_render_fins": len(rows), "n_real_outlines": len(R), "metrics": {}}
    for j, n in enumerate(names):
        res["metrics"][n] = {"real_q25_50_75": np.percentile(R[:, j], [25, 50, 75]).round(2).tolist(),
                             "render_q25_50_75": np.percentile(X[:, j], [25, 50, 75]).round(2).tolist(), "ks": round(ks(R[:, j], X[:, j]), 2)}
    inside = np.all((X >= lo) & (X <= hi), axis=1)
    real_inside = float(np.mean(np.all((R >= lo) & (R <= hi), axis=1)))
    res["share_inside_real_10_90_all_metrics"] = {"render": round(float(inside.mean()), 2), "real": round(real_inside, 2)}
    by = {}
    for r, ins in zip(rows, inside):
        mk = "R" if r["eye"] and "#R" in r["eye"] else "S" if r["eye"] and "#S" in r["eye"] else "-"
        by.setdefault(mk, []).append((r["broad"], r["sharp"], r["overhang"], ins))
    res["by_eye_mark"] = {k: {"n": len(v), "broad_median": round(float(np.median([x[0] for x in v])), 2),
                              "sharp_median": round(float(np.median([x[1] for x in v])), 2),
                              "overhang_median": round(float(np.median([x[2] for x in v])), 2),
                              "inside_real_range": round(float(np.mean([x[3] for x in v])), 2)} for k, v in by.items()}
    return res


if __name__ == "__blade__":
    pass
