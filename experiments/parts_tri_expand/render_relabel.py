"""Render dorsal label: the OLD label against the NEW one by rule A (factory layer dorsal_triA, round 21d), per whale.

    .venv/bin/python experiments/parts_tri_expand/render_relabel.py stats   -> render_relabel.json + render_relabel.md
    .venv/bin/python experiments/parts_tri_expand/render_relabel.py sheet   -> SHEETS/triA_render_labels.jpg (+ _key.json)

Sets: r06r_check20, r06r_check21 (old = parts/ class 2 of the whale's own pixels, new = dorsal_triA == k; 2560 px) and
run05 (old = the parts_seg label cache class 2, new = labels_triA class 2; the primary only; 1280 x 720).
Read-only on the factory folders; sheets never inside the repository.
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
from model_fine_tuning.parts_weak import tri_expand as TE  # noqa: E402

FACTORY = REPO.parent / "Whale-Image-Data-Factory" / "data" / "renders"
OUT = Path(__file__).resolve().parent
SHEETS = Path("/tmp/claude-1072903897/-home-sat3737-NOAA-whales/4fbba7ff-3ba5-4b50-9cb7-970c81250409/scratchpad/triA/sheets")
CACHE = Path("/home/sat3737/NOAA_whales/moonshot_cache/parts_tri_expand")
R05_OLD, R05_NEW = Path("/var/tmp/sat3737-parts-cache/run05/labels"), Path("/var/tmp/sat3737-parts-cache/run05/labels_triA")
FP = ("dorsal_fin_tip", "dorsal_fin_front", "dorsal_fin_back")
RULE = json.loads((OUT / "rule.json").read_text())["params"]


def rd(p: Path) -> np.ndarray:
    a = np.asarray(Image.open(p))
    return a[..., 0] if a.ndim == 3 else a


def view_bin(s):
    if s is None:
        return "?"
    return "side-on (<30)" if s < 30 else ("oblique (30-60)" if s < 60 else "end-on (>60)")


def below_stats(mask: np.ndarray, T, F, B) -> dict:
    """share of the mask under the base line, and its depth (5th percentile, lowest) in tip heights h."""
    fr = TE.frame(T, F, B)
    ys, xs = np.nonzero(mask)
    if fr is None or not len(ys):
        return {"below_share": None, "depth_p5_h": None, "depth_min_h": None}
    Q = np.stack([xs, ys], 1).astype(float) - fr["B"]
    up = Q @ fr["n"] / fr["h"]
    return {"below_share": float((up < 0).mean()), "depth_p5_h": float(np.percentile(up, 5)), "depth_min_h": float(up.min())}


def whales(set_name: str):
    """yield (rec, arrays) per whale with old or new dorsal pixels or with all three fin points visible."""
    ds = FACTORY / set_name / "dataset"
    r05 = set_name == "run05"
    for mp in sorted((ds / "meta").glob("*.json")):
        sid = mp.stem
        if not (ds / "dorsal_triA" / f"{sid}.png").exists():
            continue
        m = json.loads(mp.read_text())
        inst = rd(ds / "instances" / f"{sid}.png")
        did = rd(ds / "dorsal_triA" / f"{sid}.png")
        if r05:
            if not (R05_OLD / f"{sid}.png").exists():
                continue
            old_l, new_l = rd(R05_OLD / f"{sid}.png"), rd(R05_NEW / f"{sid}.png")
            ans = [{"instance_id": 1, "keypoints": m.get("keypoints") or {}}]
        else:
            old_l, new_l = rd(ds / "parts" / f"{sid}.png"), rd(ds / "parts_triA" / f"{sid}.png")
            ans = m.get("animals") or []
        for a in ans:
            k = a["instance_id"]
            own = inst == k
            old = (old_l == 2) & own
            new = (new_l == 2) & own
            states = tuple((a["keypoints"].get(n) or {}).get("state", "missing") for n in FP)
            applied = all(s == "visible" for s in states)
            if not (old.any() or new.any() or applied):
                continue
            rec = {"set": set_name, "sid": sid, "k": k, "states": states, "applied": applied, "old_px": int(old.sum()), "new_px": int(new.sum()),
                   "both_px": int((old & new).sum()), "to_ignore_px": int((old & (new_l == 255)).sum()), "own_px": int(own.sum()),
                   "n_animals": int(m.get("n_animals") or (1 + int(bool(m.get("companion"))))) if not r05 else 1 + int(bool(m.get("companion")))}
            if r05:
                b = (m.get("orientation") or {}).get("relative_bearing_deg")
                rec["sight_deg"] = None if b is None else float(abs((b % 180) - 90))
                rec["blade_vis"] = None
            else:
                rec["sight_deg"] = ((a.get("part_planes") or {}).get("fin") or {}).get("sight_deg")
                rec["blade_vis"] = (a.get("dorsal_blade_pixels") or {}).get("visible")
                rec["category"] = a.get("category")
            n_vis = sum(s == "visible" for s in states)
            rec["finless_back"] = (not r05) and ((rec["blade_vis"] or 0) < 100 or n_vis < 2)
            if applied:
                T, F, B = (np.array(a["keypoints"][n]["uv"], float) for n in FP)
                fr = TE.frame(T, F, B)
                if fr is None:                                   # base under 2 px: rule A paints nothing
                    rec["degenerate"] = True
                    yield rec
                    continue
                rec.update(base=fr["base"], h=fr["h"], base_over_h=fr["base"] / max(fr["h"], 1e-6))
                rec["old_below"] = below_stats(old, T, F, B)
                rec["new_below"] = below_stats(new, T, F, B)
                if not r05 and (inst > 0).sum() > own.sum():
                    # the helper's whale mask (any whale) against the own instance mask: where a fin stands in front of another whale
                    e_any = TE.expand(T, F, B, inst > 0, **RULE)
                    rec["any_mask_extra_px"] = int((e_any & ~own).sum())
                rec["pts"] = [T.tolist(), F.tolist(), B.tolist()]
            yield rec


def iou(r):
    u = r["old_px"] + r["new_px"] - r["both_px"]
    return r["both_px"] / u if u else np.nan


def boot(x, n=2000, seed=0):
    x = np.asarray([v for v in x if v is not None and np.isfinite(v)], float)
    if not len(x):
        return [None, None]
    rng = np.random.default_rng(seed)
    b = rng.choice(x, (n, len(x))).mean(1)
    return [round(float(np.percentile(b, 2.5)), 3), round(float(np.percentile(b, 97.5)), 3)]


def med(v, q=50):
    v = [x for x in v if x is not None and np.isfinite(x)]
    return round(float(np.percentile(v, q)), 3) if v else float("nan")


def block(rows):
    a = [r for r in rows if r["applied"] and r["old_px"] > 0 and not r.get("degenerate")]
    out = {"n_applied_with_old": len(a), "degenerate_points (base < 2 px)": sum(bool(r.get("degenerate")) for r in rows)}
    if not a:
        return out
    io = [iou(r) for r in a]
    drop = [(r["old_px"] - r["both_px"]) / r["old_px"] for r in a]
    add_new = [(r["new_px"] - r["both_px"]) / max(r["new_px"], 1) for r in a]
    ratio = [r["new_px"] / r["old_px"] for r in a]
    out.update(iou_mean=round(float(np.mean(io)), 3), iou_ci=boot(io), iou_median=round(float(np.median(io)), 3),
               old_dropped_share_mean=round(float(np.mean(drop)), 3), new_added_share_mean=round(float(np.mean(add_new)), 3),
               new_over_old_area_median=round(float(np.median(ratio)), 3),
               old_below_share_median=med([r["old_below"]["below_share"] for r in a]),
               new_below_share_median=med([r["new_below"]["below_share"] for r in a]),
               old_depth_p5_median=med([r["old_below"]["depth_p5_h"] for r in a]),
               new_depth_p5_median=med([r["new_below"]["depth_p5_h"] for r in a]),
               old_depth_min_p5=med([r["old_below"]["depth_min_h"] for r in a], 5),
               new_depth_min_p5=med([r["new_below"]["depth_min_h"] for r in a], 5))
    return out


def cmd_stats(a) -> None:
    res, md = {}, []
    allrows = {}
    for st in a.sets:
        rows = list(whales(st))
        allrows[st] = rows
        app = [r for r in rows if r["applied"]]
        nap = [r for r in rows if not r["applied"]]
        s = {"whales_considered": len(rows), "applied": len(app), "applied_without_old_dorsal": sum(r["old_px"] == 0 for r in app),
             "applied_new_empty": sum(r["new_px"] == 0 for r in app),
             "not_applied": len(nap), "not_applied_with_old_dorsal": sum(r["old_px"] > 0 for r in nap),
             "old_px_total": sum(r["old_px"] for r in rows), "old_px_not_applied": sum(r["old_px"] for r in nap),
             "not_applied_by_states": {}, "all": block(rows), "view": {}, "animals": {}, "finless_back": {}, "flat (base/h > 3: h_eff = base/3)": {}}
        by = defaultdict(list)
        for r in nap:
            key = "+".join(sorted(set(r["states"])))
            by[key].append(r)
        for kk, v in sorted(by.items(), key=lambda kv: -sum(r["old_px"] for r in kv[1])):
            s["not_applied_by_states"][kk] = {"whales": len(v), "with_old_dorsal": sum(r["old_px"] > 0 for r in v), "old_px": sum(r["old_px"] for r in v),
                                              "old_px_median_where_present": int(np.median([r["old_px"] for r in v if r["old_px"] > 0])) if any(r["old_px"] > 0 for r in v) else 0}
        for name, f, dest in (("view", lambda r: view_bin(r["sight_deg"]), "view"),
                              ("animals", lambda r: "one whale" if r["n_animals"] == 1 else "several whales", "animals"),
                              ("finless_back", lambda r: "finless back (blade < 100 px or < 2 fin points visible)" if r["finless_back"] else "fin", "finless_back"),
                              ("flat", lambda r: "base/h > 3" if r.get("base_over_h", 0) > 3 else "base/h <= 3", "flat (base/h > 3: h_eff = base/3)")):
            g = defaultdict(list)
            for r in rows:
                g[f(r)].append(r)
            for kk, v in sorted(g.items()):
                s[dest][kk] = block(v) | {"whales": len(v), "with_old_dorsal": sum(r["old_px"] > 0 for r in v),
                                          "not_applied": sum(not r["applied"] for r in v), "old_px_to_ignore": sum(r["to_ignore_px"] for r in v)}
        multi = [r for r in app if "any_mask_extra_px" in r]
        s["fin_in_front_of_another_whale"] = {"applied_in_multi_whale_shots": len(multi),
                                               "any_whale_mask_would_add_other_whale_px (whales)": sum(r["any_mask_extra_px"] > 0 for r in multi),
                                               "px_median_where_present": int(np.median([r["any_mask_extra_px"] for r in multi if r["any_mask_extra_px"] > 0])) if any(r["any_mask_extra_px"] > 0 for r in multi) else 0}
        res[st] = s
    (OUT / "render_relabel.json").write_text(json.dumps(res, indent=1))
    rows_all = [r for v in allrows.values() for r in v]
    CACHE.mkdir(parents=True, exist_ok=True)
    (CACHE / "render_relabel_rows.json").write_text(json.dumps(rows_all))
    md += table(res)
    (OUT / "render_relabel.md").write_text("\n".join(md) + "\n")
    print("\n".join(md))


def table(res):
    L = ["| set | group | whales | applied with old label | IoU old vs new, mean [95 %] | old px dropped | new px added (share of new) | new / old area (median) | under the base line old / new (median share) | 5th pct depth old / new (h, median) |",
         "|---|---|---|---|---|---|---|---|---|---|"]
    for st, s in res.items():
        for gname in ("all", "view", "animals", "finless_back", "flat (base/h > 3: h_eff = base/3)"):
            items = [("all", s["all"] | {"whales": s["whales_considered"]})] if gname == "all" else list(s[gname].items())
            for g, b in items:
                if "iou_mean" not in b:
                    L.append(f"| {st} | {g} | {b.get('whales')} | {b.get('n_applied_with_old', 0)} | - | - | - | - | - | - |")
                    continue
                L.append(f"| {st} | {g} | {b.get('whales')} | {b['n_applied_with_old']} | {b['iou_mean']:.3f} [{b['iou_ci'][0]:.2f}, {b['iou_ci'][1]:.2f}] | "
                         f"{b['old_dropped_share_mean']:.3f} | {b['new_added_share_mean']:.3f} | {b['new_over_old_area_median']:.2f} | "
                         f"{b['old_below_share_median']:.2f} / {b['new_below_share_median']:.2f} | {b['old_depth_p5_median']:.2f} / {b['new_depth_p5_median']:.2f} |")
    L += ["", "Not applied (a fin point not visible): whales, with an old dorsal label, old pixels (they become ignore, 255):", "",
          "| set | states of the three points | whales | with old dorsal | old px | old px per whale (median, where present) |", "|---|---|---|---|---|---|"]
    for st, s in res.items():
        for kk, v in s["not_applied_by_states"].items():
            L.append(f"| {st} | {kk} | {v['whales']} | {v['with_old_dorsal']} | {v['old_px']} | {v['old_px_median_where_present']} |")
        L.append(f"| {st} | **total** | {s['not_applied']} | {s['not_applied_with_old_dorsal']} | {s['old_px_not_applied']} of {s['old_px_total']} old px | |")
    return L


# ---------------------------------------------------------------------------------------------------------- sheet
COL = {1: (230, 60, 60), 2: (60, 200, 60), 3: (240, 200, 40), 4: (70, 120, 255), 5: (150, 150, 150), 255: (255, 0, 255)}


def overlay(img, lab, own):
    o = img.copy().astype(np.float32)
    for c, col in COL.items():
        m = (lab == c) & own if c != 255 else (lab == 255) & own
        o[m] = 0.45 * o[m] + 0.55 * np.array(col, np.float32)
    return o.astype(np.uint8)


def tile(r, size=150):
    st, sid, k = r["set"], r["sid"], r["k"]
    ds = FACTORY / st / "dataset"
    img = np.asarray(Image.open(ds / "images" / f"{sid}.png").convert("RGB"))
    inst = rd(ds / "instances" / f"{sid}.png")
    own = inst == k
    if st == "run05":
        old_l, new_l = rd(R05_OLD / f"{sid}.png"), rd(R05_NEW / f"{sid}.png")
    else:
        old_l, new_l = rd(ds / "parts" / f"{sid}.png"), rd(ds / "parts_triA" / f"{sid}.png")
    m = ((old_l == 2) | (new_l == 2) | (new_l == 255) & (old_l == 2)) & own
    ys, xs = np.nonzero(m)
    if not len(ys):
        return None
    cy, cx = (ys.min() + ys.max()) / 2, (xs.min() + xs.max()) / 2
    s = max(24, 0.9 * max(ys.max() - ys.min(), xs.max() - xs.min()))
    y0, y1, x0, x1 = int(max(0, cy - s)), int(min(img.shape[0], cy + s)), int(max(0, cx - s)), int(min(img.shape[1], cx + s))
    crops = [img[y0:y1, x0:x1], overlay(img, old_l, own)[y0:y1, x0:x1], overlay(img, new_l, own)[y0:y1, x0:x1]]
    t = [cv2.resize(c, (size, size), interpolation=cv2.INTER_NEAREST) for c in crops]
    if r.get("pts"):
        sc = size / max(1, (x1 - x0)), size / max(1, (y1 - y0))
        P = [(int((p[0] - x0) * sc[0]), int((p[1] - y0) * sc[1])) for p in r["pts"]]
        cv2.polylines(t[2], [np.array(P, np.int32)], True, (255, 255, 0), 1)
    out = np.hstack([np.pad(x, ((1, 1), (1, 1), (0, 0)), constant_values=255) for x in t])
    lab = f"{st[-7:]} {sid}:{k} " + (f"IoU {iou(r):.2f}" if r["applied"] and r["old_px"] else ("undef" if not r["applied"] else "no old"))
    cv2.putText(out, lab, (3, 12), cv2.FONT_HERSHEY_SIMPLEX, 0.38, (0, 0, 0), 3)
    cv2.putText(out, lab, (3, 12), cv2.FONT_HERSHEY_SIMPLEX, 0.38, (255, 255, 0), 1)
    return out


def cmd_sheet(a) -> None:
    rows = json.loads((CACHE / "render_relabel_rows.json").read_text())
    pool = [r for r in rows if r["set"] in ("r06r_check20", "r06r_check21") and (r["old_px"] > 0 or r["new_px"] > 0)]
    rng = np.random.default_rng(a.seed)
    pick = [pool[i] for i in rng.choice(len(pool), 40, replace=False)]
    worst = sorted([r for r in pool if r["applied"] and r["old_px"] >= 30 and r not in pick], key=iou)[:8]
    tiles, key = [], []
    for n, r in enumerate(pick + worst):
        t = tile(r)
        if t is None:
            continue
        cv2.putText(t, str(n + 1), (3, t.shape[0] - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1)
        tiles.append(t)
        key.append({"tile": n + 1, "group": "random (seed %d)" % a.seed if n < 40 else "differs most", **{k: r[k] for k in ("set", "sid", "k", "states", "applied", "old_px", "new_px", "sight_deg", "n_animals")},
                    "iou": None if not (r["applied"] and r["old_px"]) else round(iou(r), 3)})
    cols = 4
    while len(tiles) % cols:
        tiles.append(np.full_like(tiles[0], 255))
    sheet = np.vstack([np.hstack([np.pad(t, ((3, 3), (3, 3), (0, 0)), constant_values=255) for t in tiles[i:i + cols]]) for i in range(0, len(tiles), cols)])
    head = np.full((22, sheet.shape[1], 3), 255, np.uint8)
    cv2.putText(head, "each tile: render | OLD dorsal (green; other parts tinted) | NEW rule A (green; magenta = ignore, yellow = fin keypoint triangle). Tiles 1-40 random, 41-48 lowest IoU",
                (4, 15), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (0, 0, 0), 1)
    SHEETS.mkdir(parents=True, exist_ok=True)
    out = SHEETS / "triA_render_labels.jpg"
    cv2.imwrite(str(out), cv2.cvtColor(np.vstack([head, sheet]), cv2.COLOR_RGB2BGR), [cv2.IMWRITE_JPEG_QUALITY, 88])
    (SHEETS / "triA_render_labels_key.json").write_text(json.dumps(key, indent=1))
    print(out, len(tiles))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["stats", "sheet"])
    ap.add_argument("--sets", nargs="+", default=["r06r_check20", "r06r_check21", "run05"])
    ap.add_argument("--seed", type=int, default=21)
    a = ap.parse_args()
    {"stats": cmd_stats, "sheet": cmd_sheet}[a.cmd](a)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
