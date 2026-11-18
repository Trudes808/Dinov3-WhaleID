"""Paired test of the user's reflection idea (2026-11-17): the SAME 1 000 run06 shots rendered with the mirrored
whale in the water (r06_pair_m1, as run04 / run05) and without it (r06_pair_m2). Segmentation models trained on one
arm or the other, renders only (no real labels), compared on real photos with the measures of why.py / reflect.py.

    python -m model_fine_tuning.seg_real.pair split        split (85 / 15, by category) + masks/ byte check between arms
    python -m model_fine_tuning.seg_real.pair pairsheet    30 head-on / awash pairs m1 | m2 for the by-eye count (scratchpad)
    python -m model_fine_tuning.seg_real.pair renders      (GPU) every model on BOTH arms' validation renders
    python -m model_fine_tuning.seg_real.pair real         (CPU, cached predictions) fade / waterline / below-edge / kept
    python -m model_fine_tuning.seg_real.pair knn          (CPU) the mirror-band nearest-neighbour measure per model
    python -m model_fine_tuning.seg_real.pair realsheet    real whales, r04 | each arm (seed 0), scratchpad
    python -m model_fine_tuning.seg_real.pair summary      -> experiments/seg_pair/pair_results.json, pair_tables.md
Models: A_m{1,2}_s{0,1} = renders only from the DINOv3 weights (seg_r05_ft4 recipe, 30 epochs on the pair's train split);
B_m{1,2}_s{0,1} = seg_r05_ft4 fine-tuned 15 epochs on the arm. Plus the baselines r04, r05 (cached).
"""
from __future__ import annotations

import argparse
import hashlib
import json
import random
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np

from . import common as C
from .data import blob_1280
from .predict import load_prob, resize_1280

R = Path("/home/sat3737/NOAA_whales/Whale-Image-Data-Factory/data/renders")
ARMS = {"m1": R / "r06_pair_m1", "m2": R / "r06_pair_m2"}
DROP = {"000893"}
OUT = C.REPO / "experiments" / "seg_pair"
SPLIT = OUT / "split_pair.json"
CK = Path("/var/tmp/sat3737-parts-cache/seg_pair")
MODELS = [f"{r}_{m}_s{s}" for r in ("A", "B") for m in ("m1", "m2") for s in (0, 1)]
SH = C.SHEETS / "pair"


def ckpt(name: str) -> Path:
    return {"r04": C.SEG_R04, "r05": C.SEG_R05}.get(name, C.REPO / "experiments" / f"seg_pair_{name}" / "checkpoints" / "last.pt")


def cmd_split(a) -> None:
    from ..common.splits import filter_rows, load_manifest, make_split, summarize
    rows = load_manifest(ARMS["m2"]); rows1 = {r["id"]: r for r in load_manifest(ARMS["m1"])}
    assert {r["id"] for r in rows} == set(rows1), "the two arms have different shot ids"
    # pairs only: accepted by the quality gate in BOTH arms ("rejected" in the manifests), minus 000893 (qa_multi
    # parts_in_instance failure in both arms, r06_pair.qa_note_000893.txt); then the run05 filter (>= 800 visible px
    # for whale shots, factory-flagged shots out)
    common = [r for r in rows if not r.get("rejected") and not rows1[r["id"]].get("rejected") and r["id"] not in DROP]
    keep = filter_rows(common, 800, True); sp = make_split(keep, (0.85, 0.15, 0.0), 0)
    print(f"shots 1000 per arm; accepted in both arms {len(common)} (without {sorted(DROP)}); after the >= 800 px / flag filter {len(keep)}, "
          f"of which no-whale shots {sum(r['negative'] for r in keep)}")
    OUT.mkdir(parents=True, exist_ok=True)
    SPLIT.write_text(json.dumps({"run_dir": str(ARMS["m2"]), "seed": 0, "fractions": [0.85, 0.15, 0.0], "strata": ["category"],
                                 "note": "same ids for r06_pair_m1; shots accepted in both arms, 000893 dropped, >= 800 visible px",
                                 "pairs": len(keep), "no_whale_shots": sum(r["negative"] for r in keep), "summary": summarize(keep, sp), **sp}, indent=1))
    same = diff = 0
    for r in keep:
        p = ARMS["m2"] / "dataset" / r["mask"]
        q = ARMS["m1"] / "dataset" / r["mask"]
        if q.exists() and hashlib.md5(p.read_bytes()).digest() == hashlib.md5(q.read_bytes()).digest():
            same += 1
        else:
            diff += 1
    (OUT / "mask_check.json").write_text(json.dumps({"masks_identical": same, "masks_different_or_missing": diff}))
    print("split", {k: len(sp[k]) for k in sp}, "masks identical", same, "different", diff)
    if diff:
        raise SystemExit("masks differ between the arms")


def cmd_pairsheet(a) -> None:
    from ..common.splits import load_manifest
    ids = set(json.loads(SPLIT.read_text())["train"]) | set(json.loads(SPLIT.read_text())["val"])
    rows = [r for r in load_manifest(ARMS["m2"]) if r["id"] in ids and not r["negative"] and r["category"] in ("awash", "surface", "logging", "back_arch", "fluke_rollout", "blow", "blow_sliver")]
    head = [r for r in rows if (r.get("orientation") or {}).get("nose_toward_camera")]
    pool = head if len(head) >= 30 else rows
    pool = sorted(pool, key=lambda r: -r["through_water_pixels"] / max(r["through_water_pixels"] + r["visible_pixels"], 1))[:30]
    tiles, legend = [], []
    for i, r in enumerate(pool):
        am = cv2.imread(str(ARMS["m2"] / "dataset" / r["amodal"]), cv2.IMREAD_UNCHANGED); am = (am[..., 0] if am.ndim == 3 else am) > 0
        ys, xs = np.nonzero(am)
        if not len(xs):
            continue
        H, W = am.shape; x0, x1, y0, y1 = xs.min(), xs.max(), ys.min(), ys.max(); cx, cy = (x0 + x1) / 2, (y0 + y1) / 2
        w = max((x1 - x0) * 1.5, 200); h = max((y1 - y0) * 1.5, 133); w, h = (h * 1.5, h) if w / h < 1.5 else (w, w / 1.5); w, h = min(w, W), min(h, H)
        wx = int(min(max(cx - w / 2, 0), W - w)); wy = int(min(max(cy - h / 2, 0), H - h))
        panels = []
        for arm in ("m1", "m2"):
            im = cv2.imread(str(ARMS[arm] / "dataset" / r["image"]))[wy:int(wy + h), wx:int(wx + w)].copy()
            im = cv2.resize(im, (330, 220), interpolation=cv2.INTER_AREA if im.shape[1] > 330 else cv2.INTER_CUBIC)
            t = f"{i + 1} {arm}"; cv2.putText(im, t, (4, 16), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 3); cv2.putText(im, t, (4, 16), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            panels.append(im)
        tiles.append(cv2.copyMakeBorder(np.hstack(panels), 2, 2, 2, 2, cv2.BORDER_CONSTANT, value=(255, 255, 255)))
        legend.append({"tile": i + 1, "id": r["id"], "category": r["category"], "sea_state": r["sea_state"], "through_water_px": r["through_water_pixels"], "visible_px": r["visible_pixels"]})
    SH.mkdir(parents=True, exist_ok=True)
    for s in range(0, len(tiles), 16):
        part = tiles[s:s + 16]
        while len(part) % 2:
            part.append(np.zeros_like(tiles[0]))
        cv2.imwrite(str(SH / f"pairs_{s // 16 + 1:02d}.jpg"), np.vstack([np.hstack(part[k:k + 2]) for k in range(0, len(part), 2)]), [cv2.IMWRITE_JPEG_QUALITY, 88])
    (SH / "pairs_legend.json").write_text(json.dumps(legend, indent=0)); print(len(tiles), "pairs ->", SH)


def lower_edges(vis):
    ys, xs = np.nonzero(vis); bot = np.full(vis.shape[1], -1); np.maximum.at(bot, xs, ys)
    return bot


def cmd_renders(a) -> None:
    """Every model on both arms' VALIDATION renders (full frame, 1280 px): dataset IoU (ignore excluded), mean P and
    share >= 0.5 on the mirror side (mask 0, 1-30 px below the visible whale's lower edge), inside the whale's lowest
    10 px, and on the under-water zone (mask 255)."""
    import torch
    from ..common.data import IGNORE, WhaleRun, pad_to_multiple, to_tensor
    from ..segmentation.model import WhaleSegModel
    ids = json.loads(SPLIT.read_text())["val"]; dev = torch.device("cuda"); res = {}
    runs = {arm: WhaleRun(d) for arm, d in ARMS.items()}
    for name in a.models.split(","):
        model, _ = WhaleSegModel.from_checkpoint(ckpt(name), dev); res[name] = {}
        for arm, run in runs.items():
            acc = defaultdict(float)
            for sid in ids:
                img = run.image(sid); mask = run.mask(sid); padded, (h, w) = pad_to_multiple(img)
                with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
                    p = torch.sigmoid(model(to_tensor(padded)[None].to(dev)).float())[0, :h, :w].cpu().numpy()
                pr = p >= 0.5; vis = mask == 1; val = mask != IGNORE
                acc["tp"] += (pr & vis).sum(); acc["fp"] += (pr & ~vis & val).sum(); acc["fn"] += (~pr & vis).sum()
                bot = lower_edges(vis); cols = np.nonzero(bot >= 0)[0]
                for d in range(1, 31):
                    y = bot[cols] + d; ok = y < mask.shape[0]; cy, cx = y[ok], cols[ok]; sel = mask[cy, cx] == 0
                    acc["mir_s"] += p[cy[sel], cx[sel]].sum(); acc["mir_n"] += sel.sum(); acc["mir_hi"] += (p[cy[sel], cx[sel]] >= 0.5).sum()
                for d in range(0, 10):
                    y = bot[cols] - d; ok = y >= 0; cy, cx = y[ok], cols[ok]; sel = vis[cy, cx]
                    acc["low_s"] += p[cy[sel], cx[sel]].sum(); acc["low_n"] += sel.sum()
                u = mask == IGNORE; acc["uw_s"] += p[u].sum(); acc["uw_n"] += u.sum()
            res[name][arm] = {"dataset_iou": round(acc["tp"] / max(acc["tp"] + acc["fp"] + acc["fn"], 1), 4),
                              "mirror_side_meanP": round(acc["mir_s"] / max(acc["mir_n"], 1), 4), "mirror_side_share_whale": round(acc["mir_hi"] / max(acc["mir_n"], 1), 4),
                              "whale_lowest_10px_meanP": round(acc["low_s"] / max(acc["low_n"], 1), 4), "under_water_zone_meanP": round(acc["uw_s"] / max(acc["uw_n"], 1), 4)}
            print(name, arm, res[name][arm], flush=True)
        del model; torch.cuda.empty_cache()
    CK.mkdir(parents=True, exist_ok=True); f = OUT / "pair_renders.json"; old = json.loads(f.read_text()) if f.exists() else {}
    old.update(res); f.write_text(json.dumps(old, indent=1))


def view_rows():
    return json.loads((C.CACHE / "per_photo_whales.json").read_text())["s5"]      # the view-labelled main whales (train + val)


def cmd_real(a) -> None:
    """reflect.py real's measures plus the waterline band in pixels and the kept share of the live (r04) blob."""
    names = a.models.split(","); out = {}
    for n in names:
        acc = defaultdict(lambda: [0.0, 0]); kept = defaultdict(list)
        for r in view_rows():
            p = load_prob(n, r["day"], r["stem"]); H, W = p.shape; m = blob_1280(C.blob_image(r["day"], r["stem"]), r["b"], (H, W))
            if m.sum() < 200:
                continue
            sp = r["split"]; ys, xs = np.nonzero(m); top = np.full(W, 10 ** 6); bot = np.full(W, -1); np.minimum.at(top, xs, ys); np.maximum.at(bot, xs, ys)
            rel = (ys - top[xs]) / np.maximum(bot[xs] - top[xs], 1); hb = np.minimum((rel * 5).astype(int), 4); v = p[ys, xs]; up = bot[xs] - ys
            kept[sp].append(float((v >= 0.5).mean()))
            for b in range(5):
                s = hb == b; acc[(sp, f"height fifth {b + 1} (5 = bottom)")][0] += float(v[s].sum()); acc[(sp, f"height fifth {b + 1} (5 = bottom)")][1] += int(s.sum())
            for lo, hi in ((0, 9), (10, 19), (20, 39)):
                s = (up >= lo) & (up <= hi); k = (sp, f"{lo}-{hi} px above the lower edge")
                acc[k][0] += float(v[s].sum()); acc[k][1] += int(s.sum())
                k2 = (sp, f"{lo}-{hi} px above the lower edge, share P >= 0.5"); acc[k2][0] += float((v[s] >= 0.5).sum()); acc[k2][1] += int(s.sum())
            cols = np.nonzero(bot >= 0)[0]
            for lo, hi in ((1, 4), (5, 12)):
                for d in range(lo, hi + 1):
                    y = bot[cols] + d; ok = y < H; vv = p[y[ok], cols[ok]]
                    k = (sp, f"below the lower edge +{lo}..{hi} px, share P >= 0.5"); acc[k][0] += float((vv >= 0.5).sum()); acc[k][1] += int(ok.sum())
        out[n] = {f"{sp} | {k}": round(s / max(c, 1), 4) for (sp, k), (s, c) in acc.items()}
        for sp, v in kept.items():
            out[n][f"{sp} | kept share of the r04 blob (mean over whales)"] = round(float(np.mean(v)), 4)
        print(n, json.dumps({k: v for k, v in out[n].items() if k.startswith("val")}), flush=True)
    f = OUT / "pair_real.json"; old = json.loads(f.read_text()) if f.exists() else {}; old.update(out); f.write_text(json.dumps(old, indent=1))


def cmd_knn(a) -> None:
    """reflect.py knn with the render pool from the MIRROR arm (m1 validation) and the dropped set per model."""
    import torch
    from ..common.backbone import load_backbone
    from ..common.data import WhaleRun
    from .reflect import patch_share, tokens
    torch.set_num_threads(12); bb = load_backbone("vitb16").eval(); S = 512; run = WhaleRun(ARMS["m1"])
    ids = [i for i in json.loads(SPLIT.read_text())["val"] if not run.by_id[i]["negative"] and run.by_id[i]["visible_pixels"] > 3000]
    random.Random(1).shuffle(ids); R_f, R_c = [], []
    for sid in ids[:60]:
        img = run.image(sid); mask = run.mask(sid); vis = mask == 1; ys, xs = np.nonzero(vis); H, W = mask.shape
        cx, cy = int(np.median(xs)), int(np.median(ys)); x0 = int(np.clip(cx - S // 2, 0, W - S)); y0 = int(np.clip(cy - S // 2, 0, H - S))
        crop = img[y0:y0 + S, x0:x0 + S]; mk = mask[y0:y0 + S, x0:x0 + S]; v = vis[y0:y0 + S, x0:x0 + S]
        F = tokens(bb, crop); bot = lower_edges(v); band = np.zeros_like(v)
        for x in np.nonzero(bot >= 0)[0]:
            band[bot[x] + 1: bot[x] + 25, x] = True
        band &= mk == 0; far = cv2.distanceTransform((~v).astype(np.uint8), cv2.DIST_L2, 3) > 48
        sh = {"whale skin": patch_share(v), "under-water zone": patch_share(mk == 255), "band below waterline (mirror side)": patch_share(band), "open water": patch_share(far & (mk == 0))}
        for i in range(len(F)):
            for c, s in sh.items():
                if s[i] >= 0.6:
                    R_f.append(F[i]); R_c.append(c); break
    R_f = np.array(R_f); R_c = np.array(R_c)
    rows = sorted(view_rows(), key=lambda r: r["kept"])[:60]; names = a.models.split(","); Q = defaultdict(list)
    for r in rows:
        img = cv2.cvtColor(resize_1280(C.photo(r["day"], r["stem"])), cv2.COLOR_BGR2RGB); H, W = img.shape[:2]
        m = blob_1280(C.blob_image(r["day"], r["stem"]), r["b"], (H, W)); ys, xs = np.nonzero(m)
        cx, cy = int(np.median(xs)), int(np.median(ys)); x0 = int(np.clip(cx - S // 2, 0, W - S)); y0 = int(np.clip(cy - S // 2, 0, H - S))
        F = tokens(bb, img[y0:y0 + S, x0:x0 + S]); inb = patch_share(m[y0:y0 + S, x0:x0 + S])
        for n in names:
            pp = load_prob(n, r["day"], r["stem"])[y0:y0 + S, x0:x0 + S]; drop = patch_share(m[y0:y0 + S, x0:x0 + S] & (pp < 0.05))
            for i in range(len(F)):
                if inb[i] >= 0.6 and drop[i] >= 0.6:
                    Q[n].append(F[i])
        Q["all whale patches"] += [F[i] for i in range(len(F)) if inb[i] >= 0.6]
    res = {"render_pool (m1 val)": {c: int((R_c == c).sum()) for c in sorted(set(R_c))}}
    for q, F in Q.items():
        F = np.array(F)
        if not len(F):
            res[q] = {"patches": 0}; continue
        nn = np.argsort(-(F @ R_f.T), axis=1)[:, :10]; cls = R_c[nn]
        res[q] = {"patches": len(F), "nearest_10_shares": {c: round(float((cls == c).mean()), 3) for c in sorted(set(R_c))}}
    (OUT / "pair_knn.json").write_text(json.dumps(res, indent=1)); print(json.dumps(res, indent=1))


def cmd_realsheet(a) -> None:
    from .sheets import panel, put
    names = a.models.split(","); rows = [r for r in view_rows() if r["split"] == "val"]
    k1, k2 = names[1], names[2]                             # the two arms compared (A_m1_s0, A_m2_s0)
    def kept(n, r):
        p = load_prob(n, r["day"], r["stem"]); m = blob_1280(C.blob_image(r["day"], r["stem"]), r["b"], p.shape); return float((p[m] >= 0.5).mean())
    diff = sorted(rows, key=lambda r: -abs(kept(k1, r) - kept(k2, r)))[:12]
    rest = [r for r in rows if r not in diff]; random.Random(0).shuffle(rest); pick = diff + rest[:12]
    tiles, legend = [], []
    for t, r in enumerate(pick):
        img = resize_1280(C.photo(r["day"], r["stem"])); H, W = img.shape[:2]; blob = blob_1280(C.blob_image(r["day"], r["stem"]), r["b"], (H, W))
        ys, xs = np.nonzero(blob); x0, x1, y0, y1 = xs.min(), xs.max(), ys.min(), ys.max(); cx, cy = (x0 + x1) / 2, (y0 + y1) / 2
        w = max((x1 - x0) * 1.6, 160); h = max((y1 - y0) * 1.6, 107); w, h = (h * 1.5, h) if w / h < 1.5 else (w, w / 1.5); w, h = min(w, W), min(h, H)
        wx = int(min(max(cx - w / 2, 0), W - w)); wy = int(min(max(cy - h / 2, 0), H - h)); win = (wx, wy, int(wx + w), int(wy + h)); ps = []
        for n in names:
            pn = panel(img, load_prob(n, r["day"], r["stem"]), blob, win, size=(240, 160)); put(pn, f"{t + 1} {n}", (4, 14)); ps.append(pn)
        tiles.append(cv2.copyMakeBorder(np.hstack(ps), 2, 2, 2, 2, cv2.BORDER_CONSTANT, value=(255, 255, 255)))
        legend.append({"tile": t + 1, "day": r["day"], "stem": r["stem"], "why": "largest kept difference between the arms" if t < 12 else "random"})
    SH.mkdir(parents=True, exist_ok=True); p = SH / "real_arms.jpg"
    cv2.imwrite(str(p), np.vstack(tiles), [cv2.IMWRITE_JPEG_QUALITY, 86]); (SH / "real_arms_legend.json").write_text(json.dumps({"columns": names, "tiles": legend}, indent=0)); print(p)


def cmd_summary(a) -> None:
    Rr = json.loads((OUT / "pair_renders.json").read_text()); Re = json.loads((OUT / "pair_real.json").read_text())
    K = json.loads((OUT / "pair_knn.json").read_text()) if (OUT / "pair_knn.json").exists() else {}
    names = ["r04", "r05"] + MODELS; lines = ["# Paired reflection test (renders only): tables", ""]
    lines += ["## Renders: each model on both arms' validation renders", "", "| model | IoU m1 | IoU m2 | mirror-side P m1 | m2 | mirror-side share >= 0.5 m1 | m2 | lowest 10 px of whale P m1 | m2 | under-water zone P m1 | m2 |", "|---" * 11 + "|"]
    for n in names:
        if n in Rr:
            g = lambda arm, k: Rr[n][arm][k]
            lines.append(f"| {n} | {g('m1','dataset_iou')} | {g('m2','dataset_iou')} | {g('m1','mirror_side_meanP')} | {g('m2','mirror_side_meanP')} | {g('m1','mirror_side_share_whale')} | {g('m2','mirror_side_share_whale')} | {g('m1','whale_lowest_10px_meanP')} | {g('m2','whale_lowest_10px_meanP')} | {g('m1','under_water_zone_meanP')} | {g('m2','under_water_zone_meanP')} |")
    for sp in ("val", "train"):
        keys = sorted({k for n in names if n in Re for k in Re[n] if k.startswith(sp + " |")})
        lines += ["", f"## Real photos, {sp} days (view-labelled main whales; mean P(whale) unless a share)", "", "| measure | " + " | ".join(n for n in names if n in Re) + " |", "|---" * (1 + sum(n in Re for n in names)) + "|"]
        for k in keys:
            lines.append(f"| {k.split(' | ', 1)[1]} | " + " | ".join(f"{Re[n].get(k, float('nan')):.3f}" for n in names if n in Re) + " |")
    if K:
        lines += ["", "## Mirror-band nearest neighbours (real whale patches each model drops; pool = m1 validation renders)", "", "```", json.dumps(K, indent=1), "```"]
    (OUT / "pair_tables.md").write_text("\n".join(lines) + "\n"); (OUT / "pair_results.json").write_text(json.dumps({"renders": Rr, "real": Re, "knn": K}, indent=1))
    print("\n".join(lines))


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter); sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("split").set_defaults(fn=cmd_split); sub.add_parser("pairsheet").set_defaults(fn=cmd_pairsheet)
    for c, fn, d in (("renders", cmd_renders, ",".join(["r04", "r05"] + MODELS)), ("real", cmd_real, ",".join(["r04", "r05"] + MODELS)),
                     ("knn", cmd_knn, ",".join(["r05"] + MODELS)), ("realsheet", cmd_realsheet, "r04,A_m1_s0,A_m2_s0,B_m1_s0,B_m2_s0")):
        p = sub.add_parser(c); p.add_argument("--models", default=d); p.set_defaults(fn=fn)
    sub.add_parser("summary").set_defaults(fn=cmd_summary)
    a = ap.parse_args(argv); a.fn(a)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
