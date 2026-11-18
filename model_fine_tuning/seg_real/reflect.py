"""The reflection hypothesis (user, 2026-11-17: "is the model confident those pixels are water because of the previous
shadow / reflection?"). Every run04 / run05 render shows the whale on a dark mirror image of itself in glassy water,
labelled background. Three measurements, CPU only (no GPU lock needed; small samples).

    PYTHONPATH=. .venv/bin/python -m model_fine_tuning.seg_real.reflect renders [--n 48]
        (a) run05 VALIDATION renders: P(whale) of r04 / r05 / S0 / S5 / S6 across the waterline: per column of the visible
            whale, offsets from its lower edge (inside: -20..-1 px; below: +1..+30 px), split into the under-water zone
            (factory mask 255) and plain water (mask 0 = where the mirror image lies), by sea state.
    PYTHONPATH=. .venv/bin/python -m model_fine_tuning.seg_real.reflect real
        (c) real photos (view-labelled main whales, train + val days, cached predictions): P(whale) by height band of
            the r04 blob (top -> bottom fifth) and just below its lower edge, per model.
    PYTHONPATH=. .venv/bin/python -m model_fine_tuning.seg_real.reflect knn [--n 60]
        (b) frozen DINOv3 ViT-B/16 patch tokens (last block): real patches S5 drops (r04 whale, S5 P < 0.05) and real
            patches S5 keeps, against render patches of whale skin / under-water zone / the band just below the waterline
            (mirror image) / open water: shares of the 10 nearest render patches; a small sheet (scratchpad).
    -> experiments/seg_splash/reflect_<cmd>.json
"""
from __future__ import annotations

import argparse
import json
import random
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np

from . import common as C
from .data import blob_1280
from .predict import load_prob, resize_1280, view_keys

MODELS = {"r04": C.SEG_R04, "r05": C.SEG_R05, "s0": C.REPO / "experiments/seg_splash_s0/checkpoints/last.pt",
          "s5": C.REPO / "experiments/seg_splash_s5/checkpoints/last.pt"}
OFFS = [(-20, -11), (-10, -4), (-3, -1), (1, 4), (5, 12), (13, 30)]


def render_sample(n: int, seed: int = 0):
    from ..common import paths
    from ..common.data import open_runs
    run = open_runs(C.RUN05, paths.SPLITS_DIR / "run05.json")
    ids = [i for i in run.ids("val") if not run.by_id[i]["negative"] and run.by_id[i]["visible_pixels"] > 3000 and run.by_id[i]["category"] not in ("submerged", "breach", "spyhop")]
    rng = random.Random(seed); by = defaultdict(list)
    for i in ids:
        by["glassy/calm" if run.by_id[i]["sea_state"] in ("glassy", "calm") else "light/moderate/rough"].append(i)
    out = []
    for k, v in by.items():
        rng.shuffle(v); out += [(k, i) for i in v[: n // 2]]
    return run, out


def cpu_models(names):
    import torch
    from ..segmentation.model import WhaleSegModel
    torch.set_num_threads(12)
    return {n: WhaleSegModel.from_checkpoint(MODELS[n], torch.device("cpu"))[0] for n in names}


def prob_cpu(model, rgb):
    import torch
    from ..common.data import pad_to_multiple, to_tensor
    padded, (h, w) = pad_to_multiple(rgb)
    with torch.no_grad():
        return torch.sigmoid(model(to_tensor(padded)[None]).float())[0, :h, :w].numpy()


def lower_edges(vis):
    ys, xs = np.nonzero(vis); bot = np.full(vis.shape[1], -1); np.maximum.at(bot, xs, ys)
    return bot


def cmd_renders(a) -> None:
    names = [n for n in a.models.split(",") if n in MODELS or (C.REPO / f"experiments/seg_splash_{n}/checkpoints/last.pt").exists()]
    for n in names:
        MODELS.setdefault(n, C.REPO / f"experiments/seg_splash_{n}/checkpoints/last.pt")
    run, smp = render_sample(a.n); M = cpu_models(names)
    acc = defaultdict(lambda: [0.0, 0])          # (model, sea, zone, offset bin) -> sum P, count
    for sea, sid in smp:
        img = run.image(sid); mask = run.mask(sid); vis = mask == 1; bot = lower_edges(vis); H, W = mask.shape
        cols = np.nonzero(bot >= 0)[0]
        P = {n: prob_cpu(m, img) for n, m in M.items()}
        for (lo, hi) in OFFS:
            for d in range(lo, hi + 1):
                y = bot[cols] + d; ok = (y >= 0) & (y < H); cx, cy = cols[ok], y[ok]
                if d < 0:
                    sel = vis[cy, cx]; zones = {"whale (inside, above edge)": sel}
                else:
                    zones = {"under-water zone (255)": mask[cy, cx] == 255, "water below edge (0) = mirror side": mask[cy, cx] == 0}
                for zn, sel in zones.items():
                    for n in names:
                        v = P[n][cy[sel], cx[sel]]; k = (n, sea, zn, f"{lo}..{hi}")
                        acc[k][0] += float(v.sum()); acc[k][1] += int(sel.sum())
    out = defaultdict(dict)
    for (n, sea, zn, ob), (s, c) in acc.items():
        if c:
            out[f"{sea} | {zn} | {ob}"][n] = round(s / c, 3)
    (C.EXP / "reflect_renders.json").write_text(json.dumps({"frames": len(smp), "rows": out}, indent=1))
    for k in sorted(out):
        print(f"{k:72s}", "  ".join(f"{n} {v:.3f}" for n, v in out[k].items()))


def cmd_real(a) -> None:
    names = a.models.split(","); vk = view_keys(); acc = defaultdict(lambda: [0.0, 0]); n_ph = 0
    for k, (day, code) in vk.items():
        mp = json.loads((C.DAYS / day / "instances_v2" / "mapping.json").read_text())["old_to_new"] if False else None
    rows = json.loads((C.CACHE / "per_photo_whales.json").read_text())["s5"]
    for r in rows:
        p = {n: load_prob(n, r["day"], r["stem"]) for n in names}; H, W = p[names[0]].shape
        m = blob_1280(C.blob_image(r["day"], r["stem"]), r["b"], (H, W))
        if m.sum() < 200:
            continue
        n_ph += 1; ys, xs = np.nonzero(m); top = np.full(W, 10 ** 6); bot = np.full(W, -1); np.minimum.at(top, xs, ys); np.maximum.at(bot, xs, ys)
        rel = (ys - top[xs]) / np.maximum(bot[xs] - top[xs], 1); hb = np.minimum((rel * 5).astype(int), 4)
        for n in names:
            v = p[n][ys, xs]
            for b in range(5):
                s = hb == b; acc[(r["split"], n, f"in blob, height band {b + 1}/5 (5 = bottom)")][0] += float(v[s].sum()); acc[(r["split"], n, f"in blob, height band {b + 1}/5 (5 = bottom)")][1] += int(s.sum())
            cols = np.nonzero(bot >= 0)[0]
            for lo, hi in ((1, 4), (5, 12)):
                for d in range(lo, hi + 1):
                    y = bot[cols] + d; ok = y < H; v = p[n][y[ok], cols[ok]]
                    acc[(r["split"], n, f"below the blob's lower edge +{lo}..{hi} px")][0] += float(v.sum()); acc[(r["split"], n, f"below the blob's lower edge +{lo}..{hi} px")][1] += int(ok.sum())
    out = defaultdict(dict)
    for (sp, n, z), (s, c) in acc.items():
        out[f"{sp} | {z}"][n] = round(s / max(c, 1), 3)
    (C.EXP / "reflect_real.json").write_text(json.dumps({"photos": n_ph, "rows": out}, indent=1))
    for k in sorted(out):
        print(f"{k:60s}", "  ".join(f"{n} {v:.3f}" for n, v in out[k].items()))


def tokens(model, rgb):
    import torch
    from ..common.backbone import normalize_image
    from ..common.data import to_tensor
    with torch.no_grad():
        f = model.get_intermediate_layers(normalize_image(to_tensor(rgb)[None]), n=1, reshape=True, norm=True)[0][0]
    f = f.permute(1, 2, 0).reshape(-1, f.shape[0]).numpy()
    return f / np.linalg.norm(f, axis=1, keepdims=True)


def patch_share(mask_bool, g=16):
    H, W = mask_bool.shape; h, w = H // g, W // g
    return mask_bool[:h * g, :w * g].reshape(h, g, w, g).mean((1, 3)).ravel()


def cmd_knn(a) -> None:
    import torch
    from ..common.backbone import load_backbone
    torch.set_num_threads(12); bb = load_backbone("vitb16").eval(); S = 512
    run, smp = render_sample(a.n, seed=1)
    R_f, R_c, R_img = [], [], []          # render patch features, class, (frame idx, cx, cy)
    for j, (sea, sid) in enumerate(smp):
        img = run.image(sid); mask = run.mask(sid); vis = mask == 1; ys, xs = np.nonzero(vis)
        cx, cy = int(np.median(xs)), int(np.median(ys)); x0 = int(np.clip(cx - S // 2, 0, mask.shape[1] - S)); y0 = int(np.clip(cy - S // 2, 0, mask.shape[0] - S))
        crop = img[y0:y0 + S, x0:x0 + S]; mk = mask[y0:y0 + S, x0:x0 + S]
        if crop.shape[:2] != (S, S):
            continue
        F = tokens(bb, crop); v = vis[y0:y0 + S, x0:x0 + S]
        bot = lower_edges(v); band = np.zeros_like(v)
        for x in np.nonzero(bot >= 0)[0]:
            band[bot[x] + 1: bot[x] + 25, x] = True
        band &= mk == 0
        far = cv2.distanceTransform((~v).astype(np.uint8), cv2.DIST_L2, 3) > 48
        sh = {"whale skin": patch_share(v), "under-water zone": patch_share(mk == 255), "band below waterline (mirror side)": patch_share(band),
              "open water": patch_share(far & (mk == 0))}
        for i in range(len(F)):
            for c, s in sh.items():
                if s[i] >= 0.6:
                    R_f.append(F[i]); R_c.append(c); R_img.append((sid, x0 + (i % (S // 16)) * 16 + 8, y0 + (i // (S // 16)) * 16 + 8)); break
    R_f = np.array(R_f); R_c = np.array(R_c)
    rows = json.loads((C.CACHE / "per_photo_whales.json").read_text())["s5"]; rows = sorted(rows, key=lambda r: r["kept"])[: a.n]
    Q = defaultdict(list); Qimg = defaultdict(list)
    for r in rows:
        img = cv2.cvtColor(resize_1280(C.photo(r["day"], r["stem"])), cv2.COLOR_BGR2RGB); p = load_prob("s5", r["day"], r["stem"]); H, W = p.shape
        m = blob_1280(C.blob_image(r["day"], r["stem"]), r["b"], (H, W)); ys, xs = np.nonzero(m)
        cx, cy = int(np.median(xs)), int(np.median(ys)); x0 = int(np.clip(cx - S // 2, 0, W - S)); y0 = int(np.clip(cy - S // 2, 0, H - S))
        crop = img[y0:y0 + S, x0:x0 + S]; mm = m[y0:y0 + S, x0:x0 + S]; pp = p[y0:y0 + S, x0:x0 + S]
        F = tokens(bb, crop)
        drop = patch_share(mm & (pp < 0.05)); kept = patch_share(mm & (pp > 0.5)); inb = patch_share(mm)
        for i in range(len(F)):
            if inb[i] >= 0.6 and drop[i] >= 0.6:
                Q["real whale pixels S5 drops"].append(F[i]); Qimg["real whale pixels S5 drops"].append((r["day"], r["stem"], x0 + (i % 32) * 16 + 8, y0 + (i // 32) * 16 + 8))
            elif inb[i] >= 0.6 and kept[i] >= 0.6:
                Q["real whale pixels S5 keeps"].append(F[i])
    res = {}
    for q, F in Q.items():
        F = np.array(F); sim = F @ R_f.T; nn = np.argsort(-sim, axis=1)[:, :10]
        cls = R_c[nn]; res[q] = {"patches": len(F), "nearest_10_shares": {c: round(float((cls == c).mean()), 3) for c in sorted(set(R_c))},
                                 "mean_top1_cosine": round(float(np.take_along_axis(sim, nn[:, :1], 1).mean()), 3)}
        if q == "real whale pixels S5 drops":
            pick = np.random.default_rng(0).choice(len(F), min(12, len(F)), replace=False); tiles = []
            for t in pick:
                d, s, x, y = Qimg[q][t]; im = resize_1280(C.photo(d, s)); x0, y0 = max(0, x - 40), max(0, y - 40)
                row = [cv2.resize(im[y0:y0 + 80, x0:x0 + 80], (120, 120))]
                for k in nn[t, :5]:
                    sid, rx, ry = R_img[k]; ri = cv2.cvtColor(run.image(sid), cv2.COLOR_RGB2BGR); a0, b0 = max(0, rx - 40), max(0, ry - 40)
                    tt = cv2.resize(ri[b0:b0 + 80, a0:a0 + 80], (120, 120)); cv2.putText(tt, R_c[k][:12], (2, 112), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (255, 255, 255), 1); row.append(tt)
                tiles.append(np.hstack([cv2.copyMakeBorder(x_, 2, 2, 2, 2, cv2.BORDER_CONSTANT, value=(255, 255, 255)) for x_ in row]))
            cv2.imwrite(str(C.SHEETS / "reflect_knn.jpg"), np.vstack(tiles), [cv2.IMWRITE_JPEG_QUALITY, 88])
    res["render_patches"] = {c: int((R_c == c).sum()) for c in sorted(set(R_c))}
    (C.EXP / "reflect_knn.json").write_text(json.dumps(res, indent=1)); print(json.dumps(res, indent=1))


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter); sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("renders"); p.add_argument("--n", type=int, default=48); p.add_argument("--models", default="r04,r05,s0,s5"); p.set_defaults(fn=cmd_renders)
    p = sub.add_parser("real"); p.add_argument("--models", default="r04,r05,s0,s1,s5"); p.set_defaults(fn=cmd_real)
    p = sub.add_parser("knn"); p.add_argument("--n", type=int, default=60); p.set_defaults(fn=cmd_knn)
    a = ap.parse_args(argv); a.fn(a)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
