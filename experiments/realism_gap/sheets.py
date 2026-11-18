"""Picture sheets for a score card (all written to the scratch folder, never the repo):

    PYTHONPATH=. .venv/bin/python experiments/realism_gap/sheets.py --renders <dataset folder> [--top 5]

  heat_<k>.jpg          patch probe P(render) over crops, 6 renders and 6 real photos per sheet
                        (columns: render | its heat map | real | its heat map; red = looks rendered, blue = looks real;
                        out-of-fold: the probe never saw the photo's day / the render's batch)
  render_farthest.jpg   the 24 renders farthest from any real crop (whale-token cosine distance)
  render_nearest.jpg    the 24 renders nearest to real crops
  real_farthest_<k>.jpg real crops farthest from any render, each with its 3 nearest renders
  stat_<rank>_<stat>.jpg one sheet per large statistic gap: typical real (left two columns) against typical render
                        (right two columns) = the 12 samples of each world closest to that world's median
  regions.jpg           region maps (whale magenta, band cyan, far water clear, above-horizon orange, ignore grey)
Each sheet has a legend json next to it (tile -> world, id / key, day / batch, value).
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import common  # noqa: E402
import extract  # noqa: E402
import scorecard  # noqa: E402

T = 300          # tile side
CAP = 22         # caption strip height
TEXTURE_KINDS = ("tex_fine", "tex_medium", "tex_coarse", "psd_slope", "vgrad_share", "coherence", "foam")
CONCEPT = {"clip_hi": "exposure", "crush_lo": "exposure", "L_p99": "exposure", "L_p01": "exposure", "L_mean": "exposure",
           "hue_deg": "colour", "a_mean": "colour", "b_mean": "colour", "C_mean": "colour", "S_hsv": "colour",
           "coherence": "anisotropy", "vgrad_share": "anisotropy", "tex_fine": "texture", "tex_medium": "texture",
           "tex_coarse": "texture", "psd_slope": "texture", "lapvar": "sharpness", "edge_width": "sharpness"}
EXTRA = ("band.psd_slope", "band.foam", "whale.len_share", "whale.L_mean", "crop.lapvar")
NO_SHEET = {"frame.aspect", "frame.width_px", "whale.native_width_px", "crop.native_per_px", "frame.has_horizon"}


def _img(cache: Path, i: int, kind: str):
    return cv2.imread(str(cache / "thumbs" / f"{i}_{kind}.jpg"))


def _lab(cache: Path, i: int, kind="lab"):
    return cv2.imread(str(cache / "thumbs" / f"{i}_{kind}.png"), cv2.IMREAD_GRAYSCALE)


def fit(img, side=T):
    h, w = img.shape[:2]
    s = side / max(h, w)
    r = cv2.resize(img, (max(int(round(w * s)), 1), max(int(round(h * s)), 1)), interpolation=cv2.INTER_AREA if s < 1 else cv2.INTER_LINEAR)
    out = np.zeros((side, side, 3), np.uint8)
    y0, x0 = (side - r.shape[0]) // 2, (side - r.shape[1]) // 2
    out[y0:y0 + r.shape[0], x0:x0 + r.shape[1]] = r
    return out


def tile(img, text, color=(255, 255, 255)):
    t = fit(img) if img is not None else np.zeros((T, T, 3), np.uint8)
    cap = np.zeros((CAP, T, 3), np.uint8)
    cv2.putText(cap, text[:44], (4, 16), cv2.FONT_HERSHEY_SIMPLEX, 0.42, color, 1, cv2.LINE_AA)
    return np.vstack([t, cap])


def grid(tiles, cols=4):
    rows = []
    blank = np.zeros_like(tiles[0])
    for i in range(0, len(tiles), cols):
        r = tiles[i:i + cols] + [blank] * (cols - len(tiles[i:i + cols]))
        rows.append(np.hstack(r))
    sheet = np.vstack(rows)
    for c in range(1, cols):          # thin separators
        sheet[:, c * T - 1:c * T + 1] = 40
    return sheet


def save(sheet, legend, path: Path):
    cv2.imwrite(str(path), sheet, [cv2.IMWRITE_JPEG_QUALITY, 88])
    path.with_suffix(".json").write_text(json.dumps(legend, indent=1))
    return path


def heat_overlay(crop, p, frac):
    """crop BGR 512, p (32, 32) P(render), frac (5, 32, 32)."""
    g = cv2.cvtColor(cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY), cv2.COLOR_GRAY2BGR).astype(np.float32)
    P = cv2.resize(p.astype(np.float32), (crop.shape[1], crop.shape[0]), interpolation=cv2.INTER_CUBIC).clip(0, 1)
    usable = cv2.resize((frac[0] < 0.5).astype(np.float32), (crop.shape[1], crop.shape[0]), interpolation=cv2.INTER_NEAREST)
    # blue (real-like) -> white -> red (render-like), BGR
    blue = np.array([255, 80, 0], np.float32); white = np.array([255, 255, 255], np.float32); red = np.array([0, 0, 255], np.float32)
    t = P[..., None]
    col = np.where(t < 0.5, blue + (white - blue) * (t / 0.5), white + (red - white) * ((t - 0.5) / 0.5))
    a = 0.6 * usable[..., None]
    out = g * (1 - a) + col * a
    return out.clip(0, 255).astype(np.uint8)


def outline(img, lab, color=(0, 255, 0)):
    if lab is None:
        return img
    m = (lab == 1).astype(np.uint8)
    if m.shape[:2] != img.shape[:2]:
        m = cv2.resize(m, (img.shape[1], img.shape[0]), interpolation=cv2.INTER_NEAREST)
    cs, _ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    img = img.copy()
    cv2.drawContours(img, cs, -1, color, 2)
    return img


def window(img, lab, region, side):
    """A side x side window of img centred where `region` is densest (lab at img's size)."""
    m = (lab == region).astype(np.float32)
    if m.shape != img.shape[:2]:
        m = cv2.resize(m, (img.shape[1], img.shape[0]), interpolation=cv2.INTER_NEAREST)
    H, W = m.shape
    side = min(side, H, W)
    dens = cv2.boxFilter(m, -1, (side, side), normalize=True, borderType=cv2.BORDER_CONSTANT)
    h2 = side // 2
    dens[:h2] = 0; dens[H - (side - h2) + 1:] = 0; dens[:, :h2] = 0; dens[:, W - (side - h2) + 1:] = 0
    y, x = np.unravel_index(int(np.argmax(dens)), dens.shape)
    if dens[y, x] < 0.6:
        return None
    return img[y - h2:y - h2 + side, x - h2:x - h2 + side]


def stat_view(cache, i, stat):
    reg, kind = stat.split(".", 1)
    if reg == "whale" and kind not in TEXTURE_KINDS:
        return outline(_img(cache, i, "crop"), _lab(cache, i))
    if reg == "whale":
        return window(_img(cache, i, "crop"), _lab(cache, i), 1, 160)
    if reg == "band":
        if kind in TEXTURE_KINDS:
            return window(_img(cache, i, "crop"), _lab(cache, i), 2, 192)
        return outline(_img(cache, i, "crop"), _lab(cache, i))
    if reg in ("far", "above"):
        fr, fl = _img(cache, i, "frame"), _lab(cache, i, "flab")
        if kind in TEXTURE_KINDS:
            return window(fr, fl, 3 if reg == "far" else 4, 160)
        return fr
    return outline(_img(cache, i, "frame"), _lab(cache, i, "flab"))


def run(renders: Path, n: int = 1500, top: int = 6):
    rname = "render_" + common.tag_of(renders, n, 0)
    rcache, gcache = common.CACHE / scorecard.REAL_NAME, common.CACHE / rname
    out = common.SCRATCH / "sheets" / rname
    out.mkdir(parents=True, exist_ok=True)
    rs, rrows, rf, _ = extract.load(scorecard.REAL_NAME)
    gs, grows, gf, _ = extract.load(rname)
    card = json.loads((common.SCRATCH / f"scorecard_{rname}.json").read_text())
    lists = json.loads((gcache / "lists.json").read_text())
    pr = np.load(gcache / "probe.npz")
    written = []

    # ---- heat maps
    P_img = np.full(len(rs) + len(gs), np.nan); P_img[pr["img_oof_idx"]] = pr["img_oof_p"]

    def heat(F, idx, world):
        fold = int((pr["folds_real"] if world == "real" else pr["folds_render"])[idx])
        mu, sd, w, b = pr[f"f{fold}_mu"], pr[f"f{fold}_sd"], pr[f"f{fold}_w"], float(pr[f"f{fold}_b"])
        return ((F.astype(np.float32) - mu) / sd) @ w + b          # logit, (32, 32)

    rfull = list(zip(rf["full_idx"], rf["full_tok"])); gfull = list(zip(gf["full_idx"], gf["full_tok"]))
    nh = min(len(rfull), len(gfull), 24)
    # colour scale per world: 5th..95th percentile of that world's usable patch logits on these crops, so red marks
    # the parts that look MOST rendered relative to the world's typical patch (absolute P(render) is ~1 on every
    # render patch and ~0 on every real patch, which would draw flat maps)
    rng_w = {}
    for world, full, feats in (("render", gfull, gf), ("real", rfull, rf)):
        zs = np.concatenate([heat(F, int(i), world)[feats["frac"][int(i)][0].astype(np.float32) < 0.5].ravel() for i, F in full[:nh]])
        rng_w[world] = (float(np.percentile(zs, 5)), float(np.percentile(zs, 95)))
    for k in range(0, nh, 6):
        tiles, legend = [], []
        for j in range(k, min(k + 6, nh)):
            for (world, cache, (idx, F), samples, off) in (("render", gcache, gfull[j], gs, len(rs)), ("real", rcache, rfull[j], rs, 0)):
                idx = int(idx)
                crop = _img(cache, idx, "crop")
                zl = heat(F, idx, world)
                lo, hi = rng_w[world]
                p = (zl - lo) / max(hi - lo, 1e-6)
                fr = (gf if world == "render" else rf)["frac"][idx].astype(np.float32)
                name = samples[idx]["id"] if world == "render" else samples[idx]["day"]
                pim = P_img[off + idx]
                tiles.append(tile(outline(crop, _lab(cache, idx)), f"{world} {name}  img P(render) {pim:.2f}"))
                u = fr[0] < 0.5
                reg_txt = " ".join(f"{nm[0]}{zl[(fr[r] >= 0.5) & u].mean():+.0f}" for r, nm in common.REGIONS.items() if ((fr[r] >= 0.5) & u).any())
                tiles.append(tile(heat_overlay(crop, p, fr), f"logit {reg_txt}"))
                legend.append(dict(tile=len(tiles) - 1, world=world, idx=idx, id=samples[idx]["id"], group=samples[idx]["group"],
                                   image_p_render=round(float(pim), 3), patch_mean_logit=round(float(zl[u].mean()), 2),
                                   region_mean_logit={nm: round(float(zl[(fr[r] >= 0.5) & u].mean()), 2) for r, nm in common.REGIONS.items() if ((fr[r] >= 0.5) & u).any()},
                                   colour_range_logit=[round(lo, 2), round(hi, 2)]))
        written.append(save(grid(tiles), legend, out / f"heat_{k // 6 + 1}.jpg"))

    # ---- renders farthest / nearest
    for nm in ("render_farthest", "render_nearest"):
        tiles, legend = [], []
        for e in lists[nm][:24]:
            tiles.append(tile(outline(_img(gcache, e["i"], "crop"), _lab(gcache, e["i"])), f"{e['id']} {e['category']} d {e['d']:.3f}"))
            legend.append(dict(tile=len(tiles) - 1, **e))
        written.append(save(grid(tiles), legend, out / f"{nm}.jpg"))

    # ---- real farthest from renders, with their 3 nearest renders
    rf_list = lists["real_farthest"][:24]
    for k in range(0, len(rf_list), 6):
        tiles, legend = [], []
        for e in rf_list[k:k + 6]:
            tiles.append(tile(outline(_img(rcache, e["i"], "crop"), _lab(rcache, e["i"])), f"REAL {e['day']} d {e['d']:.3f} (real {e['d_real']:.3f})", (0, 255, 255)))
            legend.append(dict(tile=len(tiles) - 1, world="real", **{x: e[x] for x in ("key", "day", "d", "d_real")}))
            for j, gid in zip(e["nn_render_idx"], e["nn_render"]):
                tiles.append(tile(outline(_img(gcache, j, "crop"), _lab(gcache, j)), f"nearest render {gid} {gs[j].get('category')}"))
                legend.append(dict(tile=len(tiles) - 1, world="render", id=gid, category=gs[j].get("category")))
        written.append(save(grid(tiles), legend, out / f"real_farthest_{k // 6 + 1}.jpg"))

    # ---- statistic sheets: top gaps, one per kind of statistic
    used_kinds, picked = set(), []
    for r in card["stats"]:
        if r.get("ks") is None or r["stat"] in NO_SHEET:
            continue
        kind = r["stat"].split(".", 1)[1]
        kind = CONCEPT.get(kind, kind) + ("_above" if r["stat"].startswith("above.") else "")
        if kind in used_kinds:
            continue
        used_kinds.add(kind); picked.append(r)
        if len(picked) >= top:
            break
    # plus the gaps seen by eye, whatever their rank: water texture at the same scale, foam, whale size in
    # the frame, whale brightness, crop sharpness
    for st in EXTRA:
        r = next((x for x in card["stats"] if x["stat"] == st and x.get("ks") is not None), None)
        if r is not None and r not in picked:
            picked.append(r)
    for r in picked:
        st = r["stat"]
        cols = {}
        for world, cache, rows in (("real", rcache, rrows), ("render", gcache, grows)):
            v = np.array([x.get(st) if x.get(st) is not None else np.nan for x in rows], float)
            med = np.nanmedian(v)
            order = np.argsort(np.where(np.isfinite(v), np.abs(v - med), np.inf))
            got = []
            for i in order:
                if not np.isfinite(v[i]):
                    break
                im = stat_view(cache, int(i), st)
                if im is None:
                    continue
                got.append((int(i), float(v[i]), im))
                if len(got) >= 12:
                    break
            cols[world] = got
        tiles, legend = [], []
        for row in range(6):
            for world, c in (("real", 0), ("real", 1), ("render", 0), ("render", 1)):
                j = row * 2 + c
                g = cols[world]
                if j < len(g):
                    i, val, im = g[j]
                    samples = rs if world == "real" else gs
                    tiles.append(tile(im, f"{world} {r['stat']} {scorecard._fmt(val)}", (0, 255, 255) if world == "real" else (255, 255, 255)))
                    legend.append(dict(tile=len(tiles) - 1, world=world, idx=i, id=samples[i]["id"], group=samples[i]["group"], value=val))
                else:
                    tiles.append(tile(None, ""))
        save(grid(tiles), dict(stat=st, what=r["what"], unit=r["unit"], ks=r["ks"], real_median=r["real_q"][1],
                               render_median=r["render_q"][1], tiles=legend), out / f"stat_{r['rank']:02d}_{st.replace('.', '_')}.jpg")
        written.append(out / f"stat_{r['rank']:02d}_{st.replace('.', '_')}.jpg")

    # ---- region maps (QA)
    col = {0: (128, 128, 128), 1: (255, 0, 255), 2: (255, 255, 0), 4: (0, 140, 255)}
    tiles, legend = [], []
    rng = np.random.default_rng(1)
    for world, cache, samples in (("real", rcache, rs), ("render", gcache, gs)):
        for i in rng.choice(len(samples), 12, replace=False):
            fr, fl = _img(cache, int(i), "frame"), _lab(cache, int(i), "flab")
            ov = fr.copy()
            for kk, c in col.items():
                m = fl == kk
                ov[m] = (0.55 * ov[m] + 0.45 * np.array(c)).astype(np.uint8)
            tiles.append(tile(ov, f"{world} {samples[int(i)]['group']} horizon {grows[int(i)].get('_horizon_src') if world == 'render' else 'finder'}"))
            legend.append(dict(tile=len(tiles) - 1, world=world, id=samples[int(i)]["id"], group=samples[int(i)]["group"]))
    written.append(save(grid(tiles), legend, out / "regions.jpg"))
    for p in written:
        print(p)
    return written


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--renders", default=str(common.RUN05))
    ap.add_argument("--n", type=int, default=1500)
    ap.add_argument("--top", type=int, default=6)
    a = ap.parse_args()
    run(Path(a.renders), a.n, a.top)
