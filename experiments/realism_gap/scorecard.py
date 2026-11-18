"""Realism score card: real photos (train + val days) against a render dataset.

    PYTHONPATH=. .venv/bin/python experiments/realism_gap/scorecard.py --renders <dataset folder> \
        [--compare <older scorecard json>] [--n 1500]

Prints and writes (json) three things:
  1. hand-made statistics, real against render, ranked by the Kolmogorov-Smirnov statistic (KS: 0 = same
     distribution, 1 = no overlap), with medians / quartiles in plain units and the standardised mean
     difference (SMD = (render mean - real mean) / pooled sd);
  2. learned gap scores on frozen DINOv3 ViT-B/16 features (crops 512 px): image-level real-vs-render
     logistic regression (AUC, accuracy; folds hold out whole real days and whole render batches), a
     patch-level linear probe reported per region (whale, band, open water, above horizon), and coverage
     (cosine distance between whale-region mean tokens: real -> nearest renders against real -> nearest
     real photos of other days);
  3. coverage per hard situation (instances listed in the scenarios json) against the easy ones.
Features and statistics are cached under the scratch folder (common.SCRATCH/cache), so the real side is
computed once; a new render folder costs one extraction pass (~6 min for 1 500 shots on a shared GPU).
The headline numbers: image AUC, patch AUC (all regions), coverage ratio (1.0 = renders sit as close to
real photos as other real days do; higher = worse).
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import common  # noqa: E402
import extract  # noqa: E402
import stats as S  # noqa: E402

REAL_NAME, SCEN_NAME = "real_v1", "scen_v1"
NFOLD = 5


# ------------------------------------------------------------------------------------------ helpers
def _vals(rows, k):
    return np.array([r.get(k) if r.get(k) is not None else np.nan for r in rows], float)


def stat_table(real_rows, rend_rows, min_n=30):
    from scipy.stats import ks_2samp
    keys = [k for k in S.INFO if any(k in r for r in real_rows) or any(k in r for r in rend_rows)]
    out = []
    for k in keys:
        a, b = _vals(real_rows, k), _vals(rend_rows, k)
        fa, fb = np.isfinite(a), np.isfinite(b)
        row = dict(stat=k, what=S.INFO[k][0], unit=S.INFO[k][1], n_real=int(fa.sum()), n_render=int(fb.sum()),
                   share_real=round(float(fa.mean()), 3), share_render=round(float(fb.mean()), 3))
        if fa.sum() >= min_n and fb.sum() >= min_n:
            a, b = a[fa], b[fb]
            ks = ks_2samp(a, b)
            sd = math.sqrt((a.var() + b.var()) / 2) or 1e-9
            row.update(ks=round(float(ks.statistic), 4), p=float(ks.pvalue), smd=round(float((b.mean() - a.mean()) / sd), 3),
                       real_q=[round(float(x), 5) for x in np.percentile(a, [25, 50, 75])],
                       render_q=[round(float(x), 5) for x in np.percentile(b, [25, 50, 75])],
                       real_mean=round(float(a.mean()), 5), render_mean=round(float(b.mean()), 5))
        else:
            row.update(ks=None, note="too few values on one side")
        out.append(row)
    out.sort(key=lambda r: -(r["ks"] if r["ks"] is not None else -1))
    for i, r in enumerate(out):
        r["rank"] = i + 1
    return out


def _fmt(x):
    if x is None:
        return "-"
    ax = abs(x)
    if ax >= 1000:
        return f"{x:.0f}"
    if ax >= 10:
        return f"{x:.1f}"
    if ax >= 0.1:
        return f"{x:.2f}"
    if ax >= 0.001:
        return f"{x:.3f}"
    return f"{x:.1e}" if x else "0"


def fold_of_groups(groups, nfold=NFOLD, seed=0):
    """Whole groups to folds, balanced by size (largest first into the smallest fold)."""
    uniq, counts = np.unique(groups, return_counts=True)
    rng = np.random.default_rng(seed)
    order = np.lexsort((rng.random(len(uniq)), -counts))
    load = np.zeros(nfold); gf = {}
    for i in order:
        f = int(np.argmin(load)); gf[uniq[i]] = f; load[f] += counts[i]
    return np.array([gf[g] for g in groups])


def image_level(X, y, folds, C=0.1):
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import roc_auc_score
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    p = np.zeros(len(y)); z = np.zeros(len(y))
    for f in np.unique(folds):
        tr, te = folds != f, folds == f
        m = make_pipeline(StandardScaler(), LogisticRegression(C=C, max_iter=3000, class_weight="balanced"))
        m.fit(X[tr], y[tr])
        p[te] = m.predict_proba(X[te])[:, 1]; z[te] = m.decision_function(X[te])
    return dict(auc=round(float(roc_auc_score(y, p)), 4), acc=round(float(((p > 0.5) == y).mean()), 4), dprime=dprime(z, y)), p


def dprime(z, y):
    """Separation of held-out scores: (mean render - mean real) / pooled sd. Keeps falling as renders get
    closer even while the AUC still reads 1.0."""
    a, b = z[y == 0], z[y == 1]
    if len(a) < 2 or len(b) < 2:
        return None
    return round(float((b.mean() - a.mean()) / math.sqrt((a.var() + b.var()) / 2 + 1e-12)), 3)


def frechet(A, B):
    """Frechet distance between Gaussians fitted to two feature sets (as FID, here on DINOv3 tokens)."""
    A = A.astype(np.float64); B = B.astype(np.float64)
    ma, mb = A.mean(0), B.mean(0)
    ca, cb = np.cov(A, rowvar=False), np.cov(B, rowvar=False)
    w, V = np.linalg.eigh(ca)
    sa = (V * np.sqrt(np.clip(w, 0, None))) @ V.T                  # sqrt(ca)
    ev = np.linalg.eigvalsh(sa @ cb @ sa)                           # tr sqrt(ca cb) = sum sqrt(eig(sa cb sa))
    return float(((ma - mb) ** 2).sum() + np.trace(ca) + np.trace(cb) - 2 * np.sqrt(np.clip(ev, 0, None)).sum())


FD_N, FD_N_PATCH, FD_DIM, FD_REPEATS = 300, 3000, 64, 5


def fd_block(Ra, Rb, G, n_fix, seed=0):
    """Frechet distance in the 64 leading principal components of the real features (fit on real half A),
    at a FIXED sample size per side (FD grows as n shrinks, so every score card uses the same n), averaged
    over 5 random draws: floor = FD(real half A, real half B), gap = FD(real half A, renders)."""
    rng = np.random.default_rng(seed)
    Ra = Ra.astype(np.float64)
    mu = Ra.mean(0)
    _, _, Vt = np.linalg.svd(Ra[rng.choice(len(Ra), min(len(Ra), 20000), replace=False)] - mu, full_matrices=False)
    P = Vt[:FD_DIM].T
    pa, pb, pg = (Ra - mu) @ P, (Rb.astype(np.float64) - mu) @ P, (G.astype(np.float64) - mu) @ P
    n = min(n_fix, len(pa), len(pb), len(pg))
    rr, rg = [], []
    for _ in range(FD_REPEATS):
        a = pa[rng.choice(len(pa), n, replace=False)]
        rr.append(frechet(a, pb[rng.choice(len(pb), n, replace=False)])); rg.append(frechet(a, pg[rng.choice(len(pg), n, replace=False)]))
    f_rr, f_rg = float(np.mean(rr)), float(np.mean(rg))
    return dict(fd_real_real=round(f_rr, 3), fd_real_render=round(f_rg, 3), fd_ratio=round(f_rg / max(f_rr, 1e-9), 2), n=int(n),
                comparable=bool(n == n_fix))


def patch_probe(T, y, folds, reg, wd=1e-3, epochs=60, device=None):
    """Linear real-vs-render probe on patch tokens, trained with whole held-out groups.
    Returns out-of-fold logits and the fold weights (mu, sd, w, b)."""
    import torch
    dev = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(0)
    logit = np.zeros(len(y), np.float32)
    models = {}
    for f in np.unique(folds):
        tr, te = folds != f, folds == f
        Xt = torch.from_numpy(T[tr].astype(np.float32)).to(dev)
        mu = Xt.mean(0); sd = Xt.std(0) + 1e-4
        Xt = (Xt - mu) / sd
        yt = torch.from_numpy(y[tr].astype(np.float32)).to(dev)
        pos = float(yt.mean())
        wts = torch.where(yt > 0.5, 0.5 / pos, 0.5 / (1 - pos))
        w = torch.zeros(T.shape[1], device=dev, requires_grad=True); b = torch.zeros(1, device=dev, requires_grad=True)
        opt = torch.optim.Adam([w, b], lr=1e-2)
        n = Xt.shape[0]; bs = 8192
        for ep in range(epochs):
            perm = torch.randperm(n, device=dev)
            for s in range(0, n, bs):
                idx = perm[s:s + bs]
                z = Xt[idx] @ w + b
                loss = (torch.nn.functional.binary_cross_entropy_with_logits(z, yt[idx], reduction="none") * wts[idx]).mean() + wd * (w * w).sum()
                opt.zero_grad(); loss.backward(); opt.step()
        with torch.no_grad():
            Xe = (torch.from_numpy(T[te].astype(np.float32)).to(dev) - mu) / sd
            logit[te] = (Xe @ w + b).cpu().numpy()
        models[int(f)] = dict(mu=mu.cpu().numpy(), sd=sd.cpu().numpy(), w=w.detach().cpu().numpy(), b=float(b.item()))
        del Xt
    return logit, models


def _auc(y, s):
    from sklearn.metrics import roc_auc_score
    if len(np.unique(y)) < 2:
        return None
    return round(float(roc_auc_score(y, s)), 4)


def knn_dist(A, B, k=3, exclude=None):
    """Mean cosine distance from each row of A to its k nearest rows of B (rows L2-normalised);
    exclude: (ga, gb) group labels, pairs with ga == gb are skipped."""
    out = np.zeros(len(A)); nn = np.zeros((len(A), k), int)
    for s in range(0, len(A), 512):
        d = 1.0 - A[s:s + 512] @ B.T
        if exclude is not None:
            ga, gb = exclude
            d[ga[s:s + 512, None] == gb[None, :]] = np.inf
        idx = np.argpartition(d, k, axis=1)[:, :k]
        dd = np.take_along_axis(d, idx, 1)
        o = np.argsort(dd, 1)
        nn[s:s + 512] = np.take_along_axis(idx, o, 1); out[s:s + 512] = np.take_along_axis(dd, o, 1).mean(1)
    return out, nn


def _norm(F, mu):
    F = F.astype(np.float32) - mu
    return F / (np.linalg.norm(F, axis=1, keepdims=True) + 1e-8)


# ------------------------------------------------------------------------------------------ main
def run(renders: Path, n: int = 1500, compare: Path | None = None, out_json: Path | None = None, repo_json: Path | None = None):
    common.ensure_dirs()
    t0 = time.time()
    if not (common.CACHE / REAL_NAME / "feats.npz").exists():
        extract.extract(common.real_samples(1500), REAL_NAME)
    if not (common.CACHE / SCEN_NAME / "feats.npz").exists() and common.DEFAULT_SCENARIOS.exists():
        extract.extract(common.scenario_samples(common.DEFAULT_SCENARIOS), SCEN_NAME)
    rname = "render_" + common.tag_of(renders, n, 0)
    if not (common.CACHE / rname / "feats.npz").exists():
        extract.extract(common.render_samples(renders, n), rname)
    rs, rrows, rf, _ = extract.load(REAL_NAME)
    gs, grows, gf, gdir = extract.load(rname)
    rok = rf["ok"]; gok = gf["ok"]
    rrows_ok = [r for r, o in zip(rrows, rok) if o]; grows_ok = [r for r, o in zip(grows, gok) if o]

    # ---- 1. statistics
    table = stat_table(rrows_ok, grows_ok)

    # ---- 2. learned: image level
    y = np.r_[np.zeros(len(rs)), np.ones(len(gs))].astype(int)
    groups = np.array([s["group"] for s in rs] + [s["group"] for s in gs])
    ok = np.r_[rok, gok]
    folds = fold_of_groups(groups)
    learned = {}
    oof = {}
    for nm, keys in (("crop", ("crop_cls", "crop_mean")), ("frame", ("frame_cls", "frame_mean")), ("whale_only", ("whale_mean",))):
        X = np.concatenate([np.r_[rf[k], gf[k]].astype(np.float32) for k in keys], 1)
        res, p = image_level(X[ok], y[ok], folds[ok])
        learned[f"image_{nm}"] = res
        oof[nm] = (np.flatnonzero(ok), p)
    # ---- patch level
    T = np.concatenate([rf["patch_tok"], gf["patch_tok"]])
    yp = np.r_[np.zeros(len(rf["patch_tok"])), np.ones(len(gf["patch_tok"]))].astype(int)
    own = np.r_[rf["patch_own"], gf["patch_own"] + len(rs)]
    reg = np.r_[rf["patch_reg"], gf["patch_reg"]]
    pf = folds[own]
    logit, models = patch_probe(T, yp, pf, reg)
    patch = {"all": dict(auc=_auc(yp, logit), dprime=dprime(logit, yp), acc=round(float(((logit > 0) == yp).mean()), 4), n_real=int((yp == 0).sum()), n_render=int((yp == 1).sum()))}
    for r, nm in common.REGIONS.items():
        m = reg == r
        patch[nm] = dict(auc=_auc(yp[m], logit[m]), dprime=dprime(logit[m], yp[m]), acc=round(float(((logit[m] > 0) == yp[m]).mean()), 4) if m.any() else None,
                         n_real=int((m & (yp == 0)).sum()), n_render=int((m & (yp == 1)).sum()),
                         real_mean_p_render=round(float((1 / (1 + np.exp(-logit[m & (yp == 0)]))).mean()), 3) if (m & (yp == 0)).any() else None,
                         render_mean_p_render=round(float((1 / (1 + np.exp(-logit[m & (yp == 1)]))).mean()), 3) if (m & (yp == 1)).any() else None)
    learned["patch"] = patch
    # ---- Frechet DINO distances, real days split in two halves (alternate days) as the floor
    days = sorted(set(s["group"] for s in rs))
    half = np.array([days.index(s["group"]) % 2 for s in rs])
    fd = {}
    for nm, key in (("crop", "crop_mean"), ("whale", "whale_mean"), ("frame", "frame_mean")):
        fd[nm] = fd_block(rf[key][rok & (half == 0)], rf[key][rok & (half == 1)], gf[key][gok], FD_N)
    rhalf = half[rf["patch_own"]]
    rng = np.random.default_rng(0)
    for r, nm in common.REGIONS.items():
        ra = np.flatnonzero((rf["patch_reg"] == r) & (rhalf == 0)); rb = np.flatnonzero((rf["patch_reg"] == r) & (rhalf == 1))
        gg = np.flatnonzero(gf["patch_reg"] == r)
        if min(len(ra), len(rb), len(gg)) < 300:
            continue
        fd[f"patch_{nm}"] = fd_block(rf["patch_tok"][ra], rf["patch_tok"][rb], gf["patch_tok"][gg], FD_N_PATCH)
    learned["frechet"] = fd
    np.savez(gdir / "probe.npz", folds_real=folds[:len(rs)], folds_render=folds[len(rs):],
             **{f"f{f}_{k}": v for f, mdl in models.items() for k, v in mdl.items()},
             img_oof_idx=oof["crop"][0], img_oof_p=oof["crop"][1])

    # ---- coverage
    mu = rf["whale_mean"][rok].astype(np.float32).mean(0)
    A = _norm(rf["whale_mean"], mu); B = _norm(gf["whale_mean"], mu)
    A[~rok] = 0; B[~gok] = 0
    rday = np.array([s["group"] for s in rs]); gb = np.array([s["group"] for s in gs])
    d_r2g, nn_r2g = knn_dist(A, B[gok]); nn_r2g = np.flatnonzero(gok)[nn_r2g]
    d_r2r, _ = knn_dist(A, A, exclude=(rday, rday))
    d_g2r, nn_g2r = knn_dist(B, A[rok]); nn_g2r = np.flatnonzero(rok)[nn_g2r]
    d_g2g, _ = knn_dist(B, B, exclude=(gb, gb))
    rk, gk = rok, gok
    cov = dict(
        real_to_render_median=round(float(np.median(d_r2g[rk])), 4), real_to_real_other_day_median=round(float(np.median(d_r2r[rk])), 4),
        coverage_ratio=round(float(np.median(d_r2g[rk]) / np.median(d_r2r[rk])), 3),
        real_uncovered_share=round(float((d_r2g[rk] > d_r2r[rk]).mean()), 3),
        render_to_real_median=round(float(np.median(d_g2r[gk])), 4), render_to_render_other_batch_median=round(float(np.median(d_g2g[gk])), 4),
        render_isolation_ratio=round(float(np.median(d_g2r[gk]) / np.median(d_g2g[gk])), 3),
        render_far_share=round(float((d_g2r[gk] > np.percentile(d_r2r[rk], 95)).mean()), 3),
        note="cosine distance (mean of 3 nearest) between whale-region mean DINOv3 tokens, centred on the real mean; "
             "coverage_ratio = median real->render / median real->real (other days); real_uncovered_share = share of real crops "
             "whose nearest renders are farther than their nearest real crops of other days; render_far_share = share of renders "
             "farther from any real crop than 95 % of real crops are from other days")
    learned["coverage"] = cov
    # ---- scenarios
    scen = {}
    if (common.CACHE / SCEN_NAME / "feats.npz").exists():
        ss, srows, sf, _ = extract.load(SCEN_NAME)
        Csc = _norm(sf["whale_mean"], mu)
        sday = np.array([s["group"] for s in ss])
        ds_g, _ = knn_dist(Csc, B[gok])
        ds_r, _ = knn_dist(Csc, A[rok], exclude=(sday, rday[rok]))
        names = sorted(set(s["scenario"] for s in ss))
        for nm in names:
            m = np.array([s["scenario"] == nm for s in ss]) & sf["ok"]
            if m.sum() < 5:
                continue
            scen[nm] = dict(n=int(m.sum()), to_render_median=round(float(np.median(ds_g[m])), 4), to_real_median=round(float(np.median(ds_r[m])), 4),
                            ratio=round(float(np.median(ds_g[m]) / np.median(ds_r[m])), 3), uncovered_share=round(float((ds_g[m] > ds_r[m]).mean()), 3))
    # ---- lists for the sheets (scratch only)
    P_all = np.full(len(y), np.nan); P_all[oof['crop'][0]] = oof['crop'][1]; P_render = P_all[len(rs):]
    lists = dict(
        real_farthest=[dict(i=int(i), key=rs[i]["key"], day=rs[i]["day"], d=round(float(d_r2g[i]), 4), d_real=round(float(d_r2r[i]), 4),
                            nn_render=[gs[j]["id"] for j in nn_r2g[i]], nn_render_idx=[int(j) for j in nn_r2g[i]])
                       for i in np.argsort(-np.where(rk, d_r2g, -1))[:48]],
        render_farthest=[dict(i=int(i), id=gs[i]["id"], category=gs[i].get("category"), d=round(float(d_g2r[i]), 4),
                              nn_real_idx=[int(j) for j in nn_g2r[i]]) for i in np.argsort(-np.where(gk, d_g2r, -1))[:48]],
        render_nearest=[dict(i=int(i), id=gs[i]["id"], category=gs[i].get("category"), d=round(float(d_g2r[i]), 4),
                             nn_real_idx=[int(j) for j in nn_g2r[i]]) for i in np.argsort(np.where(gk, d_g2r, 9))[:48]],
        render_by_category={c: dict(n=int(m.sum()), d_median=round(float(np.median(d_g2r[m])), 4),
                                    p_render_median=round(float(np.nanmedian(P_render[m])), 3))
                            for c in sorted(set(str(s.get("category")) for s in gs)) for m in [np.array([str(s.get("category")) == c for s in gs]) & gk] if m.sum() >= 5},
    )
    (gdir / "lists.json").write_text(json.dumps(lists, indent=1))
    card = dict(
        renders=str(Path(renders).resolve()), render_cache=rname, n_real=int(rok.sum()), n_render=int(gok.sum()),
        n_real_days=len(set(rday)), n_render_groups=len(set(gb)), created=time.strftime("%Y-%m-%d %H:%M"),
        headline=dict(image_auc=learned["image_crop"]["auc"], patch_auc=patch["all"]["auc"], coverage_ratio=cov["coverage_ratio"],
                      image_dprime=learned["image_crop"]["dprime"], patch_dprime=patch["all"]["dprime"],
                      fd_crop_ratio=learned["frechet"]["crop"]["fd_ratio"]),
        learned=learned, scenarios=scen, render_by_category=lists["render_by_category"], stats=table,
        horizon_sources=dict(real=_count(rrows_ok, "_horizon_src"), render=_count(grows_ok, "_horizon_src")))
    out_json = Path(out_json or (common.SCRATCH / f"scorecard_{rname}.json"))
    out_json.write_text(json.dumps(card, indent=1))
    if repo_json:
        Path(repo_json).write_text(json.dumps(card, indent=1))
    print_card(card, compare=json.loads(Path(compare).read_text()) if compare else None)
    print(f"\nwritten: {out_json}" + (f" and {repo_json}" if repo_json else "") + f"  ({time.time() - t0:.0f}s)")
    return card


def _count(rows, k):
    out = {}
    for r in rows:
        out[r.get(k)] = out.get(r.get(k), 0) + 1
    return out


def print_card(card, compare=None, top=None):
    old = {r["stat"]: r for r in compare["stats"]} if compare else {}
    if card["n_render"] < 300:
        print(f"WARNING: only {card['n_render']} render shots; AUC / coverage are noisy and the Frechet numbers use a smaller n than run05's card")
    print(f"REALISM SCORE CARD  renders: {card['renders']}\n  real crops {card['n_real']} ({card['n_real_days']} days), "
          f"render shots {card['n_render']} ({card['n_render_groups']} batches)")
    h = card["headline"]
    print(f"\n  HEADLINE  image AUC {h['image_auc']}   patch AUC {h['patch_auc']}   coverage ratio {h['coverage_ratio']}"
          + (f"   (before: {compare['headline']['image_auc']} / {compare['headline']['patch_auc']} / {compare['headline']['coverage_ratio']})" if compare else ""))
    print(f"            (not saturating: image d' {h.get('image_dprime')}  patch d' {h.get('patch_dprime')}  Frechet crop ratio {h.get('fd_crop_ratio')})"
          + (f"   (before: {compare['headline'].get('image_dprime')} / {compare['headline'].get('patch_dprime')} / {compare['headline'].get('fd_crop_ratio')})" if compare else ""))
    print("\n  1. statistics, largest gaps first (KS 0 = same distribution, 1 = no overlap; SMD = render - real in pooled sd)")
    hdr = f"  {'#':>3} {'statistic':22s} {'unit':10s} {'real median [IQR]':>28s} {'render median [IQR]':>28s} {'KS':>5s} {'SMD':>6s}"
    if compare:
        hdr += f" {'KS before':>9s}"
    print(hdr)
    for r in card["stats"][:top] if top else card["stats"]:
        if r.get("ks") is None:
            print(f"  {r['rank']:>3} {r['stat']:22s} {r['unit'][:10]:10s} {'(n real ' + str(r['n_real']) + ', render ' + str(r['n_render']) + ')':>58s}")
            continue
        rq, gq = r["real_q"], r["render_q"]
        line = (f"  {r['rank']:>3} {r['stat']:22s} {r['unit'][:10]:10s} {_fmt(rq[1]):>9s} [{_fmt(rq[0])}, {_fmt(rq[2])}]".ljust(68)
                + f"{_fmt(gq[1]):>9s} [{_fmt(gq[0])}, {_fmt(gq[2])}]".ljust(29) + f" {r['ks']:5.2f} {r['smd']:6.2f}")
        if compare and r["stat"] in old and old[r["stat"]].get("ks") is not None:
            line += f" {old[r['stat']]['ks']:9.2f}"
        print(line)
    L = card["learned"]
    print("\n  2. learned gap (frozen DINOv3 ViT-B/16; folds hold out whole real days and render batches; 0.5 = cannot tell apart)")
    for k in ("image_crop", "image_frame", "image_whale_only"):
        line = f"     {k:18s} AUC {L[k]['auc']:.3f}  accuracy {L[k]['acc']:.3f}  d' {L[k].get('dprime')}"
        if compare:
            line += f"   (before AUC {compare['learned'][k]['auc']:.3f})"
        print(line)
    print("     patch probe per region (AUC over patches of that region; mean P(render) on real / on render patches):")
    for k, v in L["patch"].items():
        line = f"       {k:7s} AUC {v['auc'] if v['auc'] is not None else '-':<6} d' {v.get('dprime')!s:<6} n real {v['n_real']:6d} n render {v['n_render']:6d}"
        if "real_mean_p_render" in v:
            line += f"   P(render) real {v['real_mean_p_render']}  render {v['render_mean_p_render']}"
        if compare and k in compare["learned"]["patch"]:
            line += f"   (before {compare['learned']['patch'][k]['auc']})"
        print(line)
    if "frechet" in L:
        print("     Frechet DINO distance, 64 PCs, fixed n (real half A vs real half B = floor | real half A vs renders; ratio 1.0 = as close as real days are to each other):")
        for k, v in L["frechet"].items():
            line = f"       {k:12s} real-real {v['fd_real_real']:8.2f}  real-render {v['fd_real_render']:8.2f}  ratio {v['fd_ratio']:6.2f}  (n {v['n']}{'' if v.get('comparable', True) else ', SMALLER n: not comparable'})"
            if compare and k in compare["learned"].get("frechet", {}):
                line += f"   (before ratio {compare['learned']['frechet'][k]['fd_ratio']})"
            print(line)
    c = L["coverage"]
    print(f"     coverage: real->render {c['real_to_render_median']}  real->real(other days) {c['real_to_real_other_day_median']}  "
          f"ratio {c['coverage_ratio']}  real crops uncovered {c['real_uncovered_share']:.0%}")
    print(f"               render->real {c['render_to_real_median']}  render->render(other batches) {c['render_to_render_other_batch_median']}  "
          f"renders far from all real {c['render_far_share']:.0%}")
    if card.get("scenarios"):
        print("\n  3. coverage per hard situation (median cosine distance of the instance's whale tokens to its 3 nearest renders /"
              " to its 3 nearest real crops of other days)")
        for k, v in sorted(card["scenarios"].items(), key=lambda kv: -kv[1]["ratio"]):
            line = f"     {k:36s} n {v['n']:3d}  to renders {v['to_render_median']:.3f}  to real {v['to_real_median']:.3f}  ratio {v['ratio']:.2f}  uncovered {v['uncovered_share']:.0%}"
            if compare and k in compare.get("scenarios", {}):
                line += f"   (before ratio {compare['scenarios'][k]['ratio']:.2f})"
            print(line)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--renders", default=str(common.RUN05), help="render dataset folder (images/, masks/, meta/ ...)")
    ap.add_argument("--n", type=int, default=1500)
    ap.add_argument("--compare", default=None, help="an older score card json to compare with, line by line")
    ap.add_argument("--out", default=None, help="score card json (default: scratch folder)")
    ap.add_argument("--repo-json", default=None, help="also write the (aggregate-only) json here, e.g. experiments/realism_gap/scorecard_run05.json")
    ap.add_argument("--print-only", default=None, help="print an existing score card json and exit")
    a = ap.parse_args()
    if a.print_only:
        print_card(json.loads(Path(a.print_only).read_text()), compare=json.loads(Path(a.compare).read_text()) if a.compare else None)
    else:
        run(Path(a.renders), a.n, Path(a.compare) if a.compare else None, a.out, a.repo_json)
