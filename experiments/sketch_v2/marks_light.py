"""sketch_v2 C + D (2026-11-19): the marks drawn, measured three ways, with and without an exposure adjustment of the
fin before the marks are looked for (the outline is never adjusted). 21 train + 3 val days only.

  halves   within-burst repeatability (the prototype's halves.py on the v2 drawing): safe clusters with >= 4 frames that
           line up on one side are split into alternate halves, each half gets its own consensus; the normalised
           correlation of the two halves' blurred mark maps where both saw the skin, against the same correlation for two
           DIFFERENT animals' drawings of one group and side (bench.pairs, lined up with fin_sketch.register_view).
           AUC halves vs different. Also: the share of one half's drawn marks found again in the other half.
  boxes    2025-01-02 (train): the regions the researcher marked as evidence (Whale-Review-Web data pair_marks.json),
           mapped into each photo's fin tile; per box: does that photo's drawing (marks of the size drawn) hold a mark in
           it, against the same box moved to random places of the seen skin; SMALL boxes (< 10 % of the seen skin) apart.
Tone (D): none | curve (gamma on the fin cut-out so that the whale's median brightness goes to 110 of 255 when it is
darker, shadows lifted, anchored on the whale's own pixels) | raw (the cut-out re-decoded from the RAW, linear, exposure
anchored on the whale's median, shadows lifted; only frames whose whale median is below DARK: the dark fins).
    python marks_light.py halves|boxes none|curve|raw [--dark-only]"""
import json
import sys
from multiprocessing import Pool
from pathlib import Path

import cv2
import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / 'moonshot_sizing'))
sys.path.insert(0, str(HERE.parents[1]))
import bench  # noqa: E402
from size_levers import auc  # noqa: E402
from model_fine_tuning.cluster import fin_sketch as FS  # noqa: E402

DARK = 70                       # whale median luminance (0-255) below which a fin counts as dark
TARGET = 110
RAW_CACHED_ONLY = True          # in the pooled runs: only the RAW cut-outs decoded beforehand (rawcache), no decoding in a forked worker
RAWC = Path('/home/sat3737/NOAA_whales/moonshot_cache/sketch_aside/v2/raw_cutouts')


def whale_median(rgb, mask):
    Y = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    m = mask > 127 if mask.dtype != bool else mask
    return float(np.median(Y[m])) if m.sum() > 200 else 255.0


def tone_curve(rgb, mask):
    med = whale_median(rgb, mask)
    if med >= TARGET:
        return rgb
    g = float(np.clip(np.log(TARGET / 255) / np.log(max(med, 4) / 255), 0.35, 1.0))
    lut = np.clip(255 * (np.arange(256) / 255.0) ** g, 0, 255).astype(np.uint8)
    return lut[rgb]


def raw_cutout(day, key):
    """the fin cut-out re-decoded from the RAW (cached): linear 16-bit, white balance of the camera, exposure scaled so
    the whale's median goes to 0.18, then an sRGB-like curve with the shadows lifted."""
    f = RAWC / day / f'{key}.png'
    if f.exists():
        im = cv2.imread(str(f))
        return cv2.cvtColor(im, cv2.COLOR_BGR2RGB) if im is not None else None
    import rawpy
    d = FS.day_dir(day)
    stem = key.rsplit('~', 1)[0]
    src = None
    for line in open(d / 'images' / 'index.jsonl'):
        r = json.loads(line)
        if Path(r['image']).stem == stem:
            src = r['source']; break
    zp = d / 'parts_up' / f'{key}__fin.npz'
    if src is None or not zp.exists():
        return None
    z = np.load(zp, allow_pickle=True)
    with rawpy.imread(src) as raw:
        lin = raw.postprocess(use_camera_wb=True, output_bps=16, no_auto_bright=True, gamma=(1, 1), user_flip=None)
    M = z['M'].astype(np.float64)
    size = 1536
    cut = cv2.warpAffine(lin.astype(np.float32) / 65535.0, M, (size, size), flags=cv2.INTER_AREA if np.hypot(M[0, 0], M[0, 1]) < 1 else cv2.INTER_CUBIC)
    if bool(z['mirrored']):
        cut = cut[:, ::-1]
    if 'flipped_v' in z.files and bool(z['flipped_v']):
        cut = cut[::-1]
    m = z['whale']
    Y = cut.mean(2)
    med = float(np.median(Y[m])) if m.sum() > 200 else float(np.median(Y))
    x = np.clip(cut * (0.18 / max(med, 1e-5)), 0, None)
    x = x / (1 + x * 0.6)                                   # soft shoulder: highlights kept
    out = np.clip(255 * np.power(np.clip(x / 0.625, 0, 1), 1 / 2.2), 0, 255).astype(np.uint8)   # median -> ~138 of 255
    f.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(f), cv2.cvtColor(out, cv2.COLOR_RGB2BGR))
    return out


def patch_source(tone):
    """FS.source with the tone applied to the cut-out (the mask and the geometry unchanged)."""
    orig = FS.source

    def src(d, key):
        s = orig(d, key)
        if s is None or tone == 'none':
            return s
        img, m, f, name = s
        if whale_median(img, m) >= DARK and tone == 'raw':
            return s
        if tone == 'curve':
            return tone_curve(img, m), m, f, name
        if RAW_CACHED_ONLY and not (RAWC / Path(d).name / f'{key}.png').exists():
            return s
        r = raw_cutout(Path(d).name, key)
        if r is None or r.shape != img.shape:
            return s
        return r, m, f, name
    return src


DARK_LIST = Path('/home/sat3737/NOAA_whales/moonshot_cache/sketch_aside/v2/dark_clusters.json')
DARK_SET = {(r[0], int(r[1])) for r in json.loads(DARK_LIST.read_text())} if DARK_LIST.exists() else set()


def tiles_of(day, keys, P):
    d = FS.day_dir(day)
    out = []
    for k in keys:
        if k not in P:
            continue
        r, _ = FS.one(d, k, P[k])
        if r is not None:
            out.append((k, r))
    return out


def ncc(a, b, co):
    ga = cv2.GaussianBlur(a.astype(np.float32), (0, 0), 3.0) * co
    gb = cv2.GaussianBlur(b.astype(np.float32), (0, 0), 3.0) * co
    den = np.sqrt((ga ** 2).sum() * (gb ** 2).sum())
    return float((ga * gb).sum() / den) if den > 1e-6 else None


def halves_day(args):
    day, tone, dark_only = args
    cv2.setNumThreads(1)
    FS.source = patch_source(tone)
    d = FS.day_dir(day)
    P = json.loads((d / 'sketch' / 'points.json').read_text())['points']
    rows = []
    for c, ks in bench.clusters(day, 'dorsal').items():
        if len([k for k in ks if k in P]) < 4:
            continue
        T = tiles_of(day, ks, P)
        by = {}
        for k, r in T:
            by.setdefault(r[3]['side'], []).append((k, r))
        for side, rs in by.items():
            if len(rs) < 4:
                continue
            if dark_only and (day, int(c)) not in DARK_SET:      # the dark clusters, judged on the cut-outs as they are (darkcount)
                continue
            tiles = [r[:3] for _, r in rs]; infos = [r[3] for _, r in rs]
            fr = [FS.frame_drawing(*t) for t in tiles]
            ref = max(range(len(fr)), key=lambda j: fr[j]['seen'].sum() * min(infos[j]['H'], 150))
            ok = [ref]
            for j in range(len(fr)):
                if j == ref:
                    continue
                M, v = FS.register(fr[ref], fr[j])
                if v < FS.MIN_NCC:
                    continue
                rgb, mask, ins = tiles[j]
                fr[j] = FS.frame_drawing(cv2.warpAffine(rgb, M, (FS.TW, FS.TH)), cv2.warpAffine(mask, M, (FS.TW, FS.TH)), cv2.warpAffine(ins, M, (FS.TW, FS.TH), flags=cv2.INTER_NEAREST))
                ok.append(j)
            if len(ok) < 4:
                continue
            ok.sort()
            H = float(np.median([infos[j]['H'] for j in ok]))
            A = FS.consensus([fr[j] for j in ok[0::2]], None, H); B = FS.consensus([fr[j] for j in ok[1::2]], None, H)
            co = (A['seen_n'] >= 1) & (B['seen_n'] >= 1)
            v = ncc(A['marks'], B['marks'], co)
            k = np.ones((2 * FS.TOL + 1, 2 * FS.TOL + 1), np.uint8)
            nb = cv2.dilate(B['marks'].astype(np.uint8), k) > 0
            n, lab = cv2.connectedComponents((A['marks'] & co).astype(np.uint8))
            found = [bool(nb[lab == i].any()) for i in range(1, n)]
            rows.append(dict(day=day, c=int(c), side=side, ncc=v, found=found, marks=int(n - 1),
                             med=float(np.median([whale_median(tiles[j][0], tiles[j][1]) for j in ok]))))
    return rows


def diff_day(args):
    day, sub = args
    cv2.setNumThreads(1)
    dd = FS.day_dir(day) / sub
    idx = json.loads((dd / 'index.json').read_text())['clusters']
    V = {}
    out = []
    for a, b, same in bench.pairs(day, 'dorsal'):
        if same:
            continue
        for s in ('L', 'R'):
            ea, eb = (idx.get(str(a)) or {}).get(s), (idx.get(str(b)) or {}).get(s)
            if not ea or not eb or ea['n'] < 2 or eb['n'] < 2:
                continue
            va = V.setdefault((a, s), FS.load_view(dd / ea['npz'])); vb = V.setdefault((b, s), FS.load_view(dd / eb['npz']))
            M, sc = FS.register_view(va, vb)
            w = lambda x: cv2.warpAffine(x.astype(np.uint8), M, (FS.DW, FS.DH), flags=cv2.INTER_NEAREST) > 0
            co = va['seen'] & w(vb['seen'])
            v = ncc(va['marks'], w(vb['marks']), co)
            if v is not None:
                out.append(v)
    return out


if __name__ == '__main__':
    what, tone = sys.argv[1], sys.argv[2]
    dark_only = '--dark-only' in sys.argv
    days = bench.TRAIN + bench.VAL
    tag = f'{what}_{tone}' + ('_dark' if dark_only else '')
    if what == 'halves':
        with Pool(8) as p:
            R = [r for rs in p.map(halves_day, [(d, tone, dark_only) for d in days]) for r in rs]
        sub = sys.argv[sys.argv.index('--diff-sub') + 1] if '--diff-sub' in sys.argv else 'sketch/v2'
        with Pool(8) as p:
            Dn = [v for vs in p.map(diff_day, [(d, sub) for d in days]) for v in vs]
        h = np.array([r['ncc'] for r in R if r['ncc'] is not None])
        f = [x for r in R for x in r['found']]
        res = dict(tone=tone, dark_only=dark_only, halves=len(R), halves_with_marks=len(h), marks_per_half=round(float(np.mean([r['marks'] for r in R])), 2),
                   found_again=round(float(np.mean(f)), 3) if f else None, n_marks=len(f), ncc_median_halves=round(float(np.median(h)), 3),
                   ncc_median_diff=round(float(np.median(Dn)), 3), diff_pairs=len(Dn), auc_halves_vs_diff=round(float(auc(h, np.array(Dn))), 4))
    else:
        import marks_box
        res = marks_box.run(tone, patch_source)
    print(json.dumps(res))
    json.dump(res, open(HERE / f'marks_light_{tag}.json', 'w'), indent=1)
