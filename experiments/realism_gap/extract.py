"""One pass over a sample list: views, region maps, hand-made statistics, frozen DINOv3 ViT-B/16
features, small thumbnails for the sheets. Everything goes to <SCRATCH>/cache/<name>/ (never the repo):

    samples.json   the sample list (ids, groups, paths)
    stats.jsonl    one dict of statistics per sample (same order)
    feats.npz      crop_cls, crop_mean, frame_cls, frame_mean, whale_mean (N, 768) fp16;
                   patch_tok (M, 768) fp16 + patch_reg (M,) + patch_own (M,): up to 32 patch tokens per
                   region per crop; full_idx (K,) + full_tok (K, 32, 32, 768): every patch of K crops (heat
                   maps); frac (N, 5, 32, 32): region fractions per patch
    thumbs/        <i>_crop.jpg (512), <i>_frame.jpg (long side 640), <i>_lab.png (crop region map)
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import common  # noqa: E402
import horizon  # noqa: E402
import stats as S  # noqa: E402

FRAME_DINO = (1248, 832)    # full frame for DINO: centre-cropped to 3:2 in both worlds
PER_REGION = 32
K_FULL = 32


def _render_meta(s):
    p = Path(s["img"]).parent.parent / "meta" / f"{s['id']}.json"
    if p.exists():
        try:
            return json.loads(p.read_text())
        except Exception:
            return None
    return None


def process(s: dict, thumbs: Path | None, idx: int):
    rgb, whale, ign, box = common.load_sample(s)
    H, W = whale.shape
    # ---- frame view
    sv = common.FRAME_LONG / max(H, W)
    fw, fh = int(round(W * sv)), int(round(H * sv))
    frgb = common.resize_any(rgb, (fw, fh))
    fwh = common.resize_any(whale, (fw, fh), nearest=True) > 0
    fig = common.resize_any(ign, (fw, fh), nearest=True) > 0
    flab, fS = S.to_lab(frgb)
    fbox = [v * sv for v in box]
    hz, hz_src = None, "none"
    if s["world"] == "render":
        meta = _render_meta(s)
        if meta is not None and "camera" in meta:
            try:
                tr = horizon.render_truth(meta, W, H)
                hz_src = "geometry"
                if tr is not None:
                    xs = np.arange(fw) / sv
                    hz = (np.interp(xs, np.arange(W), tr) * sv).astype(np.float32)
                    if (hz < 0).all() or (hz >= fh).all():
                        hz = None
            except Exception:
                hz_src = "none"
    if hz_src == "none":
        hz, _ = horizon.detect(flab, fwh | fig, fbox)
        hz_src = "detector"
    if hz is not None and (hz <= 0).all():
        hz = None
    band_f = float(np.clip(fbox[3] - fbox[1], 8, 0.2 * fw))
    fvalid = np.ones((fh, fw), bool)
    flab_map = common.region_map(fwh, fig, fvalid, hz, band_f)
    # ---- crop view
    sx, sy, side = common.crop_params(box, W, H)
    C = common.CROP
    crgb, _ = common.warp_square(rgb, sx, sy, side, C, None)
    cwh = common.warp_square(whale.astype(np.uint8), sx, sy, side, C, cv2.INTER_NEAREST)[0] > 0
    cig = common.warp_square(ign.astype(np.uint8), sx, sy, side, C, cv2.INTER_NEAREST)[0] > 0
    cvalid = common.warp_square(np.ones((H, W), np.uint8), sx, sy, side, C, cv2.INTER_NEAREST)[0] > 0
    hzc = None
    if hz is not None:
        u = np.arange(C)
        xn = sx + (u + 0.5) * side / C
        yn = np.interp(xn * sv, np.arange(fw), hz) / sv
        hzc = ((yn - sy) * C / side).astype(np.float32)
    band_c = float(np.clip((box[3] - box[1]) * C / side, 4, 0.25 * C))
    clab_map = common.region_map(cwh, cig, cvalid, hzc, band_c)
    clab, cS = S.to_lab(crgb)
    st = S.crop_stats(crgb, clab, cS, clab_map, cvalid)
    st.update(S.frame_stats(frgb, flab, fS, flab_map, fvalid))
    bw, bh = box[2] - box[0], box[3] - box[1]
    st.update({"frame.has_horizon": float(hz is not None), "frame.aspect": W / H, "frame.width_px": float(W),
               "whale.len_share": bw / W, "whale.height_share": bh / H, "whale.area_share": float(whale.sum()) / (W * H),
               "whale.cx": (box[0] + box[2]) / 2 / W, "whale.cy": (box[1] + box[3]) / 2 / H,
               "whale.native_width_px": float(bw), "crop.native_per_px": side / C})
    st = {k: (None if (v is None or (isinstance(v, float) and not math.isfinite(v))) else round(float(v), 6)) for k, v in st.items()}
    st["_horizon_src"] = hz_src
    # ---- DINO frame input: centre crop to 3:2, resize
    ar = 1.5
    if fw / fh > ar:
        cw = int(round(fh * ar)); x0 = (fw - cw) // 2; fr = frgb[:, x0:x0 + cw]
    else:
        ch = int(round(fw / ar)); y0 = (fh - ch) // 2; fr = frgb[y0:y0 + ch]
    fr = common.resize_any(fr, FRAME_DINO)
    # region fractions per 16 px patch
    onehot = np.stack([(clab_map == r).astype(np.float32) for r in range(5)], 0)
    frac = np.stack([cv2.resize(o, (C // 16, C // 16), interpolation=cv2.INTER_AREA) for o in onehot], 0)
    if thumbs is not None:
        cv2.imwrite(str(thumbs / f"{idx}_crop.jpg"), cv2.cvtColor(crgb, cv2.COLOR_RGB2BGR), [cv2.IMWRITE_JPEG_QUALITY, 90])
        tw = 640; th = int(round(fh * tw / fw))
        cv2.imwrite(str(thumbs / f"{idx}_frame.jpg"), cv2.cvtColor(cv2.resize(frgb, (tw, th), interpolation=cv2.INTER_AREA), cv2.COLOR_RGB2BGR),
                    [cv2.IMWRITE_JPEG_QUALITY, 88])
        cv2.imwrite(str(thumbs / f"{idx}_lab.png"), clab_map)
        flm = cv2.resize(flab_map, (tw, th), interpolation=cv2.INTER_NEAREST)
        cv2.imwrite(str(thumbs / f"{idx}_flab.png"), flm)
    return crgb, fr, frac.astype(np.float16), st


class DS(torch.utils.data.Dataset):
    def __init__(self, samples, thumbs):
        self.samples, self.thumbs = samples, thumbs

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, i):
        try:
            crgb, fr, frac, st = process(self.samples[i], self.thumbs, i)
            ok = True
        except Exception as e:  # a broken file: keep the slot, mark it
            crgb = np.zeros((common.CROP, common.CROP, 3), np.uint8); fr = np.zeros((FRAME_DINO[1], FRAME_DINO[0], 3), np.uint8)
            frac = np.zeros((5, 32, 32), np.float16); st = {"_error": repr(e)[:300]}; ok = False
        return (torch.from_numpy(np.ascontiguousarray(crgb)).permute(2, 0, 1), torch.from_numpy(np.ascontiguousarray(fr)).permute(2, 0, 1),
                torch.from_numpy(frac), json.dumps(st), ok, i)


def _collate(b):
    return (torch.stack([x[0] for x in b]), torch.stack([x[1] for x in b]), torch.stack([x[2] for x in b]),
            [x[3] for x in b], [x[4] for x in b], [x[5] for x in b])


_VIT = None


def vit(device):
    global _VIT
    if _VIT is None:
        from model_fine_tuning.common.backbone import load_backbone
        _VIT = load_backbone("vitb16").to(device).eval()
    return _VIT


def extract(samples: list[dict], name: str, batch: int = 8, workers: int = 6, k_full: int = K_FULL, seed: int = 0, force: bool = False):
    out = common.CACHE / name
    if (out / "feats.npz").exists() and not force:
        return out
    out.mkdir(parents=True, exist_ok=True)
    thumbs = out / "thumbs"; thumbs.mkdir(exist_ok=True)
    (out / "samples.json").write_text(json.dumps(samples))
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = vit(dev)
    from model_fine_tuning.common.backbone import IMAGENET_MEAN, IMAGENET_STD
    mean = torch.tensor(IMAGENET_MEAN, device=dev).view(1, 3, 1, 1); std = torch.tensor(IMAGENET_STD, device=dev).view(1, 3, 1, 1)
    N = len(samples)
    rng = np.random.default_rng(seed)
    full_set = set(rng.permutation(N)[:k_full].tolist())
    D = 768
    arr = {k: np.zeros((N, D), np.float16) for k in ("crop_cls", "crop_mean", "frame_cls", "frame_mean", "whale_mean")}
    frac_all = np.zeros((N, 5, 32, 32), np.float16)
    okv = np.zeros(N, bool)
    ptok, preg, pown = [], [], []
    full_idx, full_tok = [], []
    stats_rows = [None] * N
    dl = torch.utils.data.DataLoader(DS(samples, thumbs), batch_size=batch, num_workers=workers, collate_fn=_collate, shuffle=False)
    t0 = time.time()
    done = 0
    for crop, fr, frac, sts, oks, ids in dl:
        with torch.no_grad(), torch.autocast(device_type="cuda", dtype=torch.float16, enabled=dev.type == "cuda"):
            x = (crop.to(dev).float() / 255.0 - mean) / std
            (pt, cls), = model.get_intermediate_layers(x, n=1, reshape=True, return_class_token=True, norm=True)
            y = (fr.to(dev).float() / 255.0 - mean) / std
            (pf, clsf), = model.get_intermediate_layers(y, n=1, reshape=True, return_class_token=True, norm=True)
        pt = pt.float().permute(0, 2, 3, 1).cpu().numpy()     # B, 32, 32, D
        cls = cls.float().cpu().numpy(); clsf = clsf.float().cpu().numpy(); pfm = pf.float().mean((2, 3)).cpu().numpy()
        fr_np = frac.float().numpy()
        for b, i in enumerate(ids):
            stats_rows[i] = json.loads(sts[b]); okv[i] = oks[b]; frac_all[i] = frac[b].numpy()
            f = fr_np[b]
            tok = pt[b].reshape(-1, D)
            usable = f[0].reshape(-1) < 0.5
            arr["crop_cls"][i] = cls[b]
            arr["crop_mean"][i] = tok[usable].mean(0) if usable.any() else tok.mean(0)
            arr["frame_cls"][i] = clsf[b]; arr["frame_mean"][i] = pfm[b]
            wf = f[1].reshape(-1)
            if (wf >= 0.5).any():
                arr["whale_mean"][i] = tok[wf >= 0.5].mean(0)
            elif wf.sum() > 0:
                arr["whale_mean"][i] = (tok * wf[:, None]).sum(0) / wf.sum()
            else:
                arr["whale_mean"][i] = tok.mean(0)
            reg = f[1:].reshape(4, -1)
            best = reg.argmax(0); bestv = reg.max(0)
            assign = np.where((bestv >= 0.5) & (f[0].reshape(-1) < 0.25), best + 1, 0)
            for r in (1, 2, 3, 4):
                cand = np.nonzero(assign == r)[0]
                if len(cand):
                    pick = rng.choice(cand, size=min(PER_REGION, len(cand)), replace=False)
                    ptok.append(tok[pick].astype(np.float16)); preg.append(np.full(len(pick), r, np.uint8)); pown.append(np.full(len(pick), i, np.int32))
            if i in full_set:
                full_idx.append(i); full_tok.append(pt[b].astype(np.float16))
        done += len(ids)
        if done % 200 < batch:
            print(f"  {name}: {done}/{N}  {time.time() - t0:.0f}s", flush=True)
    np.savez(out / "feats.npz", **arr, frac=frac_all, ok=okv,
             patch_tok=np.concatenate(ptok) if ptok else np.zeros((0, D), np.float16),
             patch_reg=np.concatenate(preg) if preg else np.zeros(0, np.uint8),
             patch_own=np.concatenate(pown) if pown else np.zeros(0, np.int32),
             full_idx=np.array(full_idx, np.int32), full_tok=np.stack(full_tok) if full_tok else np.zeros((0, 32, 32, D), np.float16))
    with open(out / "stats.jsonl", "w") as f:
        for r in stats_rows:
            f.write(json.dumps(r) + "\n")
    print(f"  {name}: {N} samples in {time.time() - t0:.0f}s -> {out}", flush=True)
    return out


def load(name: str):
    out = common.CACHE / name
    samples = json.loads((out / "samples.json").read_text())
    rows = [json.loads(l) for l in (out / "stats.jsonl").read_text().splitlines()]
    z = np.load(out / "feats.npz")
    return samples, rows, {k: z[k] for k in z.files}, out


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("what", choices=["real", "scenarios", "render"])
    ap.add_argument("--renders", default=str(common.RUN05))
    ap.add_argument("--n", type=int, default=1500)
    ap.add_argument("--limit", type=int, default=0, help="debug: first N samples only")
    a = ap.parse_args()
    common.ensure_dirs()
    if a.what == "real":
        ss = common.real_samples(a.n); nm = "real_v1"
    elif a.what == "scenarios":
        ss = common.scenario_samples(common.DEFAULT_SCENARIOS); nm = "scen_v1"
    else:
        ss = common.render_samples(Path(a.renders), a.n); nm = "render_" + common.tag_of(Path(a.renders), a.n, 0)
    if a.limit:
        ss = ss[:a.limit]; nm += f"_lim{a.limit}"
    print(len(ss), "samples ->", nm)
    extract(ss, nm, force=bool(a.limit))
