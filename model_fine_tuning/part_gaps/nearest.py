"""Nearest renders for real instances: frozen DINOv3 ViT-B/16 whale tokens (mean of the patch tokens on the whale, crop
1.5 x the box at 512 px) and silhouette shape (the whale mask in its box, padded square, 32 x 32, IoU).

    PYTHONPATH=. .venv/bin/python -m model_fine_tuning.part_gaps.nearest extract [--threads 12]   # features (cached)
    PYTHONPATH=. .venv/bin/python -m model_fine_tuning.part_gaps.nearest match [--model parts_seg_r05_a --variant hr]

Real side: every labelled sample instance (labels_ai.txt). Render side: every whale of the render datasets that have
render_eval outputs for --pool-model (r06 tests, run05 validation sample), plus --run05-extra run05 training shots.
Features in /var/tmp/sat3737-parts-cache/part_gaps/nearest/ (no pictures in the repo); match writes
experiments/part_gaps/nearest_<model>.json: per labelled instance its 5 nearest renders by DINO and by shape, and per
category the similarity of the nearest render for failing and for passing instances.
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import cv2
import numpy as np
import torch
from PIL import Image

from . import common as C

NN = C.CACHE / "nearest"
SIDE = 512


def crop_square(rgb, mask, box, side_px=SIDE):
    x1, y1, x2, y2 = box
    side = max(x2 - x1, y2 - y1) * 1.5 + 8
    cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
    M = np.array([[side_px / side, 0, -(cx - side / 2) * side_px / side], [0, side_px / side, -(cy - side / 2) * side_px / side]])
    c = cv2.warpAffine(rgb, M, (side_px, side_px), flags=cv2.INTER_AREA if side > side_px else cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=0)
    m = cv2.warpAffine(mask.astype(np.uint8), M, (side_px, side_px), flags=cv2.INTER_NEAREST) > 0
    return c, m


def silhouette(mask, box, n=32):
    x1, y1, x2, y2 = [int(round(v)) for v in box]
    m = mask[max(y1, 0):y2, max(x1, 0):x2].astype(np.uint8)
    h, w = m.shape
    s = max(h, w, 1)
    pad = np.zeros((s, s), np.uint8)
    pad[(s - h) // 2:(s - h) // 2 + h, (s - w) // 2:(s - w) // 2 + w] = m
    return cv2.resize(pad, (n, n), interpolation=cv2.INTER_AREA) > 0


def real_items():
    from .report import load_labels
    from .sample import instance_mask_2560
    lab = load_labels()
    for k, v in sorted(lab.items()):
        stem = k.rsplit("~", 1)[0]
        C.check_day(v["day"])
        bgr = cv2.imread(str(C.DAYS_ALL / v["day"] / "images" / f"{stem}.jpg"))
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        H, W = rgb.shape[:2]
        sc = W / 2560.0
        m = instance_mask_2560(v["day"], stem, int(k.rsplit("~", 1)[1]), 2560, int(round(H / sc)))
        if m is None:
            continue
        if sc != 1.0:
            m = cv2.resize(m.astype(np.uint8), (W, H), interpolation=cv2.INTER_NEAREST) > 0
        ys, xs = np.nonzero(m)
        if not len(xs):
            continue
        box = [xs.min(), ys.min(), xs.max() + 1, ys.max() + 1]
        yield {"id": k, "world": "real", "cat": v["cat"]}, rgb, m, box


POOL_TAGS = ("r06r_test_final", "r06r_test_check", "r06r_test_r3", "r06r_test_r4b", "r06r_test_r5", "r06r_test_r6", "run05")


def render_items(pool_model: str, run05_extra: int, tags=POOL_TAGS):
    """every whale of the run06 test sets and run05_extra random run05 shots (primary whale), from coverage.py files."""
    from .report import render_category
    for tag in tags:
        ds = C.FACTORY / tag / "dataset"
        p = C.CACHE / "coverage" / f"{tag}.jsonl"
        if not p.exists():
            continue
        recs = [json.loads(l) for l in open(p) if l.strip()]
        recs = [r for r in recs if sum(r["truth"][1:6]) > 0]
        if tag == "run05":
            recs = random.Random(1).sample(recs, min(run05_extra, len(recs)))
        by = {}
        for r in recs:
            by.setdefault(r["shot"], []).append(r)
        for sid, rs in sorted(by.items()):
            rgb = cv2.cvtColor(cv2.imread(str(ds / "images" / f"{sid}.png")), cv2.COLOR_BGR2RGB)
            inst = np.asarray(Image.open(ds / "instances" / f"{sid}.png"))
            if inst.ndim == 3:
                inst = inst[..., 0]
            for r in rs:
                m = inst == r["iid"]
                box = r["box"]
                yield {"id": f"{tag}/{sid}/{r['iid']}", "world": "render", "tag": tag, "shot": sid, "iid": r["iid"], "box": box,
                       "cat": render_category(r), "meta_category": r.get("meta_category"), "L2560": r["L2560"]}, rgb, m, box


def extract(a):
    from ..common.backbone import IMAGENET_MEAN, IMAGENET_STD, load_backbone
    torch.set_num_threads(a.threads)
    NN.mkdir(parents=True, exist_ok=True)
    model = load_backbone("vitb16").eval()
    mean = torch.tensor(IMAGENET_MEAN).view(1, 3, 1, 1)
    std = torch.tensor(IMAGENET_STD).view(1, 3, 1, 1)
    jobs = [("real", real_items()), ("render", render_items(a.pool_model, a.run05_extra))]
    if a.add:                                    # an extra render set appended to the pool: render_<tag>.npz
        jobs = [(f"render_{t}", render_items(a.pool_model, 0, (t,))) for t in a.add]
    for name, gen in jobs:
        out = NN / f"{name}.npz"
        if out.exists() and not a.force:
            print("cached", out)
            continue
        metas, feats, sils = [], [], []
        batch, bm = [], []
        def flush():
            if not batch:
                return
            x = torch.from_numpy(np.stack([b[0] for b in batch])).permute(0, 3, 1, 2).float() / 255.0
            x = (x - mean) / std
            with torch.no_grad():
                (pt, _), = model.get_intermediate_layers(x, n=1, reshape=True, return_class_token=True, norm=True)
            pt = pt.permute(0, 2, 3, 1).numpy()
            for (c, m, meta, sil), tok in zip(batch, pt):
                f = cv2.resize(m.astype(np.float32), (tok.shape[1], tok.shape[0]), interpolation=cv2.INTER_AREA)
                w = f >= 0.5
                v = tok[w].mean(0) if w.any() else (tok * f[..., None]).sum((0, 1)) / max(f.sum(), 1e-6)
                feats.append((v / (np.linalg.norm(v) + 1e-8)).astype(np.float16))
                sils.append(sil)
                metas.append(meta)
            batch.clear()
        for i, (meta, rgb, m, box) in enumerate(gen):
            c, cm = crop_square(rgb, m, box)
            batch.append((c, cm, meta, np.packbits(silhouette(m, box))))
            if len(batch) == 16:
                flush()
            if i % 200 == 0:
                print(name, i, flush=True)
        flush()
        np.savez(out, feats=np.stack(feats), sils=np.stack(sils))
        (NN / f"{name}.json").write_text(json.dumps(metas))
        print("wrote", out, len(metas), flush=True)


def load_neighbours(model: str, variant: str) -> dict:
    return json.loads((C.OUT / f"nearest_{model}_{variant}.json").read_text())


def match(a):
    rep = json.loads((C.OUT / f"report_{a.model}_{a.variant}.json").read_text())
    inst = rep["instances"]
    R = np.load(NN / "real.npz"); Rm = json.loads((NN / "real.json").read_text())
    qf, qs, Qm = [], [], []
    for f in sorted(NN.glob("render*.npz")):                 # the base pool + every set added with extract --add
        Q = np.load(f)
        qf.append(Q["feats"]); qs.append(Q["sils"]); Qm += json.loads(f.with_suffix(".json").read_text())
    fr = R["feats"].astype(np.float32); fq = np.concatenate(qf).astype(np.float32)
    sr = np.unpackbits(R["sils"], axis=1)[:, :1024].astype(bool); sq = np.unpackbits(np.concatenate(qs), axis=1)[:, :1024].astype(bool)
    sim = fr @ fq.T
    inter = sr.astype(np.float32) @ sq.T.astype(np.float32)
    union = sr.sum(1)[:, None] + sq.sum(1)[None, :] - inter
    iou = inter / np.maximum(union, 1)
    # real-to-real floor: the nearest OTHER-day real instance
    rr = fr @ fr.T
    days = np.array([m["id"].split("__")[0] if "__" in m["id"] else m["id"][:10] for m in Rm])
    res = {"model": a.model, "variant": a.variant, "n_real": len(Rm), "n_render": len(Qm),
           "render_pool": {t: sum(1 for m in Qm if m["tag"] == t) for t in sorted({m["tag"] for m in Qm})}, "real": {}, "by_category": {}}
    keys = list(inst)
    for i, m in enumerate(Rm):
        k = m["id"]
        if k not in inst:
            continue
        o = np.argsort(-sim[i])[:5]
        s = np.argsort(-iou[i])[:5]
        other = rr[i].copy()
        other[i] = -1
        res["real"][k] = {"cat": inst[k]["cat"], "ok": inst[k]["ok"], "call": inst[k]["call"], "split": inst[k]["split"],
                          "nn_sim": float(sim[i, o[0]]), "nn_real_sim": float(other.max()),
                          "nn_dino": [{"tag": Qm[j]["tag"], "shot": Qm[j]["shot"], "iid": Qm[j]["iid"], "box": Qm[j]["box"], "cat": Qm[j]["cat"],
                                       "meta_category": Qm[j]["meta_category"], "sim": float(sim[i, j])} for j in o],
                          "nn_shape": [{"tag": Qm[j]["tag"], "shot": Qm[j]["shot"], "iid": Qm[j]["iid"], "box": Qm[j]["box"], "cat": Qm[j]["cat"],
                                        "meta_category": Qm[j]["meta_category"], "iou": float(iou[i, j])} for j in s]}
    for cat in C.CATS:
        rs = [v for v in res["real"].values() if v["cat"] == cat]
        if not rs:
            continue
        def summ(xs):
            if not xs:
                return None
            return {"n": len(xs), "nn_sim_median": float(np.median([x["nn_sim"] for x in xs])),
                    "nn_real_sim_median": float(np.median([x["nn_real_sim"] for x in xs])),
                    "nn_shape_iou_median": float(np.median([x["nn_shape"][0]["iou"] for x in xs])),
                    "nn_same_category_dino": float(np.mean([x["nn_dino"][0]["cat"] == cat for x in xs])),
                    "nn_same_category_shape": float(np.mean([x["nn_shape"][0]["cat"] == cat for x in xs]))}
        res["by_category"][cat] = {"fail": summ([v for v in rs if not v["ok"]]), "pass": summ([v for v in rs if v["ok"]])}
    out = C.OUT / f"nearest_{a.model}_{a.variant}.json"
    out.write_text(json.dumps(res, indent=1))
    print(f"render pool {res['render_pool']}")
    print(f"{'category':8s} {'fail n':>6s} {'nn sim':>7s} {'real nn':>7s} {'shape':>6s} {'same cat dino/shape':>20s} | pass n  nn sim  real nn  shape  same cat")
    for cat, d in res["by_category"].items():
        f, p = d["fail"] or {}, d["pass"] or {}
        g = lambda x, k: "-" if not x else f"{x[k]:.2f}"  # noqa: E731
        print(f"{cat:8s} {f.get('n', 0):6d} {g(f, 'nn_sim_median'):>7s} {g(f, 'nn_real_sim_median'):>7s} {g(f, 'nn_shape_iou_median'):>6s} {g(f, 'nn_same_category_dino'):>9s}/{g(f, 'nn_same_category_shape'):<9s} | "
              f"{p.get('n', 0):6d} {g(p, 'nn_sim_median'):>7s} {g(p, 'nn_real_sim_median'):>7s} {g(p, 'nn_shape_iou_median'):>6s} {g(p, 'nn_same_category_dino')}/{g(p, 'nn_same_category_shape')}")
    print("wrote", out)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("what", choices=["extract", "match"])
    ap.add_argument("--threads", type=int, default=12)
    ap.add_argument("--pool-model", default="parts_seg_r05_a")
    ap.add_argument("--run05-extra", type=int, default=1500)
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--add", nargs="*", default=None, help="render set tags to append to the pool (render_<tag>.npz)")
    ap.add_argument("--model", default="parts_seg_r05_a")
    ap.add_argument("--variant", default="hr")
    a = ap.parse_args(argv)
    {"extract": extract, "match": match}[a.what](a)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
