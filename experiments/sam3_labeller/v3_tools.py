"""Scene model v3 (rules_scene_v3.md): boat labels, synthetic boat renders, development measures. Text only in the repo;
every per-photo file goes to the untracked moonshot_cache/sam3_labeller/v3/. The test day is refused here.

    V=.venv/bin/python; LOCK=/var/tmp/sat3737-parts-cache/gpu_train.lock
    flock $LOCK env PYTHONPATH=. $V experiments/sam3_labeller/v3_tools.py ade          # Mask2Former ADE20K maps: pseudo + real_s6mc + dev photos
    $V experiments/sam3_labeller/v3_tools.py labels                                   # v3 label maps -> V3/pseudo_v3, V3/real_s6mc_v3
    $V experiments/sam3_labeller/v3_tools.py renders RUN [RUN ...]                     # render label sets -> V3/renders/{train,hold}
    flock $LOCK env PYTHONPATH=. $V experiments/sam3_labeller/v3_tools.py predict --ckpt CKPT --name NAME [--set dev|renders|val|train5]
    $V experiments/sam3_labeller/v3_tools.py devscore --names A,B                     # whale on boat, dev sets -> V3/dev_scores.json

Label codes (ours): 0 water / sky, 1 whale, 2 person, 3 rail, 4 tape, 5 deck incl. bollard, 6 land, 254 NOT WHALE (any class
but whale; the loss pushes P(whale) down there and nothing else), 255 ignore.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import zlib
from multiprocessing import Pool
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
import lab_common as L  # noqa: E402

V3 = L.OUT / "v3"
NOT_WHALE = 254
FACTORY = L.NW / "Whale-Image-Data-Factory" / "data" / "renders"
# ADE20K (Mask2Former Swin-L semantic) ids: the "boat" and "vessel" groups of readymade_analyze.py
ADE_VESSEL = [76, 103]                                   # boat, ship
ADE_BOATPART = [38, 32, 95, 3, 93, 121, 53, 59, 42, 40]   # railing, fence, bannister, floor, pole, step, stairs, stairway, column, base
RENDER_MAP = np.full(256, 255, np.uint8)
RENDER_MAP[[0, 1, 2, 3, 4, 5, 6, 7, 8]] = [0, 1, 255, 3, 4, 5, 5, NOT_WHALE, 6]   # cleat -> deck; pontoon -> not whale; whale through water ignore
TRAIN5 = ["2025-01-17", "2025-02-12", "2025-03-18", "2025-01-02", "2025-03-06"]   # culled train days for the marker false-alarm check


def dev_list() -> list[tuple[str, str]]:
    return [tuple(x) for x in json.loads((V3 / "dev_real_boat.json").read_text())["photos"]]


def no_test(day: str) -> None:
    L.check_day(day)
    if day == L.TEST:
        raise RuntimeError("the test day is read only by the frozen part-C run (markers_day.py / predict on the day folder)")


# ------------------------------------------------------------------------------------------------- ADE maps
def cmd_ade(a) -> None:
    import torch
    import torch.nn.functional as F
    from PIL import Image
    sys.path.insert(0, str(L.REPO / "experiments" / "test_day"))
    from readymade import MODELS, OUT_LONG, prep
    from transformers import Mask2FormerForUniversalSegmentation
    items = {(r["day"], r["stem"]) for r in json.loads((L.OUT / "pseudo_v2" / "index.json").read_text())}
    items |= {(r["day"], r["stem"]) for r in json.loads((L.OUT / "real_s6mc" / "index.json").read_text())}
    items |= set(dev_list())
    todo = sorted(k for k in items if not (V3 / "ade" / k[0] / f"{k[1]}.png").exists())
    print(len(items), "photos;", len(todo), "to do", flush=True)
    hub, short, longmax = MODELS["ade"]; dev = torch.device("cuda")
    model = Mask2FormerForUniversalSegmentation.from_pretrained(hub).to(dev).eval()
    for n, (day, stem) in enumerate(todo):
        no_test(day)
        im = Image.open(L.photo_path(day, stem)).convert("RGB"); W, H = im.size; so = OUT_LONG / max(W, H); oh, ow = int(round(H * so)), int(round(W * so))
        x, (h, w) = prep(im, short, longmax)
        with torch.no_grad(), torch.autocast("cuda", dtype=torch.float16):
            out = model(pixel_values=x.to(dev))
            cls = out.class_queries_logits[0].float().softmax(-1)
            mk = F.interpolate(out.masks_queries_logits.float(), size=x.shape[-2:], mode="bilinear", align_corners=False)[0, :, :h, :w]
            seg = torch.einsum("qc,qhw->chw", cls[:, :-1], mk.sigmoid())
            seg = F.interpolate(seg[None], size=(oh, ow), mode="bilinear", align_corners=False)[0]
        o = V3 / "ade" / day / f"{stem}.png"; o.parent.mkdir(parents=True, exist_ok=True)
        Image.fromarray(seg.argmax(0).to(torch.uint8).cpu().numpy()).save(o)
        if n % 200 == 0:
            print(n, flush=True)
    print("done", flush=True)


def ade_regions(day: str, stem: str, hw) -> tuple[np.ndarray, np.ndarray] | None:
    f = V3 / "ade" / day / f"{stem}.png"
    if not f.exists():
        return None
    a = cv2.imread(str(f), cv2.IMREAD_UNCHANGED)
    if a.shape != tuple(hw):
        a = cv2.resize(a, (hw[1], hw[0]), interpolation=cv2.INTER_NEAREST)
    return np.isin(a, ADE_VESSEL), np.isin(a, ADE_VESSEL + ADE_BOATPART)


# ------------------------------------------------------------------------------------------------- v3 label maps
def v3_map(day: str, stem: str, lab: np.ndarray) -> tuple[np.ndarray, dict]:
    """Rule S2 of rules_scene_v3.md on a v2 pseudo map (or a real_s6mc map)."""
    import build_labels as B
    hw = lab.shape; p = L.s6_prob(day, stem)
    if p.shape != hw:
        p = cv2.resize(p, (hw[1], hw[0]), interpolation=cv2.INTER_LINEAR)
    s6h = L.hyst(p)
    keep_out = (p >= 0.1) | (cv2.dilate(s6h.astype(np.uint8), L.disk(8)) > 0)          # whale, its halo and anything S6 half-believes
    rec = B.raw_rec(day, stem)
    R = B.sam_regions(rec, B.pindex(day), hw) if rec is not None else None
    if R is not None:
        keep_out |= R["sam_whale"]
    ade = ade_regions(day, stem, hw)
    vessel, boatgrp = ade if ade is not None else (np.zeros(hw, bool), np.zeros(hw, bool))
    sure_boat = boatgrp | (R["boat"] if R is not None else False) | (R["rail"] if R is not None else False)
    out = lab.copy(); st = {"ade": ade is not None, "sam": R is not None}
    # S2a: not whale where a boat labeller is sure, outside the whale's reach, where the label was ignore
    nw = sure_boat & ~keep_out & (lab == L.IGNORE)
    out[nw] = NOT_WHALE
    # S2b: deck from ADE boat / ship touching the frame edge, minus rail / tape / person / land / whale reach
    if vessel.any():
        n, cc = cv2.connectedComponents(vessel.astype(np.uint8), connectivity=8)
        edge = np.unique(np.concatenate([cc[0], cc[-1], cc[:, 0], cc[:, -1]])); edge = edge[edge > 0]
        ours = np.isin(cc, edge)
        deck = ours & ~keep_out & np.isin(out, [0, L.IGNORE, NOT_WHALE])
        if R is not None:
            deck &= ~(R["rail"] | R["tape"] | R["person"] | R["land"] | R["rail_unsure"] | R["tape_unsure"] | R["person_unsure"] | R["gear"])
        deck = cv2.erode(deck.astype(np.uint8), L.disk(2)) > 0                          # 2-px margin to whatever is next to it
        band = (cv2.dilate(deck.astype(np.uint8), L.disk(2)) > 0) & ~deck & (out == 0)
        out[deck] = 5; out[band] = L.IGNORE
        st["deck_px"] = int(deck.sum())
    st["not_whale_px"] = int((out == NOT_WHALE).sum())
    return out, st


def labels_one(args):
    kind, day, stem, src, dst = args
    lab = cv2.imread(str(Path(src) / day / f"{stem}.png"), cv2.IMREAD_UNCHANGED)
    out, st = v3_map(day, stem, lab)
    o = Path(dst) / day; o.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(o / f"{stem}.png"), out)
    j = o / f"{stem}.jpg"
    if not j.exists():
        os.symlink(os.path.realpath(Path(src) / day / f"{stem}.jpg"), j)
    if kind == "real":
        return st
    rng = np.random.default_rng(zlib.crc32(stem.encode()))

    def pts(mask, n=40):
        ys, xs = np.nonzero(mask)
        if not len(xs):
            return []
        k = rng.choice(len(xs), min(n, len(xs)), replace=False)
        return [[int(xs[i]), int(ys[i])] for i in k]
    ex = np.isin(out, [2, 3, 4, 5, 6, NOT_WHALE])
    pc = {c: pts(out == v, 20) for c, v in [("person", 2), ("rail", 3), ("tape", 4), ("deck", 5), ("land", 6), ("not_whale", NOT_WHALE)] if (out == v).any()}
    return {"day": day, "stem": stem, "H": int(out.shape[0]), "W": int(out.shape[1]), "pts_extra": pts(ex), "pts_whale": pts(out == 1), "pts_cls": pc,
            "share": {c: float((out == v).mean()) for c, v in [("water_sky", 0), ("whale", 1), ("person", 2), ("rail", 3), ("tape", 4), ("deck", 5),
                                                                ("land", 6), ("not_whale", NOT_WHALE), ("ignore", 255)]}, "st": st}


def cmd_labels(a) -> None:
    held = set(dev_list())
    pi = [r for r in json.loads((L.OUT / "pseudo_v2" / "index.json").read_text()) if (r["day"], r["stem"]) not in held]
    ri = [r for r in json.loads((L.OUT / "real_s6mc" / "index.json").read_text()) if (r["day"], r["stem"]) not in held]
    with Pool(int(os.environ.get("NPROC", 10))) as P:
        idx = P.map(labels_one, [("pseudo", r["day"], r["stem"], str(L.OUT / "pseudo_v2"), str(V3 / "pseudo_v3")) for r in pi], chunksize=8)
        rst = P.map(labels_one, [("real", r["day"], r["stem"], str(L.OUT / "real_s6mc"), str(V3 / "real_s6mc_v3")) for r in ri], chunksize=8)
    (V3 / "pseudo_v3" / "index.json").write_text(json.dumps(idx)); (V3 / "real_s6mc_v3" / "index.json").write_text(json.dumps(ri))
    n = {c: sum(1 for r in idx if r["share"][c] > 0) for c in ["whale", "person", "rail", "tape", "deck", "land", "not_whale"]}
    s = {"pseudo_photos": len(idx), "held_out_dev": len(held), "photos_holding": n, "without_ade": sum(1 for r in idx if not r["st"]["ade"]),
         "deck_px_v2": None, "real_s6mc_photos": len(ri), "real_with_not_whale": sum(1 for x in rst if x["not_whale_px"] > 0),
         "real_with_deck": sum(1 for x in rst if x.get("deck_px", 0) > 0)}
    print(json.dumps(s, indent=1)); (V3 / "labels_summary.json").write_text(json.dumps(s, indent=1))


# ------------------------------------------------------------------------------------------------- renders
def render_one(args):
    run, rid, dst = args
    ds = FACTORY / run / "dataset"
    from PIL import Image
    img = cv2.imread(str(ds / "images" / f"{rid}.png")); fs = ds / "scene" / f"{rid}.png"
    if img is None or not fs.exists():
        return None
    sc = np.asarray(Image.open(fs))                       # palette PNG: the indices are the classes
    assert sc.ndim == 2, sc.shape
    small = L.resize_1280(img); lab = RENDER_MAP[cv2.resize(sc, (small.shape[1], small.shape[0]), interpolation=cv2.INTER_NEAREST)]
    o = Path(dst) / run; o.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(o / f"{rid}.jpg"), small, [cv2.IMWRITE_JPEG_QUALITY, 95]); cv2.imwrite(str(o / f"{rid}.png"), lab)
    rng = np.random.default_rng(zlib.crc32(f"{run}/{rid}".encode()))

    def pts(mask, n=20):
        ys, xs = np.nonzero(mask)
        if not len(xs):
            return []
        k = rng.choice(len(xs), min(n, len(xs)), replace=False)
        return [[int(xs[i]), int(ys[i])] for i in k]
    pc = {c: pts(lab == v) for c, v in [("rail", 3), ("tape", 4), ("deck", 5), ("land", 6), ("not_whale", NOT_WHALE)] if (lab == v).sum() >= 20}
    meta = json.loads((ds / "meta" / f"{rid}.json").read_text())
    return {"day": run, "stem": rid, "H": int(lab.shape[0]), "W": int(lab.shape[1]), "pts_extra": pts(np.isin(lab, [3, 4, 5, 6, NOT_WHALE]), 40),
            "pts_whale": pts(lab == 1, 40), "pts_cls": pc, "shot_type": meta.get("shot_type"), "marker": meta.get("marker")}


def cmd_renders(a) -> None:
    items = []
    for run in a.runs:
        ids = sorted(p.stem for p in (FACTORY / run / "dataset" / "scene").glob("*.png"))
        items += [(run, i) for i in ids]
    # hold-out: 20 % of each run by a hash of run / id (fixed, no look at content)
    hold = {k for k in items if zlib.crc32(f"{k[0]}/{k[1]}".encode()) % 5 == 0}
    out = {}
    with Pool(10) as P:
        for part in ("train", "hold"):
            sel = [k for k in items if (k in hold) == (part == "hold")]
            res = [r for r in P.map(render_one, [(r, i, str(V3 / "renders" / part)) for r, i in sel], chunksize=2) if r]
            (V3 / "renders" / part).mkdir(parents=True, exist_ok=True); (V3 / "renders" / part / "index.json").write_text(json.dumps(res))
            out[part] = {"n": len(res), "by_type": {t: sum(1 for r in res if r["shot_type"] == t) for t in {r["shot_type"] for r in res}}}
    print(json.dumps(out, indent=1)); (V3 / "renders" / "summary.json").write_text(json.dumps(out, indent=1))


# ------------------------------------------------------------------------------------------------- predictions
def pred_items(which: str) -> list[tuple[str, str, Path]]:
    if which == "dev":
        return [(d, s, L.photo_path(d, s)) for d, s in dev_list()]
    if which == "renders":
        idx = json.loads((V3 / "renders" / "hold" / "index.json").read_text())
        return [(r["day"], r["stem"], V3 / "renders" / "hold" / r["day"] / f"{r['stem']}.jpg") for r in idx]
    if which == "val":
        return [(d, s, L.photo_path(d, s)) for d in L.VAL for s in L.day_stems(d)]
    if which == "train5":
        return [(d, s, L.photo_path(d, s)) for d in TRAIN5 for s in L.day_stems(d)]
    raise ValueError(which)


def cmd_predict(a) -> None:
    import torch
    sys.path.insert(0, str(Path(__file__).parent))
    from predict_scene import to_map
    from model_fine_tuning.common.data import pad_to_multiple, to_tensor
    from model_fine_tuning.seg_multi.model import SceneSegModel
    items = []
    for w in a.set.split(","):
        items += pred_items(w)
    root = V3 / "pred" / a.name
    todo = [(d, s, p) for d, s, p in items if not (root / d / f"{s}.png").exists()]
    print(len(todo), "to do", flush=True)
    dev = torch.device("cuda"); model, _ = SceneSegModel.from_checkpoint(a.ckpt, dev)

    class DS(torch.utils.data.Dataset):
        def __len__(self): return len(todo)

        def __getitem__(self, j):
            return j, L.resize_1280(cv2.cvtColor(cv2.imread(str(todo[j][2])), cv2.COLOR_BGR2RGB))
    with torch.no_grad():
        for j, small in torch.utils.data.DataLoader(DS(), batch_size=None, num_workers=6):
            small = small.numpy(); padded, (h, w) = pad_to_multiple(small)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                lg = model(to_tensor(padded)[None].to(dev))
            prob = torch.softmax(lg.float(), 1)[0, :, :h, :w].cpu().numpy()
            d, s, _ = todo[int(j)]; o = root / d / f"{s}.png"; o.parent.mkdir(parents=True, exist_ok=True)
            cv2.imwrite(str(o), to_map(prob))
    print("done", flush=True)


def cmd_s6renders(a) -> None:
    """S6 P(whale) on the held-out renders (for the marker gate choice, S6.1) -> V3/s6prob_renders/<run>/<id>.png"""
    import torch
    from model_fine_tuning.common.data import pad_to_multiple, to_tensor
    from model_fine_tuning.segmentation.model import WhaleSegModel
    todo = [(d, s, p) for d, s, p in pred_items("renders") if not (V3 / "s6prob_renders" / d / f"{s}.png").exists()]
    dev = torch.device("cuda"); model, _ = WhaleSegModel.from_checkpoint(L.S6, dev)
    with torch.no_grad():
        for d, s, p in todo:
            small = L.resize_1280(cv2.cvtColor(cv2.imread(str(p)), cv2.COLOR_BGR2RGB)); padded, (h, w) = pad_to_multiple(small)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                pr = torch.sigmoid(model(to_tensor(padded)[None].to(dev)).float())[0, :h, :w].cpu().numpy()
            o = V3 / "s6prob_renders" / d / f"{s}.png"; o.parent.mkdir(parents=True, exist_ok=True)
            cv2.imwrite(str(o), np.round(pr * 255).astype(np.uint8))
    print(len(todo), "done", flush=True)


# ------------------------------------------------------------------------------------------------- dev scores
def cmd_devscore(a) -> None:
    import build_labels as B
    res = {}
    hold = json.loads((V3 / "renders" / "hold" / "index.json").read_text()) if (V3 / "renders" / "hold" / "index.json").exists() else []
    for name in a.names.split(","):
        r = {}
        # (i) real dev photos: whale px outside S6h (dilated 8) and on the boat region (ADE boat group | SAM 3 rail / deck / any-boat), not S6h
        rows = []
        for d, s in dev_list():
            f = V3 / "pred" / name / d / f"{s}.png"
            if not f.exists():
                continue
            m = cv2.imread(str(f), cv2.IMREAD_UNCHANGED); w = m == 1; hw = m.shape
            s6h = L.hyst(L.s6_prob(d, s)); near = cv2.dilate(s6h.astype(np.uint8), L.disk(8)) > 0
            rec = B.raw_rec(d, s); R = B.sam_regions(rec, B.pindex(d), hw) if rec is not None else None
            boat = ade_regions(d, s, hw)[1] if ade_regions(d, s, hw) is not None else np.zeros(hw, bool)
            if R is not None:
                boat = boat | R["rail"] | R["deck"] | R["boat"] | R["tape"]
            boat &= ~near
            rows.append({"day": d, "stem": s, "whale_out_s6h": int((w & ~near).sum()), "whale_on_boat": int((w & boat).sum()), "boat_px": int(boat.sum()),
                         "rail": float((m == 3).mean()), "deck": float((m == 5).mean())})
        r["real_dev"] = {"n": len(rows), "photos_whale_on_boat_ge50": sum(x["whale_on_boat"] >= 50 for x in rows),
                         "photos_whale_out_s6h_ge50": sum(x["whale_out_s6h"] >= 50 for x in rows),
                         "whale_on_boat_px_total": sum(x["whale_on_boat"] for x in rows), "boat_px_total": sum(x["boat_px"] for x in rows),
                         "whale_out_s6h_px_total": sum(x["whale_out_s6h"] for x in rows), "rows": rows}
        # (ii) held-out renders: per-class IoU, whale on the boat
        if hold:
            conf = np.zeros((7, 7), np.int64); wob = []; nw_shots = []
            for x in hold:
                f = V3 / "pred" / name / x["day"] / f"{x['stem']}.png"
                if not f.exists():
                    continue
                pm = cv2.imread(str(f), cv2.IMREAD_UNCHANGED); gt = cv2.imread(str(V3 / "renders" / "hold" / x["day"] / f"{x['stem']}.png"), cv2.IMREAD_UNCHANGED)
                ok = gt < 7
                conf += np.bincount(gt[ok].astype(np.int64) * 7 + pm[ok], minlength=49).reshape(7, 7)
                boat = np.isin(gt, [3, 4, 5, NOT_WHALE]); wob.append([int(((pm == 1) & boat).sum()), int(boat.sum())])
                if x["shot_type"] and x["shot_type"].startswith("marker"):
                    nw_shots.append(int((pm == 1).sum()))
            iou = {c: float(conf[k, k] / max(1, conf[k].sum() + conf[:, k].sum() - conf[k, k])) for k, c in enumerate(L.CLS)}
            rt = conf[np.ix_([3, 4], [3, 4])].sum(); rt_iou = float(rt / max(1, conf[[3, 4]].sum() + conf[:, [3, 4]].sum() - rt))
            r["renders_hold"] = {"n": len(wob), "iou": iou, "iou_rail_plus_tape": rt_iou, "whale_px_on_boat": int(sum(a_ for a_, _ in wob)),
                                 "boat_px": int(sum(b for _, b in wob)), "shots_whale_on_boat_ge50": sum(a_ >= 50 for a_, _ in wob),
                                 "marker_shots": len(nw_shots), "marker_shots_with_whale_ge50": sum(v >= 50 for v in nw_shots)}
        res[name] = r
        print(name, json.dumps({k: {kk: vv for kk, vv in v.items() if kk != "rows"} for k, v in r.items()}, indent=1))
    old = json.loads((V3 / "dev_scores.json").read_text()) if (V3 / "dev_scores.json").exists() else {}
    old.update(res); (V3 / "dev_scores.json").write_text(json.dumps(old, indent=1))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("cmd", choices=["ade", "labels", "renders", "predict", "devscore", "s6renders"])
    ap.add_argument("runs", nargs="*"); ap.add_argument("--ckpt", type=Path); ap.add_argument("--name"); ap.add_argument("--set", default="dev")
    ap.add_argument("--names", default="")
    a = ap.parse_args()
    {"ade": cmd_ade, "labels": cmd_labels, "renders": cmd_renders, "predict": cmd_predict, "devscore": cmd_devscore, "s6renders": cmd_s6renders}[a.cmd](a)


if __name__ == "__main__":
    main()
