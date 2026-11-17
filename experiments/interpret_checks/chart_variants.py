"""Would another way of turning / framing the part cut-outs give better fingerprints? Nothing of a day is changed.

    PYTHONPATH=. .venv/bin/python experiments/interpret_checks/chart_variants.py OUT_DIR DAY [DAY ...] [--workers 10]

For every whale instance whose cut-out the Labeling Assistant uses (the points of /api/interpret/<day>/fingerprints),
the dorsal fin, body and fluke cut-outs are made again from the RAW under each rule below with the production code
(cluster/parts.py extract_instance, 1536 px, jpg, then the 512 px the network takes), given to the same network
(pair_student_g) and thrown away: only the fingerprints and the transform of each cut-out are kept, in
OUT_DIR/emb_<day>.npz. No picture is written outside a temporary folder. chart_variants_score.py reads the result.

rules (pose_turn / frame_inside in cluster/parts.py):
  cur            the first rule: mirror by the flank that shows, long axis level, black outside the photo
  hd0 / hd30 / hd90   mirror by where the head points; level the axis never / up to 30 degrees / always
  sd0 / sd30 / sd90   mirror by the flank that shows; the same three
  +shift         the square moved the least that brings it inside the photo
  +fit           and made smaller when it is larger than the photo
"""
import argparse, json, math, os, shutil, sys, tempfile, time, urllib.request
from multiprocessing import get_context
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
DAYS = ROOT / "experiments" / "days_all"
CKPT = ROOT / "experiments" / "pair_student_g" / "checkpoints" / "best.pt"
SIZE, NET = 1536, 512
UP = lambda lv, mir, **fr: {"pose": "up", "level_max": float(lv), "mirror": mir, "frame": fr}
ALL3 = ("body", "fin", "fluke"); BF = ("body", "fin")
VARIANTS = {
    "cur": (None, ALL3),
    "hd0": (UP(0, "heading"), BF), "hd30": (UP(30, "heading"), BF), "hd90": (UP(90, "heading"), BF),
    "sd0": (UP(0, "side"), BF), "sd30": (UP(30, "side"), BF), "sd90": (UP(90, "side"), BF),
    "cur+shift": ({"pose": "axis", "frame": dict.fromkeys(ALL3, "shift")}, ALL3), "cur+fit": ({"pose": "axis", "frame": dict.fromkeys(ALL3, "fit")}, ALL3),
    "hd30+shift": (UP(30, "heading", **dict.fromkeys(BF, "shift")), BF), "hd30+fit": (UP(30, "heading", **dict.fromkeys(BF, "fit")), BF),
    "hd0+shift": (UP(0, "heading", **dict.fromkeys(BF, "shift")), BF), "hd0+fit": (UP(0, "heading", **dict.fromkeys(BF, "fit")), BF),
    "sd30+shift": (UP(30, "side", **dict.fromkeys(BF, "shift")), BF), "sd30+fit": (UP(30, "side", **dict.fromkeys(BF, "fit")), BF),
}
RULES = list(VARIANTS)                     # --rules narrows it ("cur" always first: the others are compared with it)


def work(job):
    rec, day, wanted, tmp, rules = job
    from PIL import Image
    from model_fine_tuning.common import geometry, paths
    from model_fine_tuning.cluster.chart import decode_full
    from model_fine_tuning.cluster.parts import extract_instance, split_instances
    cv2.setNumThreads(1)
    stem = Path(rec["image"]).stem; d = DAYS / day; out = []
    try:
        z = np.load(d / "cse" / f"{stem}_canon.npz"); x0, y0, s = z["x0y0s"]; use_all, side = z["use"], z["side"]
        signed = z["signed_mm"].astype(np.float32) / 1000.0
        if use_all.sum() < 400:
            return stem, out, None
        table = geometry.load_folded_table(paths.FOLDED_TABLE)
        full = decode_full(rec["source"]); H, W = full.shape[:2]
        scale_full = W / Image.open(d / "images" / f"{stem}.jpg").size[0]
        u = (signed[..., 0] - table["x_min_m"]) / table["length_m"]
        g = np.linspace(0, SIZE - 1, 48); xx, yy = np.meshgrid(g, g); G = np.stack([xx.ravel(), yy.ravel(), np.ones(xx.size)])
        for k, use in enumerate(split_instances(use_all, 0.1, u=u)):
            key = f"{stem}~{k}"
            if key not in wanted:
                continue
            seen = {}
            for name in rules:
                pose, parts = VARIANTS[name]
                o = Path(tmp) / f"{os.getpid()}"; (o / "charts").mkdir(parents=True, exist_ok=True)
                made = extract_instance(key, use, side, signed, u, float("nan"), full, x0, y0, s, scale_full, o, SIZE, 0.03, 0.3, parts, pose)
                for part in made:
                    zz = np.load(o / f"{key}__{part}.npz"); M = zz["M"].astype(float); mir = bool(zz["mirrored"]); fv = bool(zz["flipped_v"]) if "flipped_v" in zz else False
                    Pp = np.linalg.inv(np.vstack([M, [0, 0, 1]])) @ G
                    meta = dict(angle=float(np.degrees(np.arctan2(M[0, 1], M[0, 0]))), zoom=float(np.hypot(M[0, 0], M[0, 1])), mirrored=mir, flipped_v=fv,
                                outside=float(((Pp[0] < 0) | (Pp[0] > W) | (Pp[1] < 0) | (Pp[1] > H)).mean()))
                    sig = (part, mir, fv, tuple(np.round(M.ravel(), 3)))
                    if sig in seen:                                        # the same cut-out as an earlier rule: no need to embed it twice
                        out.append((key, part, name, seen[sig], None, meta)); continue
                    seen[sig] = name
                    im = cv2.cvtColor(cv2.imread(str(o / "charts" / f"{key}__{part}.jpg")), cv2.COLOR_BGR2RGB)
                    out.append((key, part, name, None, cv2.resize(im, (NET, NET), interpolation=cv2.INTER_AREA), meta))
                shutil.rmtree(o, ignore_errors=True)
        return stem, out, None
    except Exception as e:  # noqa: BLE001
        return stem, out, f"{type(e).__name__}: {e}"


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("out", type=Path); ap.add_argument("days", nargs="+"); ap.add_argument("--workers", type=int, default=10)
    ap.add_argument("--api", default="http://127.0.0.1:8765"); ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--rules", default=",".join(RULES))
    a = ap.parse_args(); a.out.mkdir(parents=True, exist_ok=True); rules = ["cur"] + [r for r in a.rules.split(",") if r in VARIANTS and r != "cur"]
    import torch
    from model_fine_tuning.cluster.train_id import PARTS as ID_PARTS, WhaleIdModel
    dev = torch.device("cuda"); model, ckpt = WhaleIdModel.from_checkpoint(CKPT, dev); assert int(ckpt["config"].get("args", {}).get("size", 448)) == NET

    def embed(part, ims):
        outp = []
        with torch.no_grad():
            for i in range(0, len(ims), 32):
                x = torch.from_numpy(np.stack(ims[i:i + 32]).astype(np.float32) / 255.0).permute(0, 3, 1, 2).to(dev)
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    e = model.embed(x, torch.full((len(x),), ID_PARTS.index(part), dtype=torch.long, device=dev)).float()
                outp.append(torch.nn.functional.normalize(e, dim=1).cpu().numpy())
        return np.concatenate(outp)

    for day in a.days:
        t0 = time.time(); pts = {}
        for part in ALL3:
            r = json.load(urllib.request.urlopen(f"{a.api}/api/interpret/{day}/fingerprints?part={part}&layout=pca", timeout=600))
            pts[part] = {p[0]: (int(p[1]), p[4] or "") for p in r["points"]}
        wanted = set().union(*[set(v) for v in pts.values()])
        recs = [json.loads(l) for l in open(DAYS / day / "images" / "index.jsonl")]
        recs = [r for r in recs if any(k.rsplit("~", 1)[0] == Path(r["image"]).stem for k in wanted)]
        if a.limit:
            recs = recs[:a.limit]
        tmp = tempfile.mkdtemp(prefix="chartvar_", dir=str(a.out)); rows = []; pend = {p: [] for p in ALL3}; E = {}; errs = 0
        with get_context("spawn").Pool(a.workers) as pool:              # spawn: a fork of a process that holds the GPU model hangs
            for n, (stem, out, err) in enumerate(pool.imap_unordered(work, [(r, day, wanted, tmp, rules) for r in recs], chunksize=1)):
                if err:
                    errs += 1; print("  error", stem[-30:], err[:120], flush=True)
                for key, part, name, same, im, meta in out:
                    rows.append(dict(key=key, part=part, variant=name, same=same, **meta))
                    if im is not None:
                        pend[part].append(((key, part, name), im))
                for part in ALL3:
                    if len(pend[part]) >= 256 or n == len(recs) - 1:
                        if pend[part]:
                            ee = embed(part, [im for _, im in pend[part]])
                            E.update({k: e for (k, _), e in zip(pend[part], ee)}); pend[part] = []
                if n % 50 == 49:
                    print(f"  {day}: {n + 1}/{len(recs)} photos, {len(E)} cut-outs embedded, {time.time() - t0:.0f} s", flush=True)
        for part in ALL3:
            if pend[part]:
                ee = embed(part, [im for _, im in pend[part]]); E.update({k: e for (k, _), e in zip(pend[part], ee)})
        shutil.rmtree(tmp, ignore_errors=True)
        emb = np.stack([E[(r["key"], r["part"], r["same"] or r["variant"])] for r in rows]).astype(np.float32)
        for r in rows:
            r["stack"], r["cat"] = pts[r["part"]].get(r["key"], (-1, ""))
            r["used"] = r["key"] in pts[r["part"]]                           # the Labeling Assistant counts this cut-out for this part
        f = a.out / f"emb_{day}.npz"; np.savez_compressed(f.with_suffix(".tmp.npz"), E=emb, rows=json.dumps(rows)); f.with_suffix(".tmp.npz").replace(f)
        print(f"{day}: {len(recs)} photos, {len(rows)} cut-out x rule rows, {len(E)} different cut-outs, {errs} errors, {time.time() - t0:.0f} s -> {f}", flush=True)


if __name__ == "__main__":
    main()
