"""Apply the CSE head to real photos: per photo the template vertex of every whale pixel, its confidence, the whale's
facing side, the visible fraction of every body part, and a mirror-consistency check (the photo and its mirror
image must map to mirror vertices). Crops and masks are taken from an existing canon.predict output of the same
photos (the segmentation stays as it is).

    python -m model_fine_tuning.canon.predict_cse --checkpoint experiments/cse_r04_frozen/checkpoints/best.pt \
        --canon-dir experiments/real_2024-12-18/canon_canon_r04_ft4 --images experiments/real_2024-12-18/images \
        --out experiments/real_2024-12-18/cse_r04_frozen [--overlays 40] [--limit N]

Writes <stem>_cse.npz (vertex uint16 (0 = none), conf float16, side int8, mirror_p float16, x0y0s), rows.jsonl
(coverage per part, side vote, mirror consistency per photo), summary.json and overlays (part colours; left-side
parts darker). With --as-canon it also writes the canon.predict layout (<stem>_canon.npz with signed_mm / use /
side / disagree_mm / x0y0s, and points.jsonl with crop_x0y0s, rotation_direct, side, pixel count), so cluster/parts.py,
cluster/eval_id.py, outline_metric and fit_mesh consume the CSE map unchanged (signed points = the template vertex
positions; the mirror geodesic as the disagreement).
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np
import torch

from ..common import geometry
from .cse import Template, WhaleCSEModel, predict_vertices, side_vote

PART_COLOURS = {1: (70, 130, 255), 2: (60, 200, 220), 3: (60, 230, 90), 4: (250, 200, 60), 5: (250, 80, 80)}   # fluke, peduncle, dorsal_back, mid_body, head


def crop_from(image_path: Path, x0y0s, size: int) -> np.ndarray:
    img = cv2.cvtColor(cv2.imread(str(image_path)), cv2.COLOR_BGR2RGB)
    x0, y0, s = x0y0s
    M = np.array([[s, 0, -x0 * s], [0, s, -y0 * s]], np.float64)
    return cv2.warpAffine(img, M, (size, size), flags=cv2.INTER_AREA)


def to_tensor(img: np.ndarray, device) -> torch.Tensor:
    return torch.from_numpy(np.ascontiguousarray(img)).permute(2, 0, 1).float().div(255.0)[None].to(device)


@torch.no_grad()
def run_one(model: WhaleCSEModel, template: Template, crop: np.ndarray, use: np.ndarray, device) -> dict:
    x = to_tensor(crop, device); use_t = torch.from_numpy(use).to(device)
    with torch.autocast("cuda", dtype=torch.bfloat16, enabled=device.type == "cuda"):
        a = model(x); b = model(x.flip(-1))
    emb_a, emb_b = a["emb"][0].float(), b["emb"][0].float().flip(-1)
    R = a["R"][0].float().cpu().numpy()
    ids_a, conf_a, mp_a = predict_vertices(model, emb_a, use_t, template)
    R_b = b["R"][0].float().cpu().numpy()
    ids_b, conf_b, mp_b = predict_vertices(model, emb_b, use_t, template)
    side, ids_a_v = side_vote(ids_a, conf_a, template, R, mp_a)
    ids_b_m = template.mirror[ids_b]                                             # the mirror image shows the mirrored body point
    y, x = torch.nonzero(use_t, as_tuple=True)
    geo = template.geodesic_cm[ids_a, ids_b_m]
    out = {"side": side, "n": int(len(ids_a)), "mirror_geodesic_cm_median": float(geo.median()) if len(geo) else None,
           "mirror_part_agreement": float((template.part[ids_a] == template.part[ids_b_m]).float().mean()) if len(geo) else None,
           "mirror_side_agreement": float((template.side[ids_a] == template.side[ids_b_m]).float().mean()) if len(geo) else None,
           "conf_median": float(conf_a.median()) if len(conf_a) else None, "ambiguous_fraction": float((mp_a >= 0.3 * conf_a).float().mean()) if len(conf_a) else None}
    visible = torch.zeros(template.n, dtype=torch.bool, device=device); visible[ids_a_v] = True
    out["coverage"] = {name: float(visible[template.part == k + 1].float().mean()) for k, name in enumerate(template.part_names)}
    part_px = template.part[ids_a_v]
    out["part_mix"] = {name: float((part_px == k + 1).float().mean()) for k, name in enumerate(template.part_names)} if len(ids_a_v) else {}
    out["side_mix"] = {"left": float((template.side[ids_a_v] == -1).float().mean()), "right": float((template.side[ids_a_v] == 1).float().mean())} if len(ids_a_v) else {}
    vert = np.zeros(use.shape, np.uint16); vert[y.cpu().numpy(), x.cpu().numpy()] = ids_a_v.cpu().numpy().astype(np.uint16) + 1
    conf = np.zeros(use.shape, np.float16); conf[y.cpu().numpy(), x.cpu().numpy()] = conf_a.cpu().numpy()
    mirp = np.zeros(use.shape, np.float16); mirp[y.cpu().numpy(), x.cpu().numpy()] = mp_a.cpu().numpy()
    out["_R"] = R; out["_R_b"] = R_b; out["_log_ppm"] = float(a["log_ppm"][0])
    dis = np.zeros(use.shape, np.uint16); dis[y.cpu().numpy(), x.cpu().numpy()] = np.clip(geo.cpu().numpy() * 10.0, 0, 65535).astype(np.uint16)   # mm
    out["_disagree_mm"] = dis
    return out, vert, conf, mirp


def overlay(crop: np.ndarray, vert: np.ndarray, template: Template, label: str) -> np.ndarray:
    img = crop.copy()
    part = template.part.cpu().numpy(); side = template.side.cpu().numpy()
    lab = vert > 0
    ids = vert[lab].astype(np.int64) - 1
    col = np.array([PART_COLOURS[int(p)] for p in part[ids]], np.float32)
    col[side[ids] == -1] *= 0.55                                                 # left side darker
    img[lab] = (0.35 * img[lab] + 0.65 * col).astype(np.uint8)
    cv2.putText(img, label[:95], (6, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1, cv2.LINE_AA)
    return img


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--checkpoint", type=Path, required=True); p.add_argument("--canon-dir", type=Path, required=True)
    p.add_argument("--images", type=Path, required=True); p.add_argument("--out", type=Path, required=True)
    p.add_argument("--overlays", type=int, default=40); p.add_argument("--limit", type=int, default=0)
    p.add_argument("--as-canon", action="store_true", help="also write the canon.predict layout (<stem>_canon.npz, points.jsonl)")
    a = p.parse_args(argv)
    positions_cm = None
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    template = Template(device=device)
    model, ckpt = WhaleCSEModel.from_checkpoint(a.checkpoint, device)
    a.out.mkdir(parents=True, exist_ok=True); (a.out / "overlays").mkdir(exist_ok=True)
    files = sorted(a.canon_dir.glob("*_canon.npz"))
    if a.limit:
        files = files[: a.limit]
    rows = []
    positions_cm = template.positions_cm.cpu().numpy(); side_tab = template.side.cpu().numpy()
    points = open(a.out / "points.jsonl", "w") if a.as_canon else None
    with open(a.out / "rows.jsonl", "w") as handle:
        for i, npz in enumerate(files):
            stem = npz.name[: -len("_canon.npz")]
            z = np.load(npz); use = z["use"]
            if use.sum() < 50:
                continue
            crop = crop_from(a.images / f"{stem}.jpg", z["x0y0s"], use.shape[0])
            res, vert, conf, mirp = run_one(model, template, crop, use, device)
            res["stem"] = stem
            np.savez_compressed(a.out / f"{stem}_cse.npz", vertex=vert, conf=conf, mirror_p=mirp, side=np.int8(res["side"]), x0y0s=z["x0y0s"])
            if a.as_canon:
                lab = vert > 0; ids = vert[lab].astype(np.int64) - 1
                signed = np.zeros(use.shape + (3,), np.int16); signed[lab] = np.round(positions_cm[ids] * 10.0).astype(np.int16)   # mm, canonical frame
                side_map = np.zeros(use.shape, np.int8); side_map[lab] = side_tab[ids].astype(np.int8)
                np.savez_compressed(a.out / f"{stem}_canon.npz", signed_mm=signed, use=lab, side=side_map, disagree_mm=res.pop("_disagree_mm"), x0y0s=z["x0y0s"])
                x0, y0, s = [float(v) for v in z["x0y0s"]]
                rec = {"image": str(a.images / f"{stem}.jpg"), "whale_pixels": int(lab.sum()), "status": "ok", "crop_x0y0s": [x0, y0, s],
                       "flip_disagreement_median_m": (res["mirror_geodesic_cm_median"] or 0.0) / 100.0, "side_flip_agreement": res["mirror_side_agreement"],
                       "head_side": int(np.sign(geometry.initial_side_from_rotation(res["_R"]))), "prior_side": int(res["side"]),
                       "rotation_direct": np.round(res["_R"], 4).tolist(), "rot_direct_flip_consistency_deg": round(geometry.geodesic_deg(res["_R"], geometry.flip_rotation(res["_R_b"])), 1),
                       "ppm_direct": round(float(np.exp(res["_log_ppm"])), 2), "landmarks": {}}
                points.write(json.dumps(rec) + "\n")
            for k in ("_R", "_R_b", "_log_ppm", "_disagree_mm"):
                res.pop(k, None)
            if i < a.overlays:
                mix = res["part_mix"]; top = max(mix, key=mix.get) if mix else "?"
                cv2.imwrite(str(a.out / "overlays" / f"{stem}.jpg"), cv2.cvtColor(overlay(crop, vert, template, f"{stem[-8:]} {top} side {res['side']:+d} mirror-part {res['mirror_part_agreement']:.2f} geo {res['mirror_geodesic_cm_median']:.0f}cm"), cv2.COLOR_RGB2BGR), [cv2.IMWRITE_JPEG_QUALITY, 85])
            rows.append(res); handle.write(json.dumps(res) + "\n")
            if len(rows) % 50 == 0:
                print(f"{len(rows)}/{len(files)}", flush=True)
    if points:
        points.close()

    def med(key):
        v = [r[key] for r in rows if r.get(key) is not None]
        return round(float(np.median(v)), 4) if v else None
    summary = {"photos": len(rows), "checkpoint": str(a.checkpoint), "epoch": ckpt.get("epoch"),
               "mirror_geodesic_cm_median": med("mirror_geodesic_cm_median"), "mirror_part_agreement_median": med("mirror_part_agreement"),
               "mirror_side_agreement_median": med("mirror_side_agreement"), "conf_median": med("conf_median"), "ambiguous_fraction_median": med("ambiguous_fraction"),
               "side_votes": {"left": sum(r["side"] == -1 for r in rows), "right": sum(r["side"] == 1 for r in rows)},
               "part_mix_pixels": {name: round(float(np.mean([r["part_mix"].get(name, 0) for r in rows if r["part_mix"]])), 3) for name in template.part_names}}
    (a.out / "summary.json").write_text(json.dumps(summary, indent=1))
    print(json.dumps(summary, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
