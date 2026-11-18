"""Keypoints + states on real whale instances (the part model's hr window, 512 px, mirror test-time augmentation).

    PYTHONPATH=. .venv/bin/python -m model_fine_tuning.pose_side.predict --head A/head.pt --out P.json --keys clicks:2025-01-02,clicks:2025-03-18
    --keys items: 'clicks:<day>' = every clicked or region-paired key of that click day; 'sheet:<day>[:n]' = n fluke + n
    fin cut-out keys of that day (cut-outs v3), deterministic sample.
Writes P.json {key: {name: [x, y]}} (jpg px, every point, amodal) and P_meta.json {key: {day, conf: {name: p},
state: {name: [p per STATES]}, window: [cx, cy, side]}}.
"""
from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import cv2
import numpy as np
import torch

from . import CUT, guard
from .data import CROP, FLIP, KP, STATES, _affine, load_jpg, normalise, real_window
from .model import Backbone, Head, decode
from .realdata import clicks, pairs


def key_list(spec):
    out = []
    for item in spec.split(","):
        kind, day, *rest = item.split(":"); guard([day])
        if kind == "clicks":
            ks = set(clicks(day))
            for p in pairs(day):
                ks.update([p["key_a"], p["key_b"]])
            out += [(day, k) for k in sorted(ks)]
        elif kind == "all":                          # every instance of the part model's json on that day
            from .data import _inst_json
            out += [(day, k) for k in sorted(_inst_json(day))]
        elif kind == "sheet":
            n = int(rest[0]) if rest else 6
            for part in ("fluke", "finB"):
                ks = sorted(f.stem for f in (CUT / day / part).glob("*.npz")); random.Random(day + part).shuffle(ks); out += [(day, k) for k in ks[:n]]
    return out


@torch.no_grad()
def run(head_path, items, dev, save_maps=None):
    bb = Backbone().to(dev); head = Head().to(dev); head.load_state_dict(torch.load(head_path, map_location=dev)); head.eval()
    P, META = {}, {}
    for day, key in items:
        w = real_window(day, key)
        if w is None:
            continue
        cx, cy, side = w; frame = load_jpg(day, key.rsplit("~", 1)[0])
        M = _affine(cx, cy, side, 0.0, False)
        crop = cv2.warpAffine(frame, M, (CROP, CROP), flags=cv2.INTER_AREA if side > CROP else cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)
        x = torch.stack([normalise(crop), normalise(crop[:, ::-1].copy())]).to(dev)
        hm, st = head(bb(x)); hm = hm.float(); st = st.float().softmax(-1)
        hm_f = hm[1].flip(-1)[FLIP]; hm = (hm[0] + hm_f)[None] / 2; stp = ((st[0] + st[1][FLIP]) / 2).cpu().numpy()
        pts, conf = decode(hm); pts = pts[0].cpu().numpy(); conf = conf[0].cpu().numpy()
        Mi = np.linalg.inv(np.vstack([M, [0, 0, 1]]))[:2]
        if save_maps is not None:                    # per-point probability maps (128 x 128, crop stride 4) for CPU decoding
            np.savez_compressed(save_maps / f"{key}.npz", prob=hm[0].flatten(1).softmax(-1).view_as(hm[0]).cpu().numpy().astype(np.float16), M=M, state=stp)
        P[key] = {n: [float(v) for v in Mi[:, :2] @ pts[k] + Mi[:, 2]] for k, n in enumerate(KP)}
        META[key] = dict(day=day, conf={n: float(conf[k]) for k, n in enumerate(KP)}, state={n: [round(float(v), 4) for v in stp[k]] for k, n in enumerate(KP)},
                         window=[float(cx), float(cy), float(side)])
    return P, META


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--head", type=Path, required=True); ap.add_argument("--out", type=Path, required=True); ap.add_argument("--keys", required=True)
    ap.add_argument("--save-maps", type=Path)
    a = ap.parse_args(); items = key_list(a.keys)
    if a.save_maps:
        a.save_maps.mkdir(parents=True, exist_ok=True)
    P, META = run(a.head, items, torch.device("cuda"), a.save_maps)
    a.out.write_text(json.dumps(P)); a.out.with_name(a.out.stem + "_meta.json").write_text(json.dumps(dict(states=STATES, rows=META)))
    print(f"{len(P)} instances -> {a.out}")


if __name__ == "__main__":
    main()
