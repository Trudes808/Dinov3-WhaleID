"""SAM 3 (transformers Sam3Model, text-prompted concept segmentation, image mode) on a list of photos; raw outputs kept so
every threshold and rule can be applied later without a second GPU pass.

Runs with the Underwater-WhaleID venv's interpreter, used read-only (transformers 5.x there has Sam3Model); weights from
the local HF cache only (offline, nothing is sent anywhere).

    flock /var/tmp/sat3737-parts-cache/gpu_train.lock \
      /home/sat3737/NOAA_whales/Underwater-WhaleID/.venv/bin/python experiments/sam3_compare/run_sam3.py \
      --list LIST.tsv --prompts experiments/sam3_compare/prompts_all.json --set NAME --out OUTROOT [--budget 1080]

LIST.tsv: id <tab> image path. Output OUTROOT/NAME/<id>.npz, resumable (done ids are skipped); per prompt index j:
  s{j} scores (float32, every query with score >= --min-score, at most --max-inst), b{j} boxes xyxy in input pixels,
  m{j} mask probabilities uint8 (sigmoid * 255) at the model's mask grid (288 x 288 over the whole photo, squashed like
  the input), p{j} presence score, g{j} semantic map uint8 (sigmoid * 255) at the mask grid; plus hw (input size).
  OUTROOT/NAME/_timing.jsonl: per photo the vision time, the time for all prompts, peak GPU memory.
Score = sigmoid(class logit) * sigmoid(presence logit), as post_process_instance_segmentation computes it.
"""
from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path

os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")

import numpy as np
import torch
from PIL import Image


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--list", type=Path, required=True); ap.add_argument("--prompts", type=Path, required=True)
    ap.add_argument("--set", required=True); ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--min-score", type=float, default=0.1); ap.add_argument("--max-inst", type=int, default=30)
    ap.add_argument("--budget", type=float, default=1080.0, help="seconds; stop cleanly after this (lock holds <= ~20 min)")
    ap.add_argument("--dtype", default="float16"); ap.add_argument("--limit", type=int, default=0)
    a = ap.parse_args()
    prompts = json.loads(a.prompts.read_text())["prompts"]
    out = a.out / a.set; out.mkdir(parents=True, exist_ok=True)
    (out / "_prompts.json").write_text(json.dumps(prompts))
    items = [l.rstrip("\n").split("\t") for l in open(a.list) if l.strip()]
    todo = [(i, p) for i, p in items if not (out / f"{i}.npz").exists()]
    if a.limit:
        todo = todo[: a.limit]
    print(f"{len(items)} listed, {len(todo)} to do", flush=True)
    if not todo:
        (out / "_DONE").touch(); return
    t0 = time.time()
    from transformers import Sam3Model, Sam3Processor
    dev = torch.device("cuda"); dt = getattr(torch, a.dtype)
    proc = Sam3Processor.from_pretrained("facebook/sam3")
    model = Sam3Model.from_pretrained("facebook/sam3", torch_dtype=dt).to(dev).eval()
    text = []
    with torch.no_grad():
        for q in prompts:
            t = proc(text=q, return_tensors="pt")
            text.append({k: v.to(dev) for k, v in t.items() if k in ("input_ids", "attention_mask")})
    print(f"model loaded in {time.time() - t0:.1f} s", flush=True)
    tl = open(out / "_timing.jsonl", "a")
    done = 0
    for i, path in todo:
        if time.time() - t0 > a.budget:
            print(f"budget reached after {done} photos", flush=True); break
        im = Image.open(path).convert("RGB"); W, H = im.size
        torch.cuda.reset_peak_memory_stats(); torch.cuda.synchronize(); ta = time.time()
        rec = {"hw": np.array([H, W], np.int32)}
        with torch.no_grad():
            pix = proc(images=im, return_tensors="pt")["pixel_values"].to(dev, dt)
            ve = model.get_vision_features(pixel_values=pix); torch.cuda.synchronize(); tb = time.time()
            for j, tx in enumerate(text):
                o = model(vision_embeds=ve, **tx)
                lg = o.pred_logits[0].float().sigmoid()
                pr = o.presence_logits[0].float().sigmoid().reshape(-1)[0] if o.presence_logits is not None else torch.tensor(1.0)
                sc = lg * pr
                keep = torch.nonzero(sc >= a.min_score).flatten()
                keep = keep[torch.argsort(sc[keep], descending=True)][: a.max_inst]
                bx = o.pred_boxes[0].float()[keep] * torch.tensor([W, H, W, H], device=dev, dtype=torch.float32)
                mk = (o.pred_masks[0].float()[keep].sigmoid() * 255).round().to(torch.uint8)
                rec[f"s{j}"] = sc[keep].cpu().numpy().astype(np.float32); rec[f"b{j}"] = bx.cpu().numpy().astype(np.float32)
                rec[f"m{j}"] = mk.cpu().numpy(); rec[f"p{j}"] = np.float32(pr.item())
                if getattr(o, "semantic_seg", None) is not None:
                    rec[f"g{j}"] = (o.semantic_seg[0, 0].float().sigmoid() * 255).round().to(torch.uint8).cpu().numpy()
            torch.cuda.synchronize(); tc = time.time()
        np.savez_compressed(out / f"{i}.npz", **rec)
        tl.write(json.dumps({"id": i, "vision_s": round(tb - ta, 4), "prompts_s": round(tc - tb, 4), "n_prompts": len(text),
                             "peak_mem_mb": round(torch.cuda.max_memory_allocated() / 2**20, 1), "hw": [H, W]}) + "\n"); tl.flush()
        done += 1
    left = sum(1 for i, _ in items if not (out / f"{i}.npz").exists())
    print(f"done {done} this hold, {left} left, {time.time() - t0:.0f} s", flush=True)
    if left == 0:
        (out / "_DONE").touch()


if __name__ == "__main__":
    main()
