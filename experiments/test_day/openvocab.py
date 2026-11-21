"""Open-vocabulary naming on a day (photo level: CLIP; box level: OWLv2; instance level: CLIP on the pipeline's crops).
Prompts, models and rules are fixed in prompts.json (written before any output on the test day was looked at).

    flock /var/tmp/sat3737-parts-cache/gpu_train.lock .venv/bin/python experiments/test_day/openvocab.py --day-dir DAY --out ROOT/openvocab.json

Public weights from the Hugging Face hub (downloaded once); the photos never leave the machine.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from PIL import Image

HERE = Path(__file__).parent


def main() -> None:
    p = argparse.ArgumentParser(); p.add_argument("--day-dir", type=Path, required=True); p.add_argument("--out", type=Path, required=True)
    p.add_argument("--keep-score", type=float, default=0.05); a = p.parse_args()
    cfg = json.loads((HERE / "prompts.json").read_text()); dev = "cuda"
    from transformers import CLIPModel, CLIPProcessor, Owlv2ForObjectDetection, Owlv2Processor
    rows = [json.loads(l) for l in open(a.day_dir / "images" / "index.jsonl")]
    stems = [Path(r["image"]).stem for r in rows]
    P = json.loads((a.day_dir / "safe" / "photos.json").read_text())
    clip = CLIPModel.from_pretrained(cfg["clip_model"], torch_dtype=torch.float16).to(dev).eval(); cp = CLIPProcessor.from_pretrained(cfg["clip_model"])
    classes = list(cfg["photo_classes"])
    with torch.no_grad():
        T = []
        for c in classes:
            t = cp(text=cfg["photo_classes"][c], return_tensors="pt", padding=True).to(dev)
            e = clip.text_projection(clip.text_model(input_ids=t['input_ids'], attention_mask=t['attention_mask']).pooler_output); e = e / e.norm(dim=-1, keepdim=True); m = e.mean(0); T.append(m / m.norm())
        T = torch.stack(T); scale = clip.logit_scale.exp().float()

        def clip_probs(ims):
            out = []
            for k in range(0, len(ims), 32):
                x = cp(images=ims[k:k + 32], return_tensors="pt")["pixel_values"].to(dev, torch.float16)
                f = clip.visual_projection(clip.vision_model(pixel_values=x).pooler_output); f = f / f.norm(dim=-1, keepdim=True)
                out.append((scale * f.float() @ T.float().T).softmax(-1).cpu().numpy())
            return np.concatenate(out) if out else np.zeros((0, len(classes)))

        res = {"classes": classes, "photos": {}, "instances": {}}
        ims = {}
        for s in stems:
            ims[s] = Image.open(a.day_dir / "images" / f"{s}.jpg").convert("RGB")
        pp = clip_probs([ims[s] for s in stems])
        for s, v in zip(stems, pp):
            res["photos"][s] = {"clip": [round(float(x), 4) for x in v]}
        # instances: the not-whale filter's square crop (box x 1.15, at least 64 px)
        keys, crops = [], []
        for s, ph in P.items():
            for it in ph["instances"]:
                x0, y0, x1, y1 = it["bbox"]; cx, cy = (x0 + x1) / 2, (y0 + y1) / 2; side = max(max(x1 - x0, y1 - y0) * 1.15, 64.0)
                crops.append(ims[s].crop((int(cx - side / 2), int(cy - side / 2), int(cx + side / 2), int(cy + side / 2)))); keys.append(it["key"])
        for k, v in zip(keys, clip_probs(crops)):
            res["instances"][k] = {"clip": [round(float(x), 4) for x in v]}
        del clip; torch.cuda.empty_cache()
        owl = Owlv2ForObjectDetection.from_pretrained(cfg["owl_model"], torch_dtype=torch.float16).to(dev).eval(); op = Owlv2Processor.from_pretrained(cfg["owl_model"])
        qs, qc = [], []
        for c, l in cfg["box_queries"].items():
            qs += l; qc += [c] * len(l)
        res["owl_queries"] = qs; res["owl_classes"] = qc
        for s in stems:
            im = ims[s]; inp = op(text=[qs], images=im, return_tensors="pt").to(dev)
            inp["pixel_values"] = inp["pixel_values"].half()
            o = owl(**inp)
            # OWLv2 pads to a square: boxes are relative to the padded square of side max(W, H)
            W, H = im.size; side = max(W, H)
            logits = o.logits[0].float().sigmoid().cpu().numpy(); boxes = o.pred_boxes[0].float().cpu().numpy()
            det = []
            for j in np.argwhere(logits >= a.keep_score):
                b, q = int(j[0]), int(j[1]); cx, cy, w, h = boxes[b] * side
                det.append([qc[q], qs[q], round(float(logits[b, q]), 4), [round(float(cx - w / 2), 1), round(float(cy - h / 2), 1), round(float(cx + w / 2), 1), round(float(cy + h / 2), 1)]])
            best = {c: 0.0 for c in cfg["box_queries"]}
            for q in range(len(qs)):
                best[qc[q]] = max(best[qc[q]], float(logits[:, q].max()))
            res["photos"][s]["owl_best"] = {c: round(v, 4) for c, v in best.items()}; res["photos"][s]["owl_det"] = det
    a.out.write_text(json.dumps(res)); print(len(stems), "photos,", len(keys), "instances ->", a.out)


if __name__ == "__main__":
    main()
