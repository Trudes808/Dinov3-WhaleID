"""What a render set holds, in the real-photo taxonomy, from its label files only (no model): one line per visible whale
with its truth part pixels, the shot's pose category and its size in the 2560 frame.

    PYTHONPATH=. .venv/bin/python -m model_fine_tuning.part_gaps.coverage --dataset <renders>/<run>/dataset [...]

Writes /var/tmp/sat3737-parts-cache/part_gaps/coverage/<tag>.jsonl (resumable by re-running with --force).
Truth: dataset/parts (run06 datasets) or the parts_seg label cache (run05), per whale id of dataset/instances.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from PIL import Image

from . import common as C
from .render_eval import HUMP_STATS, animals_meta, dataset_tag, truth_maps


def scan(ds: Path, force: bool = False) -> Path:
    tag = dataset_tag(ds)
    out = C.CACHE / "coverage" / f"{tag}.jsonl"
    man = [json.loads(l) for l in open(ds / "manifest.jsonl") if l.strip()]
    side = out.with_suffix(".n")
    if out.exists() and not force and side.exists() and side.read_text().strip() == str(len(man)):
        return out                                   # cached and the set has not grown since
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w") as f:
        for m in man:
            if m.get("negative") or m.get("rejected") or not (ds / "instances" / f"{m['id']}.png").exists():
                continue
            sid = m["id"]
            meta = json.loads((ds / "meta" / f"{sid}.json").read_text())
            inst = np.asarray(Image.open(ds / "instances" / f"{sid}.png"))
            if inst.ndim == 3:
                inst = inst[..., 0]
            H, W = inst.shape
            try:
                parts, inst = truth_maps(ds, sid, H, W)
            except FileNotFoundError:
                continue
            am = animals_meta(meta)
            ids = [int(v) for v in np.unique(inst) if v > 0]
            if not meta.get("animals"):             # run04 / run05: id 2 is not a separate whale (calf / through-water): primary only
                ids = [i for i in ids if i == 1]
            for iid in ids:
                msk = inst == iid
                npx = int(msk.sum())
                if npx < 400 * (W / 2560.0) ** 2:
                    continue
                ys, xs = np.nonzero(msk)
                box = [int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1]
                rec = {"shot": sid, "iid": iid, "W": W, "H": H, "box": box, "L2560": max(box[2] - box[0], box[3] - box[1]) * 2560.0 / W,
                       "px": npx, "n_animals": len(ids), "touches_border": bool(box[0] <= 0 or box[1] <= 0 or box[2] >= W or box[3] >= H),
                       **{f"meta_{k}": v for k, v in (am.get(iid) or {}).items() if k in ("category", "role", "pose", "visible_fraction")},
                       "truth": np.bincount(parts[msk], minlength=256)[:6].astype(int).tolist(),
                       "fin_hump_blade_px": HUMP_STATS.get((sid, iid))}
                f.write(json.dumps(rec) + "\n")
    side.write_text(str(len(man)))
    return out


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dataset", type=Path, nargs="+", required=True)
    ap.add_argument("--force", action="store_true")
    a = ap.parse_args(argv)
    for ds in a.dataset:
        if not (ds / "manifest.jsonl").exists():
            print("missing", ds)
            continue
        print("wrote", scan(ds, a.force), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
