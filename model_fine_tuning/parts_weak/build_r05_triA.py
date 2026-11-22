"""W3e's synthetic labels: the run05 part labels (parts_seg/labels.py cache, unchanged) with the dorsal class re-made by
rule A from the factory's additive layer run05/dataset/dorsal_triA/ (Whale-Image-Data-Factory factory/dorsal_triA.py:
rule A of experiments/parts_tri_expand/rule.json on the primary's three fin keypoints and its instance mask; proven
identical to tri_expand.py). Reads the factory folder, never writes it.

    PYTHONPATH=. .venv/bin/python -m model_fine_tuning.parts_weak.build_r05_triA [--workers 8]
        -> /var/tmp/sat3737-parts-cache/run05/labels_triA/<id>.png  (same palette / classes as run05/labels)
           /var/tmp/sat3737-parts-cache/run05/labels_triA_summary.json

Per shot, primary whale (instance 1; the calf is ignore in both caches):
  all three fin keypoints visible:  rule-A pixels that the old label calls whale (classes 1-5) -> dorsal (2);
                                    old dorsal pixels outside rule A -> the nearest non-dorsal whale class of the old
                                    label (in practice body); everything else unchanged
  otherwise (a fin point under water, under a wave, occluded, hidden, out of frame):
                                    old dorsal pixels -> ignore (255): rule A does not place the part there
Shots without whale pixels: copied.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from PIL import Image
from scipy import ndimage

from ..parts_seg.labels import IGNORE, PALETTE

FACTORY_R05 = Path("/home/sat3737/NOAA_whales/Whale-Image-Data-Factory/data/renders/run05/dataset")
OLD = Path("/var/tmp/sat3737-parts-cache/run05/labels")
NEW = Path("/var/tmp/sat3737-parts-cache/run05/labels_triA")
FIN = 2
FP = ("dorsal_fin_tip", "dorsal_fin_front", "dorsal_fin_back")


def save(a: np.ndarray, path: Path) -> None:
    im = Image.fromarray(a.astype(np.uint8), mode="P")
    pal = list(PALETTE) + [0] * (768 - len(PALETTE))
    im.putpalette(pal)
    im.save(path, optimize=True)


def one(sid: str) -> dict:
    old = np.asarray(Image.open(OLD / f"{sid}.png"))
    meta = json.loads((FACTORY_R05 / "meta" / f"{sid}.json").read_text())
    kp = meta.get("keypoints") or {}
    states = [(kp.get(n) or {}).get("state", "missing") for n in FP]
    applied = all(s == "visible" for s in states)
    new = old.copy()
    lp = FACTORY_R05 / "dorsal_triA" / f"{sid}.png"
    A = (np.asarray(Image.open(lp)) == 1) if lp.exists() else np.zeros(old.shape, bool)
    whale = (old >= 1) & (old <= 5)
    od = old == FIN
    r = {"id": sid, "states": states, "applied": applied, "old_px": int(od.sum()), "A_px": int(A.sum()),
         "A_not_whale_in_old": int((A & ~whale).sum())}
    if applied:
        drop = od & ~A
        if drop.any():
            valid = whale & ~od
            if valid.any():
                ys, xs = np.nonzero(drop | valid)
                y0, y1, x0, x1 = ys.min(), ys.max() + 1, xs.min(), xs.max() + 1
                _, (iy, ix) = ndimage.distance_transform_edt(~valid[y0:y1, x0:x1], return_indices=True)
                sub = old[y0:y1, x0:x1][iy, ix]
                d = drop[y0:y1, x0:x1]
                new[y0:y1, x0:x1][d] = sub[d]
            else:
                new[drop] = 5
            r["dropped_to"] = {str(int(v)): int(c) for v, c in zip(*np.unique(new[drop], return_counts=True))}
        put = A & whale
        new[put] = FIN
        r.update(new_px=int((new == FIN).sum()), both_px=int((A & od).sum()), dropped_px=int(drop.sum()), added_px=int((put & ~od).sum()))
    else:
        new[od] = IGNORE
        r.update(new_px=0, both_px=0, dropped_px=0, added_px=0, to_ignore_px=int(od.sum()))
    save(new, NEW / f"{sid}.png")
    return r


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--workers", type=int, default=8)
    a = ap.parse_args(argv)
    NEW.mkdir(parents=True, exist_ok=True)
    ids = sorted(p.stem for p in OLD.glob("*.png"))
    from multiprocessing import Pool
    with Pool(a.workers) as pool:
        rows = list(pool.imap(one, ids, chunksize=16))
    with_fin = [r for r in rows if r["old_px"] or r["A_px"]]
    s = {"shots": len(rows), "with_old_or_new_dorsal": len(with_fin),
         "applied": sum(r["applied"] for r in with_fin), "not_applied_with_old_dorsal": sum((not r["applied"]) and r["old_px"] > 0 for r in rows),
         "old_dorsal_px": sum(r["old_px"] for r in rows), "new_dorsal_px": sum(r["new_px"] for r in rows),
         "old_px_to_ignore": sum(r.get("to_ignore_px", 0) for r in rows), "old_px_dropped": sum(r["dropped_px"] for r in rows),
         "px_added": sum(r["added_px"] for r in rows), "A_px_not_whale_in_old": sum(r["A_not_whale_in_old"] for r in rows),
         "source": str(FACTORY_R05 / "dorsal_triA"), "old_cache": str(OLD)}
    (NEW.parent / "labels_triA_summary.json").write_text(json.dumps(s, indent=1))
    (NEW.parent / "labels_triA_rows.jsonl").write_text("\n".join(json.dumps(r) for r in rows) + "\n")
    print(json.dumps(s, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
