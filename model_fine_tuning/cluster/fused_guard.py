"""Fused-instance guard B_G for the strict tier of safe_tracklets (adopted 2026-11-23).

Sometimes two whales sit in ONE detected instance "<stem>~k": parts.split_instances joins a fluke-like mask component
and the back of another animal far away (the u-join). A strict automatic link whose end frame is such an instance can
pass from one whale to the other. flag_G marks those instances; `safe_tracklets cluster --strict --guard-fused` makes
no strict automatic link whose end instances (tracklet A's last, tracklet B's first) include a flagged one. Instances,
keys and every file before the clustering stay as they are; only links are refused (a cluster can split in two).

flag_G (geometry, no model, CPU only): the instance is a union of >= 2 of the photo's mask components (the components
parts.split_instances considers: >= MIN_PX crop px and >= MIN_FRAC of the largest) and holds a pair (c_f, c_o) with
c_f fluke-like (median canonical body fraction u < U_FLUKE), c_o not (median u >= U_OTHER), and a gap (minimum pixel
distance) between them >= G_GAP x the larger of the two components' extents (max of bbox width / height, crop px).

Source of truth for the chain: THIS module. The rule and its parameters were frozen 2026-11-23 01:06 in the
fused-instance study (experiments/fused_instances: rules_fused.md, README.md; the frozen original of the code is
experiments/fused_instances/flags.py + fcommon.py, sha256 in moonshot_cache/fused_instances/FREEZE.txt). This file
repeats that logic line for line (flags.flag_G, fcommon.components / cse, the instance list of flags.arm_instances
without the study's split arms) so that the tracked chain does not import the experiments folder. Do not change the
rule or the parameters here without a new study; the adoption check showed identical flag sets on every day.

    python -m model_fine_tuning.cluster.fused_guard --day-dir experiments/days_all/DAY [--out F.json]   # the flags of one day
"""
from __future__ import annotations

import argparse
import json
import time
from functools import lru_cache
from multiprocessing import Pool
from pathlib import Path

import cv2
import numpy as np

G_GAP = 0.20          # gap between the fluke-like and the other component, in units of the larger one's extent
U_FLUKE = 0.15        # median u below this: fluke-like component
U_OTHER = 0.30        # median u at or above this: back / fin component
MIN_FRAC = 0.1        # components considered: >= MIN_FRAC of the photo's largest ...
MIN_PX = 400          # ... and >= MIN_PX crop px (as parts.split_instances)
RULE = "B_G"


@lru_cache(maxsize=1)
def _table() -> tuple[float, float]:
    """(x_min_m, length_m) of the folded canonical table (as geometry.load_folded_table gives them: Python floats)."""
    from ..common import paths
    with np.load(paths.FOLDED_TABLE) as z:
        return z["x_min_m"].item(), z["length_m"].item()


def cse(day_dir: Path, stem: str):
    """The photo's CSE map in the canon crop: use mask and body fraction u; None without a map."""
    f = day_dir / "cse" / f"{stem}_canon.npz"
    if not f.exists():
        return None
    z = np.load(f)
    use = z["use"]; signed = z["signed_mm"].astype(np.float32) / 1000.0
    x_min, length = _table(); u = (signed[..., 0] - x_min) / length
    return {"use": use, "u": u}


def components(use: np.ndarray, min_frac: float = MIN_FRAC, min_px: int = MIN_PX) -> list:
    """The components split_instances considers (largest first), as boolean masks."""
    n, lab, st, _ = cv2.connectedComponentsWithStats(use.astype(np.uint8), connectivity=8)
    if n <= 1:
        return []
    areas = st[1:, cv2.CC_STAT_AREA]; order = np.argsort(-areas)
    return [lab == (k + 1) for k in order if areas[k] >= max(min_px, min_frac * areas[order[0]])]


def _ext(c) -> int:
    ys, xs = np.nonzero(c)
    return max(np.ptp(xs), np.ptp(ys)) + 1


def flag_G(inst: np.ndarray, comps: list, u: np.ndarray) -> list:
    """-> the fluke-like components that are far from a non-fluke component of the same instance ([] = no flag)."""
    cs = [c for c in comps if (c & inst).any()]
    if len(cs) < 2:
        return []
    med = [float(np.median(u[c])) for c in cs]
    flk = [i for i, m in enumerate(med) if m < U_FLUKE]; oth = [i for i, m in enumerate(med) if m >= U_OTHER]
    out = []
    for i in flk:
        dt = cv2.distanceTransform((~cs[i]).astype(np.uint8), cv2.DIST_L2, 3)
        for j in oth:
            gap = float(dt[cs[j]].min())
            if gap >= G_GAP * max(_ext(cs[i]), _ext(cs[j])):
                out.append(cs[i]); break
    return out


def photo_flags(day_dir: Path, stem: str) -> dict:
    """{"<stem>~k": flag_G} for the photo's instances, numbered as parts.py --instances numbers them."""
    from .parts import split_instances
    z = cse(day_dir, stem)
    if z is None or z["use"].sum() < MIN_PX:
        return {}
    insts = split_instances(z["use"], MIN_FRAC, u=z["u"])
    if not insts:
        return {}
    comps = components(z["use"])
    return {f"{stem}~{k}": bool(flag_G(m, comps, z["u"])) for k, m in enumerate(insts)}


def _pf(a):
    return photo_flags(*a)


def stems(day_dir: Path) -> list[str]:
    return [Path(json.loads(l)["image"]).stem for l in open(day_dir / "images" / "index.jsonl")]


def day_flags(day_dir: Path, workers: int = 8) -> dict:
    """flag_G of every instance of the day: {"flags": {key: bool}, "photos", "seconds"}."""
    day_dir = Path(day_dir); t = time.time(); res = {}; S = stems(day_dir)
    _table()
    if workers > 1:
        with Pool(workers) as p:
            for r in p.imap(_pf, [(day_dir, s) for s in S], chunksize=8):
                res.update(r)
    else:
        for s in S:
            res.update(photo_flags(day_dir, s))
    return {"flags": res, "photos": len(S), "seconds": round(time.time() - t, 1)}


def flagged_keys(day_dir: Path, workers: int = 8) -> tuple[frozenset, dict]:
    """(the flagged instance keys, a summary for the clustering report)."""
    r = day_flags(day_dir, workers)
    fl = frozenset(k for k, v in r["flags"].items() if v)
    return fl, {"rule": RULE, "params": {"G_GAP": G_GAP, "U_FLUKE": U_FLUKE, "U_OTHER": U_OTHER, "MIN_PX": MIN_PX, "MIN_FRAC": MIN_FRAC},
                "instances": len(r["flags"]), "flagged": len(fl), "flagged_keys": sorted(fl), "photos": r["photos"], "seconds": r["seconds"]}


def guard_allow(tracks: list[dict], pairs: list[dict], flagged) -> np.ndarray:
    """allow[j] = False where candidate pair j's end instances (A's last key, B's first key) include a flagged one."""
    by = {t["id"]: t for t in tracks}
    return np.array([not ({by[p["a"]]["keys"][-1], by[p["b"]]["keys"][0]} & flagged) for p in pairs], bool)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--day-dir", type=Path, required=True); ap.add_argument("--out", type=Path, default=None); ap.add_argument("--workers", type=int, default=8)
    a = ap.parse_args(argv)
    r = day_flags(a.day_dir, a.workers)
    if a.out:
        a.out.parent.mkdir(parents=True, exist_ok=True); a.out.write_text(json.dumps(r))
    print(f"{a.day_dir.name}: {len(r['flags'])} instances, flag_G {sum(r['flags'].values())} ({r['seconds']} s, {r['photos']} photos)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
