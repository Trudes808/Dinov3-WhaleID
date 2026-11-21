"""W3c in the live path, part 3 (2026-11-18 evening): what the Labeling Assistant page computes, end to end, on the
validation days, with five sets of inputs to its link score P and its step-2 ranking (arms fixed in
RESULTS/live_e2e/rule.json before any score of an arm other than L0 was read):

  L0  live: parts_up fin fingerprints (DAY/cluster_up), WHALE_VIEW_GATE = "blend" (P and S.A ungated)
  L1  view-class gate on P and S.A too (WHALE_VIEW_GATE = "all"); the ranking is L0's by construction
  L2  W3c gate: a fin fingerprint counts only where the W3c part model finds a dorsal region (seed 0; seed 1 repeat)
  L3  W3c centre: fin fingerprints of the "centre" cut-outs (centre.py, live network G), no gate
  L4  L3 + L2

    PYTHONPATH=. .venv/bin/python -m model_fine_tuning.cutouts_w3c.live_e2e missing            # which fin fingerprints have no centre embedding
    PYTHONPATH=. .venv/bin/python -m model_fine_tuning.cutouts_w3c.live_e2e make-missing       # their centre cut-outs (CPU) -> CACHE/<day>/<seed>x_centre
    flock /var/tmp/sat3737-parts-cache/gpu_train.lock env PYTHONPATH=. .venv/bin/python -m model_fine_tuning.cutouts_w3c.live_e2e embed-missing
    PYTHONPATH=. .venv/bin/python -m model_fine_tuning.cutouts_w3c.live_e2e repro              # the recorded numbers from the inputs they were made with
    PYTHONPATH=. .venv/bin/python -m model_fine_tuning.cutouts_w3c.live_e2e score [--arms L0]  # -> RESULTS/live_e2e/*.json

Nothing live is written: the page's backend (Whale-Review-Web/backend/labeling.py) and label_assist.py are imported
read-only; day folders are only read. Validation days only (day_list refuses the evaluation days).
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from multiprocessing import get_context
from pathlib import Path

import numpy as np

from . import CACHE, DAYS, RESULTS, day_list

VAL = ["2025-01-15", "2025-02-27", "2025-03-14"]
SEEDS = ("s0", "s1")
OUT = RESULTS / "live_e2e"
XCACHE = CACHE / "live_e2e"


def fin_up(day: str) -> dict:
    from ..cluster.safe_clusters import load_embeddings
    return load_embeddings(DAYS / day, "cluster_up").get("fin", {})


def regions(day: str, seed: str) -> dict:
    return json.loads((CACHE / "regions" / seed / f"{day}.json").read_text())


def centre_emb(day: str, seed: str) -> dict:
    """key -> G embedding of the "centre" fin cut-out (the earlier set, labelled dorsal views, plus the missing ones made here)."""
    out = {}
    z = np.load(CACHE / "emb" / "G" / f"{day}.npz")
    out.update(dict(zip([str(k) for k in z[f"fin|{seed}_centre|keys"]], z[f"fin|{seed}_centre|E"])))
    f = XCACHE / "emb_G" / f"{day}.npz"
    if f.exists():
        x = np.load(f)
        if f"fin|{seed}x_centre|keys" in x.files:
            out.update(dict(zip([str(k) for k in x[f"fin|{seed}x_centre|keys"]], x[f"fin|{seed}x_centre|E"])))
    return out


def missing(day: str, seed: str) -> tuple[list, list]:
    """(fin fingerprints with a W3c region and no centre embedding, those without a region: parts_up stays)"""
    fin = fin_up(day); R = regions(day, seed)
    z = np.load(CACHE / "emb" / "G" / f"{day}.npz"); have = {str(k) for k in z[f"fin|{seed}_centre|keys"]}
    miss = [k for k in sorted(fin) if k not in have]
    reg = [k for k in miss if (R.get(k) or {}).get("reg") is not None and (R.get(k) or {}).get("up_fin")]
    return reg, [k for k in miss if k not in set(reg)]


def cmd_missing(a):
    rep = {}
    for day in day_list(a.days):
        for seed in SEEDS:
            reg, none = missing(day, seed); fin = fin_up(day)
            rep[f"{day} {seed}"] = dict(fin_fingerprints=len(fin), no_centre_embedding=len(reg) + len(none), with_region=len(reg), without_region=len(none))
            print(day, seed, rep[f"{day} {seed}"])
    return rep


def _init(day, seed):
    from . import centre
    centre._init(day, seed)


def cmd_make_missing(a):
    from . import centre
    for day in day_list(a.days):
        src = {Path(json.loads(l)["image"]).stem: json.loads(l)["source"] for l in open(DAYS / day / "images" / "index.jsonl")}
        for seed in SEEDS:
            reg, _ = missing(day, seed); by = defaultdict(list)
            for k in reg:
                by[k.rsplit("~", 1)[0]].append(k)
            jobs = [(day, s, src[s], ks, f"{seed}x") for s, ks in sorted(by.items()) if s in src]
            rows, nerr = [], 0
            with get_context("spawn").Pool(min(a.workers, max(len(jobs), 1)), initializer=_init, initargs=(day, seed)) as pool:
                for stem, rr, err in pool.imap_unordered(centre.work, jobs, chunksize=1):
                    rows += rr
                    if err:
                        nerr += 1; print("  error", stem[-40:], err, flush=True)
            (XCACHE / day).mkdir(parents=True, exist_ok=True)
            (XCACHE / day / f"meta_{seed}x_cn.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))
            made = len(list((CACHE / day / f"{seed}x_centre").glob("*.jpg"))) if (CACHE / day / f"{seed}x_centre").exists() else 0
            print(f"{day} {seed}: missing with region {len(reg)}, jobs {len(jobs)}, rows {len(rows)}, centre jpgs {made}, errors {nerr}", flush=True)


def cmd_embed_missing(a):
    import torch
    from . import ROOT
    from ..cluster.train_id import PARTS, WhaleIdModel
    from ..cutouts_v3.embed import embed_files
    dev = torch.device("cuda"); model, _ = WhaleIdModel.from_checkpoint(ROOT / "experiments/pair_student_g/checkpoints/best.pt", dev)
    for day in day_list(a.days):
        res = {}
        for seed in SEEDS:
            files = sorted((CACHE / day / f"{seed}x_centre").glob("*.jpg"))
            E = embed_files(model, files, PARTS.index("fin"), dev)
            res[f"fin|{seed}x_centre|keys"] = np.array([f.stem for f in files]); res[f"fin|{seed}x_centre|E"] = E.astype(np.float32)
        # check: the same network on parts_up files of a few keys reproduces DAY/cluster_up
        fin = fin_up(day); ks = sorted(fin)[:24]
        E = embed_files(model, [DAYS / day / "parts_up" / "charts" / f"{k}__fin.jpg" for k in ks], PARTS.index("fin"), dev)
        cs = [float(np.dot(E[i], fin[k] / np.linalg.norm(fin[k]))) for i, k in enumerate(ks)]
        f = XCACHE / "emb_G" / f"{day}.npz"; f.parent.mkdir(parents=True, exist_ok=True); np.savez(f, **res)
        print(day, {k: len(v) for k, v in res.items() if k.endswith("|keys")}, "check cos min", round(min(cs), 4), flush=True)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("cmd", choices=["missing", "make-missing", "embed-missing", "repro", "score", "tables", "perday"])
    ap.add_argument("--days", default="val"); ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--arms", default=""); ap.add_argument("--boot", type=int, default=2000)
    a = ap.parse_args(argv)
    if a.cmd == "missing":
        cmd_missing(a)
    elif a.cmd == "make-missing":
        cmd_make_missing(a)
    elif a.cmd == "embed-missing":
        cmd_embed_missing(a)
    else:
        from . import live_e2e_score as S
        S.main_cmd(a)


if __name__ == "__main__":
    main()
