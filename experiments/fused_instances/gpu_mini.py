"""GPU steps of a split arm only for the photos the split changed (saves the shared GPU): a mini day folder holding those
photos, then the unchanged GPU commands on it (cluster_day channels with the student, link_scores instances), then a
merge with the source root's outputs for every other photo (embeddings and instance crops are computed per chart /
per instance box, so an unchanged photo's rows are the source's rows). The merged cluster_v3/channels.npz carries the
key list only (the pairwise matrices are not read by the strict chain).

    PYTHONPATH=. .venv/bin/python experiments/fused_instances/gpu_mini.py prep  --arm ARM_DAYS --day DAY
    (GPU)  cluster_day channels / link_scores instances on ARM_DAYS/_mini/DAY      (arm_chain.sh mini)
    PYTHONPATH=. .venv/bin/python experiments/fused_instances/gpu_mini.py merge --arm ARM_DAYS --day DAY [--src SHADOW_DAYS]
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import fcommon as F  # noqa: E402


PAD = 3   # cluster_day embeds a part only when the day has >= 3 charts of it: pad with unchanged photos (their rows are
          # compared with the source's in merge, a determinism check, and then dropped)


def pad_stems(D: Path, st: set) -> list:
    need = {"body": PAD, "fin": PAD, "fluke": PAD}; pad = []
    for l in open(D / "parts_hr" / "index.jsonl"):
        r = json.loads(l)
        if r["stem"] in st or not r.get("instances"):
            continue
        parts = {p for i in r["instances"] for p in i["parts"]}
        if any(need[p] > 0 for p in parts):
            pad.append(r["stem"])
            for i in r["instances"]:
                for p in i["parts"]:
                    need[p] -= 1
        if all(v <= 0 for v in need.values()):
            break
    return pad


def prep(arm: Path, day: str) -> None:
    D = arm / day; M = arm / "_mini" / day
    st = set(json.loads((D / "split_info.json").read_text())["split_stems"])
    pad = pad_stems(D, st); (arm / "_mini").mkdir(exist_ok=True); (arm / "_mini" / f"{day}.pad.json").write_text(json.dumps(pad))
    st = st | set(pad)
    (M / "parts_hr" / "charts").mkdir(parents=True, exist_ok=True); (M / "safe").mkdir(exist_ok=True); (M / "images").mkdir(exist_ok=True)
    ts = json.loads((D / "timestamps.json").read_text())
    (M / "timestamps.json").write_text(json.dumps({s: v for s, v in ts.items() if s in st}))
    with open(M / "images" / "index.jsonl", "w") as h:
        for l in open(D / "images" / "index.jsonl"):
            if Path(json.loads(l)["image"]).stem in st:
                h.write(l)
    for s in st:
        p = D / "images" / f"{s}.jpg"
        if not (M / "images" / p.name).exists():
            os.symlink(os.path.realpath(p), M / "images" / p.name)
    with open(M / "parts_hr" / "index.jsonl", "w") as h:
        for l in open(D / "parts_hr" / "index.jsonl"):
            if json.loads(l)["stem"] in st:
                h.write(l)
    for f in (D / "parts_hr").glob("*.npz"):
        if f.name.split("~")[0] in st and not (M / "parts_hr" / f.name).exists():
            os.symlink(os.path.realpath(f), M / "parts_hr" / f.name)
    for f in (D / "parts_hr" / "charts").iterdir():
        if f.name.split("~")[0] in st and not (M / "parts_hr" / "charts" / f.name).exists():
            os.symlink(os.path.realpath(f), M / "parts_hr" / "charts" / f.name)
    P = json.loads((D / "safe" / "photos.json").read_text())
    (M / "safe" / "photos.json").write_text(json.dumps({s: v for s, v in P.items() if s in st}))
    print("mini", M, len(st), "photos")


def merge(arm: Path, day: str, src: Path) -> None:
    D = arm / day; M = arm / "_mini" / day; S = src / day
    st = set(json.loads((D / "split_info.json").read_text())["split_stems"])
    pad = set(json.loads((arm / "_mini" / f"{day}.pad.json").read_text())) if st else set()
    keep = lambda k: k.rsplit("~", 1)[0] not in st
    mine = lambda k: k.rsplit("~", 1)[0] in st
    # cluster_v3
    ks = [str(k) for k in np.load(S / "cluster_v3" / "channels.npz", allow_pickle=True)["keys"]]
    km = [str(k) for k in np.load(M / "cluster_v3" / "channels.npz", allow_pickle=True)["keys"]] if st else []
    keys = [k for k in ks if keep(k)] + [k for k in km if mine(k)]; pos = {k: i for i, k in enumerate(keys)}
    zs = np.load(S / "cluster_v3" / "embeddings.npz", allow_pickle=True)
    zm = np.load(M / "cluster_v3" / "embeddings.npz", allow_pickle=True) if st else None
    out = {}
    for name in zs.files:
        if not name.endswith("|idx"):
            continue
        base = name[:-4]
        idx, E = zs[name], zs[base + "|E"]
        rows = [(pos[ks[int(i)]], E[j]) for j, i in enumerate(idx) if keep(ks[int(i)])]
        if zm is not None and name in zm.files:
            rows += [(pos[km[int(i)]], zm[base + "|E"][j]) for j, i in enumerate(zm[name]) if mine(km[int(i)])]
            src = {ks[int(i)]: E[j] for j, i in enumerate(idx)}
            d = [float(np.abs(zm[base + "|E"][j].astype(np.float32) - src[km[int(i)]].astype(np.float32)).max()) for j, i in enumerate(zm[name]) if km[int(i)] in src and km[int(i)].rsplit("~", 1)[0] in pad]
            print(f"  pad check {base}: {len(d)} rows, max |diff| {max(d) if d else None}")
        out[name] = np.array([r[0] for r in rows], np.int64); out[base + "|E"] = np.stack([r[1] for r in rows]).astype(zs[base + "|E"].dtype)
    (D / "cluster_v3").mkdir(exist_ok=True)
    np.savez_compressed(D / "cluster_v3" / "channels.npz", keys=np.array(keys))
    np.savez_compressed(D / "cluster_v3" / "embeddings.npz", **out)
    # link_scores instances
    a = np.load(S / "safe" / "inst_feat.npz")
    sel = np.array([keep(str(k)) for k in a["keys"]], bool)
    parts = {n: [a[n][sel]] for n in ("keys", "cls", "mean", "box")}
    if st:
        b = np.load(M / "safe" / "inst_feat.npz")
        sb = np.array([mine(str(k)) for k in b["keys"]], bool)
        src = {str(k): i for i, k in enumerate(a["keys"])}
        d = [float(np.abs(b["cls"][j].astype(np.float32) - a["cls"][src[str(k)]].astype(np.float32)).max()) for j, k in enumerate(b["keys"]) if str(k) in src and str(k).rsplit("~", 1)[0] in pad]
        print(f"  pad check inst_feat: {len(d)} rows, max |diff| {max(d) if d else None}")
        for n in parts:
            parts[n].append(b[n][sb])
    np.savez(D / "safe" / "inst_feat.npz", **{n: np.concatenate(v) for n, v in parts.items()})
    print("merged", D, len(keys), "keys;", int(sum(len(v) for v in parts["keys"])), "instance crops")


def main() -> None:
    ap = argparse.ArgumentParser(); ap.add_argument("cmd", choices=["prep", "merge"]); ap.add_argument("--arm", type=Path, required=True)
    ap.add_argument("--day", required=True); ap.add_argument("--src", type=Path, default=F.SR)
    a = ap.parse_args(); F.check_day(a.day)
    (prep if a.cmd == "prep" else lambda r, d: merge(r, d, a.src))(a.arm, a.day)


if __name__ == "__main__":
    main()
