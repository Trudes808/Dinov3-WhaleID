"""Switch prep (2026-11-20): is a shadow day folder complete? Compares each shadow day's file list with the live one.

Per day and per sub-folder: files and bytes live / shadow, after collapsing per-photo and per-instance names to patterns
("<stem>_canon.npz", "<stem>~k.npz", "<n>__L.png"), so the comparison says which KINDS of file one side has and the other
lacks (photo names never appear in the output). Folders the app does not read (research leftovers in the live folders,
listed in RESEARCH) are reported apart. Also the instance-level sanity numbers per day (photos, instances, kept
instances, photos without a kept instance, clusters).

    .venv/bin/python experiments/test_day/shadow_compare.py --shadow SHADOW_DAYS [--live experiments/days_all] [--days D,..] --out OUT.json
"""
from __future__ import annotations

import argparse
import json
import os
import re
from collections import defaultdict
from pathlib import Path

RESEARCH = ("instances_v2", "parts_seg_r05_a", "parts_sil", "part_landmarks_parts_v2", "part_landmarks_parts_v2r", "part_landmarks_parts_v2u",
            "signature", "signature_local_v1", "signature_local_v2", "signature_local_v3", "crops", "cluster_bursts_id_v2", "cluster_bursts_id_v3",
            "cluster_bursts_md", "cluster_bursts_whole", "dino_hr.log", "cluster_bursts_id_v2.log", "cluster_bursts_id_v3.log", "cluster_bursts_md.log",
            "cluster_bursts_whole.log", "instances_v2.log", "sketch/v1", "sketch/v2")


def pattern(rel: str, stems: list[str]) -> str:
    parts = rel.split("/"); name = parts[-1]
    for s in stems:
        if name.startswith(s):
            name = "<stem>" + name[len(s):]; break
    name = re.sub(r"~\d+", "~k", name); name = re.sub(r"^\d+__([LR])", r"<n>__\1", name); name = re.sub(r"^[LR]_\d+(-\d+)*", "<anim>", name)
    return "/".join(parts[:-1] + [name])


def listing(d: Path, stems: list[str]) -> dict:
    out = defaultdict(lambda: [0, 0])
    for root, dirs, files in os.walk(d, followlinks=False):
        rp = os.path.relpath(root, d)
        if rp == "images" or rp.startswith("images/"):
            dirs[:] = []; continue
        for f in files:
            rel = f if rp == "." else f"{rp}/{f}"; p = Path(root) / f
            k = pattern(rel, stems); out[k][0] += 1; out[k][1] += p.lstat().st_size
        for x in list(dirs):
            if (Path(root) / x).is_symlink():
                k = pattern(x if rp == "." else f"{rp}/{x}", stems) + "@"; out[k][0] += 1; dirs.remove(x)
    return dict(out)


# files the app does not read either (the backend's file names were grepped on 2026-11-20): training caches, the old
# cluster ids, tuning runs of the cluster_v3 study, older fin-model files and research folders of single days
RESEARCH_RE = re.compile(r"^(parts_hr/<stem>~k__body\.canon48\.npy|safe/clusters_v1\.json|safe/view_match_hw\.npz|part_landmarks_parts_v1(/.*)?"
                         r"|dino_frozen[^/]*(/.*)?|edges(/.*)?|signature_[^/]*(/.*)?"
                         r"|cluster_v3/(?!(channels|embeddings)\.npz$|(scores|metrics|tracklets)_student\.(npz|json)$).*)$")


def research(k: str) -> bool:
    return any(k == r or k.startswith(r + "/") for r in RESEARCH) or bool(RESEARCH_RE.match(k))


def sanity(d: Path) -> dict:
    if not (d / "parts_hr" / "index.jsonl").exists():
        return {}
    ts = json.loads((d / "timestamps.json").read_text()); photos = set(ts)
    inst = {}
    for l in open(d / "parts_hr" / "index.jsonl"):
        r = json.loads(l)
        for i in r.get("instances", []):
            inst[i["key"]] = r["stem"]
    drop = set()
    nw = d / "safe" / "not_whale.json"
    if nw.exists():
        drop = set(json.loads(nw.read_text()).get("drop", []))
    kept = {k: s for k, s in inst.items() if k not in drop}
    C = json.loads((d / "safe" / "clusters.json").read_text())["clusters"] if (d / "safe" / "clusters.json").exists() else []
    return {"photos": len(photos), "instances": len(inst), "kept": len(kept), "photos_no_kept": len(photos - set(kept.values())),
            "filtered": len(drop & set(inst)), "clusters": len(C)}


def main() -> None:
    ap = argparse.ArgumentParser(); ap.add_argument("--shadow", type=Path, required=True); ap.add_argument("--live", type=Path, default=Path("experiments/days_all"))
    ap.add_argument("--days", default=None); ap.add_argument("--live-dir", action="append", default=[], help="DAY=DIR: a live folder outside --live (the test day)")
    ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args(); extra = dict(x.split("=", 1) for x in a.live_dir)
    days = a.days.split(",") if a.days else sorted(p.name for p in a.shadow.iterdir() if p.is_dir())
    rep = {}
    for day in days:
        L = Path(extra.get(day) or a.live / day).resolve(); S = a.shadow / day
        stems = sorted(json.loads((L / "timestamps.json").read_text()), key=len, reverse=True)
        ll, ss = listing(L, stems), listing(S, stems)
        rows = {}; miss = []; extra_s = []; small = []
        for k in sorted(set(ll) | set(ss)):
            lv, sv = ll.get(k, [0, 0]), ss.get(k, [0, 0])
            if research(k):
                continue
            rows[k] = {"live": lv, "shadow": sv}
            if lv[0] and not sv[0]:
                miss.append(k)
            elif sv[0] and not lv[0]:
                extra_s.append(k)
            elif lv[1] > 0 and sv[1] < 0.5 * lv[1] * sv[0] / max(1, lv[0]):
                small.append(k)
        rep[day] = {"missing_in_shadow": miss, "only_in_shadow": extra_s, "much_smaller_in_shadow": small,
                    "research_only_live": sorted({k if RESEARCH_RE.match(k) and "/" in k and k.startswith("cluster_v3") is False else k.split("/")[0] for k in ll if research(k)}),
                    "bytes": {"live": sum(v[1] for k, v in ll.items() if not research(k)), "shadow": sum(v[1] for v in ss.values())},
                    "images": "link" if (S / "images").is_symlink() else "hardlinked dir" if (S / "images").is_dir() else "missing",
                    "sanity": {"live": sanity(L), "shadow": sanity(S)}, "files": rows}
    a.out.write_text(json.dumps(rep, indent=1))
    for day, r in rep.items():
        sl, sh = r["sanity"]["live"], r["sanity"]["shadow"]
        print(day, "missing:", r["missing_in_shadow"], "only shadow:", r["only_in_shadow"], "smaller:", r["much_smaller_in_shadow"],
              f"GB {r['bytes']['live'] / 1e9:.2f}/{r['bytes']['shadow'] / 1e9:.2f}", r["images"], sl, sh)


if __name__ == "__main__":
    main()
