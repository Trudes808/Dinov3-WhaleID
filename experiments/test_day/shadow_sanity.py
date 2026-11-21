"""Switch prep (2026-11-20): sanity table of the shadow day root against the live one, per day, nothing tuned.
Photos, instances, kept instances (not in safe/not_whale.json drop), photos without a kept instance, filtered,
clusters (safe/clusters.json); for days with a catalogue the wrong automatic links of the strict tier (safe_tracklets
evaluate as switch_eval.py runs it: safe_model.pkl, P 0.995, strict, filter on) in both roots. Report only: eval days
are counted like the others (as the app serves them) and nothing is chosen on them. No photo names are printed: a wrong
link is listed by its two catalogue animals and photo sequence numbers.

    PYTHONPATH=. .venv/bin/python experiments/test_day/shadow_sanity.py --shadow SHADOW_DAYS [--days D,..] [--live-dir DAY=DIR] --out OUT.json
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from switch_eval import LABELS, LIVE, clusters, kept_instances   # noqa: E402

EVAL = {"2024-12-18", "2024-12-26", "2025-02-19", "2025-03-21"}


def seq(stem: str) -> str:
    m = re.search(r"-(\d{3,4})-[^-]*$", stem) or re.search(r"(\d{4})$", stem)
    return m.group(1) if m else "?"


def row(root: Path, day: str, cat: dict) -> dict:
    d = root / day
    if not (d / "parts_hr" / "index.jsonl").exists():
        return {}
    stems = list(json.loads((d / "timestamps.json").read_text()))
    allk, kept, drop = kept_instances(d)
    C = json.loads((d / "safe" / "clusters.json").read_text())["clusters"]
    r = {"photos": len(stems), "instances": sum(len(v) for v in allk.values()), "kept": sum(len(v) for v in kept.values()),
         "photos_no_kept": sum(1 for s in stems if not kept.get(s)), "filtered": len(drop), "clusters": len(C)}
    labels = cat.get(day, {}).get("photos", {})
    if labels:
        r["catalogue_photos_no_kept"] = sum(1 for s in stems if labels.get(s.split("__")[-1]) and not kept.get(s))
        c = clusters(root, day); r["strict_links"] = c["links"]; r["wrong_links"] = len(c["wrong_links"])
        r["wrong_link_list"] = [re.sub(r"[^\s'\",\[\]]*__[^\s'\",~\]]*", lambda m: seq(m.group(0)), str(w))[:300] for w in c["wrong_links"]]
        r["clusters_per_animal"] = c["metrics"]["clusters_per_animal"]; r["impure_photos"] = c["metrics"]["photos_in_impure_clusters"]
    return r


def main() -> None:
    ap = argparse.ArgumentParser(); ap.add_argument("--shadow", type=Path, required=True); ap.add_argument("--days", default=None)
    ap.add_argument("--live-dir", action="append", default=[]); ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args(); extra = dict(x.split("=", 1) for x in a.live_dir)
    cat = json.loads(LABELS.read_text())["days"]
    days = a.days.split(",") if a.days else sorted(p.name for p in a.shadow.iterdir() if p.is_dir())
    out = {}
    for day in days:
        lroot = Path(extra[day]).parent if day in extra else LIVE
        out[day] = {"eval": day in EVAL, "live": row(lroot, day, cat), "shadow": row(a.shadow, day, cat)}
        L, S = out[day]["live"], out[day]["shadow"]
        print(day, {k: (L.get(k), S.get(k)) for k in ("photos", "instances", "kept", "photos_no_kept", "filtered", "clusters", "catalogue_photos_no_kept", "wrong_links")}, flush=True)
    a.out.write_text(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
