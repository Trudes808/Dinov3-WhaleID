"""Removal rule v2 on the development day (2026-04-07) against the by-eye whale-trace truth, and on culled catalogue days
(every photo kept by the researcher: any proposal there is a false one). Reads the readings of readings.py and the
SAM 3 trace-prompt outputs; the rule itself is model_fine_tuning/seg_multi/removal.py (decide). Text only.

    .venv/bin/python experiments/removal_v2/analyze.py --root /home/sat3737/NOAA_whales/moonshot_cache/removal_v2 \
        --truth .../test_days/2026-04-07/truth_whale_trace.tsv --old .../Whale-Review-Web/data/markers/2026-04-07/markers.json \
        --days 2025-01-15 2025-02-27 2025-03-14 2025-01-02 2025-01-17 2025-02-12 2025-03-06 2025-03-18 --out OUT.json
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from model_fine_tuning.seg_multi.removal import WHALE_PX, decide, sam3_read  # noqa: E402

DEV = "2026-04-07"


def rule_on_day(root: Path, day: str, markers: dict | None = None) -> dict:
    """photo -> (decision, readings) for every candidate photo of the day (readings/<day>.jsonl)"""
    out = {}
    sdir = root / "sam3" / day
    order = json.loads((sdir / "_prompts.json").read_text()) if (sdir / "_prompts.json").exists() else None
    for l in open(root / "readings" / f"{day}.jsonl"):
        r = json.loads(l); s = r["photo"]
        rd = {k: v for k, v in r.items() if k not in ("photo", "day", "seq", "scene", "ade")}
        ok = order is not None and (sdir / f"{s}.npz").exists()
        if ok:
            rd.update(sam3_read(sdir / f"{s}.npz", order))
        whale = r["s6_gate_px"] >= WHALE_PX or r.get("kept", 0) >= 1
        mk = (markers or {}).get(s)
        out[s] = (decide(whale, mk, r.get("scene") or {}, rd, ok), rd, r.get("seq"))
    return out


def main() -> None:
    ap = argparse.ArgumentParser(); ap.add_argument("--root", type=Path, required=True); ap.add_argument("--truth", type=Path, required=True)
    ap.add_argument("--old", type=Path, required=True); ap.add_argument("--days", nargs="*", default=[]); ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args()
    T = {r["seq"]: r for r in csv.DictReader((l for l in open(a.truth) if not l.startswith("#")), delimiter="\t")}
    old = json.loads(a.old.read_text())["photos"]
    mk = {s: r["kind"] for s, r in old.items() if r.get("kind")}
    dev = rule_on_day(a.root, DEV, mk)
    import re
    seq = lambda s: re.search(r"_(\d{3})_[A-Za-z0-9]+$", s).group(1)
    res = {"dev": {}, "days": {}}
    # old rule (the live file before v2) and v2, by the truth
    def tally(groups: dict) -> dict:
        c = Counter(); lists = {}
        for s, g in groups.items():
            t = T[seq(s)]; any_w = t["any_whale"] == "yes"
            c[(g, "trace" if any_w else "none")] += 1; lists.setdefault(f"{g}|{'trace' if any_w else 'none'}", []).append(seq(s))
        return {"counts": {f"{k[0]}|{k[1]}": v for k, v in sorted(c.items())}, "lists": {k: sorted(v) for k, v in sorted(lists.items())}}
    g_old = {}
    for s, r in old.items():
        g_old[s] = ("proposed:" + r["removal"]["reason"]) if r.get("removal") else "unsure" if r.get("unsure") else None
    g_old = {s: g for s, g in g_old.items() if g}
    g_new = {s: ("proposed:" + d["reason"]) if d["proposed"] else "faint" for s, (d, _, _) in dev.items() if d["proposed"] or d["faint"]}
    for k in [s for s in old if old[s].get("kind")]:          # markers: from the marker records, unchanged
        g_new[k] = "proposed:marker"
    res["dev"]["old"] = tally(g_old); res["dev"]["v2"] = tally(g_new)
    none_all = [q for q, t in T.items() if t["any_whale"] == "no"]
    res["dev"]["truth"] = {"photos": len(T), "nothing_of_a_whale": len(none_all), "eye2_looked": sum(t["source"] == "eye2" for t in T.values()),
                           "trace_classes_looked": dict(Counter(t["trace"] for t in T.values() if t["source"] == "eye2"))}
    res["dev"]["seen_on_faint"] = dict(Counter(x for d, _, _ in dev.values() if d["faint"] for x in d["seen"]))
    res["dev"]["faint"] = {seq(s): {"seen": d["seen"], "truth": T[q]["trace"], "sure": T[q]["sure"]} for s, (d, _, q) in dev.items() if d["faint"]}
    for day in a.days:
        if not (a.root / "readings" / f"{day}.jsonl").exists():      # no candidate photo on the day (every photo has an S6h whale)
            n_photos = sum(1 for _ in open(Path("/home/sat3737/NOAA_whales/moonshot_cache/switch_root/days") / day / "images" / "index.jsonl"))
            res["days"][day] = {"photos": n_photos, "candidates": 0, "proposed": {}, "faint": {}}
            continue
        D = rule_on_day(a.root, day)
        n_photos = sum(1 for _ in open(Path("/home/sat3737/NOAA_whales/moonshot_cache/switch_root/days") / day / "images" / "index.jsonl"))
        res["days"][day] = {"photos": n_photos, "candidates": len(D),
                            "proposed": {s: d["reason"] for s, (d, _, _) in D.items() if d["proposed"]},
                            "faint": {s: d["seen"] for s, (d, _, _) in D.items() if d["faint"]}}
    a.out.write_text(json.dumps(res, indent=1))
    print(json.dumps({"dev_old": res["dev"]["old"]["counts"], "dev_v2": res["dev"]["v2"]["counts"], "seen": res["dev"]["seen_on_faint"]}, indent=1))
    for d, v in res["days"].items():
        print(d, v["photos"], "cand", v["candidates"], "proposed", len(v["proposed"]), dict(Counter(v["proposed"].values())), "faint", len(v["faint"]))


if __name__ == "__main__":
    main()
