"""Switch prep: a day as the Labeling Assistant loads it (backend.labeling.load_day in a SCRATCH COPY of Whale-Review-Web),
plus the export rows the researcher's saved state gives on that load (backend.labeling.photo_rows). Read only on the day
folders; the state is read from WHALE_LABELING_STATE (point it at a scratch copy of the saved labeling.json files).

    WHALE_DAYS_ROOT=ROOT WHALE_LABELING_STATE=STATE WHALE_TRIP_LOG=1 WHALE_VIEW_MATCH_FILE=view_match_hwb6.npz \
      WHALE_HINT_FIN=mean WHALE_HINT_FIN_RULE=merge_hwb6 \
      .venv/bin/python experiments/test_day/switch_load.py --webcopy WEBCOPY --day DAY --out OUT.json
Writes {"stacks": [{id, kind, cl, items}], "photos": [{stem, inst, parts}], "rows": photo_rows, "state_keys": {...}}.
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path


def main() -> None:
    p = argparse.ArgumentParser(); p.add_argument("--webcopy", type=Path, required=True); p.add_argument("--day", required=True); p.add_argument("--out", type=Path, required=True)
    a = p.parse_args()
    if "scratchpad" not in str(a.webcopy.resolve()):
        raise SystemExit("refusing: the web copy must live in a scratch folder")
    sys.path.insert(0, str(a.webcopy))
    from backend import labeling
    if "scratchpad" not in str(labeling.STATE_ROOT.resolve()):
        raise SystemExit("refusing: WHALE_LABELING_STATE must point at a scratch copy")
    D = labeling.load_day(a.day); s = labeling.read_state(a.day)
    rows = labeling.photo_rows(D, s)
    keys = set(D["sidx"]); assign = s.get("assign", {})
    st = {"assigned": len(assign), "found": sum(k in keys for k in assign), "missing": sorted(k.split("__")[-1] for k in assign if k not in keys),
          "no_found": sum(k in keys for k in s.get("no", {})), "no": len(s.get("no", {})),
          "links": len(s.get("links", {})), "links_found": sum(r["key_a"] in keys and r["key_b"] in keys for r in s.get("links", {}).values())}
    out = {"day": a.day, "revision": D["revision"],
           "stacks": [{"id": x["id"], "kind": x["kind"], "cl": x["cl"], "items": x["items"]} for x in D["stacks"]],
           "photos": [{"stem": q["stem"], "inst": q["inst"], "parts": q["parts"], "cat": q.get("cat"), "miss": q.get("miss")} for q in D["photos"]],
           "rows": [{k: r[k] for k in ("stem", "aid", "id", "rejected", "catalog", "also")} for r in rows], "state_keys": st,
           "kinds": dict(Counter(x["kind"] for x in D["stacks"]))}
    a.out.parent.mkdir(parents=True, exist_ok=True); a.out.write_text(json.dumps(out, default=str))
    print(a.day, out["kinds"], "photos", len(out["photos"]), "state", st["assigned"], "found", st["found"], "->", a.out)


if __name__ == "__main__":
    main()
