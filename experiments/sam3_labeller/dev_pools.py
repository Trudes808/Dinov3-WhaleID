"""Rule A2.2: per class, the real dev photos (val days + nw photos, the comparison's raw SAM 3 outputs) where the union
of the class's candidate phrasings at 0.3 paints >= 0.05 % of the frame (pool P_c, blind to which phrasing), at most 80
drawn, plus 40 random photos outside P_c. Also writes every candidate's painted share per photo (for A3), on the
288 x 288 mask grid (the squashed grid covers the whole photo, so shares are kept).

    .venv/bin/python experiments/sam3_labeller/dev_pools.py      -> OUT/dev/pools.json, OUT/dev/shares.json
"""
from __future__ import annotations

import json
import random
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
import lab_common as L  # noqa: E402

GROUPS = {"rail": ["railing", "boat railing", "metal pole"], "tape": ["colored tape", "tape", "colored band"],
          "deck": ["boat deck", "cleat", "boat bow", "bollard", "mooring bollard"], "land": ["island", "land", "mountain"]}
THR = [0.3, 0.4, 0.5]


def shares(rec: dict, pidx: dict) -> dict:
    out = {}
    for c, ps in GROUPS.items():
        for p in ps:
            j = pidx[p]; s = rec[f"s{j}"]; m = rec[f"m{j}"] >= 128
            for t in THR:
                k = s >= t
                out[f"{p}@{t}"] = float(m[k].any(0).mean()) if k.any() else 0.0
            out[f"{p}@m"] = m[s >= 0.3] if (s >= 0.3).any() else None
    return out


def main() -> None:
    D = L.OUT / "dev"; D.mkdir(parents=True, exist_ok=True)
    rows = {}
    for st in ["val", "nw"]:
        pidx = {p: i for i, p in enumerate(json.loads((L.S3RAW / st / "_prompts.json").read_text()))}
        for l in open(L.NW / "moonshot_cache" / "sam3_compare" / "lists" / f"{st}.tsv"):
            pid, path = l.rstrip("\n").split("\t")
            rec = L.load_raw(L.S3RAW / st / f"{pid}.npz")
            sh = shares(rec, pidx)
            r = {k: v for k, v in sh.items() if not k.endswith("@m")}
            for c, ps in GROUPS.items():           # union of the group at 0.3 (pool rule)
                ms = [sh[f"{p}@m"] for p in ps if sh[f"{p}@m"] is not None]
                r[f"pool_{c}"] = float(np.any(np.concatenate(ms, 0), 0).mean()) if ms else 0.0
            rows[f"{st}|{pid}"] = {"path": path, **r}
    pools = {}
    for c in GROUPS:
        ins = sorted(k for k, r in rows.items() if r[f"pool_{c}"] >= 0.0005); outs = sorted(k for k in rows if k not in set(ins))
        rng = random.Random(20261121)
        pick = ins if len(ins) <= 80 else rng.sample(ins, 80)
        pools[c] = {"n_pool": len(ins), "in": sorted(pick), "out": sorted(rng.sample(outs, 40))}
        print(c, "pool", len(ins), "tagged", len(pick), "+ 40 outside")
    (D / "pools.json").write_text(json.dumps(pools, indent=1)); (D / "shares.json").write_text(json.dumps(rows))


if __name__ == "__main__":
    main()
