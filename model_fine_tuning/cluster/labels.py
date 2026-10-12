"""Identity labels for every labelled day, read from the Lightroom catalog (read-only).

    python -m model_fine_tuning.cluster.labels --lrcat Test.lrcat --out model_fine_tuning/splits/labels_all_days.json

Collections named <day>_GxxAyy (and <day>_Oxx, "other") hold the photos of one animal on one
day. "same as" notes in collection names merge animals within a day (G07A01 same as G04A01)
and across days (2025-1-20_G02A03 same as 2025-1-16_G03A06); they are applied with a
union-find so that one identity id covers all of them. Group types come from the
"<day> Group N <type>" collections (CG = competition group, MCE = mother/calf/escort, MC, MYE,
Pair, Single). Output: per day, per photo (file base name) -> identity id, plus per-day
groups with their type, camera folders, and the merge list. EVALUATION AND TRAINING LABELS:
never used by the matcher at inference.
"""

from __future__ import annotations

import argparse
import collections
import json
import re
import sqlite3
from pathlib import Path

ANIMAL = re.compile(r"^(\d{4}-\d{1,2}-\d{1,2})_(G\d+A\d+|O\d+)")
GROUP = re.compile(r"^(\d{4}-\d{1,2}-\d{1,2}) Group (\d+) (.*)$")
SAME = re.compile(r"same as (?:(\d{4}-\d{1,2}-\d{1,2})_)?(G\d+A\d+|O\d+)", re.I)


def iso_day(d: str) -> str:
    y, m, dd = d.split("-")
    return f"{int(y):04d}-{int(m):02d}-{int(dd):02d}"


class UnionFind:
    def __init__(self):
        self.p = {}

    def find(self, x):
        self.p.setdefault(x, x)
        while self.p[x] != x:
            self.p[x] = self.p[self.p[x]]
            x = self.p[x]
        return x

    def union(self, a, b):
        ra, rb = self.find(a), self.find(b)
        if ra != rb:
            self.p[max(ra, rb)] = min(ra, rb)        # the lexically smaller key becomes the id (earlier day / lower group)


def read_catalog(lrcat: Path) -> dict:
    db = sqlite3.connect(f"file:{lrcat}?mode=ro", uri=True)
    cur = db.cursor()
    rows = cur.execute("""select f.baseName, f.extension, c.name, i.captureTime, fo.pathFromRoot from AgLibraryCollectionImage ci
        join AgLibraryCollection c on c.id_local=ci.collection join Adobe_images i on i.id_local=ci.image
        join AgLibraryFile f on f.id_local=i.rootFile join AgLibraryFolder fo on fo.id_local=f.folder""").fetchall()
    names = [r[0] for r in cur.execute("select name from AgLibraryCollection").fetchall()]
    uf = UnionFind()
    merges = []
    for cname in names:
        m = ANIMAL.match(cname)
        if not m:
            continue
        day, lab = m.group(1), m.group(2)
        key = f"{iso_day(day)}_{lab}"
        s = SAME.search(cname)
        if s:
            other_day, other_lab = s.group(1) or day, s.group(2)
            other = f"{iso_day(other_day)}_{other_lab}"
            uf.union(key, other)
            merges.append((key, other, cname))
    days = collections.defaultdict(lambda: {"photos": {}, "capture_time": {}, "folders": collections.Counter(), "groups": {}, "animals": collections.defaultdict(set)})
    for base, ext, cname, ct, folder in rows:
        m = ANIMAL.match(cname)
        if not m:
            continue
        day, lab = iso_day(m.group(1)), m.group(2)
        ident = uf.find(f"{day}_{lab}")
        d = days[day]
        d["photos"][base] = ident
        d["capture_time"][base] = ct
        d["folders"][folder.rstrip("/").split("/")[-1]] += 1
        d["animals"][ident].add(base)
    for cname in names:
        g = GROUP.match(cname)
        if g:
            days[iso_day(g.group(1))]["groups"][f"G{int(g.group(2)):02d}"] = g.group(3).strip()
    out = {"_note": "LABELS FROM THE LIGHTROOM CATALOG. Training and evaluation only; the matcher never sees them at inference.",
           "merges": [{"from": a, "to": b, "collection": c} for a, b, c in sorted(set(merges))], "days": {}}
    for day in sorted(days):
        d = days[day]
        out["days"][day] = {"photos": d["photos"], "capture_time": d["capture_time"], "folders": dict(d["folders"]), "groups": d["groups"],
                            "identities": {k: sorted(v) for k, v in sorted(d["animals"].items())}}
    return out


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--lrcat", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    args = p.parse_args(argv)
    out = read_catalog(args.lrcat)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(out, indent=1))
    n_photos = sum(len(d["photos"]) for d in out["days"].values())
    idents = {i for d in out["days"].values() for i in d["identities"]}
    multi = sum(1 for d in out["days"].values() for i, v in d["identities"].items() if len(v) >= 2)
    print(f"{len(out['days'])} days, {n_photos} labelled photos, {len(idents)} identities ({multi} day-identities with >= 2 photos), {len(out['merges'])} merges -> {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
