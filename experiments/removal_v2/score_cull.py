"""Score removal proposals against the researcher's real cull of an unculled day (written 2026-11-22 without opening any
held-out day; tested on a made-up fixture: --selftest).

    .venv/bin/python experiments/removal_v2/score_cull.py --raw-dir "RAW_DAY_FOLDER" --lrcat CATALOG.lrcat --day YYYY-MM-DD \
        --records markers.json|DAY_removal.jsonl [--folder-like "2026-3-30 C1 Name"] [--out OUT.json] [--list OUT.tsv]
    .venv/bin/python experiments/removal_v2/score_cull.py --selftest

KEPT frames = raw frames whose camera counter appears among the catalogue's photos of that day (the cull happens before
the rename; renamed files keep the camera's counter as their last token: ...-24359-013-HR5_3180, ..._000_184A1234).
The catalogue is opened read-only (sqlite URI mode=ro&immutable=1); only file names, folders and capture times are read.
The catalogue's photos "of that day" = capture date == --day, or a folder path holding the day as Y-M-D; with
--folder-like (default: the raw folder's own name when a catalogue folder ends with it) only that camera's folder.
RAW frames = the file names in --raw-dir (only listed, never opened).
Records: a markers.json of the Labeling Assistant (schema v1: photos.<stem>.removal / .unsure / .faint) or the JSONL of
seg_multi/removal.py (one record per photo: photo, removal_v2.{proposed, reason, faint}). A record's photo is matched
to a raw frame by the frame's file name (stem ends with it) or, failing that, by the camera counter.

Writes: culled frames (not in the catalogue): proposed by reason / in "faint whale?" / neither; kept frames: wrongly
proposed by reason / in "faint whale?"; counter collisions and frames without a record are counted, never guessed.
"""
from __future__ import annotations

import argparse
import json
import re
import sqlite3
import sys
import tempfile
from collections import Counter
from pathlib import Path

RAW_EXT = {".cr2", ".cr3", ".nef", ".arw", ".dng", ".raf", ".jpg", ".jpeg"}
COUNTER = re.compile(r"([A-Za-z0-9]{3,4}_?\d{4})$")


def counter_of(name: str) -> str | None:
    """camera counter = the file name's last token: 184A1234, HR5_3180, IMG_1234 (case kept)"""
    m = COUNTER.search(Path(name).stem if "." in name[-6:] else name)
    return m.group(1) if m else None


def day_labels(day: str) -> list[str]:
    y, m, d = (int(x) for x in day.split("-"))
    return [f"{y}-{m}-{d}", f"{y:04d}-{m:02d}-{d:02d}"]


def catalogue_day(lrcat: Path, day: str, folder_like: str | None) -> tuple[dict, dict]:
    """counter -> [catalogue base names] of the day's photos (virtual copies counted once), and facts about the match"""
    db = sqlite3.connect(f"file:{lrcat}?mode=ro&immutable=1", uri=True)
    rows = db.execute("""select distinct f.id_local, f.baseName, i.captureTime, fo.pathFromRoot from Adobe_images i
        join AgLibraryFile f on f.id_local = i.rootFile join AgLibraryFolder fo on fo.id_local = f.folder""").fetchall()
    db.close()
    labs = day_labels(day)
    on_day = [(b, ct or "", p or "") for _, b, ct, p in rows
              if (ct or "").startswith(day) or any(f"/{l}/" in f"/{p}" or f"/{l} " in f"/{p}" for l in labs)]
    folders = Counter(p for _, _, p in on_day)
    chosen = None
    if folder_like:
        hits = [p for p in folders if p.rstrip("/").split("/")[-1].lower() == folder_like.lower()]
        if hits:
            chosen = hits
            on_day = [x for x in on_day if x[2] in hits]
    out: dict[str, list] = {}
    for b, _, _ in on_day:
        c = counter_of(b)
        if c:
            out.setdefault(c, []).append(b)
    facts = {"catalogue_photos_on_day": len(on_day), "folders_on_day": dict(folders), "folder_used": chosen,
             "without_counter": sum(1 for b, _, _ in on_day if not counter_of(b)), "counter_collisions": sum(1 for v in out.values() if len(v) > 1)}
    return out, facts


def raw_frames(raw_dir: Path) -> list[str]:
    return sorted(p.name for p in raw_dir.iterdir() if p.suffix.lower() in RAW_EXT and not p.name.startswith("."))


def load_records(path: Path) -> dict:
    """photo stem -> 'marker' | 'crew' | 'water' | 'land' | 'faint' | None"""
    out = {}
    if path.suffix == ".jsonl":
        for l in open(path):
            v = json.loads(l); r2 = v.get("removal_v2") or {}
            out[v["photo"]] = r2.get("reason") if r2.get("proposed") else "faint" if r2.get("faint") else None
        return out
    doc = json.loads(path.read_text())
    for s, r in (doc.get("photos") or {}).items():
        rem = r.get("removal")
        out[s] = (rem or {}).get("reason") if rem and not r.get("unsure") else "faint" if r.get("faint") or r.get("unsure") else None
    return out


def score(raw: list[str], cat: dict, recs: dict) -> dict:
    by_name = {}
    for s, v in recs.items():
        by_name[s] = v
    stems = list(recs)
    rows, culled, kept, no_rec = [], Counter(), Counter(), 0
    rec_by_counter: dict[str, list] = {}
    for s in stems:
        c = counter_of(s)
        if c:
            rec_by_counter.setdefault(c, []).append(s)
    for f in raw:
        base = Path(f).stem; c = counter_of(base)
        hit = [s for s in stems if s == base or s.endswith("__" + base) or s.endswith(base)]
        if not hit and c and len(rec_by_counter.get(c, [])) == 1:
            hit = rec_by_counter[c]
        v = recs[hit[0]] if len(hit) == 1 else None
        if len(hit) != 1:
            no_rec += 1
        is_kept = bool(c and c in cat)
        grp = v if v in ("marker", "crew", "water", "land", "faint") else "none"
        (kept if is_kept else culled)[grp] += 1
        rows.append({"frame": base, "counter": c, "kept": is_kept, "group": grp, "record": hit[0] if len(hit) == 1 else None})
    prop = ("marker", "crew", "water", "land")
    nk, nc = sum(kept.values()), sum(culled.values())
    summ = {"raw_frames": len(raw), "kept": nk, "culled": nc, "frames_without_one_record": no_rec,
            "culled_by_group": dict(culled), "kept_by_group": dict(kept),
            "culled_proposed": sum(culled[k] for k in prop), "kept_proposed_WRONG": sum(kept[k] for k in prop),
            "faint_culled": culled["faint"], "faint_kept": kept["faint"],
            "proposal_precision": round(sum(culled[k] for k in prop) / max(1, sum(culled[k] + kept[k] for k in prop)), 4),
            "cull_recall": round(sum(culled[k] for k in prop) / max(1, nc), 4),
            "catalogue_counters_not_in_raw": None}
    return {"summary": summ, "rows": rows}


def run(a) -> dict:
    cat, facts = catalogue_day(a.lrcat, a.day, a.folder_like or a.raw_dir.name)
    raw = raw_frames(a.raw_dir); recs = load_records(a.records)
    res = score(raw, cat, recs)
    raw_counters = {counter_of(Path(f).stem) for f in raw}
    res["summary"]["catalogue_counters_not_in_raw"] = sum(1 for c in cat if c not in raw_counters)
    res["catalogue"] = facts
    return res


def table(res: dict) -> str:
    s = res["summary"]; L = ["| | n | marker | crew | water | land | proposed | faint whale? | neither |", "|---|---|---|---|---|---|---|---|---|"]
    for name, g in (("culled by the researcher", s["culled_by_group"]), ("kept by the researcher", s["kept_by_group"])):
        n = sum(g.values()); pr = sum(g.get(k, 0) for k in ("marker", "crew", "water", "land"))
        L.append(f"| {name} | {n} | {g.get('marker', 0)} | {g.get('crew', 0)} | {g.get('water', 0)} | {g.get('land', 0)} | {pr} | {g.get('faint', 0)} | {g.get('none', 0)} |")
    return "\n".join(L)


def selftest() -> None:
    """a 2025 catalogue day as the kept list, a synthetic raw listing with extra (culled) frames, made-up records"""
    lrcat = Path("/datasets/NOAA_whales_data/Test.lrcat"); day = "2025-01-15"; folder = "2025-1-15 C3 Lyman"
    cat, facts = catalogue_day(lrcat, day, folder)
    assert facts["folder_used"] and len(cat) > 20, facts
    kept_c = sorted(cat)[:40]                                     # 40 kept frames
    pre = kept_c[0].rsplit("_", 1)[0] if "_" in kept_c[0] else kept_c[0][:4]
    extra = [f"{pre}_{9000 + k}" if "_" in kept_c[0] else f"{pre}{9000 + k}" for k in range(12)]      # 12 culled frames (counters not in the catalogue)
    with tempfile.TemporaryDirectory() as td:
        rd = Path(td) / folder; rd.mkdir()
        for i, c in enumerate(kept_c + extra):
            (rd / f"2025-1-15_C3_Lyman_{i:03d}_{c}.CR3").touch()
        names = sorted(p.stem for p in rd.iterdir())
        recs = {"version": 1, "photos": {}}
        def put(stem, reason=None, faint=False):
            recs["photos"][f"2025-1-15_C3_Lyman__{stem}"] = {"kind": "rail" if reason == "marker" else None, "removal": {"reason": reason} if reason else None, "unsure": faint, "faint": {"why": "x"} if faint else None}
        ex = [n for n in names if counter_of(n) in extra]; kp = [n for n in names if counter_of(n) in kept_c]
        for n, r in zip(ex[:3], ["marker", "crew", "water"]):
            put(n, r)                                             # 3 culled frames proposed
        put(ex[3], faint=True)                                    # 1 culled frame in faint
        put(kp[0], "water"); put(kp[1], faint=True); put(kp[2], faint=True)      # 1 kept frame wrongly proposed, 2 kept in faint
        (Path(td) / "markers.json").write_text(json.dumps(recs))
        a = argparse.Namespace(raw_dir=rd, lrcat=lrcat, day=day, records=Path(td) / "markers.json", folder_like=None)
        res = run(a); s = res["summary"]
        print(json.dumps(s, indent=1)); print(table(res)); print(json.dumps(res["catalogue"]["folder_used"]))
        assert s["raw_frames"] == 52 and s["kept"] == 40 and s["culled"] == 12, s
        assert s["culled_proposed"] == 3 and s["culled_by_group"] == {"marker": 1, "crew": 1, "water": 1, "faint": 1, "none": 8}, s
        assert s["kept_proposed_WRONG"] == 1 and s["faint_kept"] == 2 and s["kept_by_group"]["none"] == 37, s
        assert s["frames_without_one_record"] == 52 - 7, s      # only 7 frames have a record (a plain photo has none)
        assert s["catalogue_counters_not_in_raw"] == len(cat) - 40, s
    print("SELFTEST OK")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--raw-dir", type=Path); ap.add_argument("--lrcat", type=Path); ap.add_argument("--day")
    ap.add_argument("--records", type=Path); ap.add_argument("--folder-like"); ap.add_argument("--out", type=Path); ap.add_argument("--list", type=Path)
    ap.add_argument("--selftest", action="store_true")
    a = ap.parse_args()
    if a.selftest:
        selftest(); return
    if not (a.raw_dir and a.lrcat and a.day and a.records):
        sys.exit("--raw-dir, --lrcat, --day and --records are needed")
    res = run(a)
    print(json.dumps({"summary": res["summary"], "catalogue": res["catalogue"]}, indent=1)); print(table(res))
    if a.out:
        a.out.parent.mkdir(parents=True, exist_ok=True); a.out.write_text(json.dumps(res, indent=1))
    if a.list:
        with open(a.list, "w") as f:
            f.write("frame\tcounter\tkept\tgroup\n")
            for r in res["rows"]:
                f.write(f"{r['frame']}\t{r['counter']}\t{int(r['kept'])}\t{r['group']}\n")


if __name__ == "__main__":
    main()
