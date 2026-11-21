"""Switch prep (2026-11-20): migrate the researcher's saved state to the shadow day root, as COPIES.

The saved labels follow instance keys ("<stem>~k", "<stem>~x" for a photo without a tracked main whale). After the mask
switch some keys name another box or no longer exist. This script reads, per day, two Labeling Assistant loads
(experiments/test_day/switch_load.py on the live root and on the shadow root: photos with their instance boxes, stacks with
their keys) and the saved files, and writes migrated copies to OUT/<day>/ (never in place; the originals are only read):

  labeling.json   (Labeling Assistant: assign, no, links)
  pair_marks.json (Pairs / Bursts labels: key_a, key_b; cluster-level records saved under the current clustering also
                   carry set_a / set_b cluster ids, which name other clusters after the switch: re-derived from the keys)
  review_extra.json (second-look asks of the Bursts review: key_a, key_b)

Rules (photo by photo; instance boxes matched live -> shadow one to one, greedily by IoU >= 0.5):
  - the photo's main key (live ~0 or ~x) -> the shadow main key (~0 if the shadow keeps a main whale, else ~x) for
    `assign` (the photo's animal is per photo: this re-attaches a photo whose main whale appeared or vanished);
  - any other key -> the shadow instance whose box matches; a main key without a matching box -> the shadow main key
    (flagged "changed box"); a non-main key without a matching box -> left as it is (it no longer attaches: "gone").
  - a target already holding a different saved value is never overwritten ("conflict", reported).
Schema changes are additive only: every changed record keeps its old values under "mask_switch" (labeling.json:
top-level "mask_switch" with the old -> new key map). Nothing is dropped.

    .venv/bin/python experiments/test_day/switch_migrate.py --live-load DIR --shadow-load DIR --state DATA_DIR \
        --shadow-root SHADOW_DAYS --out OUT [--days D1,D2]  > report.json
LIVE/SHADOW load dirs hold <day>.json from switch_load.py. DATA_DIR = a folder of <day>/ with the saved files (read only).
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
from collections import Counter
from pathlib import Path


def iou(a, b) -> float:
    if not a or not b:
        return 0.0
    ix = max(0.0, min(a[2], b[2]) - max(a[0], b[0])); iy = max(0.0, min(a[3], b[3]) - max(a[1], b[1])); i = ix * iy
    u = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - i
    return i / u if u > 0 else 0.0


def keys_of(load: dict) -> set:
    ph = load["photos"]
    return {f"{ph[p]['stem']}~{k if k >= 0 else 'x'}" for s in load["stacks"] for p, k in s["items"]}


class Mapper:
    def __init__(self, live: dict, shadow: dict):
        self.L = {p["stem"]: p["inst"] for p in live["photos"]}; self.S = {p["stem"]: p["inst"] for p in shadow["photos"]}
        self.kl, self.ks = keys_of(live), keys_of(shadow); self.match = {}
        for stem, la in self.L.items():
            sb = self.S.get(stem) or []
            cand = sorted(((iou(a, b), i, j) for i, a in enumerate(la) for j, b in enumerate(sb) if a and b), reverse=True)
            used_i, used_j, m = set(), set(), {}
            for v, i, j in cand:
                if v < 0.5:
                    break
                if i in used_i or j in used_j or f"{stem}~{j}" not in self.ks:
                    continue
                m[i] = j; used_i.add(i); used_j.add(j)
            self.match[stem] = m

    def main_shadow(self, stem: str) -> str | None:
        for k in (f"{stem}~0", f"{stem}~x"):
            if k in self.ks:
                return k
        return None

    def is_main_live(self, key: str) -> bool:
        stem, _, k = key.rpartition("~")
        return k == "x" or (k == "0" and key in self.kl)

    def map(self, key: str, main_rule: bool) -> tuple[str | None, str]:
        """(new key or None, how): how in same | moved | x->0 | 0->x | changed box | gone | unknown photo."""
        stem, _, k = key.rpartition("~")
        if stem not in self.S:
            return None, "unknown photo"
        tgt = self.main_shadow(stem)
        if main_rule and self.is_main_live(key):
            if tgt is None:
                return None, "gone"
            if k == "x":
                return tgt, "same" if tgt == key else "x->0"
            if tgt.endswith("~x"):
                return tgt, "0->x"
            j = self.match[stem].get(0)
            return tgt, "same" if j == 0 else "changed box"
        if k == "x":
            return (tgt, "same" if tgt == key else "x->0") if tgt else (None, "gone")
        j = self.match.get(stem, {}).get(int(k))
        if j is not None:
            return f"{stem}~{j}", "same" if j == int(k) else "moved"
        if k == "0" and tgt is not None:
            return tgt, "0->x" if tgt.endswith("~x") else "changed box"
        return None, "gone"


def migrate_labeling(s: dict, M: Mapper) -> tuple[dict, dict]:
    out = copy.deepcopy(s); rep = {"assign": Counter(), "no": Counter(), "links": Counter(), "conflicts": [], "open_after": [], "changed": {}}
    keymap = {}
    # assign: main keys by the photo rule, others by box
    new_assign = {}; moved_from = {}
    order = sorted(s.get("assign", {}).items(), key=lambda kv: (not M.is_main_live(kv[0]), kv[0]))     # main keys first: they own the photo
    for key, aid in order:
        nk, how = M.map(key, main_rule=True); rep["assign"][how] += 1
        if nk is None:
            new_assign.setdefault(key, aid)
            if M.is_main_live(key):
                rep["open_after"].append(key.rpartition("__")[2])
            continue
        if nk in new_assign and new_assign[nk] != aid:
            rep["conflicts"].append({"field": "assign", "key": key.rpartition("__")[2], "to": nk.rpartition("__")[2]}); new_assign.setdefault(key, aid); continue
        new_assign[nk] = aid
        if nk != key:
            moved_from[nk] = key; keymap[key] = nk
    out["assign"] = new_assign
    # no: {key: [animals the stack is not]}
    new_no = {}
    for key, v in s.get("no", {}).items():
        nk, how = M.map(key, main_rule=False); rep["no"][how] += 1
        nk = nk or key
        if nk in new_no and nk != key:
            new_no[nk] = sorted(set(new_no[nk]) | set(v)); rep["conflicts"].append({"field": "no", "key": key.rpartition("__")[2], "merged_into": nk.rpartition("__")[2]})
        else:
            new_no[nk] = v
        if nk != key:
            keymap.setdefault(key, nk)
    out["no"] = new_no
    # links: {key_a|key_b: {key_a, key_b, regions, ...}}
    new_links = {}
    for lid, rec in s.get("links", {}).items():
        na, ha = M.map(rec["key_a"], main_rule=False); nb, hb = M.map(rec["key_b"], main_rule=False)
        how = "same" if ha == hb == "same" else "gone" if "gone" in (ha, hb) or na is None or nb is None else \
              "same keys, changed box" if (na, nb) == (rec["key_a"], rec["key_b"]) else "remapped"
        rep["links"][how] += 1
        if how != "same" and (ha in ("changed box", "0->x") or hb in ("changed box", "0->x")):
            rep["links"]["regions on a changed outline"] += 1
        if how == "remapped":
            r = dict(rec); r["mask_switch"] = {"key_a": rec["key_a"], "key_b": rec["key_b"], "how": [ha, hb]}; r["key_a"], r["key_b"] = na, nb
            nid = f"{na}|{nb}"
            if nid in new_links:
                rep["conflicts"].append({"field": "links", "key": lid}); new_links[lid] = rec; continue
            new_links[nid] = r
        else:
            new_links[lid] = rec
    out["links"] = new_links
    if keymap:
        out["mask_switch"] = {"note": "keys re-attached after the whale-mask switch (S6h + rule F); old -> new", "map": keymap}
    rep = {k: (dict(v) if isinstance(v, Counter) else v) for k, v in rep.items()}
    return out, rep


def migrate_records(recs: list, M: Mapper, cl_of: dict | None, revision: str | None) -> tuple[list, dict]:
    out = []; rep = Counter(); sets_fixed = 0; gone = []
    for r in recs:
        na, ha = M.map(r["key_a"], main_rule=False); nb, hb = M.map(r["key_b"], main_rule=False)
        how = "same" if ha == hb == "same" else "gone" if na is None or nb is None else \
              "same keys, changed box" if (na, nb) == (r["key_a"], r["key_b"]) else "remapped"
        rep[how] += 1; rep[f"a:{ha}"] += 1; rep[f"b:{hb}"] += 1
        n = dict(r)
        if how == "remapped":
            old = {"key_a": r["key_a"], "key_b": r["key_b"], "how": [ha, hb]}
            n["key_a"], n["key_b"] = na, nb
            if r.get("pair_id") == f"{r['key_a']}|{r['key_b']}":
                old["pair_id"] = r["pair_id"]; n["pair_id"] = f"{na}|{nb}"
            n["mask_switch"] = old
        elif how == "gone":
            gone.append([r["key_a"].rpartition("__")[2], r["key_b"].rpartition("__")[2], r.get("verdict")])
        if r.get("level") == "cluster" and revision and r.get("cluster_rev") == revision and cl_of is not None:
            a, b = cl_of.get(n["key_a"]), cl_of.get(n["key_b"])
            ms = n.setdefault("mask_switch", {}); ms.update({"set_a": r.get("set_a"), "set_b": r.get("set_b"), "pair_id": r.get("pair_id")})
            n["set_a"], n["set_b"] = a, b
            if a is not None and b is not None:
                n["pair_id"] = "link:cluster:" + "|".join(sorted((f"clusters:{a}", f"clusters:{b}")))
            sets_fixed += 1
        out.append(n)
    return out, {"records": len(recs), **dict(rep), "cluster_sets_rederived": sets_fixed, "gone": gone}


def revision_of(clusters: dict) -> str:
    rep = clusters.get("report", {})
    base = f"{Path(str(rep.get('model', 'model'))).stem}|P{rep.get('threshold')}|cost{rep.get('cost')}|{'strict' if rep.get('strict') else 'plain'}"
    return base + (f"|{rep['filter'].get('name')}" if rep.get("filter") else "")


def sha(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--live-load", type=Path, required=True); ap.add_argument("--shadow-load", type=Path, required=True)
    ap.add_argument("--state", type=Path, required=True); ap.add_argument("--shadow-root", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True); ap.add_argument("--days", default=None)
    a = ap.parse_args()
    if a.out.resolve() == a.state.resolve() or str(a.out.resolve()).startswith(str(a.state.resolve()) + "/"):
        raise SystemExit("refusing: --out must not be the state folder")
    days = a.days.split(",") if a.days else sorted(p.name for p in a.state.iterdir() if p.is_dir() and any((p / f).exists() for f in ("labeling.json", "pair_marks.json")))
    report = {}
    for day in days:
        sd = a.state / day; files = [f for f in ("labeling.json", "pair_marks.json", "review_extra.json") if (sd / f).exists()]
        if not files:
            continue
        before = {f: sha(sd / f) for f in files}
        live = json.loads((a.live_load / f"{day}.json").read_text()); shadow = json.loads((a.shadow_load / f"{day}.json").read_text())
        M = Mapper(live, shadow); od = a.out / day; od.mkdir(parents=True, exist_ok=True); r = {}
        cz = json.loads((a.shadow_root / day / "safe" / "clusters.json").read_text())
        cl_of = {k: c["id"] for c in cz["clusters"] for k in c["keys"]}; rev = revision_of(cz)
        if "labeling.json" in files:
            s = json.loads((sd / "labeling.json").read_text()); n, r["labeling"] = migrate_labeling(s, M)
            (od / "labeling.json").write_text(json.dumps(n, separators=(",", ":")))
        if "pair_marks.json" in files:
            s = json.loads((sd / "pair_marks.json").read_text()); n = dict(s); n["labels"], r["pair_marks"] = migrate_records(s.get("labels", []), M, cl_of, rev)
            (od / "pair_marks.json").write_text(json.dumps(n, indent=1))
        if "review_extra.json" in files:
            s = json.loads((sd / "review_extra.json").read_text()); n = dict(s); n["items"], r["review_extra"] = migrate_records(s.get("items", []), M, None, None)
            (od / "review_extra.json").write_text(json.dumps(n, indent=1))
        after = {f: sha(sd / f) for f in files}
        r["originals_unchanged"] = before == after; r["sha256"] = before
        report[day] = r
    print(json.dumps(report, indent=1, default=str))


if __name__ == "__main__":
    main()
