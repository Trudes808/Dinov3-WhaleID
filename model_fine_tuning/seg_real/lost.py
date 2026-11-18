"""Instances a candidate loses: validation photos where the r04 instances_v2 stage keeps >= 2 instances and the
candidate's run (seg_real.e2e) keeps fewer. Per such photo the r04 instance that is gone: "missing" (no candidate
instance overlaps >= 20 % of it) or "fused" (its best-overlapping candidate instance is also the best of another r04
instance). Sample up to 60 photos per candidate (seed 0), sheets drawn BLIND to the candidate (only r04 instances,
the lost one in magenta) for by-eye labels: second_whale / calf / same_whale_piece / not_whale / unclear.

    PYTHONPATH=. .venv/bin/python -m model_fine_tuning.seg_real.lost find --names s1,s4,s5
        -> experiments/seg_splash/lost_instances.json, SHEETS/lost/lost_<name>_NN.jpg (24 tiles each)
    PYTHONPATH=. .venv/bin/python -m model_fine_tuning.seg_real.lost merge --names s1,s4,s5   (SHEETS/lost_raw/lost_<name>_NN.json)
    PYTHONPATH=. .venv/bin/python -m model_fine_tuning.seg_real.lost others --names r04,s1,s4,s5
        -> SHEETS/others_17.jpg: the researcher's 17 clicked other whales (2025-01-02, 2025-03-18), each model's kept
           instances outlined in colours, clicks white (main whale) / yellow (other whales)
"""
from __future__ import annotations

import argparse
import json
import random
from collections import Counter

import cv2
import numpy as np

from . import common as C
from .e2e import E2E

OUT = C.EXP / "lost_instances.json"
COL = [(0, 0, 255), (0, 200, 0), (255, 128, 0), (0, 255, 255), (128, 0, 255), (255, 255, 0), (0, 128, 255), (200, 200, 200)]


def inst(root, day, stem, rows):
    m = cv2.imread(str(root / day / "instances_v2" / "masks" / f"{stem}.png"), cv2.IMREAD_UNCHANGED)
    return m, [i["id"] for i in rows[stem]["instances"]]


def put(img, t, org, s=0.5):
    cv2.putText(img, t, org, cv2.FONT_HERSHEY_SIMPLEX, s, (0, 0, 0), 3, cv2.LINE_AA); cv2.putText(img, t, org, cv2.FONT_HERSHEY_SIMPLEX, s, (255, 255, 255), 1, cv2.LINE_AA)


def cmd_find(a) -> None:
    R = json.loads(OUT.read_text()) if OUT.exists() else {}
    for name in a.names.split(","):
        events = []
        for day in C.VAL:
            ref = {json.loads(l)["stem"]: json.loads(l) for l in open(C.DAYS / day / "instances_v2" / "index.jsonl")}
            cand = {json.loads(l)["stem"]: json.loads(l) for l in open(E2E / name / day / "instances_v2" / "index.jsonl")}
            for stem, r in ref.items():
                if len(r["instances"]) < 2 or len(cand[stem]["instances"]) >= len(r["instances"]):
                    continue
                mr, ir = inst(C.DAYS, day, stem, ref); mc, ic = inst(E2E / name, day, stem, cand)
                best = {}
                for i in ir:
                    m = mr == i; ids, cnt = np.unique(mc[m], return_counts=True); ok = ids > 0
                    best[i] = (int(ids[ok][np.argmax(cnt[ok])]), float(cnt[ok].max() / m.sum())) if ok.any() else (None, 0.0)
                users = Counter(b for b, f in best.values() if b is not None and f >= 0.2)
                lost = [(i, "missing") for i in ir if best[i][0] is None or best[i][1] < 0.2]
                lost += [(i, "fused") for i in ir if best[i][0] is not None and best[i][1] >= 0.2 and users[best[i][0]] > 1 and i != min(j for j in ir if best[j][0] == best[i][0])]
                for i, why in lost:
                    rec = [x for x in r["instances"] if x["id"] == i][0]
                    events.append({"day": day, "stem": stem, "id": i, "key": rec["key"], "area": rec["area"], "bbox": rec["bbox"], "why": why,
                                   "n_ref": len(ir), "n_cand": len(ic)})
        photos = len({(e["day"], e["stem"]) for e in events})
        rng = random.Random(0); by_photo = {}
        for e in events:
            by_photo.setdefault((e["day"], e["stem"]), []).append(e)
        keys = sorted(by_photo); rng.shuffle(keys); smp = [rng.choice(by_photo[k]) for k in keys[:a.n]]
        for t, e in enumerate(smp):
            e["tile"] = t + 1
        R[name] = {"photos_with_loss": photos, "lost_instances": len(events), "why": dict(Counter(e["why"] for e in events)), "sample": smp}
        tiles = []
        for e in smp:
            img = C.photo(e["day"], e["stem"]); H, W = img.shape[:2]
            ref = {json.loads(l)["stem"]: json.loads(l) for l in open(C.DAYS / e["day"] / "instances_v2" / "index.jsonl")}
            mr, ir = inst(C.DAYS, e["day"], e["stem"], ref)
            ys, xs = np.nonzero(mr > 0); x0, x1, y0, y1 = xs.min(), xs.max(), ys.min(), ys.max()
            cx, cy = (x0 + x1) / 2, (y0 + y1) / 2; w = max((x1 - x0) * 1.3, 500); h = max((y1 - y0) * 1.3, 333)
            w, h = (h * 1.5, h) if w / h < 1.5 else (w, w / 1.5); w, h = min(w, W), min(h, H)
            wx = int(min(max(cx - w / 2, 0), W - w)); wy = int(min(max(cy - h / 2, 0), H - h))
            v = img.copy()
            for k, i in enumerate(ir):
                cs, _ = cv2.findContours((mr == i).astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
                cv2.drawContours(v, cs, -1, (255, 0, 255) if i == e["id"] else (0, 255, 255), 9 if i == e["id"] else 4)
            v = cv2.resize(v[wy:int(wy + h), wx:int(wx + w)], (420, 280), interpolation=cv2.INTER_AREA)
            ctx = cv2.resize(img, (210, int(210 * H / W)), interpolation=cv2.INTER_AREA); cv2.rectangle(ctx, (int(wx * 210 / W), int(wy * 210 / W)), (int((wx + w) * 210 / W), int((wy + h) * 210 / W)), (0, 255, 255), 2)
            c = np.full((280, 210, 3), 25, np.uint8); c[:ctx.shape[0]] = ctx[:280]
            put(v, str(e["tile"]), (5, 20), 0.6)
            tiles.append(cv2.copyMakeBorder(np.hstack([v, c]), 2, 2, 2, 2, cv2.BORDER_CONSTANT, value=(255, 255, 255)))
        out = C.SHEETS / "lost"; out.mkdir(parents=True, exist_ok=True)
        for s in range(0, len(tiles), 18):
            part = tiles[s:s + 18]
            while len(part) % 3:
                part.append(np.zeros_like(tiles[0]))
            cv2.imwrite(str(out / f"lost_{name}_{s // 18 + 1:02d}.jpg"), np.vstack([np.hstack(part[i:i + 3]) for i in range(0, len(part), 3)]), [cv2.IMWRITE_JPEG_QUALITY, 87])
        print(name, "photos with a lost instance", photos, "lost instances", len(events), R[name]["why"], "sample", len(smp), flush=True)
    OUT.write_text(json.dumps(R, indent=0))


def cmd_merge(a) -> None:
    R = json.loads(OUT.read_text())
    for name in a.names.split(","):
        raw = {}
        for f in sorted((C.SHEETS / "lost_raw").glob(f"lost_{name}_*.json")):
            raw.update(json.loads(f.read_text()))
        for e in R[name]["sample"]:
            v = raw.get(str(e["tile"]))
            if v:
                e["label"], e["sure"] = v[0], bool(v[1])
        lab = [e for e in R[name]["sample"] if "label" in e]
        R[name]["labels"] = {"labelled": len(lab), "counts": dict(Counter(e["label"] for e in lab).most_common()),
                             "by_why": {w: dict(Counter(e["label"] for e in lab if e["why"] == w)) for w in ("missing", "fused")}}
        print(name, R[name]["photos_with_loss"], R[name]["why"], R[name]["labels"])
    OUT.write_text(json.dumps(R, indent=0))


def cmd_others(a) -> None:
    import sys
    sys.path.insert(0, str(C.REPO / "experiments" / "instances_v2"))
    import common as IC
    names = a.names.split(","); tiles = []
    for day in ["2025-01-02", "2025-03-18"]:
        photos, oth = IC.load_marks(day)
        for key, kp in photos.items():
            ow = [w for w in (oth.get(key) or []) if w]
            if not ow:
                continue
            stem = key.rsplit("~", 1)[0]; img = C.photo(day, stem); H, W = img.shape[:2]
            pts = [(p["x"] * W, p["y"] * H, (255, 255, 255)) for p in kp.values()] + [(p["x"] * W, p["y"] * H, (0, 255, 255)) for w in ow for p in w.values()]
            P = np.array([[x, y] for x, y, _ in pts]); x0, y0 = P.min(0); x1, y1 = P.max(0)
            cx, cy = (x0 + x1) / 2, (y0 + y1) / 2; w = max((x1 - x0) * 1.8, 500); h = max((y1 - y0) * 1.8, 333)
            w, h = (h * 1.5, h) if w / h < 1.5 else (w, w / 1.5); w, h = min(w, W), min(h, H)
            wx = int(min(max(cx - w / 2, 0), W - w)); wy = int(min(max(cy - h / 2, 0), H - h))
            panels = []
            for n in names:
                root = C.DAYS if n == "r04" else E2E / n
                rows = {stem: json.loads(l) for l in open(root / day / "instances_v2" / "index.jsonl") if json.loads(l)["stem"] == stem}
                m, ids = inst(root, day, stem, rows); v = img.copy()
                for k, i in enumerate(ids):
                    cs, _ = cv2.findContours((m == i).astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE); cv2.drawContours(v, cs, -1, COL[k % len(COL)], 6)
                for x, y, col in pts:
                    cv2.circle(v, (int(x), int(y)), 12, (0, 0, 0), -1); cv2.circle(v, (int(x), int(y)), 8, col, -1)
                v = cv2.resize(v[wy:int(wy + h), wx:int(wx + w)], (240, 160), interpolation=cv2.INTER_AREA)
                put(v, f"{len(tiles) + 1} {n}: {len(ids)}", (4, 15), 0.45); panels.append(v)
            tiles.append(cv2.copyMakeBorder(np.hstack(panels), 2, 2, 2, 2, cv2.BORDER_CONSTANT, value=(255, 255, 255)))
    while len(tiles) % 2:
        tiles.append(np.zeros_like(tiles[0]))
    p = C.SHEETS / "others_17.jpg"; cv2.imwrite(str(p), np.vstack([np.hstack(tiles[i:i + 2]) for i in range(0, len(tiles), 2)]), [cv2.IMWRITE_JPEG_QUALITY, 87]); print(p, len(tiles))


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter); sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("find"); p.add_argument("--names", required=True); p.add_argument("--n", type=int, default=60); p.set_defaults(fn=cmd_find)
    p = sub.add_parser("merge"); p.add_argument("--names", required=True); p.set_defaults(fn=cmd_merge)
    p = sub.add_parser("others"); p.add_argument("--names", default="r04,s1,s4,s5"); p.set_defaults(fn=cmd_others)
    a = ap.parse_args(argv); a.fn(a)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
