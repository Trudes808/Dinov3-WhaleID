"""The two small models behind the Whale-Review-Web Photo Labeling page (/labeling), and the simulation of a researcher
sorting a day with them (the catalog plays the researcher).

    python -m model_fine_tuning.cluster.label_assist fit      [--root experiments/days_all] [--out ../Whale-Review-Web/backend/labeling_model.json]
    python -m model_fine_tuning.cluster.label_assist simulate [--model ../Whale-Review-Web/backend/labeling_model.json] [--split eval] [--taus 1.5,0.95,0.9]
    python -m model_fine_tuning.cluster.label_assist passers  [--model ...]

pair      P(two safe clusters ("stacks") of one encounter are the same whale): a logistic regression on the sequence
          model's P for the pair (safe/clusters.json suggestions), the student embedding cosine per part (max over the
          instance pairs, cluster_v3/embeddings.npz) and the gap between the two clusters. Fitted on the val days on
          every pair of labelled main-instance clusters inside one catalog group (passers-by, O ids, join the group
          whose time span holds them); reported on the eval days as the rank of the true animal when the stacks of an
          encounter are sorted one by one in capture order and the animals so far are scored by their closest stack.
boundary  P(the focal group changes across a gap between two sightings): sightings are runs of photos at most 120 s
          apart (all cameras); features are the gap, the boat's displacement from the EXIF GPS of the RAW files (the
          nearest fix within 2 min for cameras without GPS) and its speed. Fitted on the train days, reported on val +
          eval. A proposal for the researcher, who has the field log.
simulate  the page's loop with runs: the stack in hand takes along the following open stacks whose link to the run so
          far is at least tau (no other animal closer, and not across a camera-counter gap, see below); the catalog
          accepts the correct prefix. Counts the decisions per day and how often the first choice / the top three held
          the true animal. The in-page "catalogue replay" measures the same on the page's own code (and with the
          reviewer's earlier links).
passers   passers-by (O ids) photographed during an encounter: how well "links to nothing of the group nearby", a
          range jump and the two together single them out (they are rare: the numbers say what a hint is worth).

Frame traces. The photographers shoot marker frames (land = a new group follows; the boat railing after a series, its
colour the animal's role) and delete them in the cull, but the kept files still carry the camera's own frame number
(...-184A4470): a hole in that counter between two kept photos is a deleted frame. `frame_traces` reads the holes;
they enter the boundary model (a long gap without a hole is almost never a new group) and stop the runs (a hole right
before a stack means a series ended there). A camera body without any hole all day is "unknown" (an unculled set).

The page's backend (Whale-Review-Web/backend/labeling.py) computes the same features; `fit` writes the weights it reads.
The catalog is used for fitting on val / train days and for the reports only.
"""
from __future__ import annotations

import argparse
import bisect
import io
import json
import math
import re
import struct
from collections import Counter, OrderedDict, defaultdict
from pathlib import Path

import numpy as np

from .safe_clusters import PARTS, load_embeddings

PAIR_NAMES = ["p_seq", "has_p_seq", "app_fin", "has_fin", "app_fluke", "has_fluke", "app_body", "has_body", "log_gap", "log_gap_cap60"]
BOUNDARY_NAMES = ["log_gap", "log1p_dist", "speed", "has_gps", "del_some", "log1p_del", "del_none"]
# ...-184A4470 / ...-HR5_2949 / ..._000_184A5044 (2026 naming): the camera's own file name survives the rename; the token before
# the counter joins by "_" only when it holds a letter, so a sequence number is never part of the body (same groups as
# the earlier rule on every name it matched; Whale-Review-Web backend/labeling.py has the same rule)
CAMERA_NUMBER = re.compile(r"(?:^|[-_])((?:[A-Za-z0-9]*[A-Za-z][A-Za-z0-9]*_)?[A-Za-z0-9]*?)(\d{4})$")
SIGHTING_GAP = 120.0
RAW_ROOT = Path("/datasets/NOAA_whales_data/Lightroom images/Topside research/2025 Research")
RAW_EXT = (".cr2", ".cr3", ".dng", ".arw", ".nef", ".jpg", ".jpeg", ".tif", ".tiff")


# ---- frame traces --------------------------------------------------------------------------------------------------
def frame_traces(photos: list[tuple]) -> dict:
    """stem -> frames deleted on its camera body since that body's previous kept photo. `photos` = (time, stem, folder)
    in capture order. None = unknown: the body's first photo, a name without a camera number, a counter roll-over, or a
    body whose counter shows no hole all day (nothing was culled: the counter says nothing)."""
    last, miss, body, holes = {}, {}, {}, defaultdict(int)
    for _, s, folder in photos:
        m = CAMERA_NUMBER.search(s)
        if not m:
            miss[s] = None; continue
        key = (folder, m.group(1)); n = int(m.group(2)); p = last.get(key); body[s] = key
        miss[s] = (n - p - 1) if p is not None and 0 <= n - p - 1 < 2000 else None
        holes[key] += miss[s] or 0; last[key] = n
    for s, key in body.items():
        if not holes[key]:
            miss[s] = None
    return miss


def day_photos(day_dir: Path) -> list[tuple]:
    ts = json.loads((day_dir / "timestamps.json").read_text())
    return sorted((float(t["time"]), s, t.get("folder", "?")) for s, t in ts.items() if t.get("time") is not None)


# ---- stacks and pair features ------------------------------------------------------------------------------------
def major(c: dict) -> str | None:
    e = sorted((c.get("catalog") or {}).items(), key=lambda kv: -kv[1])
    return e[0][0] if e else None


def load_stacks(day_dir: Path) -> dict:
    """The labelled main-instance clusters of a day with the pair feature matrices between them."""
    z = json.loads((day_dir / "safe" / "clusters.json").read_text()); emb = load_embeddings(day_dir)
    cl = [c for c in z["clusters"] if c["main_photos"] and major(c)]
    cl.sort(key=lambda c: (c["start"], c["end"], c["id"])); n = len(cl); pos = {c["id"]: i for i, c in enumerate(cl)}
    t0 = np.array([c["start"] for c in cl]); t1 = np.array([c["end"] for c in cl])
    gap = np.maximum(0.0, np.maximum(t0[:, None], t0[None, :]) - np.minimum(t1[:, None], t1[None, :]))
    app = {}
    for part in PARTS:
        E = emb.get(part) or {}; rows, own = [], []
        for i, c in enumerate(cl):
            for k in c["keys"]:
                if k in E:
                    rows.append(E[k]); own.append(i)
        A = np.full((n, n), np.nan, np.float32)
        if rows:
            own = np.asarray(own); M = np.asarray(rows, np.float32); starts = np.flatnonzero(np.r_[True, own[1:] != own[:-1]]); ids = own[starts]
            A[np.ix_(ids, ids)] = np.maximum.reduceat(np.maximum.reduceat(M @ M.T, starts, axis=0), starts, axis=1)
        app[part] = A
    pseq = np.zeros((n, n), np.float32)
    for g in z.get("suggestions", []):
        if g.get("p") is not None and g["a"] in pos and g["b"] in pos:
            a, b = pos[g["a"]], pos[g["b"]]; pseq[a, b] = pseq[b, a] = max(pseq[a, b], float(g["p"]))
    photos = day_photos(day_dir); miss = frame_traces(photos); when = {s_: t for t, s_, _ in photos}
    trace = [miss.get(min(c["photos"], key=lambda s_: when.get(s_, 0.0))) for c in cl]          # frames deleted right before the stack's first photo
    return {"clusters": cl, "true": [major(c) for c in cl], "gap": gap, "app": app, "pseq": pseq, "n_photos": sum(len(c["main_photos"]) for c in cl), "trace": trace}


def pair_features(D: dict) -> np.ndarray:
    """(n, n, len(PAIR_NAMES)) feature tensor, the definition the web backend repeats (labeling._pair_matrix)."""
    n = len(D["clusters"]); lg = np.log1p(D["gap"]); F = np.zeros((n, n, len(PAIR_NAMES)), np.float32)
    F[..., 0] = D["pseq"]; F[..., 1] = D["pseq"] > 0
    for j, part in ((2, "fin"), (4, "fluke"), (6, "body")):
        A = D["app"][part]; has = np.isfinite(A); F[..., j] = np.where(has, np.nan_to_num(A), 0.0); F[..., j + 1] = has
    F[..., 8] = lg; F[..., 9] = np.minimum(lg, math.log1p(60.0))
    return F


def scopes(D: dict) -> dict:
    """Encounter scopes from the catalog: the clusters of each G group; an O animal's clusters join the group whose
    time span holds them, the others chain into scopes of their own (gaps under 10 min)."""
    cl, true = D["clusters"], D["true"]; grp = lambda i: true[i].split("_")[1]
    spans = defaultdict(lambda: [1e18, -1e18])
    for i, c in enumerate(cl):
        if grp(i).startswith("G"):
            s = spans[grp(i)[:3]]; s[0] = min(s[0], c["start"]); s[1] = max(s[1], c["end"])
    out = defaultdict(list); rest = []
    for i, c in enumerate(cl):
        g = grp(i)
        if g.startswith("G"):
            out[g[:3]].append(i); continue
        inside = [(min(abs(c["start"] - cl[j]["start"]) for j in range(len(cl)) if grp(j)[:3] == k), k) for k, s in spans.items() if s[0] <= c["start"] <= s[1]]
        if inside:
            out[min(inside)[1]].append(i)
        else:
            rest.append(i)
    k, prev = 0, None
    for i in sorted(rest, key=lambda i: cl[i]["start"]):
        if prev is not None and cl[i]["start"] - prev > 600:
            k += 1
        out[f"O{k}"].append(i); prev = cl[i]["end"]
    return out


def pair_training(D: dict) -> tuple[np.ndarray, np.ndarray]:
    F = pair_features(D); X, y = [], []
    for sc in scopes(D).values():
        for a in range(len(sc)):
            for b in range(a + 1, len(sc)):
                X.append(F[sc[a], sc[b]]); y.append(D["true"][sc[a]] == D["true"][sc[b]])
    return np.asarray(X, np.float32), np.asarray(y, bool)


def pair_prob(D: dict, coef, intercept) -> np.ndarray:
    P = 1.0 / (1.0 + np.exp(-(pair_features(D) @ np.asarray(coef, np.float32) + intercept))); np.fill_diagonal(P, 0.0)
    return P


def simulate(D: dict, P: np.ndarray, tau: float = 1.5, max_run: int = 8, stop_at_traces: bool = True) -> Counter:
    """Sort every scope's stacks in capture order with the catalog as the researcher; tau <= 1 offers runs, which do
    not cross a camera-counter hole (a deleted marker frame) unless stop_at_traces is off."""
    t = Counter(); true = D["true"]; hole = [bool(x) for x in D.get("trace") or [0] * len(true)] if stop_at_traces else [False] * len(true)
    for sc in scopes(D).values():
        sc = sorted(sc, key=lambda i: (D["clusters"][i]["start"], i)); animals: OrderedDict = OrderedDict(); done = set(); k = 0
        while k < len(sc):
            i = sc[k]; k += 1
            if i in done:
                continue
            who = true[i]; t["decisions"] += 1
            names = list(animals); score = [max(P[i, j] for j in animals[a]) for a in names]
            order = [names[x] for x in np.argsort(-np.asarray(score))] if names else []
            if who not in animals:
                t["new"] += 1
            else:
                r = order.index(who) + 1; t["existing"] += 1; t["first"] += r == 1; t["top3"] += r <= 3
            top = order[0] if order else None
            others = [j for a, js in animals.items() if a != top for j in js]      # as on the page: every animal but the model's first choice
            animals.setdefault(who, []).append(i); done.add(i)
            if tau <= 1.0:
                offered, cur = [], [i]
                for j in sc[k:]:                       # the run the model offers: a chain of links >= tau, no other animal closer
                    if j in done:
                        continue
                    if len(offered) >= max_run or hole[j]:
                        break
                    p = max(P[x, j] for x in cur)
                    if p < tau or (others and max(P[j, o] for o in others) > p):
                        break
                    offered.append(j); cur.append(j)
                taken = 0
                for j in offered:                      # the researcher cuts the run at the first wrong stack
                    if true[j] != who:
                        break
                    taken += 1; animals[who].append(j); done.add(j)
                t["run_offered"] += len(offered); t["run_taken"] += taken; t["run_cut"] += len(offered) > taken
        t["stacks"] += len(sc)
    t["photos"] += D["n_photos"]
    return t


# ---- encounter boundaries ---------------------------------------------------------------------------------------
def _dms(v: str, ref: str) -> float | None:
    try:
        parts = []
        for x in v.strip("[] ").split(","):
            a, _, b = x.strip().partition("/"); parts.append(float(a) / float(b) if b else float(a))
        deg = parts[0] + (parts[1] if len(parts) > 1 else 0.0) / 60.0 + (parts[2] if len(parts) > 2 else 0.0) / 3600.0
        return -deg if ref.strip().upper() in ("S", "W") else deg
    except (ValueError, ZeroDivisionError, IndexError):
        return None


def _tiff_gps(buf: bytes):
    """Latitude / longitude from a bare GPS IFD (the CMT4 box of a CR3 file)."""
    if len(buf) < 16 or buf[:2] not in (b"II", b"MM"):
        return None
    e = "<" if buf[:2] == b"II" else ">"
    try:
        off = struct.unpack(e + "I", buf[4:8])[0]; n = struct.unpack(e + "H", buf[off:off + 2])[0]; tags = {}
        for i in range(n):
            tag, typ, cnt, val = struct.unpack(e + "HHI4s", buf[off + 2 + 12 * i: off + 14 + 12 * i])
            if tag in (1, 3):
                tags[tag] = val[:1].decode("ascii", "replace")
            elif tag in (2, 4) and typ == 5 and cnt == 3:
                o = struct.unpack(e + "I", val)[0]; r = struct.unpack(e + "6I", buf[o:o + 24]); tags[tag] = [r[2 * k] / r[2 * k + 1] if r[2 * k + 1] else 0.0 for k in range(3)]
        if 2 not in tags or 4 not in tags:
            return None
        lat = tags[2][0] + tags[2][1] / 60 + tags[2][2] / 3600; lon = tags[4][0] + tags[4][1] / 60 + tags[4][2] / 3600
        return None if lat == 0 and lon == 0 else (-lat if tags.get(1) == "S" else lat, -lon if tags.get(3) == "W" else lon)
    except (struct.error, IndexError):
        return None


def exif_gps(path: Path):
    try:
        if path.suffix.lower() == ".cr3":
            data = open(path, "rb").read(4_000_000); i = data.find(b"CMT4")
            if i < 4:
                return None
            return _tiff_gps(data[i + 4:i - 4 + struct.unpack(">I", data[i - 4:i])[0]])
        import exifread
        with open(path, "rb") as fh:
            tags = exifread.process_file(fh, details=False)
        la, lo = tags.get("GPS GPSLatitude"), tags.get("GPS GPSLongitude")
        if la is None or lo is None:
            return None
        lat = _dms(str(la), str(tags.get("GPS GPSLatitudeRef", "N"))); lon = _dms(str(lo), str(tags.get("GPS GPSLongitudeRef", "E")))
        return None if lat is None or lon is None or (lat == 0 and lon == 0) else (lat, lon)
    except Exception:
        return None


def haversine(a, b) -> float:
    p1, p2 = math.radians(a[0]), math.radians(b[0]); dl = math.radians(b[1] - a[1])
    x = math.sin((p2 - p1) / 2) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(dl / 2) ** 2
    return 2 * 6371000.0 * math.asin(math.sqrt(x))


def raw_dirs(raw_root: Path) -> dict:
    out, level = {}, [raw_root]
    for _ in range(3):
        nxt = []
        for d in level:
            try:
                subs = sorted(x for x in d.iterdir() if x.is_dir())
            except OSError:
                continue
            for x in subs:
                out.setdefault(x.name, x); nxt.append(x)
        level = nxt
    return out


def day_gaps(day_dir: Path, labels: dict, dirs: dict, cache: Path) -> list[tuple]:
    """(gap s, displacement m or None, frames deleted at the gap or None, the focal group changes) for every gap between
    two sightings of a labelled day."""
    photos = day_photos(day_dir); miss = frame_traces(photos)
    f = cache / f"{day_dir.name}.json"
    if f.exists():
        gps = json.loads(f.read_text())
    else:
        files = {}
        for folder in {p[2] for p in photos}:
            d = dirs.get(folder)
            if d is not None:
                files.update({x.stem: x for x in d.iterdir() if x.suffix.lower() in RAW_EXT})
        gps = {}
        for _, s, _ in photos:
            r = files.get(s.split("__")[-1]); g = exif_gps(r) if r else None
            if g:
                gps[s] = [round(g[0], 6), round(g[1], 6)]
        cache.mkdir(parents=True, exist_ok=True); f.write_text(json.dumps(gps))
    fixes = sorted((t, gps[s]) for t, s, _ in photos if s in gps); ft = [x[0] for x in fixes]

    def where(t, s):
        if s in gps:
            return gps[s]
        if not fixes:
            return None
        j = bisect.bisect_left(ft, t); c = [fixes[k] for k in (j - 1, j) if 0 <= k < len(fixes)]; x = min(c, key=lambda x: abs(x[0] - t))
        return x[1] if abs(x[0] - t) <= 120 else None
    group = lambda s: (lambda ident: ident.split("_")[1][:3] if ident and ident.split("_")[1].startswith("G") else None)(labels.get(s.split("__")[-1]))
    sights = [[photos[0]]] if photos else []
    for a, b in zip(photos, photos[1:]):
        if b[0] - a[0] > SIGHTING_GAP:
            sights.append([b])
        else:
            sights[-1].append(b)
    G = [[g for g in (group(p[1]) for p in s) if g] for s in sights]; out = []
    for k in range(len(sights) - 1):
        left = next((g[-1] for g in reversed(G[: k + 1]) if g), None); right = next((g[0] for g in G[k + 1:] if g), None)
        if left is None or right is None:
            continue
        a, b = sights[k][-1], sights[k + 1][0]; pa, pb = where(a[0], a[1]), where(b[0], b[1])
        out.append((b[0] - a[0], haversine(pa, pb) if pa and pb else None, miss.get(b[1]), left != right))
    return out


def boundary_x(gap: float, dist: float | None, miss: int | None) -> list[float]:
    """miss = frames deleted before the first photo after the gap (None = unknown)."""
    return [math.log(max(gap, 1e-3)), math.log1p(dist) if dist is not None else 0.0, dist / gap if dist is not None else 0.0, float(dist is not None),
            float(bool(miss)), math.log1p(miss or 0), float(miss == 0)]


# ---- commands -----------------------------------------------------------------------------------------------------
def main(argv=None) -> int:
    here = Path(__file__).resolve().parents[2]
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("cmd", choices=["fit", "simulate", "passers"])
    ap.add_argument("--root", type=Path, default=here / "experiments" / "days_all")
    ap.add_argument("--days", type=Path, default=here / "model_fine_tuning" / "splits" / "days.json")
    ap.add_argument("--labels", type=Path, default=here / "model_fine_tuning" / "splits" / "labels_all_days.json")
    ap.add_argument("--raw-root", type=Path, default=RAW_ROOT)
    ap.add_argument("--out", type=Path, default=here.parent / "Whale-Review-Web" / "backend" / "labeling_model.json")
    ap.add_argument("--model", type=Path, default=here.parent / "Whale-Review-Web" / "backend" / "labeling_model.json")
    ap.add_argument("--split", default="eval")
    ap.add_argument("--taus", default="1.5,0.95,0.9")
    ap.add_argument("--propose", type=float, default=0.35)
    a = ap.parse_args(argv)
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import roc_auc_score
    split = json.loads(a.days.read_text()); ready = lambda d: (a.root / d / "safe" / "clusters.json").exists()

    if a.cmd == "simulate":
        m = json.loads(a.model.read_text())["pair"]; data = {d: load_stacks(a.root / d) for d in split[a.split] if ready(d)}
        for tau in [float(x) for x in a.taus.split(",")]:
            for stop in ((True, False) if tau <= 1 else (True,)):
                tot = Counter()
                for d, D in data.items():
                    tot.update(simulate(D, pair_prob(D, m["coef"], m["intercept"]), tau, stop_at_traces=stop))
                print(f"{a.split} tau {tau}{'' if tau > 1 else ' runs stop at a deleted frame' if stop else ' runs ignore the frame traces'}: photos {tot['photos']} stacks {tot['stacks']} animals {tot['new']} -> decisions {tot['decisions']} "
                      f"({tot['photos'] / max(tot['decisions'], 1):.1f} photos each) | first choice {tot['first'] / max(tot['existing'], 1):.3f} top three {tot['top3'] / max(tot['existing'], 1):.3f} | "
                      f"run stacks taken {tot['run_taken']} of {tot['run_offered']} ({tot['run_taken'] / max(tot['run_offered'], 1):.3f}), runs cut {tot['run_cut']}")
        return 0

    if a.cmd == "passers":
        # Passers-by photographed during an encounter: stacks with an O id inside a group's scope that have at least 3
        # of the group's stacks within 5 min. Cues: isolation = 1 - the best link to any of those stacks; farther = the
        # stack's log range (safe/photos.json) above the median of those stacks.
        m = json.loads(a.model.read_text())["pair"]; rows = []
        for part in ("train", "val", "eval"):
            for d in split[part]:
                if not ready(d):
                    continue
                D = load_stacks(a.root / d); P = pair_prob(D, m["coef"], m["intercept"]); ph = json.loads((a.root / d / "safe" / "photos.json").read_text()); cl = D["clusters"]
                rng = [float(np.nanmedian([ph[s_].get("log_range") if ph.get(s_, {}).get("log_range") is not None else np.nan for s_ in c["main_photos"]])) for c in cl]
                for name, sc in scopes(D).items():
                    if not name.startswith("G"):
                        continue
                    for i in sc:
                        ctx = [j for j in sc if j != i and D["true"][j].split("_")[1].startswith("G") and max(cl[i]["start"] - cl[j]["end"], cl[j]["start"] - cl[i]["end"], 0) <= 300]
                        if len(ctx) >= 3:
                            rows.append((part, D["true"][i].split("_")[1].startswith("O"), 1.0 - float(max(P[i, j] for j in ctx)), rng[i] - float(np.nanmedian([rng[j] for j in ctx]))))
        y = np.array([r[1] for r in rows]); iso = np.array([r[2] for r in rows]); far = np.nan_to_num(np.array([r[3] for r in rows]), nan=0.0)
        print(f"stacks inside an encounter with >= 3 of its stacks within 5 min: {len(y)}; passers-by {int(y.sum())} ({y.mean():.2%}); AUC isolation {roc_auc_score(y, iso):.3f}, farther {roc_auc_score(y, far):.3f}")
        for name, f in (("isolation >= 0.3", iso >= 0.3), ("isolation >= 0.4", iso >= 0.4), ("isolation >= 0.5", iso >= 0.5), ("isolation >= 0.4 and farther by 0.4", (iso >= 0.4) & (far >= 0.4)), ("farther by 0.7 alone", far >= 0.7)):
            print(f"  {name:<38} flags {int(f.sum()):>4} stacks ({f.mean():.1%}): {int((f & y).sum())} passers-by, precision {(f & y).sum() / max(f.sum(), 1):.2f}, recall {(f & y).sum() / max(y.sum(), 1):.2f}")
        pr = lambda f: {"flagged": round(float(f.mean()), 3), "precision": round(float((f & y).sum() / max(f.sum(), 1)), 2), "recall": round(float((f & y).sum() / max(y.sum(), 1)), 2)}
        model = json.loads(a.out.read_text())
        model["passer_by"] = {"what": "a stack that links to none of the encounter's stacks within 5 min (isolation = 1 - best link) may be a passer-by or a group animal seen once; 'farther' = log range above the group's median",
                              "isolation": 0.4, "farther": 0.4, "window_s": 300, "stacks": len(y), "passers_by": int(y.sum()), "auc_isolation": round(float(roc_auc_score(y, iso)), 3), "auc_farther": round(float(roc_auc_score(y, far)), 3),
                              "isolated": pr(iso >= 0.4), "isolated_and_farther": pr((iso >= 0.4) & (far >= 0.4)), "note": "all 28 labelled days (30 passers-by: too few to hold days out); scene / lighting change alone was no better than 2 % precision"}
        a.out.write_text(json.dumps(model, indent=1)); print("wrote the passer_by section of", a.out)
        return 0

    # ---- fit: the pair model on the val days
    data = {d: load_stacks(a.root / d) for d in split["val"] + split["eval"] if ready(d)}
    X = np.concatenate([pair_training(data[d])[0] for d in split["val"]]); y = np.concatenate([pair_training(data[d])[1] for d in split["val"]])
    lr = LogisticRegression(C=1.0, max_iter=3000).fit(X, y); coef = [round(float(c), 4) for c in lr.coef_[0]]; b0 = round(float(lr.intercept_[0]), 4)
    print(f"pair model: {len(y)} pairs of {len(split['val'])} val days ({y.mean():.2f} same); weights {dict(zip(PAIR_NAMES, coef))} bias {b0}")
    rep = {}
    for name in ("val", "eval"):
        tot = Counter()
        for d in split[name]:
            D = data[d]
            if name == "val":                              # leave the day out of its own fit
                others = [x for x in split["val"] if x != d]
                Xo = np.concatenate([pair_training(data[x])[0] for x in others]); yo = np.concatenate([pair_training(data[x])[1] for x in others])
                m2 = LogisticRegression(C=1.0, max_iter=3000).fit(Xo, yo); P = pair_prob(D, m2.coef_[0], float(m2.intercept_[0]))
            else:
                P = pair_prob(D, coef, b0)
            tot.update(simulate(D, P))
        rep[name] = {"days": split[name], "top1": round(tot["first"] / tot["existing"], 3), "top3": round(tot["top3"] / tot["existing"], 3), "stacks": tot["stacks"], "photos": tot["photos"], "animals": tot["new"]}
        print(f"  {name}: stacks sorted one by one, true animal first {rep[name]['top1']} / in the top three {rep[name]['top3']} ({tot['existing']} stacks with their animal present, {tot['new']} animals, {tot['photos']} photos)")
    # ---- fit: the boundary model on the train days
    labels = json.loads(a.labels.read_text())["days"]; dirs = raw_dirs(a.raw_root); cache = here / "experiments" / "label_assist" / "gps"
    gaps = {d: day_gaps(a.root / d, labels.get(d, {}).get("photos", {}), dirs, cache) for d in split["train"] + split["val"] + split["eval"] if (a.root / d / "timestamps.json").exists()}
    tr = [g for d in split["train"] for g in gaps.get(d, [])]; te = [g for d in split["val"] + split["eval"] for g in gaps.get(d, [])]
    Xb = np.asarray([boundary_x(*g[:3]) for g in tr]); yb = np.asarray([g[3] for g in tr]); Xt = np.asarray([boundary_x(*g[:3]) for g in te]); yt = np.asarray([g[3] for g in te])
    lb = LogisticRegression(C=1.0, max_iter=5000).fit(Xb, yb); p = lb.predict_proba(Xt)[:, 1]
    old_p = LogisticRegression(C=1.0, max_iter=5000).fit(Xb[:, :4], yb).predict_proba(Xt[:, :4])[:, 1]          # without the frame traces, for the record
    none = Xt[:, 6] > 0
    tp = int(((p >= a.propose) & yt).sum()); fp = int(((p >= a.propose) & ~yt).sum()); fn = int(((p < a.propose) & yt).sum())
    brep = {"auc": round(float(roc_auc_score(yt, p)), 3), "precision": round(tp / max(tp + fp, 1), 2), "recall": round(tp / max(tp + fn, 1), 2), "gaps": len(yt), "days": len(split["val"]) + len(split["eval"]),
            "gap_only_auc": round(float(roc_auc_score(yt, Xt[:, 0])), 3), "with_gps": round(float(Xt[:, 3].mean()), 3), "without_traces_auc": round(float(roc_auc_score(yt, old_p)), 3),
            "none_deleted": {"gaps": int(none.sum()), "group_changes": int(yt[none].sum())}, "changes_with_deleted_frames": round(float((Xt[yt, 4] > 0).mean()), 2)}
    print(f"boundary model: {len(yb)} gaps of {len(split['train'])} train days ({yb.mean():.2f} group changes); held-out {len(yt)} gaps: AUC {brep['auc']} (gap alone {brep['gap_only_auc']}, gap + GPS {brep['without_traces_auc']}; "
          f"{brep['none_deleted']['group_changes']} group changes among the {brep['none_deleted']['gaps']} gaps without a deleted frame), "
          f"at P >= {a.propose}: precision {brep['precision']} recall {brep['recall']} (tp {tp} fp {fp} fn {fn}); weights {dict(zip(BOUNDARY_NAMES, np.round(lb.coef_[0], 3)))} bias {lb.intercept_[0]:.3f}")
    old = json.loads(a.out.read_text()) if a.out.exists() else {}
    out = {"note": "Weights of the two small assistants of the Photo Labeling page (backend/labeling.py). Written by Dinov3-WhaleID model_fine_tuning/cluster/label_assist.py fit; the run and confidence figures come from the page's catalogue replay (experiments/LOG.md 2026-11-11).",
           "pair": {"what": "P(two safe clusters of one encounter are the same whale): logistic on the sequence model's P, the student embedding cosine per part (max over instance pairs) and the gap between the clusters",
                    "names": PAIR_NAMES, "coef": coef, "intercept": b0, "fit_days": split["val"], "val": rep["val"],
                    "eval": rep["eval"] | {"note": "stack-level top-1 / top-3 inside catalogue encounters, every stack decided on its own (no runs)"}},
           "boundary": {"what": "P(the focal group changes across a gap between two sightings): logistic on the gap, the boat's displacement (EXIF GPS), its speed and the frames deleted at the gap (holes in the camera's frame counter: the culled marker photos)",
                        "names": BOUNDARY_NAMES, "coef": [round(float(c), 4) for c in lb.coef_[0]], "intercept": round(float(lb.intercept_[0]), 4), "sighting_gap": SIGHTING_GAP,
                        "propose": a.propose, "maybe": (old.get("boundary") or {}).get("maybe", 0.15), "fit_days": split["train"], "eval": brep},
           "run": old.get("run") or {"what": "following stacks offered together with the head stack when their link to the run so far is at least tau", "tau": 0.9, "max": 8},
           "passer_by": old.get("passer_by"),
           "blend": old.get("blend"), "order": old.get("order"), "collect": old.get("collect"), "merge": old.get("merge"),          # the page's similarity blend and order of work (fitted / measured outside this script, see experiments/LOG.md 2026-11-12)
           "confidence": old.get("confidence") or {"lead": {"top": 0.7, "margin": 0.35, "first_right": {"val": None, "eval": None}}, "open": {"first_right": {"val": None, "eval": None}, "top3": {"val": None, "eval": None}, "new_animal": {"val": None, "eval": None}}}}
    a.out.write_text(json.dumps(out, indent=1)); print("wrote", a.out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
