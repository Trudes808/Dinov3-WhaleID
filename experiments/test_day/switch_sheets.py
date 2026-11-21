"""Switch prep sheets (written to a scratch folder only; they show crew and file content: never into the repository).

    PYTHONPATH=. .venv/bin/python experiments/test_day/switch_sheets.py lost   --eval EVAL.json --s6 S6ROOT --out SHEET.jpg
    PYTHONPATH=. .venv/bin/python experiments/test_day/switch_sheets.py gained --s6 S6ROOT --out SHEET.jpg [--legend L.json]
    PYTHONPATH=. .venv/bin/python experiments/test_day/switch_sheets.py tiles  --list LIST.json --out SHEET.jpg
lost: every val-day catalogue photo without a kept S6-src instance (lost against live, or empty in both). gained: what S6-src removes or
un-fuses (live instances dropped by the not-whale filter, live instances without an S6 counterpart, live instances that
S6 splits in two) and the photos it gains. Tile = the photo (live kept instances red, live filtered orange, S6 kept
green) + a zoom on the instance in question with the raw S6 mask in cyan.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np

REPO = Path(__file__).resolve().parents[2]
LIVE = REPO / "experiments" / "days_all"
VAL = ["2025-01-15", "2025-02-27", "2025-03-14"]
EVAL = {"2024-12-18", "2024-12-26", "2025-02-19", "2025-03-21"}
_T = None


def table():
    global _T
    if _T is None:
        from model_fine_tuning.common import geometry, paths
        _T = geometry.load_folded_table(paths.FOLDED_TABLE)
    return _T


def inst(d: Path, stem: str, W: int, H: int):
    from model_fine_tuning.cluster.instances_v2 import old_instances
    nwf = d / "safe" / "not_whale.json"; nw = json.loads(nwf.read_text()) if nwf.exists() else {}
    keys, masks = old_instances(d, stem, W, H, table())
    drop = set(nw.get("drop", [])); sc = nw.get("scores", {})
    return [(k, m, k not in drop, sc.get(k)) for k, m in zip(keys, masks)]


def outline(img, m, col, th):
    cs, _ = cv2.findContours(m.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(img, cs, -1, col, th, cv2.LINE_AA)


def put(img, text, org, scale=0.5):
    cv2.putText(img, text, org, cv2.FONT_HERSHEY_SIMPLEX, scale, (0, 0, 0), 3, cv2.LINE_AA)
    cv2.putText(img, text, org, cv2.FONT_HERSHEY_SIMPLEX, scale, (255, 255, 255), 1, cv2.LINE_AA)


def box_of(m):
    ys, xs = np.nonzero(m)
    return (xs.min(), ys.min(), xs.max() + 1, ys.max() + 1) if len(xs) else None


def tile(day: str, stem: str, s6: Path, focus=None, lines=(), ctx_w=480, zoom=320, live_dir=None):
    assert day not in EVAL
    L = live_dir or (LIVE / day)
    img = cv2.imread(str(L / "images" / f"{stem}.jpg")); H, W = img.shape[:2]
    li = inst(L, stem, W, H); si = inst(s6 / day, stem, W, H) if (s6 / day / "cse").exists() else []
    raw = cv2.imread(str(s6 / day / "canon" / f"{stem}_mask.png"), cv2.IMREAD_GRAYSCALE); raw = raw > 0 if raw is not None else np.zeros((H, W), bool)
    c = img.copy(); th = max(3, W // 400)
    for k, m, kept, sc in li:
        outline(c, m, (0, 0, 255) if kept else (0, 140, 255), th)
    for k, m, kept, sc in si:
        outline(c, m, (0, 220, 0), th)
    if focus is None:
        ms = [m for _, m, _, _ in li] + [m for _, m, _, _ in si] + ([raw] if raw.any() else [])
        bs = [box_of(m) for m in ms if m.any()]
        focus = (min(b[0] for b in bs), min(b[1] for b in bs), max(b[2] for b in bs), max(b[3] for b in bs)) if bs else (0, 0, W, H)
    x0, y0, x1, y1 = focus; side = max(1.6 * max(x1 - x0, y1 - y0), 200); cx, cy = (x0 + x1) / 2, (y0 + y1) / 2
    wx0, wy0 = int(max(0, min(cx - side / 2, W - side))), int(max(0, min(cy - side / 2, H - side))); wx1, wy1 = int(min(W, wx0 + side)), int(min(H, wy0 + side))
    z = img.copy()
    for k, m, kept, sc in li:
        outline(z, m, (0, 0, 255) if kept else (0, 140, 255), max(1, int(side / 300)))
    outline(z, raw, (255, 255, 0), max(1, int(side / 300)))
    zc = z[wy0:wy1, wx0:wx1]; s = zoom / max(zc.shape[:2]); zc = cv2.resize(zc, (max(1, int(zc.shape[1] * s)), max(1, int(zc.shape[0] * s))), interpolation=cv2.INTER_AREA if s < 1 else cv2.INTER_CUBIC)
    cv2.rectangle(c, (wx0, wy0), (wx1, wy1), (0, 255, 255), th)
    ch = int(H * ctx_w / W); c = cv2.resize(c, (ctx_w, ch), interpolation=cv2.INTER_AREA)
    hh = max(ch, zoom) + 18 * (len(lines) + 1)
    out = np.full((hh, ctx_w + zoom + 4, 3), 30, np.uint8); out[:ch, :ctx_w] = c; out[:zc.shape[0], ctx_w + 4:ctx_w + 4 + zc.shape[1]] = zc
    for i, t in enumerate(lines):
        put(out, t, (5, max(ch, zoom) + 16 + 18 * i))
    return out


def grid(tiles, cols, out: Path, title: str):
    if not tiles:
        tiles = [np.full((100, 400, 3), 30, np.uint8)]
    h = max(t.shape[0] for t in tiles); w = max(t.shape[1] for t in tiles)
    tiles = [cv2.copyMakeBorder(t, 0, h - t.shape[0], 0, w - t.shape[1], cv2.BORDER_CONSTANT, value=(30, 30, 30)) for t in tiles]
    tiles = [cv2.copyMakeBorder(t, 3, 3, 3, 3, cv2.BORDER_CONSTANT, value=(255, 255, 255)) for t in tiles]
    while len(tiles) % cols:
        tiles.append(np.zeros_like(tiles[0]))
    sheet = np.vstack([np.hstack(tiles[i:i + cols]) for i in range(0, len(tiles), cols)])
    head = np.full((40, sheet.shape[1], 3), 30, np.uint8); put(head, title, (8, 27), 0.7)
    out.parent.mkdir(parents=True, exist_ok=True); cv2.imwrite(str(out), np.vstack([head, sheet]), [cv2.IMWRITE_JPEG_QUALITY, 85]); print(out, len(tiles))


def short(stem: str) -> str:
    n = stem.split("__")[-1]; parts = n.replace("_", "-").split("-")
    return "-".join(parts[-2:])


def cmd_lost(a) -> None:
    E = json.loads(Path(a.eval).read_text()); s6 = Path(a.s6); tiles = []; legend = []
    for day in VAL:
        for r in E["days"][day]["rows"]:
            if not r[a.arm]["kept"] and r["cat"]:
                kind = "LOST against live" if r["L"]["kept"] else "empty in both arms"
                lines = [f"{day} {short(r['stem'])} {r['cat'].split('_')[-1]}: {kind}", f"L: {r['L']['n']} inst, {r['L']['kept']} kept, mask {r['L']['mask_px']} px | {a.arm}: mask {r[a.arm]['mask_px']} px, use {r[a.arm]['use_px']} crop px"]
                tiles.append(tile(day, r["stem"], s6, lines=lines)); legend.append({"tile": len(tiles), "day": day, "stem": r["stem"], **{k: r[k] for k in ("cat", "L", a.arm)}})
    grid(tiles, 3, Path(a.out), f"val days: every catalogue photo {a.arm} leaves without a kept instance (red live kept, orange live filtered, cyan raw {a.arm} mask): {len(tiles)}")
    Path(a.out).with_suffix(".json").write_text(json.dumps(legend, indent=0))


def iou_box(a, b):
    ix = max(0, min(a[2], b[2]) - max(a[0], b[0])); iy = max(0, min(a[3], b[3]) - max(a[1], b[1])); i = ix * iy
    u = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - i
    return i / u if u else 0.0


def cmd_gained(a) -> None:
    s6 = Path(a.s6); tiles = []; legend = []; counts = {}
    for day in VAL:
        lid = {json.loads(l)["stem"]: json.loads(l) for l in open(LIVE / day / "parts_hr" / "index.jsonl")}
        sid = {json.loads(l)["stem"]: json.loads(l) for l in open(s6 / day / "parts_hr" / "index.jsonl")}
        nw = json.loads((LIVE / day / "safe" / "not_whale.json").read_text()); drop = set(nw["drop"])
        for stem, r in lid.items():
            lb = [(i["key"], i["bbox_2560"]) for i in r.get("instances", [])]; sb = [(i["key"], i["bbox_2560"]) for i in sid.get(stem, {}).get("instances", [])]
            why = []
            for k, b in lb:
                ov = [iou_box(b, q) for _, q in sb]; cont = [sum(1 for _, q in sb if (max(0, min(b[2], q[2]) - max(b[0], q[0])) * max(0, min(b[3], q[3]) - max(b[1], q[1]))) > 0.5 * (q[2] - q[0]) * (q[3] - q[1]))]
                if k in drop:
                    why.append(("filtered live", k, b))
                elif not ov or max(ov) < 0.1:
                    why.append(("no S6 counterpart", k, b))
                elif cont[0] >= 2:
                    why.append(("split by S6", k, b))
            if not lb and sb:
                why.append(("gained by S6", sb[0][0], sb[0][1]))
            for w, k, b in why:
                counts[w] = counts.get(w, 0) + 1
                lines = [f"{day} {short(stem)} {w}", f"live {len(lb)} inst ({sum(1 for x, _ in lb if x in drop)} filtered), S6 {len(sb)}; nw {nw['scores'].get(k, float('nan')):.1f}" if w != "gained by S6" else f"live {len(lb)} inst, S6 {len(sb)}"]
                tiles.append(tile(day, stem, s6, focus=b, lines=lines)); legend.append({"tile": len(tiles), "day": day, "stem": stem, "why": w, "key": k})
    grid(tiles, 3, Path(a.out), f"val days: what S6-src removes, splits or gains (red live kept, orange live filtered, green S6 kept, cyan raw S6): {counts}")
    Path(a.out).with_suffix(".json").write_text(json.dumps({"counts": counts, "tiles": legend}, indent=0))


def cmd_tiles(a) -> None:
    """LIST.json: {"title": str, "items": [{"day", "stem", "lines": [...], "s6": root, "live": day folder (optional)}]}"""
    L = json.loads(Path(a.list).read_text()); tiles = []
    for it in L["items"]:
        tiles.append(tile(it["day"], it["stem"], Path(it["s6"]), lines=it.get("lines", []), live_dir=Path(it["live"]) if it.get("live") else None))
    grid(tiles, a.cols, Path(a.out), L["title"])


def main() -> None:
    p = argparse.ArgumentParser(); sub = p.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("lost"); s.add_argument("--eval", required=True); s.add_argument("--s6", required=True); s.add_argument("--out", required=True); s.add_argument("--arm", default="S6")
    s = sub.add_parser("gained"); s.add_argument("--s6", required=True); s.add_argument("--out", required=True)
    s = sub.add_parser("tiles"); s.add_argument("--list", required=True); s.add_argument("--out", required=True); s.add_argument("--cols", type=int, default=3)
    a = p.parse_args(); {"lost": cmd_lost, "gained": cmd_gained, "tiles": cmd_tiles}[a.cmd](a)


if __name__ == "__main__":
    main()
