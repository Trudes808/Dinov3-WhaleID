"""Contact sheets of composites (pictures: written OUTSIDE the repository only).

    PYTHONPATH=. .venv/bin/python -m model_fine_tuning.synth_aug.sheets triplets --out DIR --n 48 [--mode rw_rs]
    PYTHONPATH=. .venv/bin/python -m model_fine_tuning.synth_aug.sheets grid --out DIR --n 120        (numbered, for judging)
    PYTHONPATH=. .venv/bin/python -m model_fine_tuning.synth_aug.sheets pick --out DIR --ids 3,17,... --notes notes.json

triplets: per sample  render | composite | composite with the label outlines (fluke red, fin green, pectoral yellow,
head blue, body grey, ignore magenta) -- the labels must still sit on the whale.
grid: composites only, numbered, to judge failures by eye; pick: a sheet of chosen grid numbers with notes.
Samples are drawn from the run05 TRAIN split with a fixed seed; sample k of a sheet uses rng (seed, k).
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np

from ..common import paths
from ..common.data import WhaleRun
from ..parts_seg.data import DEFAULT_CACHE, load_label
from ..parts_seg.labels import PALETTE
from .composite import Compositor, Options
from .train_rw import MODES

RUN = paths.WHALE_FACTORY_ROOT / "data" / "renders" / "run05"
COLOURS = np.asarray(PALETTE, np.uint8).reshape(-1, 3)


def outline(img: np.ndarray, label: np.ndarray) -> np.ndarray:
    out = img.copy()
    for c in list(range(1, 6)) + [255]:
        m = (label == c).astype(np.uint8)
        if not m.any():
            continue
        cs, _ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        col = (255, 0, 255) if c == 255 else tuple(int(v) for v in COLOURS[c])
        cv2.drawContours(out, cs, -1, col, 2)
    return out


def fit(img: np.ndarray, w: int, h: int) -> np.ndarray:
    """letterbox into w x h (dark grey)."""
    s = min(w / img.shape[1], h / img.shape[0])
    r = cv2.resize(img, (max(1, int(img.shape[1] * s)), max(1, int(img.shape[0] * s))), interpolation=cv2.INTER_AREA)
    out = np.full((h, w, 3), 40, np.uint8)
    out[: r.shape[0], : r.shape[1]] = r
    return out


INDEX_OVERRIDE: Path | None = None          # --index: another plate index (modelled-sea plates)


def samples(mode: str, n: int, seed: int, colour: bool = True, band: float = 1.0, whales_only: bool = True):
    run = WhaleRun(RUN, paths.SPLITS_DIR / "run05.json")
    ids = run.ids("train")
    opts = Options(colour_match=colour, band=band, **MODES[mode])
    comp = Compositor(opts, index=INDEX_OVERRIDE) if INDEX_OVERRIDE else Compositor(opts)
    rng0 = np.random.default_rng(seed)
    order = rng0.permutation(len(ids))
    out = []
    for i in order:
        if len(out) >= n:
            break
        sid = ids[i]
        label = load_label(run, sid, DEFAULT_CACHE)
        if whales_only and not ((label >= 1) & (label <= 5)).any():
            continue
        image, mask = run.image(sid), run.mask(sid)
        rng = np.random.default_rng([seed, len(out)])
        ci, cl, info = comp(image, label, mask, rng)
        out.append(dict(k=len(out), id=sid, image=image, label=label, comp=ci, clabel=cl, info=info, category=run.by_id[sid]["category"]))
    return out


def triplets(out_dir: Path, mode: str, n: int, seed: int, per_sheet: int = 24, tw: int = 420, th: int = 300,
             prefix: str = "composites") -> list[str]:
    ss = samples(mode, n, seed)
    made = []
    for s0 in range(0, len(ss), per_sheet):
        rows = []
        for s in ss[s0:s0 + per_sheet]:
            a = fit(s["image"], tw, th)
            b = fit(s["comp"], tw, th)
            c = fit(outline(s["comp"], s["clabel"]), tw, th)
            row = np.hstack([a, b, c])
            inf = s["info"]
            txt = f"#{s['k']} {s['id']} {s['category']} x{inf.get('scale', 1):.2f} len {inf.get('render_length_share', 0):.2f}->{inf.get('out_length_share', 0):.2f} {inf.get('plate_day', '')}"
            cv2.putText(row, txt, (6, 20), 0, 0.55, (255, 255, 255), 2)
            rows.append(row)
        pairs = [np.hstack([rows[i], rows[i + 1] if i + 1 < len(rows) else np.zeros_like(rows[i])]) for i in range(0, len(rows), 2)]
        sheet = np.vstack(pairs)
        p = out_dir / f"{prefix}_{mode}_{s0 // per_sheet + 1:02d}.jpg"
        cv2.imwrite(str(p), cv2.cvtColor(sheet, cv2.COLOR_RGB2BGR), [cv2.IMWRITE_JPEG_QUALITY, 85])
        made.append(str(p))
    (out_dir / f"{prefix}_{mode}_info.json").write_text(json.dumps([dict(k=s["k"], id=s["id"], category=s["category"], info=s["info"]) for s in ss], indent=1, default=str))
    return made


def grid(out_dir: Path, mode: str, n: int, seed: int, cols: int = 6, tw: int = 400, th: int = 260) -> list[str]:
    ss = samples(mode, n, seed)
    made = []
    for s0 in range(0, len(ss), 60):
        tiles = []
        for s in ss[s0:s0 + 60]:
            t = fit(s["comp"], tw, th)
            cv2.putText(t, f"{s['k']}", (6, 26), 0, 0.9, (255, 255, 0), 2)
            tiles.append(t)
        while len(tiles) % cols:
            tiles.append(np.zeros_like(tiles[0]))
        sheet = np.vstack([np.hstack(tiles[i:i + cols]) for i in range(0, len(tiles), cols)])
        p = out_dir / f"judge_{mode}_{s0 // 60 + 1:02d}.jpg"
        cv2.imwrite(str(p), cv2.cvtColor(sheet, cv2.COLOR_RGB2BGR), [cv2.IMWRITE_JPEG_QUALITY, 85])
        made.append(str(p))
    return made


def pick(out_dir: Path, mode: str, n: int, seed: int, notes: dict, name: str, tw: int = 420, th: int = 300) -> str:
    ss = samples(mode, n, seed)
    rows = []
    for k, note in notes.items():
        s = ss[int(k)]
        row = np.hstack([fit(s["image"], tw, th), fit(s["comp"], tw, th), fit(outline(s["comp"], s["clabel"]), tw, th)])
        cv2.putText(row, f"#{k} {note}", (6, 20), 0, 0.55, (255, 255, 0), 2)
        rows.append(row)
    pairs = [np.hstack([rows[i], rows[i + 1] if i + 1 < len(rows) else np.zeros_like(rows[i])]) for i in range(0, len(rows), 2)]
    p = out_dir / name
    cv2.imwrite(str(p), cv2.cvtColor(np.vstack(pairs), cv2.COLOR_RGB2BGR), [cv2.IMWRITE_JPEG_QUALITY, 85])
    return str(p)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("cmd", choices=["triplets", "grid", "pick"])
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--mode", default="rw_rs", choices=sorted(MODES))
    ap.add_argument("--n", type=int, default=48)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--notes", type=Path, default=None, help="pick: json {grid number: note}")
    ap.add_argument("--name", default="failures.jpg")
    ap.add_argument("--index", type=Path, default=None, help="another plate index (e.g. modelled-sea plates)")
    ap.add_argument("--prefix", default="composites", help="triplets: file name prefix")
    a = ap.parse_args(argv)
    if str(a.out.resolve()).startswith(str(paths.REPO_ROOT.resolve())):
        raise SystemExit("sheets show real photos: write them outside the repository")
    a.out.mkdir(parents=True, exist_ok=True)
    global INDEX_OVERRIDE
    INDEX_OVERRIDE = a.index
    if a.cmd == "triplets":
        print("\n".join(triplets(a.out, a.mode, a.n, a.seed, prefix=a.prefix)))
    elif a.cmd == "grid":
        print("\n".join(grid(a.out, a.mode, a.n, a.seed)))
    else:
        print(pick(a.out, a.mode, a.n, a.seed, json.loads(a.notes.read_text()), a.name))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
