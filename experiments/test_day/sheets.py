"""Contact sheets for the by-eye pass over a test day (output only to an untracked folder: the sheets show crew).

    .venv/bin/python experiments/test_day/sheets.py plain --day-dir DAY --out SHEET_DIR [--per 12 --cols 4 --tile 480]

`plain`: every photo in frame order, the tile labelled with its 3-digit sequence number and the EXIF time; nothing from
any model on it (the blind pass). The library functions (`tile`, `grid`) are used by the other scripts here.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

FONT = "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"


def font(size: int):
    try:
        return ImageFont.truetype(FONT, size)
    except Exception:
        return ImageFont.load_default()


def tile(im: Image.Image, w: int, h: int, text: str = "", border=None, bw: int = 6, text2: str = "") -> Image.Image:
    """The photo fitted into w x h (letterboxed), an optional coloured border and a label strip on top."""
    im = im.convert("RGB"); s = min(w / im.width, h / im.height)
    im = im.resize((max(1, round(im.width * s)), max(1, round(im.height * s))), Image.LANCZOS)
    out = Image.new("RGB", (w, h), (30, 30, 30)); out.paste(im, ((w - im.width) // 2, (h - im.height) // 2))
    d = ImageDraw.Draw(out)
    if border is not None:
        for k in range(bw):
            d.rectangle([k, k, w - 1 - k, h - 1 - k], outline=border)
    if text:
        f = font(max(14, h // 16)); tb = d.textbbox((0, 0), text, font=f)
        d.rectangle([bw, bw, bw + tb[2] + 8, bw + tb[3] + 6], fill=(0, 0, 0)); d.text((bw + 4, bw + 2), text, fill=(255, 255, 0), font=f)
    if text2:
        f = font(max(12, h // 20)); tb = d.textbbox((0, 0), text2, font=f)
        d.rectangle([bw, h - bw - tb[3] - 6, bw + tb[2] + 8, h - bw], fill=(0, 0, 0)); d.text((bw + 4, h - bw - tb[3] - 3), text2, fill=(255, 255, 255), font=f)
    return out


def grid(tiles: list, cols: int, title: str = "", pad: int = 4) -> Image.Image:
    w, h = tiles[0].size; rows = (len(tiles) + cols - 1) // cols; th = 36 if title else 0
    out = Image.new("RGB", (cols * (w + pad) + pad, rows * (h + pad) + pad + th), (12, 12, 12))
    if title:
        ImageDraw.Draw(out).text((8, 6), title, fill=(255, 255, 255), font=font(22))
    for i, t in enumerate(tiles):
        out.paste(t, (pad + (i % cols) * (w + pad), th + pad + (i // cols) * (h + pad)))
    return out


def photos(day_dir: Path) -> list[dict]:
    rows = [json.loads(l) for l in open(day_dir / "images" / "index.jsonl")]
    ts = json.loads((day_dir / "timestamps.json").read_text()) if (day_dir / "timestamps.json").exists() else {}
    out = []
    for r in rows:
        stem = Path(r["image"]).stem; parts = stem.split("_")
        out.append({"stem": stem, "seq": parts[-2], "frame": parts[-1], "path": day_dir / "images" / r["image"],
                    "time": (ts.get(stem) or {}).get("datetime", "")})
    return out


COLOURS = {"whale": (60, 180, 75), "whale_crew": (245, 130, 48), "under": (70, 200, 220), "water": (40, 90, 255),
           "crew_nowhale": (240, 50, 230), "marker_rail": (230, 25, 75), "marker_land": (170, 110, 40), "bow": (255, 225, 25)}
SHORT = {"whale": "whale", "whale_crew": "whale+crew", "under": "under water", "water": "water only", "crew_nowhale": "crew",
         "marker_rail": "rail marker", "marker_land": "land marker", "bow": "bow cleat"}


def truth(root: Path) -> dict:
    import csv
    return {r["seq"]: r for r in csv.DictReader(open(root / "truth.tsv"), delimiter="\t")}


def legend(width: int) -> Image.Image:
    out = Image.new("RGB", (width, 34), (12, 12, 12)); d = ImageDraw.Draw(out); x = 8; f = font(18)
    for k, c in COLOURS.items():
        d.rectangle([x, 8, x + 18, 26], fill=c); d.text((x + 24, 7), SHORT[k], fill=(255, 255, 255), font=f); x += 34 + d.textbbox((0, 0), SHORT[k], font=f)[2]
    return out


def main() -> None:
    p = argparse.ArgumentParser(); p.add_argument("cmd", choices=["plain", "overview"]); p.add_argument("--day-dir", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True); p.add_argument("--per", type=int, default=12); p.add_argument("--cols", type=int, default=4)
    p.add_argument("--tile", type=int, default=480); p.add_argument("--root", type=Path, default=None, help="folder with truth.tsv (overview)")
    p.add_argument("--pipe", type=Path, default=None, help="photos.tsv of the pipeline's answer (overview: shown under the tile)"); a = p.parse_args()
    a.out.mkdir(parents=True, exist_ok=True); P = photos(a.day_dir); w, h = a.tile, round(a.tile * 2 / 3)
    if a.cmd == "overview":
        import csv
        T = truth(a.root); pipe = {r["seq"]: r for r in csv.DictReader(open(a.pipe), delimiter="\t")} if a.pipe else {}
        for k in range(0, len(P), a.per):
            chunk = P[k:k + a.per]; tiles = []
            for x in chunk:
                t = T[x["seq"]]; mk = t["marker"].replace("rail_", "").rstrip("?")
                lab = f"{x['seq']} {SHORT[t['class']]}" + (f" ({mk})" if mk and t["class"] == "marker_rail" else "")
                q = pipe.get(x["seq"]); low = f"model: {q['answer']}" if q else x["time"][11:]
                tiles.append(tile(Image.open(x["path"]), w, h, lab, border=COLOURS[t["class"]], bw=7, text2=low))
            g = grid(tiles, a.cols, f"test day 2026-04-07, photos {chunk[0]['seq']}-{chunk[-1]['seq']} (border = by-eye truth)")
            L = legend(g.width); out = Image.new("RGB", (g.width, g.height + L.height)); out.paste(g, (0, 0)); out.paste(L, (0, g.height))
            out.save(a.out / f"testday_overview_{k // a.per + 1:02d}.jpg", quality=85)
        print(len(P), "photos ->", a.out); return
    for k in range(0, len(P), a.per):
        chunk = P[k:k + a.per]
        tiles = [tile(Image.open(x["path"]), w, h, x["seq"], text2=x["time"][11:]) for x in chunk]
        grid(tiles, a.cols, f"{chunk[0]['seq']}-{chunk[-1]['seq']}").save(a.out / f"plain_{chunk[0]['seq']}.jpg", quality=85)
    print(len(P), "photos,", (len(P) + a.per - 1) // a.per, "sheets ->", a.out)


if __name__ == "__main__":
    main()
