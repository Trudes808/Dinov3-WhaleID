"""Tape colour, round 2: one sheet of every real rail marker's tape band (both unculled days) + the three crew photos of
2026-03-30 with a band, with the measured values, the truth colour, the reader's call and the new call. The sheet shows
the boat and crew: written to the scratchpad / review_sheets only, never into the repo.

    .venv/bin/python experiments/sam3_labeller/colour2_sheet.py --real SCRATCH/colour2/bands_real.json --out SHEET.jpg
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import colour2_measure as CM  # noqa: E402
import colour2_rules as CR  # noqa: E402

TW, TH, CAP = 420, 300, 92
BGR = {"red": (150, 120, 235), "yellow": (60, 200, 230), "green": (60, 170, 40), "blue": (220, 120, 30), "other": (160, 160, 160), "none": (90, 90, 90)}


def main() -> None:
    ap = argparse.ArgumentParser(); ap.add_argument("--real", required=True); ap.add_argument("--out", required=True); a = ap.parse_args()
    R = json.loads(Path(a.real).read_text())
    proto = CR.fit("lab_nn", CR.labelled_bands(R))
    tiles = []
    for r in R:
        day = r["day"]; stems = {CM.seq_of(p.stem): p.stem for p in (CM.RUNS / "days" / day / "images").glob("*.jpg")}
        stem = stems[r["seq"]]
        photo = cv2.imread(str(CM.RUNS / "days" / day / "images" / f"{stem}.jpg"))
        lab = cv2.imread(str(CM.RUNS / "days" / day / "scene3" / "scene" / f"{stem}.png"), cv2.IMREAD_UNCHANGED)
        H, W = photo.shape[:2]; s = W / lab.shape[1]
        bands = r["bands"]; old = CR.pick(bands, "largest"); new = CR.pick(bands, "centre")
        newcall = CR.classify(new, "lab_nn", proto)
        x0 = min(b["bbox"][0] for b in bands); y0 = min(b["bbox"][1] for b in bands)
        x1 = max(b["bbox"][2] for b in bands); y1 = max(b["bbox"][3] for b in bands)
        cx, cy = (x0 + x1) / 2 * s, (y0 + y1) / 2 * s; half = max((x1 - x0) * s, (y1 - y0) * s * TW / TH) * 0.75 + 60
        hx, hy = half, half * TH / TW
        X0, X1 = int(max(0, cx - hx)), int(min(W, cx + hx)); Y0, Y1 = int(max(0, cy - hy)), int(min(H, cy + hy))
        tape = np.where(lab == 255, 0, lab) == 4; n, cc = cv2.connectedComponents(tape.astype(np.uint8), connectivity=8)
        vis = photo.copy()
        for b in bands:
            bx, by_ = b["bbox"][0], b["bbox"][1]
            ids = np.unique(cc[by_:b["bbox"][3] + 1, bx:b["bbox"][2] + 1]); ids = ids[ids > 0]
            k = max(ids, key=lambda i: (cc[by_:b["bbox"][3] + 1, bx:b["bbox"][2] + 1] == i).sum())
            m = cv2.resize((cc == k).astype(np.uint8), (W, H), interpolation=cv2.INTER_NEAREST)
            cnts, _ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            col = (255, 255, 0) if b is new else (255, 255, 255)
            cv2.drawContours(vis, cnts, -1, col, 6 if b is new else 3)
            if b is old:
                cv2.drawContours(vis, cnts, -1, (0, 0, 255), 2)
        crop = vis[Y0:Y1, X0:X1]; sc = min(TW / crop.shape[1], TH / crop.shape[0])
        crop = cv2.resize(crop, (max(1, int(crop.shape[1] * sc)), max(1, int(crop.shape[0] * sc))), interpolation=cv2.INTER_AREA)
        tile = np.zeros((TH + CAP, TW, 3), np.uint8); tile[:crop.shape[0], :crop.shape[1]] = crop
        L, aa, bb = CR.lab_ab(new); hab = CR.feat(new, "lab_nn")
        ok_old = r["old_call"] == r["truth"]; ok_new = newcall == r["truth"]
        t1 = f"{day[5:]} #{r['seq']} {'CREW ' if r['role'] == 'crew' else ''}truth {r['truth']}"
        t2 = f"old {r['old_call']} {'ok' if ok_old else 'WRONG'} -> new {newcall} {'ok' if ok_new else 'WRONG'}  bands {len(bands)}"
        t3 = f"read: hsvH {new['hue_reader'] if new['hue_reader'] is not None else '-'} S {new['S_mean']} V {new['V_mean']}  ctr {new['dist']:.2f}"
        t4 = f"Lab L {L:.0f} a {aa:+.0f} b {bb:+.0f} hab {hab:.0f}  tube {new['white_rgb']}"
        y = TH + 16
        for t, c in ((t1, (255, 255, 255)), (t2, (80, 255, 80) if ok_new and not ok_old else (0, 200, 255) if not ok_new else (220, 220, 220)), (t3, (200, 200, 200)), (t4, (200, 200, 200))):
            cv2.putText(tile, t, (5, y), cv2.FONT_HERSHEY_SIMPLEX, 0.42, c, 1, cv2.LINE_AA); y += 19
        cv2.rectangle(tile, (TW - 26, TH + 4), (TW - 6, TH + 24), BGR.get(r["truth"], (90, 90, 90)), -1)
        tiles.append(tile)
    head = np.zeros((70, TW * 6, 3), np.uint8)
    cv2.putText(head, "Tape bands of every real rail marker (2026-04-07, 2026-03-30) + 3 crew photos. Cyan outline = band the new rule reads "
                "(nearest frame centre), red inner line = band the old reader read (largest), white = other band.", (8, 24),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
    cv2.putText(head, "hsvH = old reader's hue (S>=.25,V>=.2); Lab a/b = CIE a*, b* of the band mean; hab = Lab hue angle; new protos "
                + ", ".join(f"{c} {v:.0f}" for c, v in proto.items()) + "; tube = white tube RGB beside the band", (8, 50),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
    tiles += [np.zeros_like(tiles[0])] * ((-len(tiles)) % 6)
    img = np.vstack([head] + [np.hstack(tiles[i:i + 6]) for i in range(0, len(tiles), 6)])
    Path(a.out).parent.mkdir(parents=True, exist_ok=True); cv2.imwrite(a.out, img, [cv2.IMWRITE_JPEG_QUALITY, 88]); print(a.out, img.shape)


if __name__ == "__main__":
    main()
