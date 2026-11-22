"""Rule A3: choose the labeller's phrasing (single or a pair's union) and threshold per extra class on non-test data:
the boat renders (exact labels) and the by-eye photo tags of the real dev photos (val days + nw photos).

    .venv/bin/python experiments/sam3_labeller/dev_choose.py     -> OUT/dev/choice.json, OUT/dev/choice_table.md
"""
from __future__ import annotations

import itertools
import json
import sys
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).parent))
import lab_common as L  # noqa: E402

GROUPS = {"rail": ["railing", "boat railing", "metal pole"], "tape": ["colored tape", "tape", "colored band"],
          "deck": ["boat deck", "cleat", "boat bow", "bollard", "mooring bollard"], "land": ["island", "land", "mountain"]}
RENDER_CLS = {"rail": [3], "tape": [4], "deck": [5, 6], "land": [8]}
TAG = {"rail": "R", "tape": "T", "deck": "D", "land": "L"}
AREA = {"rail": 0.005, "tape": 0.0005, "deck": 0.005, "land": 0.005}
FP_AREA = {"rail": 0.005, "tape": 0.0005, "deck": 0.005, "land": 0.005}
THR = [0.3, 0.4, 0.5]
RENDERS = {"b": L.NW / "Whale-Image-Data-Factory/data/renders/r06r_boat20b_look/dataset", "a": L.NW / "Whale-Image-Data-Factory/data/renders/r06r_boat20_look/dataset"}
PORDER = L.PRM


def candidates(c):
    ps = GROUPS[c]
    return [(p,) for p in ps] + list(itertools.combinations(ps, 2))


def render_stats():
    """per class, candidate, threshold: pooled inter / union on renders with the class, FP renders without it."""
    raw = L.OUT / "raw" / "renders"; pidx = {p: i for i, p in enumerate(json.loads((raw / "_prompts.json").read_text()))}
    acc = {}
    ids = [l.split("\t")[0] for l in open(L.OUT / "lists" / "renders.tsv")]
    for rid in ids:
        lab = np.array(Image.open(RENDERS[rid[0]] / "scene" / f"{rid[1:]}.png"))
        hw = lab.shape[:2]; rec = L.load_raw(raw / f"{rid}.npz")
        small = (hw[0] // 2, hw[1] // 2); lab = cv2.resize(lab, (small[1], small[0]), interpolation=cv2.INTER_NEAREST)
        masks = {(p, t): L.class_mask(rec, [pidx[p]], t, small) for c in GROUPS for p in GROUPS[c] for t in THR}
        for c in GROUPS:
            truth = np.isin(lab, RENDER_CLS[c]); has = truth.mean() >= 0.001
            for cand in candidates(c):
                for t in THR:
                    m = np.zeros(small, bool)
                    for p in cand:
                        m |= masks[(p, t)]
                    a = acc.setdefault((c, cand, t), {"i": 0, "u": 0, "n_has": 0, "n_not": 0, "fp": 0})
                    if has:
                        a["i"] += int((m & truth).sum()); a["u"] += int((m | truth).sum()); a["n_has"] += 1
                    else:
                        a["n_not"] += 1; a["fp"] += int(m.mean() >= FP_AREA[c])
    return acc


def real_stats():
    order = json.loads((L.OUT / "dev" / "tag_order.json").read_text()); rows = json.loads((L.OUT / "dev" / "shares.json").read_text())
    tags = {int(l.split()[0]): l.split()[1] for l in open(L.OUT / "dev" / "tags.txt") if not l.startswith("#")}
    out = {}
    # union of a pair: shares are per phrasing; the union share is bounded below by the max, so for the >= area test
    # the pair is counted painted when either phrasing reaches the area (the photo-level test is an OR)
    for c in GROUPS:
        for cand in candidates(c):
            for t in THR:
                yes = no = ry = fn = 0
                for i, k in enumerate(order):
                    painted = any(rows[k][f"{p}@{t}"] >= AREA[c] for p in cand)
                    if TAG[c] in tags[i]:
                        yes += 1; ry += painted
                    else:
                        no += 1; fn += painted
                out[(c, cand, t)] = {"yes": yes, "no": no, "Rc": ry / max(1, yes), "Fc": fn / max(1, no)}
    return out


def main() -> None:
    R = render_stats(); Q = real_stats(); res = {}; lines = ["| class | candidate | thr | render IoU | render FP | real Rc | real Fc | J |", "|---|---|---|---|---|---|---|---|"]
    for c in GROUPS:
        rows = []
        for cand in candidates(c):
            for t in THR:
                a = R[(c, cand, t)]; q = Q[(c, cand, t)]
                iou = a["i"] / max(1, a["u"]); fp = a["fp"] / max(1, a["n_not"])
                use_real = q["yes"] >= 5
                J = iou - fp + ((q["Rc"] - q["Fc"]) if use_real else 0.0)
                rows.append({"cand": cand, "thr": t, "iou_r": iou, "fp_r": fp, "n_r_has": a["n_has"], "n_r_not": a["n_not"], **q, "J": J})
        best = max(r["J"] for r in rows)
        tied = [r for r in rows if r["J"] >= best - 0.02]
        tied.sort(key=lambda r: (-r["thr"], len(r["cand"]), min(PORDER.index(p) for p in r["cand"])))
        res[c] = {"chosen": {"phrasings": list(tied[0]["cand"]), "thr": tied[0]["thr"], "J": tied[0]["J"]}, "best_J": best,
                  "tied": [[list(r["cand"]), r["thr"]] for r in tied], "rows": [{**r, "cand": list(r["cand"])} for r in rows]}
        for r in sorted(rows, key=lambda r: -r["J"]):
            lines.append(f"| {c} | {' + '.join(r['cand'])} | {r['thr']} | {r['iou_r']:.3f} | {r['fp_r']:.2f} | {r['Rc']:.2f} ({r['yes']}) | {r['Fc']:.2f} ({r['no']}) | {r['J']:.3f} |")
        print(c, "->", res[c]["chosen"], "n renders with class", rows[0]["n_r_has"], "without", rows[0]["n_r_not"], "tagged yes", rows[0]["yes"])
    (L.OUT / "dev" / "choice.json").write_text(json.dumps(res, indent=1)); (L.OUT / "dev" / "choice_table.md").write_text("\n".join(lines) + "\n")


if __name__ == "__main__":
    main()
