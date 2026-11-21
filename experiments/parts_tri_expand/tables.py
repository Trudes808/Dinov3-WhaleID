"""Markdown tables from render_scores.json (and the other score files when present) -> stdout / tables.md."""
from __future__ import annotations

import json
import sys
from pathlib import Path

OUT = Path(__file__).resolve().parent


def f(x, ci=None):
    if x is None:
        return "-"
    s = f"{x:.3f}"
    if ci:
        s += f" [{ci[0]:.2f}, {ci[1]:.2f}]"
    return s


def render_tables(sc, st):
    lines = [f"### Renders {st} (truth exact; IoU mean [95 % bootstrap], recall / precision means)", ""]
    s = sc["sets"][st]
    for tname, tl in (("fin_hump", "fin + hump (run06 dorsal label)"), ("fin", "fin only (blade sub-label)")):
        lines += [f"Against {tl}:", "", "| group | n | triangle IoU | triangle R / P | A, exact mask | R / P | A, S6h | R / P | A, live | R / P | B, exact mask | R / P | B, S6h | R / P |",
                  "|---|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
        rows = [("all", s["all"])]
        for grp in ("view", "water", "flat_triangle", "animals", "whitewater"):
            rows += [(f"{grp}: {k}", v) for k, v in s[grp].items()]
        for name, b in rows:
            if b.get("n", 0) == 0:
                continue
            t = b[f"tri_{tname}"]
            cells = [name, str(b["n"]), f(t["iou"], t["iou_ci"]), f"{t['recall']:.2f} / {t['precision']:.2f}"]
            for mk in ("gt", "s6h", "r04", "gt_B", "s6h_B"):
                e = b.get(f"exp_{mk}_{tname}")
                cells += [f(e["iou"], e["iou_ci"]), f"{e['recall']:.2f} / {e['precision']:.2f}"] if e else ["-", "-"]
            lines.append("| " + " | ".join(cells) + " |")
        lines.append("")
    return lines


def main() -> int:
    sc = json.loads((OUT / "render_scores.json").read_text())
    lines = [f"Rule A (fixed on fin + hump): {sc['params']}; rule B (fixed on fin only): {sc.get('params_B')}", ""]
    for st in sc["sets"]:
        lines += render_tables(sc, st)
    txt = "\n".join(lines)
    (OUT / "tables_renders.md").write_text(txt + "\n")
    print(txt)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
