"""Simple / complex triage from cheap signals (splash.py output), shares on training and validation days, and how the
rule relates to what the researcher could click on the two click days.

    PYTHONPATH=. .venv/bin/python -m model_fine_tuning.pose_side.triage [--splash S.json] [--out experiments/pose_side/triage.json]

Rule v0 (thresholds set by eye on a training-day sheet of five splash levels, nothing fitted; validation days only
reported; 0.4 keeps water streaming off a raised fluke and a fin's bow wash as simple, above about 0.8 it is splash):
  simple = view classifier sure (P >= 0.9) of fluke or dorsal fin
           and white water round the whale <= SPLASH x the whale's own area
           and, for fins, white on the whale <= WHITE_ON of its pixels (foam on the back; a fluke's white underside is pigment)
           and the whale not cut by the photo's edge
The keypoint head adds its own gate later (all three points of the part found with confidence, fin not end-on).
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from . import CACHE, CLICK_DAYS, OUT, ROOT
from .realdata import clicks

SPLASH, WHITE_ON, PSURE = 0.40, 0.10, 0.9


def view(r):
    v = r.get("view") or [0.0, 0.0]
    return "fluke" if v[0] >= 0.5 else "dorsal" if v[1] >= 0.5 else "other"


def reasons(r, splash=SPLASH, white_on=WHITE_ON):
    v = r.get("view") or [0.0, 0.0]; out = []
    if max(v) < PSURE:
        out.append("view not sure / other")
    if r["splash"] > splash:
        out.append("splash")
    if r["white_on"] > white_on and v[1] >= 0.5:          # fins only: a fluke's white underside is pigment, not water
        out.append("white on whale")
    if r["photo_edge"]:
        out.append("photo edge")
    return out


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--splash", type=Path, default=CACHE / "splash.json"); ap.add_argument("--out", type=Path, default=OUT / "triage.json")
    a = ap.parse_args(); z = json.loads(a.splash.read_text())
    split = json.loads((ROOT / "model_fine_tuning" / "splits" / "days.json").read_text())
    res = {}
    for name in ("train", "val"):
        R = [r for d in split[name] if d in z for r in z[d].values()]
        tab = {}
        for vw in ("fluke", "dorsal", "other", "all"):
            S = [r for r in R if vw == "all" or view(r) == vw]
            if not S:
                continue
            simple = [not reasons(r) for r in S]
            why = {}
            for r in S:
                for x in reasons(r):
                    why[x] = why.get(x, 0) + 1
            tab[vw] = dict(n=len(S), simple=round(float(np.mean(simple)), 3), complex_reasons={k: round(v / len(S), 3) for k, v in sorted(why.items(), key=lambda t: -t[1])})
        # sensitivity of the share to the splash threshold
        tab["share_simple_by_splash_threshold"] = {str(t): round(float(np.mean([not reasons(r, splash=t) for r in R])), 3) for t in (0.1, 0.2, 0.4, 0.8)}
        res[name] = tab
    # click days: among photos the researcher clicked, are the clicks complete more often for 'simple'?
    comp = {"fluke": {True: [], False: []}, "dorsal": {True: [], False: []}}
    for d in CLICK_DAYS:
        for key, kp in clicks(d).items():
            r = z.get(d, {}).get(key)
            if not r or view(r) not in comp:
                continue
            need = ("fluke_tip_left", "fluke_tip_right", "fluke_notch") if view(r) == "fluke" else ("dorsal_fin_front", "dorsal_fin_back", "dorsal_fin_tip")
            comp[view(r)][not reasons(r)].append(all(n in kp for n in need))
    res["click_days_complete_clicks"] = {vw: {("simple" if s else "complex"): dict(n=len(v), complete=round(float(np.mean(v)), 3) if v else None) for s, v in c.items()} for vw, c in comp.items()}
    res["rule"] = dict(splash=SPLASH, white_on=WHITE_ON, view_sure=PSURE)
    a.out.write_text(json.dumps(res, indent=1)); print(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
