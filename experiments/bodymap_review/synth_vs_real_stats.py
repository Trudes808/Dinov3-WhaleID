"""Read-only: what the run05 renders hold, shot by shot (from dataset/meta), next to the same quantities on real photos
(bodymap_stats.py output, training + validation days): what a frame shows, how large the whale is in it, from where it
is seen, white water, light, more than one animal.

    .venv/bin/python experiments/bodymap_review/synth_vs_real_stats.py REAL_STATS.json CLICKS.json [OUT.json]
"""
import glob, json, sys
from collections import Counter
from multiprocessing import Pool
from pathlib import Path
import numpy as np
SYN = Path("/home/sat3737/NOAA_whales/Whale-Image-Data-Factory/data/renders/run05/dataset")


def one(f):
    m = json.load(open(f))
    if m.get("negative") or not (m.get("visible") or {}).get("visible_pixels"):
        return {"neg": True}
    v = m["visible"]; kp = m["keypoints"]; st = lambda n: (kp.get(n) or {}).get("state")
    x0, y0, x1, y1 = v["bbox_xyxy"]; P = m.get("parts") or {}; o = m.get("orientation") or {}; ww = m.get("whitewater") or {}; sun = m.get("sun") or {}
    vis = {k: P[k]["amodal_pixels"] * P[k]["visible_fraction"] for k in P}
    r = {"cat": m["category"], "px": v["visible_pixels"], "long": max(x1 - x0, y1 - y0) / 1280.0, "border": bool(v.get("mask_touches_border")), "fill": v["visible_fraction_of_image"],
         "fluke_px": vis.get("fluke", 0.0), "share_fluke": vis.get("fluke", 0.0) / max(v["visible_pixels"], 1), "fin_tip": st("dorsal_fin_tip") == "visible", "tips": (st("fluke_tip_left") == "visible") + (st("fluke_tip_right") == "visible"),
         "notch": st("fluke_notch") == "visible", "head_share": vis.get("head", 0.0) / max(v["visible_pixels"], 1), "bearing": o.get("relative_bearing_deg"), "elev": o.get("view_elevation_deg"), "dist": o.get("distance_m"),
         "ww": ww.get("amount", 0.0) if ww else 0.0, "impact": ww.get("impact", 0.0) if ww else 0.0, "splash": bool(m.get("splash")), "sun_rel": sun.get("relative_azimuth_deg"), "sun_el": sun.get("elevation_deg"), "overcast": bool(m.get("overcast")),
         "sea": m.get("sea_state"), "calf_px": v.get("companion_pixels", 0) if m.get("companion") else 0, "frac_whale": m.get("visible_fraction_of_whale"), "ppm": m.get("pixels_per_metre_at_whale")}
    if r["tips"] == 2:
        r["from_behind"] = kp["fluke_tip_left"]["uv"][0] < kp["fluke_tip_right"]["uv"][0]
    return r


def q(v, p=(10, 50, 90)): return " / ".join(f"{x:.2f}" for x in np.percentile(np.array(v, float), p))


if __name__ == "__main__":
    with Pool(12) as pool:
        S = pool.map(one, sorted(glob.glob(str(SYN / "meta" / "*.json"))), chunksize=64)
    neg = sum(1 for r in S if r.get("neg")); S = [r for r in S if not r.get("neg")]
    R = [r for r in json.load(open(sys.argv[1])) if "inst" in r and r.get("blobs", 0) > 0]; I = [(r, x) for r in R for x in r["inst"] if not x.get("not_whale") and x.get("vc")]
    print(f"renders: {len(S)} shots with a visible whale ({neg} without) | real: {len(R)} photos, {len(I)} whale instances with a view answer")
    # what a frame shows. Renders: a fluke counts when a tip or the notch is visible or the fluke is >= 30 % of the visible whale; a dorsal fin when its tip is visible
    def syn_view(r):
        f = r["tips"] >= 1 or r["notch"] or r["share_fluke"] >= 0.3; return "fluke" if f else "dorsal fin" if r["fin_tip"] else "neither"
    cs = Counter(syn_view(r) for r in S); cr = Counter("fluke" if x["vc"][0] >= 0.5 else "dorsal fin" if x["vc"][1] >= 0.5 else "neither" for _, x in I)
    print("\nwhat it shows            renders   real"); [print(f"   {k:20s} {cs[k] / len(S):6.1%}  {cr[k] / len(I):6.1%}") for k in ("fluke", "dorsal fin", "neither")]
    both = sum(1 for r in S if (r["tips"] >= 1 or r["notch"]) and r["fin_tip"]); print(f"   renders with a fluke point AND the fin tip visible: {both / len(S):.1%}")
    fb = [r["from_behind"] for r in S if "from_behind" in r]; C = json.load(open(sys.argv[2])); F = [r for r in C if "fluke_tip_left" in r["pts"] and "fluke_tip_right" in r["pts"]]
    print(f"\nflukes with both tips in view: renders {len(fb)} ({len(fb) / len(S):.1%} of shots), seen from behind (the whale's left tip on the left) {np.mean(fb):.1%} | real clicked flukes {len(F)}, from behind {np.mean([r['pts']['fluke_tip_left']['c'][0] < r['pts']['fluke_tip_right']['c'][0] for r in F]):.1%}")
    rl = [max(x["long_px"] for x in r["inst"]) / r["s"] / 2560.0 for r in R]
    print(f"\nlength of the visible whale as a share of the frame's width (10 / 50 / 90 pct): renders {q([r['long'] for r in S])} | real {q(rl)}")
    print(f"   under a tenth of the frame: renders {np.mean([r['long'] < 0.1 for r in S]):.1%}, real {np.mean(np.array(rl) < 0.1):.1%} | over half the frame: renders {np.mean([r['long'] > 0.5 for r in S]):.1%}, real {np.mean(np.array(rl) > 0.5):.1%}")
    print(f"   visible whale pixels at the size the networks see (long side 1280): renders median {np.median([r['px'] for r in S]):.0f}, real median {np.median([sum(x['area'] for x in r['inst']) / r['s'] ** 2 / 4 for r in R]):.0f}")
    print(f"   whale touches the frame border: renders {np.mean([r['border'] for r in S]):.1%}")
    print("\nrender pose categories:", ", ".join(f"{k} {v / len(S):.0%}" for k, v in Counter(r["cat"] for r in S).most_common()))
    b = np.array([r["bearing"] for r in S if r["bearing"] is not None]) % 360; rel = np.minimum(b, 360 - b)          # 0 = camera ahead of the whale, 180 = behind
    print(f"camera around the whale (0 = ahead of it, 90 = abeam, 180 = behind): ahead (<45) {np.mean(rel < 45):.0%}, abeam (45-135) {np.mean((rel >= 45) & (rel <= 135)):.0%}, behind (>135) {np.mean(rel > 135):.0%}")
    for k in ("fluke", "dorsal fin"):
        bb = np.array([r["bearing"] for r in S if syn_view(r) == k and r["bearing"] is not None]) % 360; rr = np.minimum(bb, 360 - bb)
        print(f"   {k} shots: ahead {np.mean(rr < 45):.0%}, abeam {np.mean((rr >= 45) & (rr <= 135)):.0%}, behind {np.mean(rr > 135):.0%}")
    print(f"camera elevation above the horizon line of the whale, deg (10 / 50 / 90): {q([r['elev'] for r in S if r['elev'] is not None])} | distance m: {q([r['dist'] for r in S if r['dist'] is not None])}")
    print(f"white water amount 0-1 (10 / 50 / 90): {q([r['ww'] for r in S])}, none {np.mean([r['ww'] <= 0.01 for r in S]):.0%}, heavy (> 0.6) {np.mean([r['ww'] > 0.6 for r in S]):.0%} | impact splash {np.mean([r['impact'] > 0.05 for r in S]):.0%} | splash object {np.mean([r['splash'] for r in S]):.0%}")
    sr = np.array([r["sun_rel"] for r in S if r["sun_rel"] is not None]) % 360; srel = np.minimum(sr, 360 - sr)
    print(f"light: overcast {np.mean([r['overcast'] for r in S]):.0%} | sun elevation deg {q([r['sun_el'] for r in S if r['sun_el'] is not None])} | sun azimuth against the camera's view (10 / 50 / 90): {q(srel)}")
    print(f"sea: {dict(Counter(r['sea'] for r in S).most_common())}")
    print(f"a calf in view (>= 200 px): {np.mean([r['calf_px'] >= 200 for r in S]):.1%} | two adults: 0 | boats, people, drones, rods: 0")
    print(f"share of the whale's surface that is visible (10 / 50 / 90): {q([r['frac_whale'] for r in S if r['frac_whale'] is not None])}")
    if len(sys.argv) > 3: json.dump(S, open(sys.argv[3], "w"))
