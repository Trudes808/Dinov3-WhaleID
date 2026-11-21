"""Dorsal-fin match embeddings from HW-b6, the public Happywhale dorsal model, raw or fine-tuned on our train days.

HW-b6 = liu-peilin/happywhale_b6_pseudo_charm_round2_1024_v2 (tf_efficientnet_b6_ns at 1024 px, GeM, BatchNorm
embedding of 2304), a re-run of the Happywhale 2022 1st-place recipe. On our days it finds far more same-animal fin
pairs at high precision than the live matcher (cluster/view_match.py) and across both flanks (experiments
moonshot_sizing, LOG 2026-11-18). This module writes it for ANY day folder in the layout of DAY/safe/view_match.npz, so
the Labeling Assistant reads it as it reads that file (Whale-Review-Web backend/labeling.py: WHALE_VIEW_MATCH_FILE for
the ranking of step 2, WHALE_HINT_FIN / WHALE_HINT_FIN_FILE for the merge hints of the Review step).

Per day, reading no catalogue labels: the instances the view classifier says show a dorsal fin (DAY/safe/view_class.json,
P >= its threshold, with the instance box it was computed for) are cut as a square of the box x 1.15 from
DAY/images/<stem>.jpg (2560 px; outside the photo = black), resized to 1024 px, normalised (ImageNet) and embedded in
bfloat16. Then: unit length, centred on the day's mean (the day's light and water; about +0.025 pair AUC, no labels),
unit length again. P(same whale) = sigmoid(a x best cosine + b) for the best cosine between two stacks, (a, b) fitted
by `fit` on catalogue cluster pairs of TRAIN days only (same group, more than 60 s apart: the bench's pairs) and never
on the days a fine-tuned model was trained on.

    # GPU (one shared card: under the lock, <= 15 min per hold; --max-minutes stops starting new days)
    flock /var/tmp/sat3737-parts-cache/gpu_train.lock env OMP_NUM_THREADS=8 .venv/bin/python -m model_fine_tuning.cluster.fin_hw \
        embed --day-dir experiments/days_all/DAY [DAY2 ...] [--models hwb6,hwb6_s16_0,hwb6_s16_1]
    # CPU: the calibration (once per model, train days only) and the page files (embed writes them too when it can)
    .venv/bin/python -m model_fine_tuning.cluster.fin_hw fit --model hwb6
    .venv/bin/python -m model_fine_tuning.cluster.fin_hw write --day-dir experiments/days_all/DAY [...]

Files: DAY/safe/fin_hw/<model>.npz = keys, box, X (float16, the BatchNorm embedding as the net gives it: the cache
`write` and `fit` start from) and DAY/safe/view_match_<model>.npz = the page file: "dorsal|keys", "dorsal|E" (float16,
unit, day-centred), "dorsal|cal" (a, b); the fluke part, "student_in_sample" and "view_model" copied from
DAY/safe/view_match.npz (the live file) so that the ranking switch changes the dorsal fin only; "fin_in_sample" = a
fine-tuned model saw this day's catalogue animals in training; "model" = the model's name.
Models and calibrations: model_fine_tuning/cluster/fin_hw.json (checkpoint paths, the days each fine-tuned run was
trained on, the fitted (a, b) with the days and pairs they come from).
"""
from __future__ import annotations

import argparse
import itertools
import json
import re
import time
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
DAYS = REPO / "experiments" / "days_all"
REG = Path(__file__).with_name("fin_hw.json")
SPLITS = REPO / "model_fine_tuning" / "splits" / "days.json"
RES, MULT = 1024, 1.15
MEAN, STD = (0.485, 0.456, 0.406), (0.229, 0.224, 0.225)
GAP = 60.0                                   # calibration pairs: stacks of one catalogue group more than this many seconds apart


def registry() -> dict:
    return json.loads(REG.read_text())


def _unit(X: np.ndarray) -> np.ndarray:
    X = np.asarray(X, np.float32)
    return X / np.maximum(np.linalg.norm(X, axis=1, keepdims=True), 1e-9)


def day_centred(X: np.ndarray) -> np.ndarray:
    """unit -> minus the day's mean -> unit (what the page compares)."""
    if not len(X):
        return np.zeros((0, X.shape[1] if X.ndim == 2 else 2304), np.float32)
    U = _unit(X)
    return _unit(U - U.mean(0))


def dorsal_instances(day_dir: Path) -> tuple[list[str], dict]:
    """The instance keys the view classifier says show a dorsal fin, and their boxes (2560-px image)."""
    V = json.loads((day_dir / "safe" / "view_class.json").read_text()); thr = float(V.get("threshold", 0.5)); box = V.get("box") or {}
    keys = sorted(k for k, v in V["views"].items() if v[1] >= thr and box.get(k) is not None)
    return keys, {k: [float(x) for x in box[k]] for k in keys}


# ---- the network -------------------------------------------------------------------------------------------------
def _net(ckpt: str | None, public: str):
    """HW-b6 rebuilt from the public checkpoint's tensors (weights_only=True; heads dropped), then a fine-tuned state
    dict on top when given (moonshot_sizing/fin_tune: a state dict of this same module)."""
    import timm
    import torch
    import torch.nn.functional as F

    class GeM(torch.nn.Module):
        def __init__(self, p=3, eps=1e-6):
            super().__init__()
            self.p, self.eps = torch.nn.Parameter(torch.ones(1) * p), eps

        def forward(self, x):
            return F.avg_pool2d(x.clamp(min=self.eps).pow(self.p), (x.size(-2), x.size(-1))).pow(1. / self.p)

    class HappyWhaleB6(torch.nn.Module):     # as moonshot_sizing/public_reid/extract.py (same parameter names)
        def __init__(self):
            super().__init__()
            ck = torch.load(public, map_location="cpu", weights_only=True)["model"]
            self.backbone = timm.create_model("tf_efficientnet_b6.ns_jft_in1k", pretrained=False, num_classes=0, global_pool="")
            self.backbone.load_state_dict({k[9:]: v for k, v in ck.items() if k.startswith("backbone.")}, strict=True)
            self.pool = GeM(); self.pool.load_state_dict({"p": ck["pool.p"]})
            self.bn = torch.nn.BatchNorm1d(2304); self.bn.load_state_dict({k[12:]: v for k, v in ck.items() if k.startswith("embedding.0.")})

        def forward(self, x):
            return self.bn(self.pool(self.backbone.forward_features(x)).flatten(1))

    m = HappyWhaleB6()
    if ckpt:
        m.load_state_dict(torch.load(ckpt, map_location="cpu", weights_only=True), strict=True)
    return m.eval()


def _crop(day_dir: Path, key: str, b: list[float]) -> np.ndarray:
    from PIL import Image
    im = Image.open(day_dir / "images" / f"{key.rsplit('~', 1)[0]}.jpg").convert("RGB")
    cx, cy = (b[0] + b[2]) / 2, (b[1] + b[3]) / 2; s = max(max(b[2] - b[0], b[3] - b[1]) * MULT, 64.0)
    r = (int(round(cx - s / 2)), int(round(cy - s / 2)), int(round(cx + s / 2)), int(round(cy + s / 2)))
    return np.asarray(im.crop(r).resize((RES, RES), Image.BICUBIC if s < RES else Image.LANCZOS), np.uint8)


def cmd_embed(a) -> None:
    import torch
    torch.set_num_threads(2)
    reg = registry(); names = a.models.split(","); dev = torch.device("cuda")
    torch.cuda.set_per_process_memory_fraction(min(1.0, a.mem_gb * 2**30 / torch.cuda.get_device_properties(0).total_memory))
    nets = {n: _net(reg["models"][n].get("ckpt"), reg["public_checkpoint"]).to(dev) for n in names}
    mean, std = torch.tensor(MEAN, device=dev).view(1, 3, 1, 1), torch.tensor(STD, device=dev).view(1, 3, 1, 1)

    class Crops(torch.utils.data.Dataset):
        def __init__(self, d, keys, box):
            self.d, self.keys, self.box = d, keys, box

        def __len__(self):
            return len(self.keys)

        def __getitem__(self, i):
            return torch.from_numpy(_crop(self.d, self.keys[i], self.box[self.keys[i]]).copy()).permute(2, 0, 1)

    t0 = time.time()
    for dd in a.day_dir:
        d = Path(dd)
        if not (d / "safe" / "view_class.json").exists():
            print(f"{d.name}: no DAY/safe/view_class.json (run the safe clusters and view_class first), skipped", flush=True); continue
        todo = [n for n in names if a.redo or not (d / "safe" / "fin_hw" / f"{n}.npz").exists()]
        if not todo:
            print(f"{d.name}: done before", flush=True); continue
        if time.time() - t0 > a.max_minutes * 60:
            print(f"time budget used, stopping before {d.name}", flush=True); break
        keys, box = dorsal_instances(d); out = {n: [] for n in todo}
        if keys:
            with torch.no_grad():
                for x in torch.utils.data.DataLoader(Crops(d, keys, box), batch_size=a.bs, num_workers=a.workers):
                    x = (x.to(dev).float() / 255. - mean) / std
                    for n in todo:
                        with torch.autocast("cuda", dtype=torch.bfloat16):
                            out[n].append(nets[n](x).float().cpu().numpy())
        (d / "safe" / "fin_hw").mkdir(exist_ok=True)
        for n in todo:
            X = np.concatenate(out[n]) if out[n] else np.zeros((0, 2304), np.float32); p = d / "safe" / "fin_hw" / f"{n}.npz"
            tmp = p.with_suffix(".tmp.npz"); np.savez(tmp, keys=np.array(keys), box=np.array([box[k] for k in keys], np.float32).reshape(-1, 4), X=X.astype(np.float16), model=np.array(n)); tmp.replace(p)
            if reg["models"][n].get("cal"):
                write_day(d, n, reg)
        print(f"{d.name}: {len(keys)} dorsal-fin instances x {len(todo)} models, {time.time() - t0:.0f} s, peak GPU {torch.cuda.max_memory_allocated() / 2**30:.2f} GB", flush=True)


# ---- the page file -------------------------------------------------------------------------------------------------
def write_day(d: Path, name: str, reg: dict | None = None) -> Path:
    reg = reg or registry(); m = reg["models"][name]; cal = m.get("cal")
    if not cal:
        raise SystemExit(f"{name}: no calibration yet (run fit)")
    z = np.load(d / "safe" / "fin_hw" / f"{name}.npz"); keys = [str(k) for k in z["keys"]]
    E = day_centred(z["X"].astype(np.float32)).astype(np.float16)
    out = {"model": np.array(name), "dorsal|keys": np.array(keys), "dorsal|E": E, "dorsal|cal": np.array([cal["a"], cal["b"]], np.float32),
           "fin_in_sample": np.array(d.name in m.get("train_days", []))}
    live = d / "safe" / "view_match.npz"
    if live.exists():                        # the fluke part and the student flag as the live file has them: the switch changes the dorsal fin only
        L = np.load(live, allow_pickle=False)
        for k in ("fluke|keys", "fluke|E", "fluke|cal", "student_in_sample", "view_model"):
            out[k] = L[k]
    else:
        V = json.loads((d / "safe" / "view_class.json").read_text())
        out.update({"fluke|keys": np.array([], "<U8"), "fluke|E": np.zeros((0, 32), np.float16), "fluke|cal": np.array([0.0, -10.0], np.float32),
                    "student_in_sample": np.array(False), "view_model": np.array(str(V.get("model")))})
    p = d / "safe" / f"view_match_{name}.npz"; tmp = d / "safe" / f"view_match_{name}.tmp.npz"; np.savez(tmp, **out); tmp.replace(p)
    return p


def cmd_write(a) -> None:
    reg = registry()
    for dd in a.day_dir:
        d = Path(dd)
        for n in a.models.split(","):
            if (d / "safe" / "fin_hw" / f"{n}.npz").exists():
                p = write_day(d, n, reg); print(f"{d.name}: {p.name}", flush=True)
            else:
                print(f"{d.name}: no embedding for {n} (run embed)", flush=True)


# ---- calibration (train days only) ------------------------------------------------------------------------------------
def _major(c: dict):
    e = sorted((c.get("catalog") or {}).items(), key=lambda kv: -kv[1])
    return e[0][0] if e else None


def _group(a: str) -> str:
    m = re.search(r"_(G\d+)A\d+", a)
    return m.group(1) if m else "O"


def cal_pairs(d: Path, name: str) -> tuple[list[float], list[int]]:
    """(best cosine, same animal) for the safe-cluster pairs of one catalogue group, more than GAP s apart, that both
    show a dorsal fin: the bench's pairs (moonshot_sizing/bench.py pairs), cosines as the page computes them."""
    z = np.load(d / "safe" / "fin_hw" / f"{name}.npz"); keys = [str(k) for k in z["keys"]]
    if not keys:
        return [], []
    E = day_centred(z["X"].astype(np.float32)); row = {k: i for i, k in enumerate(keys)}
    C = json.loads((d / "safe" / "clusters.json").read_text())["clusters"]; by = {}
    for c in C:
        idx = [row[k] for k in c["keys"] if k in row]; maj = _major(c)
        if idx and maj:
            by[c["id"]] = (idx, maj, float(c["start"]), float(c["end"]))
    s, y = [], []
    for a_, b_ in itertools.combinations(sorted(by), 2):
        (ia, ma, sa, ea), (ib, mb, sb, eb) = by[a_], by[b_]
        if _group(ma) != _group(mb) or max(0.0, max(sa, sb) - min(ea, eb)) <= GAP:
            continue
        s.append(float((E[ia] @ E[ib].T).max())); y.append(int(ma == mb))
    return s, y


def cmd_fit(a) -> None:
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import roc_auc_score
    reg = registry(); m = reg["models"][a.model]; sp = json.loads(SPLITS.read_text())
    days = [x for x in sp["train"] if x not in set(m.get("train_days", []))]      # train days only, never a day the model was trained on
    S, Y, used = [], [], []
    for x in days:
        if (DAYS / x / "safe" / "fin_hw" / f"{a.model}.npz").exists():
            s, y = cal_pairs(DAYS / x, a.model); S += s; Y += y; used.append(x)
    S, Y = np.array(S), np.array(Y)
    lr = LogisticRegression(C=1e6, max_iter=1000).fit(S[:, None], Y)
    m["cal"] = {"a": round(float(lr.coef_[0, 0]), 4), "b": round(float(lr.intercept_[0]), 4), "days": used, "pairs": int(len(Y)), "same": int(Y.sum()),
                "auc": round(float(roc_auc_score(Y, S)), 4), "fitted": time.strftime("%Y-%m-%d"),
                "how": "Platt (logistic, no penalty) of same animal on the best day-centred cosine, safe-cluster pairs of one catalogue group > 60 s apart that both show a dorsal fin"}
    REG.write_text(json.dumps(reg, indent=1) + "\n")
    print(f"{a.model}: a {m['cal']['a']} b {m['cal']['b']} on {len(Y)} pairs ({int(Y.sum())} same) of {len(used)} train days, AUC {m['cal']['auc']}", flush=True)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter); sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("embed"); p.add_argument("--day-dir", nargs="+", required=True); p.add_argument("--models", default="hwb6"); p.add_argument("--bs", type=int, default=8)
    p.add_argument("--workers", type=int, default=6); p.add_argument("--mem-gb", type=float, default=8.0); p.add_argument("--max-minutes", type=float, default=12.0)
    p.add_argument("--redo", action="store_true"); p.set_defaults(fn=cmd_embed)
    p = sub.add_parser("write"); p.add_argument("--day-dir", nargs="+", required=True); p.add_argument("--models", default="hwb6"); p.set_defaults(fn=cmd_write)
    p = sub.add_parser("fit"); p.add_argument("--model", required=True); p.set_defaults(fn=cmd_fit)
    a = ap.parse_args(argv); a.fn(a)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
