"""The cut-outs the Labeling Assistant counts per part (view-classifier gated), with stack and catalogue animal, from
the running web app (GET /api/interpret/<day>/fingerprints), cached in CACHE/points/<day>.json.

    PYTHONPATH=. .venv/bin/python -m model_fine_tuning.cutouts_v3.points train,val
"""
import json, sys, urllib.request
from . import CACHE, day_list

API = "http://127.0.0.1:8765"


def load(day: str) -> dict:
    """part -> {key: (stack, catalogue animal or "")}"""
    f = CACHE / "points" / f"{day}.json"
    z = json.loads(f.read_text())
    return {p: {k: tuple(v) for k, v in d.items()} for p, d in z.items()}


def main():
    (CACHE / "points").mkdir(parents=True, exist_ok=True)
    for day in day_list(sys.argv[1]):
        out = {}
        for part in ("fin", "body", "fluke"):
            r = json.load(urllib.request.urlopen(f"{API}/api/interpret/{day}/fingerprints?part={part}&layout=pca", timeout=900))
            out[part] = {p[0]: [int(p[1]), p[4] or ""] for p in r["points"]}
        (CACHE / "points" / f"{day}.json").write_text(json.dumps(out))
        print(day, {p: len(v) for p, v in out.items()}, flush=True)


if __name__ == "__main__":
    main()
