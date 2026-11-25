"""The two GPU commands of the chain (cluster_day channels with the student, link_scores instances) on the CPU, for the
small mini folders of gpu_mini.py (a few photos: seconds to a minute on the CPU, no wait in the shared GPU-lock queue).
The commands are the unchanged module mains; torch's .cuda() / .half() are made no-ops so the same code runs in float32
on the CPU. gpu_mini.py merge compares the padding photos' rows with the shadow root's GPU rows (the CPU / GPU
difference is printed there and reported in README.md).

    PYTHONPATH=. .venv/bin/python experiments/fused_instances/cpu_steps.py MINI_DAY_DIR
"""
from __future__ import annotations

import sys
from pathlib import Path

import torch

torch.Tensor.cuda = lambda self, *a, **k: self
torch.Tensor.half = lambda self, *a, **k: self
torch.nn.Module.cuda = lambda self, *a, **k: self
torch.nn.Module.half = lambda self, *a, **k: self
torch.set_num_threads(16)

from model_fine_tuning.cluster import cluster_day, link_scores  # noqa: E402

M = Path(sys.argv[1])
cluster_day.main(["channels", "--day-dir", str(M), "--out", str(M / "cluster_v3"), "--sources", "id:experiments/pair_student_g/checkpoints/best.pt",
                  "--local", "", "--candidates", "0", "--device", "cpu"])
sys.argv = ["link_scores", "instances", "--day-dir", str(M)]
link_scores.main() if hasattr(link_scores, "main") else None
