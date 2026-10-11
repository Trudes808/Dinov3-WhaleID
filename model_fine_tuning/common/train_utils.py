"""Small training helpers shared by both trainers."""

from __future__ import annotations

import json
import math
import random
import time
from pathlib import Path

import numpy as np
import torch


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def lr_at(step: int, total: int, warmup: int, base: float, floor: float = 0.02) -> float:
    if step < warmup:
        return base * (step + 1) / max(warmup, 1)
    progress = min(1.0, (step - warmup) / max(total - warmup, 1))
    return base * (floor + (1 - floor) * 0.5 * (1 + math.cos(math.pi * progress)))


def apply_lr(optimizer: torch.optim.Optimizer, factor: float) -> None:
    for group in optimizer.param_groups:
        group["lr"] = group["base_lr"] * factor


class Logger:
    def __init__(self, out_dir: Path):
        self.out_dir = Path(out_dir)
        self.out_dir.mkdir(parents=True, exist_ok=True)
        self.path = self.out_dir / "log.jsonl"
        self.t0 = time.time()

    def log(self, **record):
        record["elapsed_s"] = round(time.time() - self.t0, 1)
        with open(self.path, "a") as handle:
            handle.write(json.dumps(record) + "\n")
        printable = {k: (round(v, 4) if isinstance(v, float) else v) for k, v in record.items() if not isinstance(v, (dict, list))}
        print(json.dumps(printable), flush=True)


def save_checkpoint(path: Path, model: torch.nn.Module, config: dict, extra: dict | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    state = {k: v for k, v in model.state_dict().items()}
    torch.save({"model": state, "config": config, **(extra or {})}, path)


def load_checkpoint(path: Path, map_location="cpu") -> dict:
    return torch.load(path, map_location=map_location, weights_only=False)


def count_params(model: torch.nn.Module) -> dict:
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return {"total_M": round(total / 1e6, 2), "trainable_M": round(trainable / 1e6, 2)}


def move(batch: dict, device) -> dict:
    return {k: (v.to(device, non_blocking=True) if torch.is_tensor(v) else v) for k, v in batch.items()}
