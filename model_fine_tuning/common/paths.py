"""Default locations. Everything can be overridden by environment variables so the code
runs unchanged on another machine.

    DINOV3_REPO       facebookresearch/dinov3 checkout (hubconf.py at its root)
    DINOV3_WEIGHTS    backbone checkpoint (.pth) for the chosen architecture
    WHALE_RUN_DIR     a Whale-Image-Data-Factory run (contains dataset/manifest.jsonl)
    WHALE_MESH_OBJ    the humpback mesh the run was rendered with (mesh frame, cm)
"""

from __future__ import annotations

import os
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
FINE_TUNING_DIR = REPO_ROOT / "model_fine_tuning"
EXPERIMENTS_DIR = REPO_ROOT / "experiments"
ASSETS_DIR = FINE_TUNING_DIR / "assets"
SPLITS_DIR = FINE_TUNING_DIR / "splits"

DINOV3_REPO = Path(os.environ.get("DINOV3_REPO", Path.home() / "dinov3"))
DINOV3_WEIGHTS = Path(os.environ.get("DINOV3_WEIGHTS", DINOV3_REPO / "weights" / "dinov3_vitb16_pretrain_lvd1689m-73cec8be.pth"))
WHALE_RUN_DIR = Path(os.environ.get("WHALE_RUN_DIR", Path.home() / "NOAA_whales" / "Whale-Image-Data-Factory" / "data" / "renders" / "run01"))
WHALE_MESH_OBJ = Path(os.environ.get("WHALE_MESH_OBJ", Path.home() / "NOAA_whales" / "Whale-Image-Data-Factory" / "third_party_assets" / "humpback_k9239" / "converted" / "SM_Humpback.obj"))
BODY_TABLE = ASSETS_DIR / "humpback_body_table.npz"
TEST_DATA_DIR = Path(os.environ.get("WHALE_TEST_DATA", Path.home() / "NOAA_whales" / "test_data_do_not_use"))
