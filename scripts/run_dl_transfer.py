#!/usr/bin/env python
"""Run the three-condition (A / B / C-head / C-full) 1D-CNN comparison.

Usage:
    uv run python scripts/run_dl_transfer.py

Set BIESL_DL_EPOCHS to override the per-fold epoch budget (default 60).
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.dl_train import run_three_way_dl   # noqa: E402


if __name__ == "__main__":
    epochs = int(os.environ.get("BIESL_DL_EPOCHS", "60"))
    run_three_way_dl(epochs=epochs)
