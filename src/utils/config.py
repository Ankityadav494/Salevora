"""Configuration management for Salevora."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

BASE_DIR = Path(__file__).resolve().parents[2]
CONFIG_PATH = BASE_DIR / "config.yaml"


def load_config(path: Path | None = None) -> dict[str, Any]:
    cfg_path = path or CONFIG_PATH
    if not cfg_path.exists():
        return {}
    with open(cfg_path, encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def get_data_path(key: str = "processed_data_path") -> Path:
    cfg = load_config()
    rel = cfg.get("data", {}).get(key, "data/processed/live_sales.csv")
    return BASE_DIR / rel
