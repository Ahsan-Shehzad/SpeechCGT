from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from typing import Any

import yaml

from .constants import DEFAULT_CONFIG

TOP_LEVEL_NAMESPACES = tuple(DEFAULT_CONFIG.keys())


def _deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    merged = deepcopy(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = _deep_merge(merged[key], value)
        else:
            merged[key] = value
    return merged


def validate_config(config: dict[str, Any]) -> None:
    missing = [key for key in TOP_LEVEL_NAMESPACES if key not in config]
    if missing:
        raise ValueError(f"Missing config namespaces: {missing}")


def load_config(config_path: str | Path | None = None) -> dict[str, Any]:
    config = deepcopy(DEFAULT_CONFIG)
    if config_path is not None:
        with Path(config_path).open("r", encoding="utf-8") as handle:
            loaded = yaml.safe_load(handle) or {}
        config = _deep_merge(config, loaded)
    validate_config(config)
    return config


def save_config(config: dict[str, Any], output_path: str | Path) -> None:
    validate_config(config)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        yaml.safe_dump(config, handle, sort_keys=False)

