from __future__ import annotations

import copy
import os
from typing import Any

import yaml


def _merge_dict(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    merged = copy.deepcopy(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = _merge_dict(merged[key], value)
        else:
            merged[key] = copy.deepcopy(value)
    return merged


def load_config(path: str) -> dict[str, Any]:
    """Load YAML configuration, including an optional parent configuration."""
    config_path = os.path.abspath(path)
    with open(config_path) as config_file:
        config = yaml.safe_load(config_file) or {}

    parent = config.pop("defaults_from", None)
    if parent is None:
        return config

    parent_path = os.path.join(os.path.dirname(config_path), parent)
    return _merge_dict(load_config(parent_path), config)
