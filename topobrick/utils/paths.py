from __future__ import annotations

import os
from typing import Any


def repository_root() -> str:
    return os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))


def resolve_path(*parts: str) -> str:
    path = os.path.join(*parts)
    if not os.path.isabs(path):
        path = os.path.join(repository_root(), path)
    return os.path.abspath(path)


def ensure_directory(path: str) -> str:
    os.makedirs(path, exist_ok=True)
    return path


def resolve_processed_root(path: str | None = None) -> str:
    return path or os.environ.get(
        "TOPOBRICK_DATA_ROOT", resolve_path("data", "processed")
    )


def processed_data_directory(config: dict[str, Any], dataset: str | None = None) -> str:
    subdirectory = dataset or config["dataset"]["processed_subdir"]
    return resolve_path(config["paths"]["processed_root"], subdirectory)


def raw_data_directory(config: dict[str, Any], dataset: str | None = None) -> str:
    subdirectory = dataset or config["dataset"]["raw_subdir"]
    return resolve_path(config["paths"]["raw_root"], subdirectory)
