"""Load cached topology subgraphs and aligned forecast windows."""

from __future__ import annotations

import os
from typing import Any, Dict, Sequence

import pandas as pd
import torch

from topobrick.forecast.data.timeseries import TimeSeriesSlicer
from topobrick.forecast.inputs.selection import VALID_SELECTION_MODES
from topobrick.utils.paths import resolve_path
from topobrick.utils.progress import log


class ForecastDataset:
    def __init__(
        self,
        config: Dict[str, Any],
        dataset_name: str,
        split: str,
        horizon: int,
        windows_filename: str | None = None,
        exogenous_selection: str = "all",
        max_exogenous: int | None = None,
        target_node_id_subset: Sequence[str] | None = None,
    ):
        if exogenous_selection not in VALID_SELECTION_MODES:
            raise ValueError(
                f"exogenous_selection must be one of {sorted(VALID_SELECTION_MODES)}"
            )

        self.config = config
        self.dataset_name = dataset_name
        self.split = split
        self.horizon = int(horizon)
        self.lookback = int(config["L"])
        self.exogenous_selection = exogenous_selection
        self.max_exogenous = max_exogenous

        processed_dir = resolve_path(config["paths"]["processed_root"], dataset_name)
        cache_filename = os.environ.get("KG_CACHE_FILENAME", "kg_cache_pull.pt")
        cache_path = os.path.join(processed_dir, cache_filename)
        if not os.path.exists(cache_path):
            raise FileNotFoundError(
                f"no topology cache at {cache_path}; run the unified workflow "
                "through the cache stage first"
            )
        cache = torch.load(cache_path, weights_only=False)
        self.subgraphs = cache["subgraphs"]
        log(
            f"loaded topology cache ({len(self.subgraphs)} subgraphs) "
            f"exogenous_selection={self.exogenous_selection} "
            f"max_exogenous={self.max_exogenous}"
        )

        filename = windows_filename or f"windows_H{self.horizon}_{split}.parquet"
        windows_path = os.path.join(processed_dir, filename)
        self.windows = pd.read_parquet(windows_path)
        log(f"loaded {len(self.windows)} {split} windows from {windows_path}")
        if target_node_id_subset is not None:
            wanted = set(target_node_id_subset)
            self.windows = self.windows[
                self.windows["target_node_id"].isin(wanted)
            ].reset_index(drop=True)
            log(f"  retained {len(self.windows)} windows for {len(wanted)} targets")

        target_sensor_ids = set(self.windows["sensor_id"].dropna().unique().tolist())
        selected_targets = (
            set(target_node_id_subset)
            if target_node_id_subset is not None
            else set(self.subgraphs)
        )
        exogenous_node_ids = {
            node_id
            for target_node_id in selected_targets
            if target_node_id in self.subgraphs
            for index, node_id in enumerate(self.subgraphs[target_node_id]["node_ids"])
            if bool(self.subgraphs[target_node_id]["has_ts_mask"][index])
        }
        self.slicer = TimeSeriesSlicer(
            processed_dir=processed_dir,
            target_sensor_ids=target_sensor_ids,
            exogenous_node_ids=exogenous_node_ids,
            lookback=self.lookback,
            horizon=self.horizon,
        )

    def __len__(self) -> int:
        return len(self.windows)
