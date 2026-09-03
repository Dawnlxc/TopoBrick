"""Build topology-aware inputs under deployment-time availability."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Mapping

import numpy as np
import torch

from topobrick.forecast.inputs.meteorology import MeteorologicalFutureProvider
from topobrick.forecast.inputs.selection import select_exogenous_mask
from topobrick.utils.calendar import hour_of_day_features


DATASET_FREQUENCY_MINUTES = {
    "LBNL59": 15,
    "BTS_Site_B": 15,
    "BTS_Site_C": 15,
}

TARGET_HISTORY = "target_history"
PAST_KNOWN_EXOGENOUS = "past_known_exogenous"
TOPOLOGY_AWARE_EXOGENOUS = "topology_aware_exogenous"


@dataclass(frozen=True)
class InputBuilderConfig:
    frequency_minutes: int = 15
    add_calendar: bool = False
    past_correlation_threshold: float = 0.0
    meteorological_future: str = "none"
    meteorological_noise_std: float = 0.0
    meteorological_noise_std_normalized: float = 0.0
    meteorological_forecaster_uses_calendar: bool = False
    seed: int = 0

    def __post_init__(self) -> None:
        allowed = {"oracle", "none", "persistence", "forecast"}
        if self.meteorological_future not in allowed:
            raise ValueError(
                f"meteorological_future must be one of {sorted(allowed)}, "
                f"got {self.meteorological_future!r}"
            )


@dataclass
class ForecastWindow:
    target_history: np.ndarray
    target_future: np.ndarray
    trusted_mask: np.ndarray
    exogenous_history: Dict[str, np.ndarray]


@dataclass
class PreparedForecastInput:
    model_input: Dict[str, Any]
    target_history: np.ndarray
    target_future: np.ndarray
    trusted_mask: np.ndarray
    counts: Dict[str, int] = field(default_factory=dict)


def load_forecast_window(dataset, index: int) -> ForecastWindow:
    row = dataset.windows.iloc[index]
    target_node_id = row["target_node_id"]
    target_history, _, _, target_future, trusted_mask = (
        dataset.slicer.get_target_window(
            row["sensor_id"],
            row["input_start"],
            row["input_end"],
            row["target_start"],
            row["target_end"],
        )
    )
    subgraph = dataset.subgraphs.get(target_node_id)
    if subgraph is None:
        return ForecastWindow(target_history, target_future, trusted_mask, {})

    keep_timeseries = select_exogenous_mask(
        subgraph,
        mode=dataset.exogenous_selection,
        max_exogenous=dataset.max_exogenous,
    )
    exogenous_history: Dict[str, np.ndarray] = {}
    n_nodes = int(subgraph["node_mask"].sum())
    target_local_index = int(subgraph["target_local_index"])
    for local_index in range(n_nodes):
        if local_index == target_local_index:
            continue
        if not bool(keep_timeseries[local_index]):
            continue
        if not bool(subgraph["has_ts_mask"][local_index]):
            continue
        node_id = subgraph["node_ids"][local_index]
        history, _, valid = dataset.slicer.get_context_window(
            node_id,
            row["input_start"],
            row["input_end"],
        )
        if valid:
            exogenous_history[f"cov_{local_index:02d}"] = history.astype(np.float32)
    return ForecastWindow(
        target_history,
        target_future,
        trusted_mask,
        exogenous_history,
    )


def classify_exogenous_availability(brick_class: str) -> str:
    if "Setpoint" in brick_class or brick_class.endswith("_Command"):
        return "operational_schedule"
    meteorological_quantity = ("Temperature", "Humidity", "Dewpoint", "Enthalpy")
    is_outdoor_air = brick_class.startswith(("Outside_Air_", "Outdoor_Air_"))
    if is_outdoor_air and any(name in brick_class for name in meteorological_quantity):
        return "meteorological"
    return "past_known"


def _peak_absolute_cross_correlation(
    first: np.ndarray,
    second: np.ndarray,
    max_lag: int = 24,
) -> float:
    first = np.asarray(first, dtype=np.float64)
    second = np.asarray(second, dtype=np.float64)
    length = len(first)
    if length < 8:
        return 0.0

    peak = 0.0
    for lag in range(-max_lag, max_lag + 1):
        if lag >= 0:
            left, right = first[lag:length], second[: length - lag]
        else:
            left, right = first[: length + lag], second[-lag:length]
        if len(left) < max(8, length // 3):
            continue
        left_std, right_std = left.std(), right.std()
        if left_std < 1e-8 or right_std < 1e-8:
            continue
        correlation = float(
            np.mean((left - left.mean()) * (right - right.mean()))
            / (left_std * right_std)
        )
        peak = max(peak, abs(correlation))
    return peak


class AvailabilityAwareInputBuilder:
    def __init__(self, config: InputBuilderConfig, meteorological_forecaster=None):
        self.config = config
        self.meteorological = MeteorologicalFutureProvider(
            mode=config.meteorological_future,
            frequency_minutes=config.frequency_minutes,
            pipeline=meteorological_forecaster,
            noise_std=config.meteorological_noise_std,
            normalized_noise_std=config.meteorological_noise_std_normalized,
            use_calendar=config.meteorological_forecaster_uses_calendar,
            seed=config.seed,
        )

    def clear_forecast_cache(self) -> None:
        self.meteorological.clear_cache()

    def build(
        self,
        dataset,
        index: int,
        horizon: int,
        brick_class_by_node: Mapping[str, str],
    ) -> PreparedForecastInput:
        row = dataset.windows.iloc[index]
        window = load_forecast_window(dataset, index)
        calendar = self._calendar_covariates(row, len(window.target_history), horizon)
        subgraph = dataset.subgraphs.get(row["target_node_id"])

        if subgraph is None or not window.exogenous_history:
            model_input: Dict[str, Any] = {
                "target": torch.from_numpy(window.target_history.astype(np.float32))
            }
            if calendar:
                model_input["past_covariates"] = calendar["past"]
                model_input["future_covariates"] = calendar["future"]
            calendar_count = len(calendar.get("past", {}))
            return PreparedForecastInput(
                model_input,
                window.target_history,
                window.target_future,
                window.trusted_mask,
                {
                    "n_past": calendar_count,
                    "n_future": calendar_count,
                    "n_operational_schedule": 0,
                    "n_meteorological": 0,
                    "n_past_known": 0,
                    "n_pruned": 0,
                },
            )

        past_exogenous: Dict[str, np.ndarray] = {}
        future_exogenous: Dict[str, np.ndarray] = {}
        counts = {
            "n_operational_schedule": 0,
            "n_meteorological": 0,
            "n_past_known": 0,
            "n_pruned": 0,
        }
        n_nodes = int(subgraph["node_mask"].sum())
        target_local_index = int(subgraph["target_local_index"])
        for local_index in range(n_nodes):
            if local_index == target_local_index or not bool(
                subgraph["has_ts_mask"][local_index]
            ):
                continue
            node_id = subgraph["node_ids"][local_index]
            key = f"cov_{local_index:02d}"
            if key not in window.exogenous_history:
                continue
            history = window.exogenous_history[key]
            brick_class = brick_class_by_node.get(node_id, "")
            availability = classify_exogenous_availability(brick_class)
            if (
                availability == "past_known"
                and self.config.past_correlation_threshold > 0
                and _peak_absolute_cross_correlation(window.target_history, history)
                < self.config.past_correlation_threshold
            ):
                counts["n_pruned"] += 1
                continue

            past_exogenous[key] = history
            if availability == "operational_schedule":
                future_exogenous[key] = dataset.slicer.get_future_window(
                    node_id,
                    row["target_start"],
                    row["target_end"],
                )
                counts["n_operational_schedule"] += 1
            elif availability == "meteorological":
                meteorological_future = self.meteorological.get(
                    dataset,
                    node_id,
                    history,
                    row["input_start"],
                    row["target_start"],
                    row["target_end"],
                    horizon,
                )
                if meteorological_future is not None:
                    future_exogenous[key] = meteorological_future
                counts["n_meteorological"] += 1
            else:
                counts["n_past_known"] += 1

        if calendar:
            past_exogenous.update(calendar["past_numpy"])
            future_exogenous.update(calendar["future_numpy"])

        model_input = {
            "target": torch.from_numpy(window.target_history.astype(np.float32))
        }
        if past_exogenous:
            model_input["past_covariates"] = {
                key: torch.from_numpy(np.asarray(value, dtype=np.float32))
                for key, value in past_exogenous.items()
            }
        if future_exogenous:
            model_input["future_covariates"] = {
                key: torch.from_numpy(np.asarray(value, dtype=np.float32))
                for key, value in future_exogenous.items()
            }
        counts["n_past"] = len(past_exogenous)
        counts["n_future"] = len(future_exogenous)
        return PreparedForecastInput(
            model_input,
            window.target_history,
            window.target_future,
            window.trusted_mask,
            counts,
        )

    @staticmethod
    def select_context(prepared: PreparedForecastInput, context: str):
        if context == TARGET_HISTORY:
            selected = {"target": prepared.model_input["target"]}
            for group in ("past_covariates", "future_covariates"):
                calendar = {
                    key: value
                    for key, value in prepared.model_input.get(group, {}).items()
                    if key.startswith("tod_")
                }
                if calendar:
                    selected[group] = calendar
            return selected if len(selected) > 1 else selected["target"]
        if context == TOPOLOGY_AWARE_EXOGENOUS:
            return prepared.model_input
        if context != PAST_KNOWN_EXOGENOUS:
            raise ValueError(f"unknown forecast context: {context!r}")

        selected = {
            key: value
            for key, value in prepared.model_input.items()
            if key != "future_covariates"
        }
        calendar_future = {
            key: value
            for key, value in prepared.model_input.get("future_covariates", {}).items()
            if key.startswith("tod_")
        }
        if calendar_future:
            selected["future_covariates"] = calendar_future
        return selected

    def _calendar_covariates(self, row, lookback: int, horizon: int) -> Dict[str, Dict]:
        if not self.config.add_calendar:
            return {}
        past_sin, past_cos = hour_of_day_features(
            row["input_start"], self.config.frequency_minutes, lookback
        )
        future_sin, future_cos = hour_of_day_features(
            row["target_start"], self.config.frequency_minutes, horizon
        )
        past_numpy = {"tod_sin": past_sin, "tod_cos": past_cos}
        future_numpy = {"tod_sin": future_sin, "tod_cos": future_cos}
        return {
            "past_numpy": past_numpy,
            "future_numpy": future_numpy,
            "past": {key: torch.from_numpy(value) for key, value in past_numpy.items()},
            "future": {
                key: torch.from_numpy(value) for key, value in future_numpy.items()
            },
        }
