"""Forecast metrics reported in the paper."""

from __future__ import annotations

from typing import Dict, Mapping

import numpy as np
import pandas as pd


def compute_forecast_metrics(
    results: Mapping[str, np.ndarray],
    selection: np.ndarray | None = None,
    train_std_by_sensor: Mapping[str, float] | None = None,
) -> Dict[str, float | int]:
    target = results["target"]
    prediction = results["prediction"]
    trusted = results["trusted_mask"]
    sensor_ids = results["sensor_id"]

    if selection is not None:
        target = target[selection]
        prediction = prediction[selection]
        trusted = trusted[selection]
        sensor_ids = sensor_ids[selection]
    if len(target) == 0:
        return {"n": 0}

    error = (prediction - target) * trusted.astype(np.float32)
    metrics: Dict[str, float | int] = {
        "n": int(len(target)),
        "MAE_overall": float(np.abs(error).sum() / max(trusted.sum(), 1)),
        "MSE_overall": float((error**2).sum() / max(trusted.sum(), 1)),
    }
    if train_std_by_sensor is None:
        return metrics

    scales = np.array(
        [
            float(train_std_by_sensor.get(str(sensor_id), 1.0))
            for sensor_id in sensor_ids
        ],
        dtype=np.float32,
    ).clip(min=0.01)
    normalized_error = error / scales[:, None]
    metrics.update(
        {
            "nMAE_overall": float(
                np.abs(normalized_error).sum() / max(trusted.sum(), 1)
            ),
            "nMSE_overall": float((normalized_error**2).sum() / max(trusted.sum(), 1)),
        }
    )
    return metrics


def aggregate_per_sensor(
    results: Mapping[str, np.ndarray],
    train_std_by_sensor: Mapping[str, float] | None = None,
) -> pd.DataFrame:
    target = results["target"]
    prediction = results["prediction"]
    trusted = results["trusted_mask"]
    sensor_ids = np.asarray(results["sensor_id"]).astype(str)
    brick_classes = np.asarray(
        results.get("target_class", np.array([""] * len(sensor_ids)))
    ).astype(str)
    error = (prediction - target) * trusted.astype(np.float32)

    per_window = pd.DataFrame(
        {
            "sensor_id": sensor_ids,
            "brick_class": brick_classes,
            "n_trusted": trusted.sum(axis=1),
            "sum_abs": np.abs(error).sum(axis=1),
            "sum_sq": (error**2).sum(axis=1),
        }
    )
    grouped = (
        per_window.groupby(["sensor_id", "brick_class"], sort=False)
        .agg(
            n_windows=("n_trusted", "size"),
            n_trusted=("n_trusted", "sum"),
            sum_abs=("sum_abs", "sum"),
            sum_sq=("sum_sq", "sum"),
        )
        .reset_index()
    )
    if train_std_by_sensor is None:
        grouped["train_std"] = 1.0
    else:
        grouped["train_std"] = grouped["sensor_id"].map(
            lambda sensor_id: max(
                float(train_std_by_sensor.get(str(sensor_id), 1.0)), 0.01
            )
        )
    return grouped
