"""TimesFM forecasting backend."""

from __future__ import annotations

from importlib import import_module
from typing import Dict

import numpy as np

from topobrick.utils.arrays import as_float_array


def _convert_input(model_input: Dict, horizon: int):
    target = as_float_array(model_input["target"])
    past = model_input.get("past_covariates", {}) or {}
    future = model_input.get("future_covariates", {}) or {}
    dynamic = {}
    for key in sorted(set(past) & set(future)):
        history = as_float_array(past[key])
        forecast = as_float_array(future[key])[:horizon]
        if len(forecast) < horizon:
            last_value = float(forecast[-1]) if len(forecast) else 0.0
            forecast = np.concatenate(
                [
                    forecast,
                    np.full(horizon - len(forecast), last_value, dtype=np.float32),
                ]
            )
        dynamic[key] = np.concatenate([history, forecast]).astype(np.float32)
    return target, dynamic


class TimesFMBackend:
    name = "timesfm"
    default_model = "google/timesfm-2.0-500m-pytorch"

    def __init__(
        self,
        model_id: str,
        device: str,
        lookback: int,
        horizon: int,
        batch_size: int,
    ):
        del batch_size
        timesfm = import_module("timesfm")
        backend = "gpu" if str(device).startswith("cuda") else "cpu"
        self.pipeline = timesfm.TimesFm(
            hparams=timesfm.TimesFmHparams(
                backend=backend,
                per_core_batch_size=32,
                horizon_len=horizon,
                context_len=lookback,
                num_layers=50,
                use_positional_embedding=False,
            ),
            checkpoint=timesfm.TimesFmCheckpoint(huggingface_repo_id=model_id),
        )

    def predict(self, model_inputs: list[Dict], horizon: int) -> np.ndarray:
        converted = [_convert_input(value, horizon) for value in model_inputs]
        targets = [target for target, _ in converted]
        covariates = [values for _, values in converted]

        if not any(covariates):
            point_forecast, _ = self.pipeline.forecast(
                inputs=targets,
                freq=[0] * len(targets),
            )
            return np.asarray(point_forecast)[:, :horizon].astype(np.float32)

        keys = sorted({key for record in covariates for key in record})
        batch = {key: [] for key in keys}
        for target, record in converted:
            for key in keys:
                batch[key].append(
                    record.get(
                        key,
                        np.zeros(len(target) + horizon, dtype=np.float32),
                    )
                )
        point_forecast, _ = self.pipeline.forecast_with_covariates(
            inputs=targets,
            dynamic_numerical_covariates=batch,
            freq=[0] * len(targets),
            xreg_mode="xreg + timesfm",
            normalize_xreg_target_per_input=True,
        )
        return np.stack(
            [np.asarray(forecast)[:horizon] for forecast in point_forecast]
        ).astype(np.float32)
