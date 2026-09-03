"""Moirai forecasting backend."""

from __future__ import annotations

from importlib import import_module
from typing import Dict

import numpy as np

from topobrick.utils.arrays import as_float_array


def _convert_input(model_input: Dict, horizon: int):
    target = as_float_array(model_input["target"])
    lookback = len(target)
    past = model_input.get("past_covariates", {}) or {}
    future = model_input.get("future_covariates", {}) or {}
    future_known_keys = sorted(set(past) & set(future))
    past_known_keys = sorted(set(past) - set(future))

    if future_known_keys:
        history = np.stack([as_float_array(past[key]) for key in future_known_keys])
        forecast = np.stack([as_float_array(future[key]) for key in future_known_keys])
        if forecast.shape[1] < horizon:
            forecast = np.concatenate(
                [
                    forecast,
                    np.repeat(forecast[:, -1:], horizon - forecast.shape[1], axis=1),
                ],
                axis=1,
            )
        future_known = np.concatenate([history, forecast[:, :horizon]], axis=1)
    else:
        future_known = np.zeros((0, lookback + horizon), dtype=np.float32)

    past_known = (
        np.stack([as_float_array(past[key]) for key in past_known_keys])
        if past_known_keys
        else np.zeros((0, lookback), dtype=np.float32)
    )
    return target, future_known, past_known


def _pad_rows(values: np.ndarray, row_count: int) -> np.ndarray:
    if values.shape[0] == row_count:
        return values
    padding = np.zeros(
        (row_count - values.shape[0], values.shape[1]),
        dtype=np.float32,
    )
    return np.concatenate([values, padding], axis=0)


class MoiraiBackend:
    name = "moirai"
    default_model = "Salesforce/moirai-2.0-R-small"

    def __init__(
        self,
        model_id: str,
        device: str,
        lookback: int,
        horizon: int,
        batch_size: int,
    ):
        del horizon
        moirai = import_module("uni2ts.model.moirai2")
        self.module = moirai.Moirai2Module.from_pretrained(model_id)
        self.forecast_class = moirai.Moirai2Forecast
        self.device = device
        self.lookback = lookback
        self.batch_size = batch_size

    def predict_all(self, model_inputs, starts, horizon: int) -> np.ndarray:
        rows = []
        max_future_known = 0
        max_past_known = 0
        for model_input, start in zip(model_inputs, starts):
            target, future_known, past_known = _convert_input(model_input, horizon)
            max_future_known = max(max_future_known, future_known.shape[0])
            max_past_known = max(max_past_known, past_known.shape[0])
            rows.append((target, future_known, past_known, start))

        if not rows:
            return np.empty((0, horizon), dtype=np.float32)
        lookback = len(rows[0][0])
        model = self.forecast_class(
            prediction_length=horizon,
            target_dim=1,
            feat_dynamic_real_dim=max_future_known,
            past_feat_dynamic_real_dim=max_past_known,
            context_length=lookback,
            module=self.module,
        )
        predictor = model.create_predictor(
            batch_size=self.batch_size,
            device=self.device,
        )
        entries = []
        for index, (target, future_known, past_known, start) in enumerate(rows):
            entry = {"target": target, "start": start, "item_id": str(index)}
            if max_future_known:
                entry["feat_dynamic_real"] = _pad_rows(
                    future_known,
                    max_future_known,
                )
            if max_past_known:
                entry["past_feat_dynamic_real"] = _pad_rows(
                    past_known,
                    max_past_known,
                )
            entries.append(entry)

        forecasts = list(predictor.predict(entries))
        if len(forecasts) != len(entries):
            raise RuntimeError(
                f"Moirai returned {len(forecasts)} forecasts for {len(entries)} inputs"
            )
        output = np.zeros((len(entries), horizon), dtype=np.float32)
        for index, forecast in enumerate(forecasts):
            median = forecast.quantile(0.5).astype(np.float32)
            if len(median) < horizon:
                last_value = float(median[-1]) if len(median) else 0.0
                median = np.concatenate(
                    [
                        median,
                        np.full(horizon - len(median), last_value, dtype=np.float32),
                    ]
                )
            output[index] = median[:horizon]
        return output
