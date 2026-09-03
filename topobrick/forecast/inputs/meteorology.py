"""Meteorological forecasts used as future-known exogenous variables."""

from __future__ import annotations

from typing import Dict

import numpy as np
import pandas as pd
import torch

from topobrick.utils.arrays import median_sample_forecast
from topobrick.utils.calendar import hour_of_day_features


def persistence_forecast(history: np.ndarray, horizon: int) -> np.ndarray:
    if history.size == 0:
        return np.zeros(horizon, dtype=np.float32)
    if np.isnan(history[-1]):
        observed = history[~np.isnan(history)]
        last_value = float(observed.mean()) if observed.size else 0.0
    else:
        last_value = float(history[-1])
    return np.full(horizon, last_value, dtype=np.float32)


class MeteorologicalFutureProvider:
    def __init__(
        self,
        *,
        mode: str,
        frequency_minutes: int,
        pipeline=None,
        noise_std: float = 0.0,
        normalized_noise_std: float = 0.0,
        use_calendar: bool = False,
        seed: int = 0,
    ):
        self.mode = mode
        self.frequency_minutes = frequency_minutes
        self.pipeline = pipeline
        self.noise_std = noise_std
        self.normalized_noise_std = normalized_noise_std
        self.use_calendar = use_calendar
        self._rng = np.random.RandomState(seed)
        self._forecast_cache: Dict[tuple, np.ndarray] = {}

    def clear_cache(self) -> None:
        self._forecast_cache.clear()

    def get(
        self,
        dataset,
        node_id: str,
        history: np.ndarray,
        input_start,
        target_start,
        target_end,
        horizon: int,
    ) -> np.ndarray | None:
        if self.mode == "none":
            return None
        if self.mode == "persistence":
            return persistence_forecast(history, horizon)
        if self.mode == "forecast":
            return self._forecast(node_id, history, input_start, target_start, horizon)

        future = dataset.slicer.get_future_window(node_id, target_start, target_end)
        if self.noise_std > 0:
            future = future + self._rng.normal(
                0.0,
                self.noise_std,
                size=future.shape,
            ).astype(np.float32)
        if self.normalized_noise_std > 0:
            scale = dataset.slicer.observed_std_for_node(node_id)
            future = future + self._rng.normal(
                0.0,
                self.normalized_noise_std * scale,
                size=future.shape,
            ).astype(np.float32)
        return future

    def _forecast(
        self,
        node_id: str,
        history: np.ndarray,
        input_start,
        target_start,
        horizon: int,
    ) -> np.ndarray:
        cache_key = (node_id, int(pd.Timestamp(target_start).value), horizon)
        if self.use_calendar:
            cache_key += ("calendar",)
        cached = self._forecast_cache.get(cache_key)
        if cached is not None:
            return cached
        if history.size == 0:
            forecast = np.zeros(horizon, dtype=np.float32)
            self._forecast_cache[cache_key] = forecast
            return forecast
        if self.pipeline is None:
            raise ValueError(
                "meteorological_future='forecast' requires a forecaster pipeline"
            )

        target = torch.from_numpy(history.astype(np.float32))
        if self.use_calendar:
            past_sin, past_cos = hour_of_day_features(
                input_start,
                self.frequency_minutes,
                len(history),
            )
            future_sin, future_cos = hour_of_day_features(
                target_start,
                self.frequency_minutes,
                horizon,
            )
            model_input = {
                "target": target,
                "past_covariates": {
                    "tod_sin": torch.from_numpy(past_sin),
                    "tod_cos": torch.from_numpy(past_cos),
                },
                "future_covariates": {
                    "tod_sin": torch.from_numpy(future_sin),
                    "tod_cos": torch.from_numpy(future_cos),
                },
            }
        else:
            model_input = target

        with torch.no_grad():
            predictions = self.pipeline.predict(
                [model_input],
                prediction_length=horizon,
                batch_size=1,
            )
        forecast = median_sample_forecast(predictions[0], horizon)
        forecast[np.isnan(forecast)] = 0.0
        self._forecast_cache[cache_key] = forecast
        return forecast
