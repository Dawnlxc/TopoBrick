"""Array helpers shared by forecasting backends."""

from __future__ import annotations

import numpy as np


def as_float_array(value) -> np.ndarray:
    if hasattr(value, "detach"):
        value = value.detach().cpu().numpy()
    return np.asarray(value, dtype=np.float32)


def median_sample_forecast(prediction, horizon: int) -> np.ndarray:
    values = prediction.cpu().numpy().astype(np.float32)
    if values.ndim == 3:
        values = np.median(values, axis=1).squeeze()
    elif values.ndim == 2:
        values = np.median(values, axis=0)
    return np.asarray(values[:horizon], dtype=np.float32)
