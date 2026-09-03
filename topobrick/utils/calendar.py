"""Deterministic calendar features."""

from __future__ import annotations

import numpy as np
import pandas as pd


def hour_of_day_features(
    start,
    frequency_minutes: int,
    length: int,
) -> tuple[np.ndarray, np.ndarray]:
    timestamps = pd.date_range(
        pd.Timestamp(start),
        periods=length,
        freq=f"{frequency_minutes}min",
    )
    fraction_of_day = (timestamps.hour * 60 + timestamps.minute) / (24 * 60)
    return (
        np.sin(2 * np.pi * fraction_of_day.values).astype(np.float32),
        np.cos(2 * np.pi * fraction_of_day.values).astype(np.float32),
    )
