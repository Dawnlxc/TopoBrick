"""Zero-shot forecasting backends."""

from topobrick.forecast.backends.moirai import MoiraiBackend
from topobrick.forecast.backends.timesfm import TimesFMBackend

BACKENDS = {
    TimesFMBackend.name: TimesFMBackend,
    MoiraiBackend.name: MoiraiBackend,
}

__all__ = ["BACKENDS", "MoiraiBackend", "TimesFMBackend"]
