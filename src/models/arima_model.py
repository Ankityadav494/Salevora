"""ARIMA time series model."""

from __future__ import annotations

import warnings

import numpy as np
import pandas as pd

from src.utils.config import load_config

try:
    from statsmodels.tsa.arima.model import ARIMA
    HAS_ARIMA = True
except ImportError:
    HAS_ARIMA = False


def _order() -> tuple[int, int, int]:
    cfg = load_config().get("models", {}).get("arima", {}).get("order", [1, 1, 1])
    return tuple(int(x) for x in cfg[:3])


def _linear_fallback(train: np.ndarray, horizon: int) -> np.ndarray:
    x = np.arange(len(train))
    slope, intercept = np.polyfit(x, train, 1)
    future_x = np.arange(len(train), len(train) + horizon)
    return np.maximum(0, slope * future_x + intercept)


def fit_predict(
    series: pd.Series,
    horizon: int = 12,
    order: tuple[int, int, int] | None = None,
) -> np.ndarray:
    values = series.astype(float).values
    if len(values) < 8:
        return _linear_fallback(values, horizon)

    if not HAS_ARIMA:
        return _linear_fallback(values, horizon)

    order = order or _order()
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            model = ARIMA(values, order=order)
            fitted = model.fit()
            forecast = fitted.forecast(steps=horizon)
            return np.maximum(0, np.asarray(forecast))
    except Exception:
        return _linear_fallback(values, horizon)
