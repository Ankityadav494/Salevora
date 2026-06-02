"""Facebook Prophet model with seasonal fallback."""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.utils.config import load_config

try:
    from prophet import Prophet
    HAS_PROPHET = True
except ImportError:
    HAS_PROPHET = False


def _prophet_cfg() -> dict:
    return load_config().get("models", {}).get("prophet", {})


def _seasonal_fallback(series: pd.Series, horizon: int) -> np.ndarray:
    values = series.astype(float).values
    n = len(values)
    x = np.arange(n)
    slope, intercept = np.polyfit(x, values, 1)
    last = values[-min(52, n):]
    seasonal = last - last.mean()
    preds = []
    for i in range(horizon):
        trend = slope * (n + i) + intercept
        season = seasonal[i % len(seasonal)]
        preds.append(max(0, trend + season * 0.3))
    return np.array(preds)


def fit_predict(
    series: pd.Series,
    horizon: int = 12,
    freq: str = "W",
) -> np.ndarray:
    if len(series) < 10:
        return _seasonal_fallback(series, horizon)

    if not HAS_PROPHET:
        return _seasonal_fallback(series, horizon)

    cfg = _prophet_cfg()
    df = pd.DataFrame({"ds": series.index, "y": series.values})
    try:
        model = Prophet(
            yearly_seasonality=cfg.get("yearly_seasonality", True),
            weekly_seasonality=cfg.get("weekly_seasonality", False),
            daily_seasonality=cfg.get("daily_seasonality", False),
            seasonality_mode=cfg.get("seasonality_mode", "multiplicative"),
            changepoint_prior_scale=cfg.get("changepoint_prior_scale", 0.05),
        )
        model.fit(df)
        future = model.make_future_dataframe(periods=horizon, freq=freq)
        forecast = model.predict(future)
        preds = forecast["yhat"].tail(horizon).values
        return np.maximum(0, preds)
    except Exception:
        return _seasonal_fallback(series, horizon)
