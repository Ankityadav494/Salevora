"""Ensemble of ARIMA, Prophet, and neural network models."""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.models import arima_model, lstm_model, prophet_model
from src.utils.config import load_config


def _weights() -> list[float]:
    cfg = load_config()
    return cfg.get("models", {}).get("ensemble", {}).get("weights", [0.3, 0.3, 0.4])


def fit_predict(series: pd.Series, horizon: int = 12) -> dict[str, np.ndarray]:
    w = _weights()
    total = sum(w)
    w = [x / total for x in w]

    arima_pred = arima_model.fit_predict(series, horizon)
    prophet_pred = prophet_model.fit_predict(series, horizon)
    lstm_pred = lstm_model.fit_predict(series, horizon)

    ensemble = w[0] * arima_pred + w[1] * prophet_pred + w[2] * lstm_pred

    return {
        "arima": arima_pred,
        "prophet": prophet_pred,
        "lstm": lstm_pred,
        "ensemble": np.maximum(0, ensemble),
    }
