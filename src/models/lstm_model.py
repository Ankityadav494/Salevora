"""Neural network forecaster (MLP) — lightweight LSTM alternative."""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler


def _build_sequences(values: np.ndarray, seq_len: int) -> tuple[np.ndarray, np.ndarray]:
    x, y = [], []
    for i in range(seq_len, len(values)):
        x.append(values[i - seq_len : i])
        y.append(values[i])
    return np.array(x), np.array(y)


def fit_predict(
    series: pd.Series,
    horizon: int = 12,
    sequence_length: int = 8,
) -> np.ndarray:
    values = series.astype(float).values
    n = len(values)

    if n < sequence_length + 4:
        x = np.arange(n)
        slope, intercept = np.polyfit(x, values, 1)
        future_x = np.arange(n, n + horizon)
        return np.maximum(0, slope * future_x + intercept)

    seq_len = min(sequence_length, n // 2)
    x_train, y_train = _build_sequences(values, seq_len)

    scaler_x = StandardScaler()
    scaler_y = StandardScaler()
    x_scaled = scaler_x.fit_transform(x_train)
    y_scaled = scaler_y.fit_transform(y_train.reshape(-1, 1)).ravel()

    model = MLPRegressor(
        hidden_layer_sizes=(50, 50),
        activation="relu",
        max_iter=500,
        random_state=42,
        early_stopping=True,
        validation_fraction=0.1,
    )
    model.fit(x_scaled, y_scaled)

    window = values[-seq_len:].copy()
    preds = []
    for _ in range(horizon):
        x_in = scaler_x.transform(window.reshape(1, -1))
        y_hat = scaler_y.inverse_transform(model.predict(x_in).reshape(-1, 1))[0, 0]
        y_hat = max(0, float(y_hat))
        preds.append(y_hat)
        window = np.roll(window, -1)
        window[-1] = y_hat

    return np.array(preds)
