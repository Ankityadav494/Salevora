"""Evaluation metrics for forecasting models."""

from __future__ import annotations

from typing import Callable

import numpy as np
import pandas as pd


def mae(actual: np.ndarray, predicted: np.ndarray) -> float:
    return float(np.mean(np.abs(actual - predicted)))


def rmse(actual: np.ndarray, predicted: np.ndarray) -> float:
    return float(np.sqrt(np.mean((actual - predicted) ** 2)))


def mape(actual: np.ndarray, predicted: np.ndarray) -> float:
    mask = actual != 0
    if not mask.any():
        return 0.0
    return float(np.mean(np.abs((actual[mask] - predicted[mask]) / actual[mask])) * 100)


def accuracy_from_mape(mape_val: float) -> float:
    return max(0.0, 100.0 - mape_val)


def evaluate(actual: np.ndarray, predicted: np.ndarray) -> dict[str, float]:
    mape_val = mape(actual, predicted)
    return {
        "mae": round(mae(actual, predicted), 2),
        "rmse": round(rmse(actual, predicted), 2),
        "mape": round(mape_val, 2),
        "accuracy": round(accuracy_from_mape(mape_val), 1),
    }


def rolling_backtest(
    series: pd.Series,
    predict_fn: Callable[[pd.Series, int], np.ndarray],
    holdout: int = 4,
    folds: int = 5,
) -> dict[str, float]:
    """Rolling-window backtest — more reliable than a single holdout split."""
    if len(series) <= holdout + 12:
        return {"mae": 0, "rmse": 0, "mape": 0, "accuracy": 0, "folds": 0}

    maes, rmses, mapes = [], [], []
    used = 0
    for i in range(folds):
        end = len(series) - i
        if end - holdout < 12:
            break
        train = series.iloc[: end - holdout]
        test = series.iloc[end - holdout : end]
        try:
            pred = np.asarray(predict_fn(train, holdout))
        except Exception:
            continue
        if len(pred) != len(test):
            continue
        maes.append(mae(test.values, pred))
        rmses.append(rmse(test.values, pred))
        mapes.append(mape(test.values, pred))
        used += 1

    if not used:
        return {"mae": 0, "rmse": 0, "mape": 0, "accuracy": 0, "folds": 0}

    avg_mape = float(np.mean(mapes))
    return {
        "mae": round(float(np.mean(maes)), 2),
        "rmse": round(float(np.mean(rmses)), 2),
        "mape": round(avg_mape, 2),
        "accuracy": round(accuracy_from_mape(avg_mape), 1),
        "folds": used,
    }
