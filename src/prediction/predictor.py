"""Prediction utilities for Salevora forecasting."""

from __future__ import annotations

from datetime import timedelta
from typing import Any

import numpy as np
import pandas as pd

from src.data.data_loader import aggregate_weekly, load_sales_csv
from src.data.data_preprocessor import aggregate_daily_company, clean_sales_data
from src.models import arima_model, ensemble_model, lstm_model, prophet_model
from src.utils.metrics import evaluate, rolling_backtest


MODELS = {
    "arima": arima_model.fit_predict,
    "prophet": prophet_model.fit_predict,
    "lstm": lstm_model.fit_predict,
    "ensemble": None,
}

FORECAST_TARGET = "revenue"  # dashboard forecasts revenue ($)


def _predict_fn(model_name: str):
    def _fn(train: pd.Series, horizon: int) -> np.ndarray:
        if model_name == "ensemble":
            return ensemble_model.fit_predict(train, horizon)["ensemble"]
        return MODELS[model_name](train, horizon)

    return _fn


def _future_dates(last_date: pd.Timestamp, horizon: int) -> list[str]:
    return [
        (last_date + timedelta(weeks=i + 1)).strftime("%Y-%m-%d")
        for i in range(horizon)
    ]


def _weekly_revenue_series(cleaned: pd.DataFrame) -> pd.Series:
    daily = aggregate_daily_company(cleaned)
    weekly = aggregate_weekly(daily, FORECAST_TARGET)
    return weekly.set_index("date")[FORECAST_TARGET]


def _holdout_metrics(series: pd.Series, model_name: str, holdout: int = 4) -> dict[str, float]:
    if len(series) <= holdout + 8:
        return {"mae": 0, "rmse": 0, "mape": 0, "accuracy": 0}

    train, test = series.iloc[:-holdout], series.iloc[-holdout:]
    if model_name == "ensemble":
        predicted = ensemble_model.fit_predict(train, holdout)["ensemble"]
    else:
        predicted = MODELS[model_name](train, holdout)

    return evaluate(test.values, np.asarray(predicted))


def model_metrics(series: pd.Series, model_name: str) -> dict[str, float]:
    """Single holdout + rolling backtest metrics."""
    single = _holdout_metrics(series, model_name)
    rolling = rolling_backtest(series, _predict_fn(model_name), holdout=4, folds=5)
    return {"holdout": single, "rolling": rolling}


def run_forecast(
    horizon: int = 12,
    model: str = "ensemble",
    data_path: str | None = None,
) -> dict[str, Any]:
    raw = load_sales_csv(data_path)
    cleaned = clean_sales_data(raw)
    series = _weekly_revenue_series(cleaned)
    weekly = series.reset_index()
    weekly.columns = ["date", FORECAST_TARGET]

    model = model.lower()
    if model not in MODELS:
        model = "ensemble"

    if model == "ensemble":
        preds_dict = ensemble_model.fit_predict(series, horizon)
        predicted = preds_dict["ensemble"]
        components = {
            k: [round(float(v), 2) for v in vals]
            for k, vals in preds_dict.items()
        }
    else:
        predicted = MODELS[model](series, horizon)
        components = {model: [round(float(v), 2) for v in predicted]}

    predicted = np.maximum(0, np.asarray(predicted))
    metrics = model_metrics(series, model)
    rmse = metrics["rolling"]["rmse"] or metrics["holdout"]["rmse"]

    historical = [
        {"date": d.strftime("%Y-%m-%d"), "sales": round(float(s), 2)}
        for d, s in zip(weekly["date"], weekly[FORECAST_TARGET])
    ]

    last_date = weekly["date"].iloc[-1]
    forecast = [
        {
            "date": dt,
            "sales": round(float(v), 2),
            "lower": round(max(0.0, float(v) - 1.96 * rmse), 2),
            "upper": round(float(v) + 1.96 * rmse, 2),
        }
        for dt, v in zip(_future_dates(last_date, horizon), predicted)
    ]

    cat_totals = cleaned.groupby("category")["revenue"].sum().sort_values(ascending=False)
    categories = [
        {"name": k, "revenue": round(float(v), 2)}
        for k, v in cat_totals.items()
    ]

    return {
        "model": model,
        "target": FORECAST_TARGET,
        "horizon": horizon,
        "historical": historical,
        "forecast": forecast,
        "components": components,
        "metrics": metrics["holdout"],
        "metrics_rolling": metrics["rolling"],
        "summary": {
            "rows": int(len(cleaned)),
            "weeks": int(len(series)),
            "date_start": cleaned["date"].min().strftime("%Y-%m-%d"),
            "date_end": cleaned["date"].max().strftime("%Y-%m-%d"),
            "total_revenue": round(float(cleaned["revenue"].sum()), 2),
            "total_sales": round(float(cleaned["sales"].sum()), 2),
            "categories": categories,
        },
    }


def analytics_summary(data_path: str | None = None) -> dict[str, Any]:
    raw = load_sales_csv(data_path)
    cleaned = clean_sales_data(raw)
    series = _weekly_revenue_series(cleaned)

    end = cleaned["date"].max()
    last30 = cleaned[cleaned["date"] >= end - pd.Timedelta(days=30)]["revenue"].sum()
    prev30 = cleaned[
        (cleaned["date"] >= end - pd.Timedelta(days=60))
        & (cleaned["date"] < end - pd.Timedelta(days=30))
    ]["revenue"].sum()
    mom = ((last30 - prev30) / prev30 * 100) if prev30 else None

    return {
        "rows": int(len(cleaned)),
        "weeks": int(len(series)),
        "date_range": {
            "start": cleaned["date"].min().strftime("%Y-%m-%d"),
            "end": end.strftime("%Y-%m-%d"),
        },
        "total_revenue": round(float(cleaned["revenue"].sum()), 2),
        "total_sales": round(float(cleaned["sales"].sum()), 2),
        "categories": sorted(cleaned["category"].dropna().unique().tolist()),
        "mom_change_pct": round(mom, 2) if mom is not None else None,
    }
