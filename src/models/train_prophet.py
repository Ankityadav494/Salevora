"""
Prophet model training for Sales Forecasting Project
"""

import pandas as pd
from pathlib import Path
from prophet import Prophet
from sklearn.metrics import mean_squared_error
import numpy as np


def train_prophet():
    # Paths
    data_path = Path("data/processed/processed_sales_data.csv")
    model_path = Path("models/prophet_model.csv")

    # Load data
    df = pd.read_csv(data_path)
    df["date"] = pd.to_datetime(df["date"])

    # Aggregate by date (same logic as ARIMA)
    df = df.groupby("date", as_index=False)["sales"].sum()

    # Prophet requires columns: ds (date), y (value)
    prophet_df = df.rename(columns={
        "date": "ds",
        "sales": "y"
    })

    # Train-test split
    split = int(len(prophet_df) * 0.8)
    train = prophet_df.iloc[:split]
    test = prophet_df.iloc[split:]

    # Train Prophet model
    model = Prophet(
        yearly_seasonality=True,
        weekly_seasonality=True,
        daily_seasonality=False
    )
    model.fit(train)

    # Forecast on test data
    future = model.make_future_dataframe(periods=len(test), freq="W")
    forecast = model.predict(future)

    # Extract predictions for test period
    y_pred = forecast.iloc[-len(test):]["yhat"].values
    y_true = test["y"].values

    rmse = np.sqrt(mean_squared_error(y_true, y_pred))

    # Save forecast results
    forecast[["ds", "yhat"]].to_csv(model_path, index=False)

    print("✅ Prophet model trained successfully")
    print(f"📉 Prophet RMSE: {rmse:.2f}")
    print(f"💾 Forecast saved at: {model_path}")


if __name__ == "__main__":
    train_prophet()
