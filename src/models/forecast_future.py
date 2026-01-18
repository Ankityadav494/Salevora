"""
Future sales forecasting using trained ARIMA model
"""

import pandas as pd
from pathlib import Path
import matplotlib.pyplot as plt
from statsmodels.tsa.arima.model import ARIMAResults


def forecast_future(steps=12):
    # Paths
    model_path = Path("models/arima_model.pkl")
    data_path = Path("data/processed/processed_sales_data.csv")

    # Load processed data
    df = pd.read_csv(data_path)
    df["date"] = pd.to_datetime(df["date"])

    # Aggregate by date (same as training)
    df = df.groupby("date", as_index=False)["sales"].sum()
    df = df.set_index("date").sort_index()

    # Load trained ARIMA model
    model_fit = ARIMAResults.load(model_path)

    # Forecast future values
    forecast = model_fit.forecast(steps=steps)

    # Create future date index
    last_date = df.index[-1]
    future_dates = pd.date_range(
        start=last_date,
        periods=steps + 1,
        freq="W"
    )[1:]

    forecast_df = pd.DataFrame({
        "date": future_dates,
        "forecasted_sales": forecast.values
    })

    print("\n🔮 Future Sales Forecast")
    print(forecast_df)

    # Plot forecast
    plt.figure(figsize=(10, 5))
    plt.plot(df.index, df["sales"], label="Historical Sales")
    plt.plot(forecast_df["date"], forecast_df["forecasted_sales"],
             linestyle="--", marker="o", label="Future Forecast")
    plt.title("Future Sales Forecast (Next 12 Weeks)")
    plt.xlabel("Date")
    plt.ylabel("Sales")
    plt.legend()
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    forecast_future(steps=12)
