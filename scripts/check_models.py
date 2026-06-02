"""Quick health check for all Salevora ML models."""

import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.data.data_loader import load_sales_csv
from src.data.data_preprocessor import clean_sales_data
from src.inventory.catalog import build_catalog, load_sales
from src.inventory.demand import daily_demand, forecast_weekly_units
from src.prediction.predictor import model_metrics, run_forecast

DATA = "data/processed/live_sales.csv"


def main() -> None:
    print("=" * 60)
    print("SALEVORA MODEL HEALTH CHECK (improved pipeline)")
    print("=" * 60)

    artifact = Path("models/forecast_artifact.joblib")
    print(f"\n[1] Training artifact: {'FOUND' if artifact.exists() else 'MISSING'}")

    raw = load_sales_csv(DATA)
    cleaned = clean_sales_data(raw)
    print(f"\n[2] Data quality:")
    print(f"    Raw rows:     {len(raw):,}")
    print(f"    Clean rows:   {len(cleaned):,}")
    if "product_id" in cleaned.columns:
        cats_per = cleaned.groupby("product_id")["category"].nunique().max()
        print(f"    Max categories per SKU: {cats_per} (target: 1)")

    from src.prediction.predictor import _weekly_revenue_series
    series = _weekly_revenue_series(cleaned)
    print(f"    Weekly revenue points: {len(series)}")

    print("\n[3] Holdout + rolling backtest (revenue target):")
    for name in ("arima", "prophet", "lstm", "ensemble"):
        t0 = time.time()
        try:
            m = model_metrics(series, name)
            h, r = m["holdout"], m["rolling"]
            print(
                f"    {name:10s}  holdout={h['accuracy']}%  "
                f"rolling={r['accuracy']}% ({r['folds']} folds)  ({time.time()-t0:.1f}s) OK"
            )
        except Exception as exc:
            print(f"    {name:10s}  FAILED: {exc}")

    print("\n[4] Forecast API (8-week ensemble, revenue):")
    t0 = time.time()
    r = run_forecast(horizon=8, model="ensemble", data_path=DATA)
    total = sum(f["sales"] for f in r["forecast"])
    print(
        f"    target={r['target']}  weeks={len(r['forecast'])}  total=${total:,.0f}  "
        f"rolling accuracy={r['metrics_rolling']['accuracy']}%  ({time.time()-t0:.1f}s) OK"
    )

    print("\n[5] Per-SKU quantity forecasts:")
    sales_df = load_sales(DATA)
    cat = build_catalog(sales_df)
    for sku in cat["sku"].head(5):
        t0 = time.time()
        d = daily_demand(sales_df, sku)
        w = forecast_weekly_units(sales_df, sku, 4)
        print(f"    {sku:12s}  daily={d:.1f} units  4wk={sum(w):,.0f}  ({time.time()-t0:.1f}s) OK")

    print("\n" + "=" * 60)


if __name__ == "__main__":
    main()
