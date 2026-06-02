"""Model training pipeline."""

from __future__ import annotations

import json
import sys
from datetime import datetime, timezone
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(BASE_DIR))

import joblib

from src.data.data_loader import load_sales_csv
from src.data.data_preprocessor import clean_sales_data
from src.models import ensemble_model
from src.prediction.predictor import _weekly_revenue_series, model_metrics
from src.utils.config import load_config


def train(save: bool = True) -> dict:
    cfg = load_config()
    model_dir = BASE_DIR / cfg.get("training", {}).get("model_save_path", "models")
    log_dir = BASE_DIR / cfg.get("training", {}).get("log_path", "logs")
    model_dir.mkdir(parents=True, exist_ok=True)
    log_dir.mkdir(parents=True, exist_ok=True)

    raw = load_sales_csv()
    cleaned = clean_sales_data(raw)
    series = _weekly_revenue_series(cleaned)

    results = {}
    for name in ("arima", "prophet", "lstm", "ensemble"):
        m = model_metrics(series, name)
        results[name] = m
        h, r = m["holdout"], m["rolling"]
        print(
            f"  {name:10s}  holdout={h['accuracy']}%  "
            f"rolling={r['accuracy']}% ({r['folds']} folds)  RMSE={r['rmse']:,.0f}"
        )

    artifact = {
        "trained_at": datetime.now(timezone.utc).isoformat(),
        "forecast_target": "revenue",
        "rows": len(cleaned),
        "weeks": len(series),
        "metrics": results,
        "series_tail": series.tail(52).tolist(),
        "series_dates": [d.strftime("%Y-%m-%d") for d in series.tail(52).index],
    }

    if save:
        joblib.dump(artifact, model_dir / "forecast_artifact.joblib")
        with open(log_dir / "training_log.json", "w", encoding="utf-8") as f:
            json.dump(artifact, f, indent=2, default=str)

    ensemble_model.fit_predict(series, 12)
    print(f"\nSaved artifact -> {model_dir / 'forecast_artifact.joblib'}")
    return artifact


def main() -> None:
    print("=" * 60)
    print("Salevora — Model Training Pipeline (revenue target)")
    print("=" * 60)
    train()


if __name__ == "__main__":
    main()
