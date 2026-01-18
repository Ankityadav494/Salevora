"""
Run preprocessing pipeline for Walmart sales data
"""

import sys
from pathlib import Path
import pandas as pd

# Add project root to Python path
sys.path.append(str(Path(__file__).resolve().parents[2]))

from src.data.data_preprocessor import preprocess_sales_data


def main():
    raw_path = Path("data/raw/sales_data.csv")
    processed_path = Path("data/processed/processed_sales_data.csv")

    # Load raw data
    df = pd.read_csv(raw_path)

    # Rename Walmart columns
    df = df.rename(columns={
        'Date': 'date',
        'Weekly_Sales': 'sales'
    })

    # Apply preprocessing
    df_processed = preprocess_sales_data(
        df,
        date_column='date',
        target_column='sales'
    )

    # Ensure output directory exists
    processed_path.parent.mkdir(parents=True, exist_ok=True)

    # Save processed data
    df_processed.to_csv(processed_path)

    print("✅ Preprocessing completed successfully")
    print(f"📁 Saved to: {processed_path}")


if __name__ == "__main__":
    main()
