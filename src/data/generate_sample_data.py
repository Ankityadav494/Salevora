"""
Generate sample sales data for demonstration
"""
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime, timedelta


def generate_sample_sales_data(start_date: str = '2020-01-01', 
                              end_date: str = '2024-12-31',
                              n_products: int = 5,
                              output_path: str = 'data/raw/sales_data.csv') -> pd.DataFrame:
    """
    Generate synthetic sales data with realistic patterns
    
    Args:
        start_date: Start date for the data
        end_date: End date for the data
        n_products: Number of different products
        output_path: Path to save the generated data
        
    Returns:
        pd.DataFrame: Generated sales dataframe
    """
    # Create date range
    dates = pd.date_range(start=start_date, end=end_date, freq='D')
    n_days = len(dates)
    
    # Initialize data list
    data = []
    
    # Generate data for each product
    for product_id in range(1, n_products + 1):
        # Base sales level (different for each product)
        base_sales = np.random.uniform(100, 500)
        
        # Trend component (slight upward trend)
        trend = np.linspace(0, 0.5, n_days)
        
        # Seasonal component (yearly seasonality)
        seasonal = 50 * np.sin(2 * np.pi * np.arange(n_days) / 365.25)
        
        # Weekly seasonality
        weekly = 20 * np.sin(2 * np.pi * np.arange(n_days) / 7)
        
        # Random noise
        noise = np.random.normal(0, 30, n_days)
        
        # Special events (promotions, holidays)
        events = np.zeros(n_days)
        n_events = np.random.randint(10, 30)
        event_days = np.random.choice(n_days, n_events, replace=False)
        events[event_days] = np.random.uniform(50, 150, n_events)
        
        # Calculate sales
        sales = base_sales + trend * base_sales + seasonal + weekly + noise + events
        sales = np.maximum(sales, 0)  # Ensure non-negative
        
        # Create product data
        product_data = pd.DataFrame({
            'date': dates,
            'product_id': f'Product_{product_id}',
            'sales': sales.round(2),
            'quantity': (sales / np.random.uniform(10, 50)).astype(int),
            'price': np.random.uniform(20, 100, n_days).round(2),
            'category': np.random.choice(['Electronics', 'Clothing', 'Food', 'Books', 'Toys'], n_days)
        })
        
        data.append(product_data)
    
    # Combine all products
    df = pd.concat(data, ignore_index=True)
    
    # Add some additional features
    df['revenue'] = df['sales'] * df['quantity']
    df['day_of_week'] = pd.to_datetime(df['date']).dt.day_name()
    df['month'] = pd.to_datetime(df['date']).dt.month_name()
    
    # Ensure output directory exists
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Save to CSV
    df.to_csv(output_path, index=False)
    print(f"Sample data generated successfully! Saved to {output_path}")
    print(f"Shape: {df.shape}")
    print(f"Date range: {df['date'].min()} to {df['date'].max()}")
    
    return df


if __name__ == "__main__":
    # Generate sample data
    df = generate_sample_sales_data()
    print("\nFirst few rows:")
    print(df.head())
    print("\nData info:")
    print(df.info())
