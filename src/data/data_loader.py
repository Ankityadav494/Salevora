"""
Data loading utilities
"""
import pandas as pd
from pathlib import Path
from typing import Optional


def load_data(file_path: str, **kwargs) -> pd.DataFrame:
    """
    Load data from various file formats
    
    Args:
        file_path: Path to the data file
        **kwargs: Additional arguments for pd.read_csv or pd.read_excel
        
    Returns:
        pd.DataFrame: Loaded dataframe
    """
    file_path = Path(file_path)
    
    if not file_path.exists():
        raise FileNotFoundError(f"Data file not found: {file_path}")
    
    # Determine file type and load accordingly
    if file_path.suffix == '.csv':
        df = pd.read_csv(file_path, **kwargs)
    elif file_path.suffix in ['.xlsx', '.xls']:
        df = pd.read_excel(file_path, **kwargs)
    else:
        raise ValueError(f"Unsupported file format: {file_path.suffix}")
    
    return df


def load_sample_data() -> pd.DataFrame:
    """
    Load sample sales data (if available)
    
    Returns:
        pd.DataFrame: Sample dataframe
    """
    sample_path = Path("data/raw/sales_data.csv")
    
    if sample_path.exists():
        return load_data(sample_path)
    else:
        print("Sample data not found. Please generate sample data first.")
        return pd.DataFrame()
