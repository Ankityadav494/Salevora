"""
Configuration management utilities
"""
import yaml
import os
from pathlib import Path


def load_config(config_path="config.yaml"):
    """
    Load configuration from YAML file
    
    Args:
        config_path: Path to configuration file
        
    Returns:
        dict: Configuration dictionary
    """
    config_file = Path(config_path)
    
    if not config_file.exists():
        raise FileNotFoundError(f"Configuration file not found: {config_path}")
    
    with open(config_file, 'r') as f:
        config = yaml.safe_load(f)
    
    return config


def get_data_path(config, path_key="raw_data_path"):
    """
    Get data path from configuration
    
    Args:
        config: Configuration dictionary
        path_key: Key for the path in config
        
    Returns:
        Path: Path object
    """
    data_path = config['data'][path_key]
    return Path(data_path)


def ensure_directories():
    """
    Ensure all necessary directories exist
    """
    directories = [
        "data/raw",
        "data/processed",
        "data/external",
        "models",
        "logs",
        "reports"
    ]
    
    for directory in directories:
        Path(directory).mkdir(parents=True, exist_ok=True)
