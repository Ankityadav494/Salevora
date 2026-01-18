"""
Base model class for all forecasting models
"""
from abc import ABC, abstractmethod
import joblib
from pathlib import Path


class BaseForecastModel(ABC):
    """
    Abstract base class for all forecasting models
    """
    
    def __init__(self, name: str):
        self.name = name
        self.model = None
        self.is_trained = False
    
    @abstractmethod
    def train(self, X, y):
        """
        Train the model
        
        Args:
            X: Features
            y: Target variable
        """
        pass
    
    @abstractmethod
    def predict(self, X):
        """
        Make predictions
        
        Args:
            X: Features
            
        Returns:
            Predictions
        """
        pass
    
    def save(self, filepath: str):
        """
        Save the model to disk
        
        Args:
            filepath: Path to save the model
        """
        filepath = Path(filepath)
        filepath.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self.model, filepath)
        print(f"Model saved to {filepath}")
    
    def load(self, filepath: str):
        """
        Load the model from disk
        
        Args:
            filepath: Path to load the model from
        """
        self.model = joblib.load(filepath)
        self.is_trained = True
        print(f"Model loaded from {filepath}")
