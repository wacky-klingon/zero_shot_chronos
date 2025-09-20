"""
Chronos predictor module for zero-shot time series forecasting.

This module provides the ChronosPredictor class for fitting and using Chronos models
in zero-shot mode for time series forecasting.
"""

from autogluon.timeseries import TimeSeriesPredictor, TimeSeriesDataFrame
from pathlib import Path
from typing import Optional, Union
import yaml
import pandas as pd


class ChronosPredictor:
    """Wrapper for Chronos zero-shot forecasting."""
    
    def __init__(self, config_path: str = "config/settings.yaml") -> None:
        """
        Initialize the Chronos predictor with configuration.
        
        Args:
            config_path: Path to the YAML configuration file
        """
        with open(config_path, 'r') as file:
            self.config = yaml.safe_load(file)
        
        self.prediction_length = self.config['model']['prediction_length']
        self.model_path = self.config['model']['model_path']
        self.model_name = self.config['model']['model_name']
        self.model_type = self.config['model']['model_type']
        self.version = self.config['model']['version']
        self.loading_mode = self.config['model']['loading_mode']
        self.auto_detect_mode = self.config['model']['auto_detect_mode']
        self.predictions_dir = Path(self.config['data']['predictions_dir'])
        self.predictions_dir.mkdir(parents=True, exist_ok=True)
        
        self.predictor: Optional[TimeSeriesPredictor] = None
        self.predictions: Optional[TimeSeriesDataFrame] = None
    
    def fit(self, train_data: TimeSeriesDataFrame) -> 'ChronosPredictor':
        """
        Fit the Chronos model in zero-shot mode.
        
        Args:
            train_data: Training data for the model
            
        Returns:
            Self for method chaining
            
        Raises:
            ValueError: If training data is invalid
        """
        try:
            # Check if model path exists for inference mode
            model_path_exists = Path(self.model_path).exists()
            
            if (self.loading_mode == "inference" and model_path_exists) or (self.auto_detect_mode and model_path_exists):
                # Check if predictor subdirectory exists
                predictor_path = Path(self.model_path) / "predictor"
                if predictor_path.exists():
                    # Load existing AutoGluon predictor
                    self.predictor = TimeSeriesPredictor.load(str(predictor_path))
                    print(f"Chronos model ({self.model_type} v{self.version}) loaded from {predictor_path}")
                else:
                    # Fall back to preset mode
                    print(f"Warning: No AutoGluon predictor found at {predictor_path}. Using preset mode.")
                    raise ValueError("No AutoGluon predictor found")
            else:
                # Use preset for training or when model doesn't exist
                if self.loading_mode == "inference" and not model_path_exists:
                    print(f"Warning: Model path {self.model_path} does not exist. Using preset instead.")
                
                preset_name = f"chronos_{self.model_type.replace('-', '_')}"
                self.predictor = TimeSeriesPredictor(
                    prediction_length=self.prediction_length
                ).fit(
                    train_data, 
                    presets=preset_name
                )
                print(f"Chronos model ({preset_name}) fitted successfully")
            return self
        except Exception as e:
            raise ValueError(f"Error fitting Chronos model: {e}")
    
    def predict(self, data: TimeSeriesDataFrame) -> 'ChronosPredictor':
        """
        Generate predictions.
        
        Args:
            data: Data to make predictions on
            
        Returns:
            Self for method chaining
            
        Raises:
            ValueError: If model is not fitted or prediction fails
        """
        if self.predictor is None:
            raise ValueError("Model must be fitted before making predictions")
        
        try:
            self.predictions = self.predictor.predict(data)
            print(f"Predictions generated for {len(data.item_ids)} series")
            return self
        except Exception as e:
            raise ValueError(f"Error generating predictions: {e}")
    
    def save_predictions(self, filename: str = "chronos_predictions.csv") -> None:
        """
        Save predictions to file.
        
        Args:
            filename: Name of the output file
            
        Raises:
            ValueError: If no predictions are available to save
        """
        if self.predictions is None:
            raise ValueError("No predictions available to save")
        
        try:
            output_path = self.predictions_dir / filename
            self.predictions.to_csv(output_path)
            print(f"Predictions saved to: {output_path}")
        except Exception as e:
            raise ValueError(f"Error saving predictions: {e}")
    
    def get_leaderboard(self, test_data: TimeSeriesDataFrame) -> pd.DataFrame:
        """
        Display model performance leaderboard.
        
        Args:
            test_data: Test data for evaluation
            
        Returns:
            DataFrame containing model performance metrics
            
        Raises:
            ValueError: If model is not fitted or evaluation fails
        """
        if self.predictor is None:
            raise ValueError("Model must be fitted before evaluation")
        
        try:
            leaderboard = self.predictor.leaderboard(test_data)
            print("Model Performance Leaderboard:")
            print(leaderboard)
            return leaderboard
        except Exception as e:
            raise ValueError(f"Error generating leaderboard: {e}")
    
    def plot_predictions(self, data: TimeSeriesDataFrame, 
                        item_ids: Optional[list] = None,
                        max_history_length: int = 200) -> None:
        """
        Plot predictions using the predictor's built-in plotting functionality.
        
        Args:
            data: Original data for plotting
            item_ids: List of item IDs to plot (default: first 2)
            max_history_length: Maximum history length to show
        """
        if self.predictor is None or self.predictions is None:
            raise ValueError("Model must be fitted and predictions generated before plotting")
        
        try:
            if item_ids is None:
                item_ids = data.item_ids[:2]  # Plot first 2 series by default
            
            self.predictor.plot(
                data=data,
                predictions=self.predictions,
                item_ids=item_ids,
                max_history_length=max_history_length
            )
            print(f"Plotted predictions for items: {item_ids}")
        except Exception as e:
            raise ValueError(f"Error plotting predictions: {e}")
    
    def is_fitted(self) -> bool:
        """
        Check if the model is fitted.
        
        Returns:
            True if model is fitted, False otherwise
        """
        return self.predictor is not None
    
    def has_predictions(self) -> bool:
        """
        Check if predictions are available.
        
        Returns:
            True if predictions are available, False otherwise
        """
        return self.predictions is not None
