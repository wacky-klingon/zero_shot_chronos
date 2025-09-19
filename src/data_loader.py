"""
Data loading and preprocessing module for Chronos time series forecasting.

This module provides the TimeSeriesDataLoader class for loading and preprocessing
time series data from local CSV files, with support for both single and multi-series data.
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Union, Optional, Tuple
from autogluon.timeseries import TimeSeriesDataFrame
import yaml


class TimeSeriesDataLoader:
    """Load and preprocess time series data from local files."""
    
    def __init__(self, config_path: str = "config/settings.yaml") -> None:
        """
        Initialize the data loader with configuration.
        
        Args:
            config_path: Path to the YAML configuration file
        """
        with open(config_path, 'r') as file:
            self.config = yaml.safe_load(file)
        
        self.input_dir = Path(self.config['data']['input_dir'])
        self.output_dir = Path(self.config['data']['output_dir'])
        self.output_dir.mkdir(parents=True, exist_ok=True)
    
    def load_from_csv(self, file_path: Union[str, Path]) -> TimeSeriesDataFrame:
        """
        Load time series data from CSV file.
        
        Args:
            file_path: Path to the CSV file containing time series data
            
        Returns:
            TimeSeriesDataFrame: Processed time series data
            
        Raises:
            FileNotFoundError: If the specified file doesn't exist
            ValueError: If required columns are missing
        """
        file_path = Path(file_path)
        
        if not file_path.exists():
            raise FileNotFoundError(f"Data file not found: {file_path}")
        
        try:
            df = pd.read_csv(file_path)
        except Exception as e:
            raise ValueError(f"Error reading CSV file: {e}")
        
        # Ensure proper time series format
        if 'timestamp' in df.columns:
            df['timestamp'] = pd.to_datetime(df['timestamp'])
        elif 'date' in df.columns:
            df['timestamp'] = pd.to_datetime(df['date'])
            df = df.drop('date', axis=1)
        else:
            raise ValueError("CSV must contain either 'timestamp' or 'date' column")
        
        # Check for required value column
        if 'value' not in df.columns:
            raise ValueError("CSV must contain 'value' column")
        
        # Rename 'value' to 'target' for AutoGluon compatibility
        df = df.rename(columns={'value': 'target'})
        
        # Create TimeSeriesDataFrame
        try:
            ts_df = TimeSeriesDataFrame.from_data_frame(
                df, 
                id_column="item_id" if "item_id" in df.columns else None,
                timestamp_column="timestamp"
            )
        except Exception as e:
            raise ValueError(f"Error creating TimeSeriesDataFrame: {e}")
        
        return ts_df
    
    def train_test_split(self, data: TimeSeriesDataFrame, 
                        prediction_length: int) -> Tuple[TimeSeriesDataFrame, TimeSeriesDataFrame]:
        """
        Split data into train and test sets.
        
        Args:
            data: TimeSeriesDataFrame to split
            prediction_length: Number of time steps to use for testing
            
        Returns:
            Tuple of (train_data, test_data)
        """
        try:
            return data.train_test_split(prediction_length)
        except Exception as e:
            raise ValueError(f"Error splitting data: {e}")
    
    def save_processed_data(self, data: TimeSeriesDataFrame, 
                           filename: str) -> None:
        """
        Save processed data to output directory.
        
        Args:
            data: TimeSeriesDataFrame to save
            filename: Name of the output file
        """
        try:
            output_path = self.output_dir / filename
            data.to_csv(output_path)
            print(f"Data saved to: {output_path}")
        except Exception as e:
            raise ValueError(f"Error saving data: {e}")
    
    def get_data_info(self, data: TimeSeriesDataFrame) -> dict:
        """
        Get information about the loaded data.
        
        Args:
            data: TimeSeriesDataFrame to analyze
            
        Returns:
            Dictionary containing data statistics
        """
        info = {
            'total_records': len(data),
            'num_series': len(data.item_ids) if hasattr(data, 'item_ids') else 1,
            'date_range': (data.index.get_level_values('timestamp').min(), 
                          data.index.get_level_values('timestamp').max()),
            'columns': list(data.columns)
        }
        return info
