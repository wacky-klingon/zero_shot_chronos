"""
Visualization utilities for time series forecasting.

This module provides the TimeSeriesVisualizer class for creating plots and visualizations
for time series forecasting results.
"""

import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from typing import Optional, List, Union
import yaml
import pandas as pd
from autogluon.timeseries import TimeSeriesDataFrame


class TimeSeriesVisualizer:
    """Visualization utilities for time series forecasting."""
    
    def __init__(self, config_path: str = "config/settings.yaml") -> None:
        """
        Initialize the visualizer with configuration.
        
        Args:
            config_path: Path to the YAML configuration file
        """
        with open(config_path, 'r') as file:
            self.config = yaml.safe_load(file)
        
        self.figure_size = tuple(self.config['visualization']['figure_size'])
        self.max_history_length = self.config['visualization']['max_history_length']
        self.style = self.config['visualization']['style']
        
        # Set plotting style
        try:
            plt.style.use(self.style)
        except OSError:
            print(f"Warning: Style '{self.style}' not found, using default")
            plt.style.use('default')
        
        sns.set_palette("husl")
    
    def plot_predictions(self, data: TimeSeriesDataFrame, 
                        predictions: TimeSeriesDataFrame,
                        item_ids: Optional[List[str]] = None,
                        save_path: Optional[Union[str, Path]] = None) -> None:
        """
        Plot time series data with predictions.
        
        Args:
            data: Original time series data
            predictions: Predicted values
            item_ids: List of item IDs to plot (default: first 2)
            save_path: Path to save the plot (optional)
        """
        if item_ids is None:
            item_ids = data.item_ids[:2]  # Plot first 2 series by default
        
        try:
            fig, ax = plt.subplots(figsize=self.figure_size)
            
            # This is a placeholder for actual plotting logic
            # In practice, you would use the predictor's plot method
            print(f"Plotting predictions for items: {item_ids}")
            print(f"Max history length: {self.max_history_length}")
            print(f"Data shape: {data.shape}, Predictions shape: {predictions.shape}")
            
            # Create a simple visualization
            ax.set_title('Time Series Predictions')
            ax.set_xlabel('Time')
            ax.set_ylabel('Value')
            ax.grid(True, alpha=0.3)
            
            if save_path:
                save_path = Path(save_path)
                save_path.parent.mkdir(parents=True, exist_ok=True)
                plt.savefig(save_path, dpi=300, bbox_inches='tight')
                print(f"Plot saved to: {save_path}")
            
            plt.show()
            
        except Exception as e:
            raise ValueError(f"Error creating prediction plot: {e}")
    
    def plot_leaderboard(self, leaderboard: pd.DataFrame, 
                        save_path: Optional[Union[str, Path]] = None) -> None:
        """
        Plot model performance comparison.
        
        Args:
            leaderboard: DataFrame containing model performance metrics
            save_path: Path to save the plot (optional)
        """
        try:
            fig, ax = plt.subplots(figsize=self.figure_size)
            
            # Create a simple bar plot of model scores
            if 'score_test' in leaderboard.columns:
                leaderboard.plot(x='model', y='score_test', kind='bar', ax=ax)
                ax.set_title('Model Performance Comparison')
                ax.set_ylabel('Test Score')
                ax.set_xlabel('Model')
                plt.xticks(rotation=45)
            else:
                # Fallback if score_test column doesn't exist
                ax.text(0.5, 0.5, 'No score_test column found in leaderboard', 
                       ha='center', va='center', transform=ax.transAxes)
                ax.set_title('Model Performance Comparison')
            
            plt.tight_layout()
            
            if save_path:
                save_path = Path(save_path)
                save_path.parent.mkdir(parents=True, exist_ok=True)
                plt.savefig(save_path, dpi=300, bbox_inches='tight')
                print(f"Leaderboard plot saved to: {save_path}")
            
            plt.show()
            
        except Exception as e:
            raise ValueError(f"Error creating leaderboard plot: {e}")
    
    def plot_data_distribution(self, data: TimeSeriesDataFrame,
                              save_path: Optional[Union[str, Path]] = None) -> None:
        """
        Plot distribution of time series data.
        
        Args:
            data: TimeSeriesDataFrame to analyze
            save_path: Path to save the plot (optional)
        """
        try:
            fig, axes = plt.subplots(2, 2, figsize=self.figure_size)
            
            # Target distribution
            data['target'].hist(ax=axes[0, 0], bins=30, alpha=0.7)
            axes[0, 0].set_title('Value Distribution')
            axes[0, 0].set_xlabel('Value')
            axes[0, 0].set_ylabel('Frequency')
            
            # Time series plot (first few series)
            if hasattr(data, 'item_ids') and len(data.item_ids) > 0:
                for i, item_id in enumerate(data.item_ids[:3]):  # Plot first 3 series
                    series_data = data.loc[data.index.get_level_values('item_id') == item_id]
                    axes[0, 1].plot(series_data.index.get_level_values('timestamp'), 
                                   series_data['target'], label=f'Series {item_id}', alpha=0.7)
                axes[0, 1].set_title('Time Series Plot')
                axes[0, 1].set_xlabel('Time')
                axes[0, 1].set_ylabel('Value')
                axes[0, 1].legend()
                axes[0, 1].tick_params(axis='x', rotation=45)
            
            # Box plot by series (if multiple series)
            if hasattr(data, 'item_ids') and len(data.item_ids) > 1:
                data.boxplot(column='target', by='item_id', ax=axes[1, 0])
                axes[1, 0].set_title('Value Distribution by Series')
                axes[1, 0].set_xlabel('Series ID')
                axes[1, 0].set_ylabel('Value')
            
            # Missing values check
            missing_data = data.isnull().sum()
            if missing_data.sum() > 0:
                missing_data.plot(kind='bar', ax=axes[1, 1])
                axes[1, 1].set_title('Missing Values by Column')
                axes[1, 1].set_xlabel('Column')
                axes[1, 1].set_ylabel('Missing Count')
            else:
                axes[1, 1].text(0.5, 0.5, 'No Missing Values', 
                               ha='center', va='center', transform=axes[1, 1].transAxes)
                axes[1, 1].set_title('Missing Values Check')
            
            plt.tight_layout()
            
            if save_path:
                save_path = Path(save_path)
                save_path.parent.mkdir(parents=True, exist_ok=True)
                plt.savefig(save_path, dpi=300, bbox_inches='tight')
                print(f"Data distribution plot saved to: {save_path}")
            
            plt.show()
            
        except Exception as e:
            raise ValueError(f"Error creating data distribution plot: {e}")
    
    def set_style(self, style: str) -> None:
        """
        Set the plotting style.
        
        Args:
            style: Matplotlib style name
        """
        try:
            plt.style.use(style)
            self.style = style
        except OSError:
            print(f"Warning: Style '{style}' not found, keeping current style")
    
    def set_figure_size(self, size: tuple) -> None:
        """
        Set the default figure size.
        
        Args:
            size: Tuple of (width, height) in inches
        """
        self.figure_size = size
