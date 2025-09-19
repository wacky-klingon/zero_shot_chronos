#!/usr/bin/env python3
"""
Main script for Chronos zero-shot forecasting implementation.

This script demonstrates the complete workflow for time series forecasting
using Chronos models in zero-shot mode.
"""

from src.data_loader import TimeSeriesDataLoader
from src.chronos_predictor import ChronosPredictor
from src.visualization import TimeSeriesVisualizer
from pathlib import Path
import sys


def main():
    """Main execution function."""
    print("Chronos Zero-Shot Forecasting Implementation")
    print("=" * 50)
    
    try:
        # Initialize components
        print("Initializing components...")
        data_loader = TimeSeriesDataLoader()
        predictor = ChronosPredictor()
        visualizer = TimeSeriesVisualizer()
        print("Components initialized successfully")
        
        # Load data (using sample data file)
        data_file = "data/raw/sample_timeseries_data.csv"
        
        if not Path(data_file).exists():
            print(f"Data file not found: {data_file}")
            print("Please place your time series data in the data/raw/ directory")
            print("Expected format: CSV with columns 'timestamp', 'value', and optionally 'item_id'")
            print("\nExample data format:")
            print("timestamp,value,item_id")
            print("2020-01-01,100.5,series_1")
            print("2020-01-02,102.3,series_1")
            print("2020-01-03,98.7,series_1")
            return
        
        # Load and prepare data
        print("\nLoading time series data...")
        try:
            data = data_loader.load_from_csv(data_file)
            data_info = data_loader.get_data_info(data)
            print(f"Data loaded successfully:")
            print(f"  - Total records: {data_info['total_records']}")
            print(f"  - Number of series: {data_info['num_series']}")
            print(f"  - Date range: {data_info['date_range'][0]} to {data_info['date_range'][1]}")
            print(f"  - Columns: {data_info['columns']}")
        except Exception as e:
            print(f"Error loading data: {e}")
            return
        
        # Split data
        print("\nSplitting data into train/test sets...")
        try:
            train_data, test_data = data_loader.train_test_split(
                data, 
                predictor.prediction_length
            )
            print(f"Train data: {len(train_data)} records")
            print(f"Test data: {len(test_data)} records")
        except Exception as e:
            print(f"Error splitting data: {e}")
            return
        
        # Save processed data
        print("\nSaving processed data...")
        try:
            data_loader.save_processed_data(train_data, "train_data.csv")
            data_loader.save_processed_data(test_data, "test_data.csv")
        except Exception as e:
            print(f"Warning: Error saving processed data: {e}")
        
        # Fit Chronos model (zero-shot)
        print(f"\nFitting Chronos model ({predictor.model_preset})...")
        try:
            predictor.fit(train_data)
            print("Model fitted successfully")
        except Exception as e:
            print(f"Error fitting model: {e}")
            return
        
        # Generate predictions
        print("\nGenerating predictions...")
        try:
            predictor.predict(train_data)
            print("Predictions generated successfully")
        except Exception as e:
            print(f"Error generating predictions: {e}")
            return
        
        # Save predictions
        print("\nSaving predictions...")
        try:
            predictor.save_predictions()
        except Exception as e:
            print(f"Warning: Error saving predictions: {e}")
        
        # Evaluate model
        print("\nEvaluating model performance...")
        try:
            leaderboard = predictor.get_leaderboard(test_data)
        except Exception as e:
            print(f"Warning: Error generating leaderboard: {e}")
            leaderboard = None
        
        # Create visualizations
        print("\nCreating visualizations...")
        try:
            # Data distribution plot
            visualizer.plot_data_distribution(
                data,
                save_path="data/predictions/data_distribution.png"
            )
            
            # Prediction plots
            visualizer.plot_predictions(
                data, 
                predictor.predictions,
                save_path="data/predictions/forecast_plot.png"
            )
            
            # Leaderboard plot (if available)
            if leaderboard is not None:
                visualizer.plot_leaderboard(
                    leaderboard,
                    save_path="data/predictions/leaderboard.png"
                )
            
        except Exception as e:
            print(f"Warning: Error creating visualizations: {e}")
        
        print("\n" + "=" * 50)
        print("Implementation complete!")
        print("Check the following directories for outputs:")
        print("  - Processed data: data/processed/")
        print("  - Predictions: data/predictions/")
        print("  - Visualizations: data/predictions/*.png")
        
    except KeyboardInterrupt:
        print("\n\nOperation cancelled by user")
        sys.exit(1)
    except Exception as e:
        print(f"\nUnexpected error: {e}")
        print("Please check your configuration and data format")
        sys.exit(1)


if __name__ == "__main__":
    main()
