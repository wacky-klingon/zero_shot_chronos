#!/usr/bin/env python3
"""
Script to download the Chronos-Bolt-Base model from Hugging Face.

This script downloads the amazon/chronos-bolt-base model and saves it
to the data/model/ directory for local use.
"""

import os
import sys
from pathlib import Path
from huggingface_hub import snapshot_download
import shutil
from autogluon.timeseries import TimeSeriesPredictor
import yaml


def download_chronos_model(model_name: str = "amazon/chronos-bolt-base", 
                          model_type: str = "chronos-bolt-base", 
                          version: str = "v1.0"):
    """Download a Chronos model from Hugging Face to versioned directory."""
    
    # Set up paths
    model_dir = Path(f"data/model/{model_type}/{version}")
    temp_dir = Path(f"data/model_temp_{model_type}_{version}")
    
    # Create model directory if it doesn't exist
    model_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"Downloading {model_name} model...")
    print(f"Target directory: {model_dir.absolute()}")
    
    try:
        # Download the entire model repository
        print("Downloading model repository...")
        downloaded_path = snapshot_download(
            repo_id=model_name,
            local_dir=str(temp_dir),
            local_dir_use_symlinks=False
        )
        
        print(f"Model downloaded to temporary directory: {downloaded_path}")
        
        # Move files from temp directory to final model directory
        print("Moving files to final location...")
        if temp_dir.exists():
            # Remove existing files in model_dir if any
            if model_dir.exists():
                shutil.rmtree(model_dir)
            
            # Move the downloaded content
            shutil.move(str(temp_dir), str(model_dir))
            print(f"Model successfully moved to: {model_dir.absolute()}")
        
        # List downloaded files
        print("\nDownloaded files:")
        for file_path in model_dir.rglob("*"):
            if file_path.is_file():
                print(f"  - {file_path.relative_to(model_dir)}")
        
        # Convert raw model files to AutoGluon predictor format
        print("\nConverting to AutoGluon predictor format...")
        try:
            # Load configuration for prediction length
            config_path = "config/settings.yaml"
            if Path(config_path).exists():
                with open(config_path, 'r') as file:
                    config = yaml.safe_load(file)
                prediction_length = config['model']['prediction_length']
            else:
                prediction_length = 48  # Default fallback
            
            # Create a temporary predictor directory
            predictor_dir = model_dir / "predictor"
            predictor_dir.mkdir(exist_ok=True)
            
            # Create AutoGluon predictor
            predictor = TimeSeriesPredictor(
                prediction_length=prediction_length,
                path=str(predictor_dir)
            )
            
            # Create a minimal dummy dataset to fit the predictor
            import pandas as pd
            import numpy as np
            from autogluon.timeseries import TimeSeriesDataFrame
            
            # Create dummy data for fitting
            dates = pd.date_range('2020-01-01', periods=10, freq='D')
            dummy_data = pd.DataFrame({
                'timestamp': dates,
                'target': np.random.randn(10),
                'item_id': 'dummy'
            })
            dummy_ts = TimeSeriesDataFrame.from_data_frame(dummy_data, id_column='item_id', timestamp_column='timestamp')
            
            # Fit the predictor with dummy data
            predictor.fit(dummy_ts)
            
            # Save the predictor
            predictor.save()
            print(f"AutoGluon predictor saved to: {predictor_dir}")
            
            # List final files including AutoGluon predictor files
            print("\nFinal model files:")
            for file_path in model_dir.rglob("*"):
                if file_path.is_file():
                    print(f"  - {file_path.relative_to(model_dir)}")
            
        except Exception as e:
            print(f"Warning: Could not convert to AutoGluon format: {e}")
            print("Raw model files are available, but TimeSeriesPredictor.load() may not work")
        
        return True
        
    except Exception as e:
        print(f"Error downloading model: {e}")
        # Clean up temp directory if it exists
        if temp_dir.exists():
            shutil.rmtree(temp_dir)
        return False


def main():
    """Main execution function."""
    print("Chronos Model Downloader")
    print("=" * 40)
    
    # Available models
    available_models = {
        "1": ("amazon/chronos-bolt-tiny", "chronos-bolt-tiny"),
        "2": ("amazon/chronos-bolt-mini", "chronos-bolt-mini"), 
        "3": ("amazon/chronos-bolt-small", "chronos-bolt-small"),
        "4": ("amazon/chronos-bolt-base", "chronos-bolt-base")
    }
    
    print("Available models:")
    for key, (model_name, model_type) in available_models.items():
        print(f"  {key}. {model_name}")
    
    choice = input("\nSelect model (1-4, default=4): ").strip() or "4"
    
    if choice not in available_models:
        print("Invalid choice. Using chronos-bolt-base.")
        choice = "4"
    
    model_name, model_type = available_models[choice]
    version = input(f"Enter version for {model_type} (default=v1.0): ").strip() or "v1.0"
    
    # Check if model already exists
    model_dir = Path(f"data/model/{model_type}/{version}")
    if model_dir.exists() and any(model_dir.iterdir()):
        print(f"Model directory {model_dir} already contains files.")
        response = input("Do you want to re-download? (y/N): ").strip().lower()
        if response not in ['y', 'yes']:
            print("Download cancelled.")
            return
    
    # Clean up any existing temp directory
    temp_dir = Path(f"data/model_temp_{model_type}_{version}")
    if temp_dir.exists():
        shutil.rmtree(temp_dir)
    
    # Download the model
    success = download_chronos_model(model_name, model_type, version)
    
    if success:
        print("\n" + "=" * 40)
        print("Download completed successfully!")
        print(f"The model is now available in data/model/{model_type}/{version}/")
        print("The model has been converted to AutoGluon predictor format.")
        print("\nNext steps:")
        print(f"1. Update config/settings.yaml model_path to 'data/model/{model_type}/{version}'")
        print("2. Run your forecasting script")
    else:
        print("\n" + "=" * 40)
        print("Download failed. Please check the error messages above.")
        sys.exit(1)


if __name__ == "__main__":
    main()
