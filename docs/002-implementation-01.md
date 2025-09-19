# Implementation Plan: Zero-Shot Forecasting with Chronos

This document outlines a simplified implementation plan for setting up univariate zero-shot forecasting using AutoGluon-TimeSeries with Chronos models, using local files instead of remote data sources.

## Overview

The implementation will focus on:
- Setting up the environment with AutoGluon-TimeSeries
- Preparing local data files for time series forecasting
- Implementing zero-shot forecasting with Chronos-Bolt models
- Generating and visualizing predictions

## Project Structure

```
chronos-raw/
├── data/
│   ├── raw/                    # Original time series data
│   ├── processed/              # Cleaned and formatted data
│   └── predictions/            # Generated forecasts
├── src/
│   ├── data_loader.py          # Data loading and preprocessing
│   ├── chronos_predictor.py    # Chronos model wrapper
│   └── visualization.py        # Plotting utilities
├── config/
│   └── settings.yaml           # Configuration parameters
└── docs/
    ├── 001-RAW.md
    ├── 002-implementation-01.md
    └── 003-finetuning.md
```

## Implementation Steps

### 1. Environment Setup

Create a modern Python project configuration:

**File: `pyproject.toml`**
```toml
[build-system]
requires = ["setuptools>=61.0", "wheel"]
build-backend = "setuptools.build_meta"

[project]
name = "chronos-raw"
version = "0.1.0"
description = "Chronos time series forecasting with AutoGluon"
authors = [
    {name = "Your Name", email = "your.email@example.com"}
]
readme = "README.md"
requires-python = ">=3.8"
classifiers = [
    "Development Status :: 3 - Alpha",
    "Intended Audience :: Developers",
    "License :: OSI Approved :: MIT License",
    "Programming Language :: Python :: 3",
    "Programming Language :: Python :: 3.8",
    "Programming Language :: Python :: 3.9",
    "Programming Language :: Python :: 3.10",
    "Programming Language :: Python :: 3.11",
]

dependencies = [
    "autogluon.timeseries>=1.2.0",
    "pandas>=1.5.0",
    "numpy>=1.21.0",
    "matplotlib>=3.5.0",
    "seaborn>=0.11.0",
    "pyyaml>=6.0",
]

[project.optional-dependencies]
dev = [
    "pytest>=7.0",
    "pytest-cov>=4.0",
    "black>=22.0",
    "isort>=5.0",
    "flake8>=5.0",
    "mypy>=1.0",
]

[project.urls]
Homepage = "https://github.com/yourusername/chronos-raw"
Repository = "https://github.com/yourusername/chronos-raw.git"
Issues = "https://github.com/yourusername/chronos-raw/issues"

[tool.setuptools.packages.find]
where = ["src"]

[tool.setuptools.package-dir]
"" = "src"

[tool.black]
line-length = 88
target-version = ['py38']

[tool.isort]
profile = "black"
multi_line_output = 3

[tool.mypy]
python_version = "3.8"
warn_return_any = true
warn_unused_configs = true
disallow_untyped_defs = true
```

**File: `requirements.txt` (optional, for pip install)**
```
autogluon.timeseries>=1.2.0
pandas>=1.5.0
numpy>=1.21.0
matplotlib>=3.5.0
seaborn>=0.11.0
pyyaml>=6.0
```

### 2. Configuration Management

**File: `config/settings.yaml`**
```yaml
data:
  input_dir: "data/raw"
  output_dir: "data/processed"
  predictions_dir: "data/predictions"
  
model:
  prediction_length: 48
  model_preset: "bolt_small"
  
visualization:
  max_history_length: 200
  figure_size: [12, 8]
  style: "seaborn-v0_8"
```

### 3. Data Loading Module

**File: `src/data_loader.py`**
```python
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Union, Optional
from autogluon.timeseries import TimeSeriesDataFrame
import yaml

class TimeSeriesDataLoader:
    """Load and preprocess time series data from local files."""
    
    def __init__(self, config_path: str = "config/settings.yaml"):
        with open(config_path, 'r') as file:
            self.config = yaml.safe_load(file)
        
        self.input_dir = Path(self.config['data']['input_dir'])
        self.output_dir = Path(self.config['data']['output_dir'])
        self.output_dir.mkdir(parents=True, exist_ok=True)
    
    def load_from_csv(self, file_path: Union[str, Path]) -> TimeSeriesDataFrame:
        """Load time series data from CSV file."""
        df = pd.read_csv(file_path)
        
        # Ensure proper time series format
        if 'timestamp' in df.columns:
            df['timestamp'] = pd.to_datetime(df['timestamp'])
        elif 'date' in df.columns:
            df['timestamp'] = pd.to_datetime(df['date'])
            df = df.drop('date', axis=1)
        
        # Create TimeSeriesDataFrame
        ts_df = TimeSeriesDataFrame.from_data_frame(
            df, 
            id_column="item_id" if "item_id" in df.columns else None,
            timestamp_column="timestamp"
        )
        
        return ts_df
    
    def train_test_split(self, data: TimeSeriesDataFrame, 
                        prediction_length: int) -> tuple:
        """Split data into train and test sets."""
        return data.train_test_split(prediction_length)
    
    def save_processed_data(self, data: TimeSeriesDataFrame, 
                           filename: str) -> None:
        """Save processed data to output directory."""
        output_path = self.output_dir / filename
        data.to_csv(output_path)
        print(f"Data saved to: {output_path}")
```

### 4. Chronos Predictor Module

**File: `src/chronos_predictor.py`**
```python
from autogluon.timeseries import TimeSeriesPredictor
from pathlib import Path
from typing import Optional
import yaml

class ChronosPredictor:
    """Wrapper for Chronos zero-shot forecasting."""
    
    def __init__(self, config_path: str = "config/settings.yaml"):
        with open(config_path, 'r') as file:
            self.config = yaml.safe_load(file)
        
        self.prediction_length = self.config['model']['prediction_length']
        self.model_preset = self.config['model']['model_preset']
        self.predictions_dir = Path(self.config['data']['predictions_dir'])
        self.predictions_dir.mkdir(parents=True, exist_ok=True)
        
        self.predictor = None
    
    def fit(self, train_data) -> 'ChronosPredictor':
        """Fit the Chronos model in zero-shot mode."""
        self.predictor = TimeSeriesPredictor(
            prediction_length=self.prediction_length
        ).fit(
            train_data, 
            presets=self.model_preset
        )
        print(f"Chronos model ({self.model_preset}) fitted successfully")
        return self
    
    def predict(self, data) -> 'ChronosPredictor':
        """Generate predictions."""
        if self.predictor is None:
            raise ValueError("Model must be fitted before making predictions")
        
        self.predictions = self.predictor.predict(data)
        return self
    
    def save_predictions(self, filename: str = "chronos_predictions.csv") -> None:
        """Save predictions to file."""
        if not hasattr(self, 'predictions'):
            raise ValueError("No predictions available to save")
        
        output_path = self.predictions_dir / filename
        self.predictions.to_csv(output_path)
        print(f"Predictions saved to: {output_path}")
    
    def get_leaderboard(self, test_data) -> None:
        """Display model performance leaderboard."""
        if self.predictor is None:
            raise ValueError("Model must be fitted before evaluation")
        
        leaderboard = self.predictor.leaderboard(test_data)
        print(leaderboard)
        return leaderboard
```

### 5. Visualization Module

**File: `src/visualization.py`**
```python
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from typing import Optional
import yaml

class TimeSeriesVisualizer:
    """Visualization utilities for time series forecasting."""
    
    def __init__(self, config_path: str = "config/settings.yaml"):
        with open(config_path, 'r') as file:
            self.config = yaml.safe_load(file)
        
        self.figure_size = tuple(self.config['visualization']['figure_size'])
        self.max_history_length = self.config['visualization']['max_history_length']
        self.style = self.config['visualization']['style']
        
        # Set plotting style
        plt.style.use(self.style)
        sns.set_palette("husl")
    
    def plot_predictions(self, data, predictions, 
                        item_ids: Optional[list] = None,
                        save_path: Optional[str] = None) -> None:
        """Plot time series data with predictions."""
        if item_ids is None:
            item_ids = data.item_ids[:2]  # Plot first 2 series by default
        
        fig, ax = plt.subplots(figsize=self.figure_size)
        
        # This would use the predictor's plot method in practice
        # For now, we'll create a simple visualization framework
        print(f"Plotting predictions for items: {item_ids}")
        print(f"Max history length: {self.max_history_length}")
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Plot saved to: {save_path}")
        
        plt.show()
    
    def plot_leaderboard(self, leaderboard, save_path: Optional[str] = None) -> None:
        """Plot model performance comparison."""
        fig, ax = plt.subplots(figsize=self.figure_size)
        
        # Create a simple bar plot of model scores
        if 'score_test' in leaderboard.columns:
            leaderboard.plot(x='model', y='score_test', kind='bar', ax=ax)
            ax.set_title('Model Performance Comparison')
            ax.set_ylabel('Test Score')
            plt.xticks(rotation=45)
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Leaderboard plot saved to: {save_path}")
        
        plt.show()
```

### 6. Main Implementation Script

**File: `main.py`**
```python
#!/usr/bin/env python3
"""
Main script for Chronos zero-shot forecasting implementation.
"""

from src.data_loader import TimeSeriesDataLoader
from src.chronos_predictor import ChronosPredictor
from src.visualization import TimeSeriesVisualizer
from pathlib import Path

def main():
    """Main execution function."""
    print("Chronos Zero-Shot Forecasting Implementation")
    print("=" * 50)
    
    # Initialize components
    data_loader = TimeSeriesDataLoader()
    predictor = ChronosPredictor()
    visualizer = TimeSeriesVisualizer()
    
    # Load data (replace with your local data file)
    data_file = "data/raw/your_timeseries_data.csv"
    
    if not Path(data_file).exists():
        print(f"Data file not found: {data_file}")
        print("Please place your time series data in the data/raw/ directory")
        print("Expected format: CSV with columns 'timestamp', 'value', and optionally 'item_id'")
        return
    
    # Load and prepare data
    print("Loading time series data...")
    data = data_loader.load_from_csv(data_file)
    print(f"Data loaded: {len(data)} records")
    
    # Split data
    train_data, test_data = data_loader.train_test_split(
        data, 
        predictor.prediction_length
    )
    print(f"Train data: {len(train_data)} records")
    print(f"Test data: {len(test_data)} records")
    
    # Fit Chronos model (zero-shot)
    print("Fitting Chronos model...")
    predictor.fit(train_data)
    
    # Generate predictions
    print("Generating predictions...")
    predictor.predict(train_data)
    
    # Save predictions
    predictor.save_predictions()
    
    # Evaluate model
    print("Evaluating model performance...")
    leaderboard = predictor.get_leaderboard(test_data)
    
    # Visualize results
    print("Creating visualizations...")
    visualizer.plot_predictions(
        data, 
        predictor.predictions,
        save_path="data/predictions/forecast_plot.png"
    )
    
    visualizer.plot_leaderboard(
        leaderboard,
        save_path="data/predictions/leaderboard.png"
    )
    
    print("Implementation complete!")

if __name__ == "__main__":
    main()
```

## Data Requirements

### Expected Data Format

Place your time series data in `data/raw/your_timeseries_data.csv` with the following structure:

```csv
timestamp,value,item_id
2020-01-01,100.5,series_1
2020-01-02,102.3,series_1
2020-01-03,98.7,series_1
...
```

**Required columns:**
- `timestamp`: Date/time values
- `value`: Numeric time series values

**Optional columns:**
- `item_id`: Series identifier (for multivariate data)

## Usage Instructions

1. **Install dependencies:**
   ```bash
   # Using pyproject.toml (recommended)
   pip install -e .
   
   # Or using requirements.txt
   pip install -r requirements.txt
   ```

2. **Prepare your data:**
   - Place time series CSV file in `data/raw/`
   - Update `main.py` with your filename

3. **Run the implementation:**
   ```bash
   python main.py
   ```

4. **Check outputs:**
   - Processed data: `data/processed/`
   - Predictions: `data/predictions/`
   - Visualizations: `data/predictions/*.png`

## Next Steps

This implementation provides the foundation for zero-shot forecasting. The next phase (documented in `003-finetuning.md`) will cover:
- Fine-tuning Chronos models on custom data
- Incorporating covariates and static features
- Advanced model configuration and hyperparameter tuning
