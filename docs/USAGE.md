# Chronos Time Series Forecasting - Usage Guide

This guide provides instructions for using the Chronos time series forecasting system with versioned model management.

## Quick Start

1. **Install dependencies**: `poetry install`
2. **Download a model**: `python src/download_chronos_model.py`
3. **Run forecasting**: `python main.py`

## Model Management

### Downloading Models

The system supports all Chronos-Bolt model sizes:

```bash
python src/download_chronos_model.py
```

Available models:
- **chronos-bolt-tiny** (9M parameters) - Fastest, least accurate
- **chronos-bolt-mini** (21M parameters) - Good balance
- **chronos-bolt-small** (48M parameters) - Better accuracy
- **chronos-bolt-base** (205M parameters) - Best accuracy, slower

Models are organized in versioned directories: `data/model/{model-type}/{version}/`

### Managing Models

Use the model manager to list, switch, and manage model versions:

```bash
python src/model_manager.py
```

**Key operations:**
- List all available models and versions
- Switch between different model versions
- Create backups of model versions
- Remove old or unused model versions
- View model information and file sizes

### Switching Models

To use a different model, update `config/settings.yaml`:

```yaml
model:
  model_path: "data/model/chronos-bolt-small/v1.0"
  model_type: "chronos-bolt-small"
  version: "v1.0"
```

## Data Preparation

### Input Data Format

The system supports two data input formats:

#### CSV Format (Traditional)
Place your time series data in `data/raw/` as CSV files with these columns:
- `timestamp` - Date/time values
- `value` - Numeric time series values
- `item_id` - (Optional) Series identifier for multi-series data

**Example:**
```csv
timestamp,value,item_id
2020-01-01,100.5,series_1
2020-01-02,102.3,series_1
2020-01-03,98.7,series_1
```

#### Parquet Format (New - Recommended)
For high-performance data loading, use the new parquet loader with structured data:

**Directory Structure:**
```
data/parquet/
├── 2014/
│   ├── 01/
│   │   └── SYMBOL_1min_h15_2014_01_769b531dbb2ff3cb.parquet
│   └── 02/
│       └── SYMBOL_1min_h15_2014_02_769b531dbb2ff3cb.parquet
└── 2015/
    └── 01/
        └── SYMBOL_1min_h15_2015_01_769b531dbb2ff3cb.parquet
```

**Configuration:**
```yaml
# config/parquet_loader_config.yaml
data_paths:
  root_dir: "data/parquet"
  file_pattern: "*.parquet"

file_patterns:
  naming_regex: "^(?P<symbol>\\w+)_(?P<timeframe>\\d+min)_h(?P<horizon>\\d+)_(?P<year>\\d{4})_(?P<month>\\d{2})_(?P<hash>\\w+)\\.parquet$"

schema:
  timestamp: "datetime64[ns]"
  value: "float64"
  item_id: "string"
```

**Usage:**
```python
from src.parquet_loader import ParquetDataLoader

# Load data for specific year range
loader = ParquetDataLoader("config/parquet_loader_config.yaml")
data = loader.load_training_data(
    symbol="SYMBOL",
    year_range=(2014, 2016),
    month_range=(1, 6)
)
```

### Supported Data Types

- **Single series**: One time series in a CSV file
- **Multi-series**: Multiple series with different `item_id` values
- **Date formats**: ISO dates, timestamps, or any pandas-readable format
- **Parquet files**: High-performance structured data with automatic discovery and idempotency

## Running Forecasts

### Basic Forecasting

```bash
python main.py
```

This will:
1. Load data from `data/raw/sample_timeseries_data.csv`
2. Split into train/test sets
3. Load the configured Chronos model
4. Generate predictions
5. Save results to `data/predictions/`
6. Create visualizations

### Output Files

The system generates several output files:

- **`data/processed/train_data.csv`** - Training data
- **`data/processed/test_data.csv`** - Test data  
- **`data/predictions/chronos_predictions.csv`** - Forecast results
- **`data/predictions/forecast_plot.png`** - Prediction visualization
- **`data/predictions/data_distribution.png`** - Data distribution plot
- **`data/predictions/leaderboard.png`** - Model performance metrics

## Configuration

### Model Settings

Edit `config/settings.yaml` to customize:

```yaml
model:
  prediction_length: 48        # Number of future steps to predict
  model_path: "data/model/chronos-bolt-base/v1.0"
  loading_mode: "inference"    # Options: "train", "inference", "auto"
  auto_detect_mode: true       # Auto-detect model existence
```

### Data Settings

```yaml
data:
  input_dir: "data/raw"        # Input data directory
  output_dir: "data/processed" # Processed data directory
  predictions_dir: "data/predictions" # Output directory
  model_dir: "data/model"      # Model storage directory
```

### Visualization Settings

```yaml
visualization:
  max_history_length: 200      # Max history to show in plots
  figure_size: [12, 8]         # Plot dimensions
  style: "seaborn-v0_8"        # Matplotlib style
```

## Advanced Usage

### Parquet Data Loading

The new parquet loader provides advanced features for large-scale time series data:

**Key Features:**
- **Range-based loading**: Load specific year/month ranges
- **Idempotency**: Automatic tracking of processed files
- **Audit logging**: Complete trace of processing activities
- **High performance**: Optimized for large datasets

**Example - Range Loading:**
```python
# Load data for Q1-Q2 2014-2015
data = loader.load_training_data(
    symbol="SYMBOL",
    year_range=(2014, 2015),
    month_range=(1, 6)
)

# Load single year, all months
data = loader.load_training_data(
    symbol="SYMBOL",
    year=2014
)
```

**Example - Prediction Data:**
```python
# Load recent data for prediction
prediction_data = loader.load_prediction_data(
    symbol="SYMBOL",
    year=2024,
    month=12,
    context_length=100
)
```

### Custom Model Paths

For custom models or different locations:

```yaml
model:
  model_path: "/path/to/your/custom/model"
  model_name: "your-custom-model"
  loading_mode: "inference"
```

### Training Mode

To retrain or fine-tune models:

```yaml
model:
  loading_mode: "train"
  auto_detect_mode: false
```

### Multi-Series Forecasting

The system automatically handles multiple time series when `item_id` column is present. Each series will be forecast independently.

## Troubleshooting

### Common Issues

**Model not found**: Ensure the model path in config matches the actual model location
**Data format errors**: Check that CSV has required columns (`timestamp`, `value`)
**Memory issues**: Use smaller models (tiny/mini) for large datasets
**Import errors**: Run `poetry install` to ensure all dependencies are installed

### Getting Help

- Check the logs in `data/predictions/` for detailed error messages
- Verify model files exist in the specified path
- Ensure data format matches the expected structure
- Use the model manager to verify model integrity

## Performance Tips

- **Small datasets**: Use chronos-bolt-tiny for fastest results
- **Large datasets**: Use chronos-bolt-base for best accuracy
- **Memory constraints**: Reduce `prediction_length` or use smaller models
- **Speed optimization**: Enable `auto_detect_mode` to skip unnecessary checks

## File Structure

```
chronos-raw/
├── data/
│   ├── raw/                   # Input time series data
│   ├── processed/             # Cleaned and split data
│   ├── predictions/           # Forecast outputs and plots
│   └── model/                 # Versioned model storage
│       └── chronos-bolt-base/
│           └── v1.0/
├── src/                       # Source code
├── config/                    # Configuration files
├── docs/                      # Documentation
└── main.py                    # Main execution script
```
