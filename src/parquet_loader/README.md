# Parquet Loader for Chronos

A robust parquet data loader for the Chronos forecasting system with range specification, idempotency, and audit logging.

## Features

- **Range Specification**: Load data for specific year/month ranges
- **Idempotency**: Avoid reprocessing already processed files
- **Audit Logging**: Complete trace of all processing activities
- **Fail-Fast Configuration**: No defaults, explicit configuration required
- **Error Handling**: Comprehensive error handling with clear messages

## Quick Start

### 1. Configuration

Create a configuration file `config/parquet_loader_config.yaml`:

```yaml
parquet_loader:
  data_paths:
    root_dir: "/path/to/your/data/directory"  # REQUIRED
  schema:
    datetime_column: "ds"
    target_columns:
      - "target_close"
      - "target_volatility"
    feature_columns:
      features:
        - "feature_1"
        - "feature_2"
```

### 2. Basic Usage

```python
from parquet_loader import ParquetDataLoader

# Initialize loader
loader = ParquetDataLoader("config/parquet_loader_config.yaml")

# Discover files with range specification
files = loader.discover_files(
    symbol="SYMBOL",
    year_range=(2020, 2022),  # 2020-2022
    month_range=(1, 6)        # January-June
)

# Load training data
training_data = loader.load_training_data(
    symbol="SYMBOL",
    year_range=(2020, 2022),
    month_range=(1, 6)
)
```

### 3. Idempotency

The loader automatically tracks processed files and skips already processed ones:

```python
# First run - processes all files
data1 = loader.load_training_data(symbol="SYMBOL", year=2024)

# Second run - skips already processed files
data2 = loader.load_training_data(symbol="SYMBOL", year=2024)
# data2 will have files_skipped > 0
```

### 4. Audit Logging

Every processing run is logged with complete traceability:

```python
# Start audit session
session_id = loader.start_audit_session("SYMBOL", year_range=(2024, 2024))

# Process data
data = loader.load_training_data(symbol="SYMBOL", year=2024)

# End session
loader.end_audit_session(session_id, "completed")

# Get statistics
stats = loader.get_audit_stats()
print(f"Processed {stats['total_files_processed']} files")
```

## API Reference

### ParquetDataLoader

Main class for loading parquet data.

#### Methods

- `discover_files(symbol, year=None, year_range=None, month=None, month_range=None, timeframes=None)`
  - Discover files matching criteria with range support
  
- `load_training_data(symbol, year=None, year_range=None, month=None, month_range=None, target_columns=None, feature_columns=None)`
  - Load and prepare data for training
  
- `load_prediction_data(symbol, year=None, year_range=None, month=None, month_range=None, context_length=None)`
  - Load data for prediction/inference
  
- `load_incremental_data(symbol, last_timestamp)`
  - Load only new data since last timestamp
  
- `is_already_processed(file_info)`
  - Check if file has already been processed
  
- `mark_as_processed(file_info, status="completed")`
  - Mark file as processed
  
- `start_audit_session(symbol, year_range=None, month_range=None)`
  - Start audit session and return session ID
  
- `end_audit_session(session_id, status="completed")`
  - End audit session with final status

### FileDiscovery

Discovers parquet files in year/month directory structure.

### IdempotencyTracker

Tracks processed files to ensure idempotency.

### AuditLogger

Simple JSON-based audit logging for progress tracking.

## Configuration

### Required Fields

- `data_paths.root_dir`: Root directory for parquet files (must exist)
- `schema.target_columns`: List of target columns for forecasting
- `schema.feature_columns`: Dictionary of feature columns by category

### Optional Fields

- `idempotency.enabled`: Enable idempotency tracking (default: true)
- `audit.enabled`: Enable audit logging (default: true)
- `processing.null_handling`: Null handling strategy (default: "forward_fill")
- `performance.use_parallel`: Enable parallel processing (default: true)

## Error Handling

The loader uses a comprehensive exception hierarchy:

- `ParquetLoaderError`: Base exception
- `ConfigError`: Configuration-related errors
- `DataNotFoundError`: Missing data errors
- `InvalidFilenameError`: Filename pattern mismatches
- `DataQualityError`: Data validation errors
- `IncompleteContextError`: Insufficient context for prediction
- `SchemaValidationError`: Schema validation errors

## Examples

See the `examples/` directory for complete usage examples:

- `test_parquet_loader.py`: Basic functionality tests
- `chronos_integration_example.py`: Integration with Chronos components

## File Structure

```
src/parquet_loader/
├── __init__.py          # Package initialization
├── config.py            # Configuration management
├── exceptions.py         # Exception hierarchy
├── file_discovery.py    # File discovery with ranges
├── idempotency.py       # Idempotency tracking
├── audit.py             # Audit logging
├── loader.py            # Main loader class
└── README.md            # This file
```

## Development

### Running Tests

```bash
python examples/test_parquet_loader.py
```

### Integration Example

```bash
python examples/chronos_integration_example.py
```

## License

This project is part of the Chronos forecasting system.
