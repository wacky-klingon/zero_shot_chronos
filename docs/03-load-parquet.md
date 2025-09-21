# 03 - Load Parquet Data Design

## Overview

This document outlines the design for implementing parquet data loading as a first-class feature in the Chronos forecasting system. The design supports both training and prediction time data loading with robust schema validation, type conversion, and performance optimization.

## Data Schema

Based on the provided template, the system will handle financial time series data with the following structure:

### Core Time Series Columns
- `ds` (VARCHAR) - Datetime index
- `target_*` columns - Primary forecasting targets (bid, ask, open, close, high, low, mid, volumes, volatility, returns, spreads)
- `__index_level_0__` (BIGINT) - Row index

### External Feature Columns
- **Yield Curve Spreads**: `spread_10Y_2Y_*`, `spread_10Y_3M_*` with associated timestamps
- **Commodity Data**: `gold_xauusd_*`, `oil_brent_*` (bid, ask, volumes)
- **Market Indices**: `dxy_index_*`, `vix_index_*` (prices, volumes)
- **Derived Features**: `close`, `target` (processed values)

## Architecture Design

### 1. Core Components

#### 1.1 ParquetDataLoader
```python
class ParquetDataLoader:
    """Primary interface for loading parquet data with schema validation."""
    
    def __init__(self, config_path: str):
        """Initialize with config path - NO DEFAULTS, FAIL FAST."""
        self.config = self._load_config(config_path)
        self._validate_config()
        self.schema_validator = SchemaValidator()
        self.type_converter = TypeConverter()
        self.feature_processor = FeatureProcessor()
        self.file_discovery = FileDiscovery(self.config)
        self.audit_logger = AuditLogger(self.config)
        self.idempotency_tracker = IdempotencyTracker(self.config)
    
    def _load_config(self, config_path: str) -> ParquetLoaderConfig:
        """Load configuration - FAIL FAST if missing."""
        if not Path(config_path).exists():
            raise ConfigError(f"Configuration file not found: {config_path}")
        # Load and validate config...
    
    def _validate_config(self) -> None:
        """Validate configuration - FAIL FAST on missing required fields."""
        if not self.config.data_paths.root_dir:
            raise ConfigError("root_dir is required and cannot be empty")
        if not Path(self.config.data_paths.root_dir).exists():
            raise ConfigError(f"Root directory does not exist: {self.config.data_paths.root_dir}")
    
    def discover_files(self, 
                      symbol: str, 
                      year: int = None,
                      year_range: Tuple[int, int] = None,
                      month: int = None,
                      month_range: Tuple[int, int] = None,
                      timeframes: List[str] = None) -> List[ParquetFileInfo]:
        """Discover parquet files matching criteria with range support."""
        
    def load_training_data(self, 
                          symbol: str,
                          year: int = None,
                          year_range: Tuple[int, int] = None,
                          month: int = None,
                          month_range: Tuple[int, int] = None,
                          target_columns: List[str] = None,
                          feature_columns: List[str] = None) -> TrainingDataset:
        """Load and prepare data for training from discovered files."""
        
    def load_prediction_data(self, 
                            symbol: str,
                            year: int,
                            month: int = None,
                            context_length: int = None) -> PredictionDataset:
        """Load data for prediction/inference from discovered files."""
        
    def load_incremental_data(self, 
                             symbol: str,
                             last_timestamp: datetime) -> IncrementalDataset:
        """Load only new data since last timestamp."""
    
    def is_already_processed(self, file_info: ParquetFileInfo) -> bool:
        """Check if file has already been processed (idempotency)."""
        
    def mark_as_processed(self, file_info: ParquetFileInfo, status: str = "completed") -> None:
        """Mark file as processed in idempotency tracker."""
        
    def start_audit_session(self, symbol: str, year_range: Tuple[int, int] = None, 
                           month_range: Tuple[int, int] = None) -> str:
        """Start audit session and return session ID."""
        
    def end_audit_session(self, session_id: str, status: str = "completed") -> None:
        """End audit session with final status."""
```

#### 1.2 FileDiscovery
```python
class FileDiscovery:
    """Discovers and manages parquet files in the year/month directory structure."""
    
    def __init__(self, config: ParquetLoaderConfig):
        self.config = config
        self.root_dir = Path(config.data_paths.root_dir)
        self.naming_regex = re.compile(config.file_patterns.naming_regex)
    
    def discover_files(self, 
                      symbol: str, 
                      year: int = None,
                      year_range: Tuple[int, int] = None,
                      month: int = None,
                      month_range: Tuple[int, int] = None,
                      timeframes: List[str] = None) -> List[ParquetFileInfo]:
        """Discover parquet files matching criteria with range support."""
        files = []
        
        # Determine year range to search
        if year_range:
            years_to_search = range(year_range[0], year_range[1] + 1)
        elif year:
            years_to_search = [year]
        else:
            raise ValueError("Either 'year' or 'year_range' must be specified")
        
        for search_year in years_to_search:
            year_dir = self.root_dir / str(search_year)
            
            if not year_dir.exists():
                continue  # Skip missing years instead of failing
            
            # Determine month range to search
            if month_range:
                months_to_search = range(month_range[0], month_range[1] + 1)
            elif month:
                months_to_search = [month]
            else:
                months_to_search = range(1, 13)  # All months
            
            for search_month in months_to_search:
                month_dir = year_dir / f"{search_month:02d}"
                if not month_dir.exists():
                    continue  # Skip missing months
                
                for file_path in month_dir.glob("*.parquet"):
                    file_info = self._parse_filename(file_path)
                    if self._matches_criteria(file_info, symbol, timeframes):
                        files.append(file_info)
        
        return sorted(files, key=lambda x: (x.year, x.month, x.symbol))
    
    def _parse_filename(self, file_path: Path) -> ParquetFileInfo:
        """Parse filename to extract metadata."""
        match = self.naming_regex.match(file_path.name)
        if not match:
            raise InvalidFilenameError(f"Filename does not match expected pattern: {file_path.name}")
        
        symbol, timeframe, horizon, year, month, hash_id, extension = match.groups()
        
        return ParquetFileInfo(
            file_path=file_path,
            symbol=symbol,
            timeframe=timeframe,
            horizon=int(horizon),
            year=int(year),
            month=int(month),
            hash_id=hash_id,
            extension=extension
        )
    
    def _matches_criteria(self, file_info: ParquetFileInfo, 
                         symbol: str, 
                         timeframes: List[str] = None) -> bool:
        """Check if file matches discovery criteria."""
        if file_info.symbol != symbol:
            return False
        if timeframes and file_info.timeframe not in timeframes:
            return False
        return True

@dataclass
class ParquetFileInfo:
    """Metadata about a discovered parquet file."""
    file_path: Path
    symbol: str
    timeframe: str
    horizon: int
    year: int
    month: int
    hash_id: str
    extension: str
    
    @property
    def is_parquet(self) -> bool:
        return self.extension == "parquet"
    
    @property
    def is_json(self) -> bool:
        return self.extension == "json"

#### 1.3 SchemaValidator
```python
class SchemaValidator:
    """Validates parquet schema against expected structure."""
    
    def validate_schema(self, schema: pa.Schema) -> ValidationResult:
        """Validate parquet schema matches expected columns and types."""
        
    def validate_data_quality(self, df: pd.DataFrame) -> QualityReport:
        """Check for missing values, outliers, and data consistency."""
```

#### 1.3 TypeConverter
```python
class TypeConverter:
    """Handles type conversion and optimization."""
    
    def convert_timestamps(self, df: pd.DataFrame) -> pd.DataFrame:
        """Convert timestamp columns to proper datetime format."""
        
    def optimize_dtypes(self, df: pd.DataFrame) -> pd.DataFrame:
        """Optimize data types for memory efficiency."""
        
    def handle_nulls(self, df: pd.DataFrame, strategy: str) -> pd.DataFrame:
        """Apply null handling strategy (forward_fill, interpolate, drop)."""
```

#### 1.4 IdempotencyTracker
```python
class IdempotencyTracker:
    """Tracks processed files to ensure idempotency."""
    
    def __init__(self, config: ParquetLoaderConfig):
        self.config = config
        self.state_file = Path(config.data_paths.root_dir) / ".processed_files.json"
        self.processed_files = self._load_state()
    
    def _load_state(self) -> Dict[str, Dict[str, Any]]:
        """Load processed files state from disk."""
        if self.state_file.exists():
            with open(self.state_file, 'r') as f:
                return json.load(f)
        return {"processed_files": {}}
    
    def _save_state(self) -> None:
        """Save processed files state to disk."""
        with open(self.state_file, 'w') as f:
            json.dump(self.processed_files, f, indent=2)
    
    def is_processed(self, file_info: ParquetFileInfo) -> bool:
        """Check if file has already been processed."""
        file_key = f"{file_info.symbol}_{file_info.year}_{file_info.month:02d}_{file_info.hash_id}"
        return file_key in self.processed_files["processed_files"]
    
    def mark_processed(self, file_info: ParquetFileInfo, status: str = "completed") -> None:
        """Mark file as processed."""
        file_key = f"{file_info.symbol}_{file_info.year}_{file_info.month:02d}_{file_info.hash_id}"
        self.processed_files["processed_files"][file_key] = {
            "file_path": str(file_info.file_path),
            "processed_at": datetime.now().isoformat(),
            "checksum": self._calculate_checksum(file_info.file_path),
            "status": status
        }
        self._save_state()
    
    def _calculate_checksum(self, file_path: Path) -> str:
        """Calculate file checksum for change detection."""
        import hashlib
        with open(file_path, 'rb') as f:
            return hashlib.md5(f.read()).hexdigest()
```

#### 1.5 AuditLogger
```python
class AuditLogger:
    """Simple audit logging for progress tracking (KISS approach)."""
    
    def __init__(self, config: ParquetLoaderConfig):
        self.config = config
        self.audit_dir = Path(config.data_paths.root_dir) / "audit_logs"
        self.audit_dir.mkdir(exist_ok=True)
    
    def start_session(self, symbol: str, year_range: Tuple[int, int] = None, 
                     month_range: Tuple[int, int] = None) -> str:
        """Start audit session and return session ID."""
        session_id = f"{symbol}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        session_data = {
            "session_id": session_id,
            "start_time": datetime.now().isoformat(),
            "symbol": symbol,
            "year_range": year_range,
            "month_range": month_range,
            "files_discovered": 0,
            "files_processed": 0,
            "files_skipped": 0,
            "status": "running"
        }
        
        audit_file = self.audit_dir / f"{session_id}.json"
        with open(audit_file, 'w') as f:
            json.dump(session_data, f, indent=2)
        
        return session_id
    
    def update_session(self, session_id: str, **updates) -> None:
        """Update session with progress information."""
        audit_file = self.audit_dir / f"{session_id}.json"
        
        if audit_file.exists():
            with open(audit_file, 'r') as f:
                session_data = json.load(f)
            
            session_data.update(updates)
            
            with open(audit_file, 'w') as f:
                json.dump(session_data, f, indent=2)
    
    def end_session(self, session_id: str, status: str = "completed") -> None:
        """End audit session with final status."""
        self.update_session(session_id, 
                          end_time=datetime.now().isoformat(),
                          status=status)
```

### 2. Configuration System

#### 2.1 ParquetLoaderConfig
```yaml
# config/parquet_loader_config.yaml
parquet_loader:
  # Data paths - NO DEFAULTS, FAIL FAST
  data_paths:
    root_dir: "/path/to/your/data/directory"
    # Optional: specific subdirectories for different data types
    training_subdir: null  # Use root_dir if null
    prediction_subdir: null  # Use root_dir if null
    incremental_subdir: null  # Use root_dir if null
  
  # File naming patterns
  file_patterns:
    parquet_extension: ".parquet"
    json_extension: ".json"
    # Pattern: {SYMBOL}_{TIMEFRAME}_{HORIZON}_{YEAR}_{MONTH}_{HASH}
    # Example: SYMBOL_1min_h15_2014_03_769b531dbb2ff3cb.parquet
    naming_regex: "([A-Z]+)_([0-9]+min)_h([0-9]+)_([0-9]{4})_([0-9]{2})_([a-f0-9]+)\\.(parquet|json)$"
    
  # Directory structure
  directory_structure:
    year_format: "YYYY"  # 2014, 2015, etc.
    month_format: "MM"   # 01, 02, 03, etc.
    # Structure: {root_dir}/{YEAR}/{MONTH}/{filename}
  
  # Schema configuration
  schema:
    datetime_column: "ds"
    target_columns:
      - "target_bid_first"
      - "target_ask_first"
      - "target_open"
      - "target_close"
      - "target_high"
      - "target_low"
      - "target_mid"
      - "target_volatility"
      - "target_returns"
      - "target_spread"
    
    feature_columns:
      yield_curve:
        - "spread_10Y_2Y_GS10"
        - "spread_10Y_2Y_GS2"
        - "spread_10Y_2Y_10Y_2Y"
        - "spread_10Y_2Y_GS3M"
        - "spread_10Y_2Y_10Y_3M"
        - "spread_10Y_3M_GS10"
        - "spread_10Y_3M_GS2"
        - "spread_10Y_3M_10Y_2Y"
        - "spread_10Y_3M_GS3M"
        - "spread_10Y_3M_10Y_3M"
      
      commodities:
        - "gold_xauusd_bid"
        - "gold_xauusd_ask"
        - "gold_xauusd_bid_volume"
        - "gold_xauusd_ask_volume"
        - "oil_brent_bid"
        - "oil_brent_ask"
        - "oil_brent_bid_volume"
        - "oil_brent_ask_volume"
      
      indices:
        - "dxy_index_bid"
        - "dxy_index_ask"
        - "dxy_index_bid_volume"
        - "dxy_index_ask_volume"
        - "vix_index_vix_close"
        - "vix_index_vix_high"
        - "vix_index_vix_low"
        - "vix_index_vix_open"
        - "vix_index_volume"
    
    timestamp_columns:
      - "spread_10Y_2Y_date"
      - "spread_10Y_3M_date"
  
  # Data processing options
  processing:
    null_handling: "forward_fill"  # forward_fill, interpolate, drop
    outlier_detection: true
    outlier_method: "iqr"  # iqr, zscore, isolation_forest
    outlier_threshold: 3.0
    
    # Memory optimization
    optimize_dtypes: true
    use_categorical: true
    chunk_size: 10000
    
  # Performance settings
  performance:
    use_parallel: true
    n_workers: 4
    memory_limit: "2GB"
    cache_intermediate: true
    cache_dir: "data/cache/parquet"
  
  # Idempotency settings
  idempotency:
    enabled: true
    state_file: ".processed_files.json"
    checksum_algorithm: "md5"  # md5, sha1, sha256
    
  # Audit logging settings
  audit:
    enabled: true
    log_dir: "audit_logs"
    session_timeout_hours: 24
    max_log_files: 100
```

### 3. Data Loading Patterns

#### 3.1 Training Time Loading
```python
def load_training_data_pattern():
    """Pattern for loading training data with full feature engineering."""
    
    # 1. Initialize loader with config - FAIL FAST
    loader = ParquetDataLoader("config/parquet_loader_config.yaml")
    
    # 2. Start audit session
    session_id = loader.start_audit_session(
        symbol="SYMBOL",
        year_range=(2014, 2016),  # Multi-year range
        month_range=(1, 12)       # All months
    )
    
    try:
        # 3. Discover files for training period
        training_files = loader.discover_files(
            symbol="SYMBOL",
            year_range=(2014, 2016),  # Range specification
            month_range=(1, 12),
            timeframes=["1min"]
        )
        
        if not training_files:
            raise DataNotFoundError("No training files found for SYMBOL in range 2014-2016")
        
        # 4. Update audit with discovered files
        loader.audit_logger.update_session(session_id, files_discovered=len(training_files))
        
        # 5. Process files with idempotency checks
        processed_count = 0
        skipped_count = 0
        
        for file_info in training_files:
            if loader.is_already_processed(file_info):
                logger.info(f"Skipping already processed file: {file_info.file_path}")
                skipped_count += 1
                continue
            
            # Load and process file
            raw_data = loader.load_training_data(
                symbol="SYMBOL",
                year=file_info.year,
                month=file_info.month,
                target_columns=config.schema.target_columns,
                feature_columns=config.schema.feature_columns
            )
            
            # Apply data quality checks
            quality_report = loader.validate_data_quality(raw_data)
            if not quality_report.is_valid:
                logger.warning(f"Data quality issues in {file_info.file_path}: {quality_report.issues}")
                continue
            
            # Feature engineering
            processed_data = loader.feature_processor.process_training_features(raw_data)
            
            # Mark as processed
            loader.mark_as_processed(file_info, "completed")
            processed_count += 1
        
        # 6. Update audit with final counts
        loader.audit_logger.update_session(session_id, 
                                         files_processed=processed_count,
                                         files_skipped=skipped_count)
        
        # 7. Create train/validation splits from all processed data
        train_data, val_data = loader.create_splits(
            processed_data, 
            validation_split=0.2,
            time_based_split=True
        )
        
        return TrainingDataset(train_data, val_data)
        
    except Exception as e:
        # End session with error status
        loader.end_audit_session(session_id, "failed")
        raise
    finally:
        # End session
        loader.end_audit_session(session_id, "completed")
```

#### 3.2 Prediction Time Loading
```python
def load_prediction_data_pattern():
    """Pattern for loading data for real-time prediction."""
    
    # 1. Initialize loader with config - FAIL FAST
    loader = ParquetDataLoader("config/parquet_loader_config.yaml")
    
    # 2. Discover most recent files
    current_year = datetime.now().year
    current_month = datetime.now().month
    
    prediction_files = loader.discover_files(
        symbol="SYMBOL",
        year=current_year,
        month=current_month,
        timeframes=["1min"]
    )
    
    if not prediction_files:
        # Fallback to previous month
        prev_month = current_month - 1 if current_month > 1 else 12
        prev_year = current_year if current_month > 1 else current_year - 1
        prediction_files = loader.discover_files(
            symbol="SYMBOL",
            year=prev_year,
            month=prev_month,
            timeframes=["1min"]
        )
    
    if not prediction_files:
        raise DataNotFoundError("No prediction files found for SYMBOL")
    
    # 3. Load context data from most recent file
    context_data = loader.load_prediction_data(
        symbol="SYMBOL",
        year=prediction_files[-1].year,
        month=prediction_files[-1].month,
        context_length=config.inference.context_length
    )
    
    # 4. Apply same preprocessing as training
    processed_context = loader.feature_processor.process_prediction_features(context_data)
    
    # 5. Validate context completeness
    if not loader.validate_context_completeness(processed_context):
        raise IncompleteContextError("Insufficient context data for prediction")
    
    return PredictionDataset(processed_context)
```

#### 3.3 Incremental Loading
```python
def load_incremental_data_pattern():
    """Pattern for loading only new data since last update."""
    
    # 1. Initialize loader with config - FAIL FAST
    loader = ParquetDataLoader("config/parquet_loader_config.yaml")
    
    # 2. Determine last processed timestamp
    last_timestamp = get_last_processed_timestamp()
    last_year = last_timestamp.year
    last_month = last_timestamp.month
    
    # 3. Discover files since last timestamp
    new_files = []
    current_year = datetime.now().year
    current_month = datetime.now().month
    
    # Check current year
    for year in range(last_year, current_year + 1):
        start_month = last_month if year == last_year else 1
        end_month = current_month if year == current_year else 12
        
        for month in range(start_month, end_month + 1):
            files = loader.discover_files(
                symbol="SYMBOL",
                year=year,
                month=month,
                timeframes=["1min"]
            )
            new_files.extend(files)
    
    if not new_files:
        raise DataNotFoundError("No new data found since last timestamp")
    
    # 4. Load incremental data
    new_data = loader.load_incremental_data(
        symbol="SYMBOL",
        last_timestamp=last_timestamp
    )
    
    # 5. Merge with existing data
    merged_data = loader.merge_incremental_data(existing_data, new_data)
    
    return merged_data
```

### 4. Feature Engineering Pipeline

#### 4.1 Temporal Feature Engineering
```python
class TemporalFeatureProcessor:
    """Handles temporal feature creation and alignment."""
    
    def align_timestamps(self, df: pd.DataFrame) -> pd.DataFrame:
        """Align all timestamp columns to primary datetime index."""
        
    def create_lag_features(self, df: pd.DataFrame, lags: List[int]) -> pd.DataFrame:
        """Create lagged features for time series."""
        
    def create_rolling_features(self, df: pd.DataFrame, windows: List[int]) -> pd.DataFrame:
        """Create rolling window statistics."""
        
    def create_difference_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create first and second differences."""
```

#### 4.2 Cross-Asset Feature Engineering
```python
class CrossAssetFeatureProcessor:
    """Handles cross-asset feature creation."""
    
    def create_spread_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create bid-ask spreads and yield curve spreads."""
        
    def create_correlation_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create rolling correlations between assets."""
        
    def create_volatility_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create volatility measures and indicators."""
```

### 5. Performance Optimizations

#### 5.1 Memory Management
- **Chunked Loading**: Process large files in chunks to manage memory
- **Type Optimization**: Use appropriate dtypes (float32 vs float64, categoricals)
- **Lazy Loading**: Load only required columns and time ranges
- **Caching**: Cache processed intermediate results

#### 5.2 Parallel Processing
- **Multi-threaded Reading**: Use pyarrow's parallel reading capabilities
- **Feature Engineering**: Parallelize independent feature calculations
- **Validation**: Parallel data quality checks

#### 5.3 Storage Optimization
- **Partitioning**: Partition parquet files by date for efficient time-based queries
- **Compression**: Use snappy or zstd compression
- **Column Pruning**: Only read required columns

### 6. Error Handling and Validation

#### 6.1 Custom Exceptions
```python
class ParquetLoaderError(Exception):
    """Base exception for parquet loader errors."""
    pass

class ConfigError(ParquetLoaderError):
    """Configuration-related errors."""
    pass

class DataNotFoundError(ParquetLoaderError):
    """Data file or directory not found errors."""
    pass

class InvalidFilenameError(ParquetLoaderError):
    """Filename does not match expected pattern."""
    pass

class DataQualityError(ParquetLoaderError):
    """Data quality validation errors."""
    pass

class IncompleteContextError(ParquetLoaderError):
    """Insufficient context data for prediction."""
    pass

class SchemaValidationError(ParquetLoaderError):
    """Schema validation errors."""
    pass
```

#### 6.2 Fail-Fast Configuration Validation
```python
def validate_config_fail_fast(config: Dict[str, Any]) -> None:
    """Validate configuration and fail fast on missing required fields."""
    required_fields = [
        "data_paths.root_dir",
        "file_patterns.naming_regex",
        "schema.datetime_column"
    ]
    
    for field in required_fields:
        if not get_nested_value(config, field):
            raise ConfigError(f"Required configuration field missing: {field}")
    
    # Validate root directory exists
    root_dir = Path(config["data_paths"]["root_dir"])
    if not root_dir.exists():
        raise ConfigError(f"Root directory does not exist: {root_dir}")
    
    if not root_dir.is_dir():
        raise ConfigError(f"Root path is not a directory: {root_dir}")

def get_nested_value(config: Dict[str, Any], field_path: str) -> Any:
    """Get nested value from config using dot notation."""
    keys = field_path.split(".")
    value = config
    for key in keys:
        if not isinstance(value, dict) or key not in value:
            return None
        value = value[key]
    return value
```

### 7. Data Quality Checks
```python
class DataQualityValidator:
    """Comprehensive data quality validation."""
    
    def check_missing_values(self, df: pd.DataFrame) -> MissingValueReport:
        """Check for missing values and patterns."""
        
    def check_temporal_consistency(self, df: pd.DataFrame) -> TemporalReport:
        """Validate temporal ordering and gaps."""
        
    def check_outliers(self, df: pd.DataFrame) -> OutlierReport:
        """Detect statistical outliers."""
        
    def check_schema_compliance(self, df: pd.DataFrame) -> SchemaReport:
        """Validate against expected schema."""
```

#### 7.1 Error Recovery
- **Graceful Degradation**: Continue processing with available data
- **Data Imputation**: Apply intelligent imputation strategies
- **Fallback Mechanisms**: Use alternative data sources when available

### 8. Integration Points

#### 8.1 Training Integration
```python
# Integration with existing ChronosTrainer
class ChronosTrainer:
    def __init__(self, parquet_config_path: str = None):
        if parquet_config_path:
            self.parquet_loader = ParquetDataLoader(parquet_config_path)
        else:
            # Fallback to dummy data generation
            self.dummy_data_generator = DummyDataGenerator()
    
    def load_training_data_from_parquet(self, file_path: str) -> np.ndarray:
        """Load training data from parquet files."""
        dataset = self.parquet_loader.load_training_data(file_path)
        return self.convert_to_chronos_format(dataset)
```

#### 8.2 Prediction Integration
```python
# Integration with existing ChronosLoader
class ChronosLoader:
    def __init__(self, parquet_config_path: str = None):
        if parquet_config_path:
            self.parquet_loader = ParquetDataLoader(parquet_config_path)
    
    def load_context_from_parquet(self, file_path: str) -> np.ndarray:
        """Load prediction context from parquet files."""
        dataset = self.parquet_loader.load_prediction_data(file_path)
        return self.convert_to_chronos_format(dataset)
```

### 9. Implementation Phases

#### Phase 1: Core Infrastructure
1. Implement `ParquetDataLoader` base class
2. Create schema validation system
3. Implement basic type conversion
4. Add configuration management

#### Phase 2: Feature Engineering
1. Implement temporal feature processing
2. Add cross-asset feature creation
3. Create feature selection and ranking
4. Add feature scaling and normalization

#### Phase 3: Performance Optimization
1. Implement chunked loading
2. Add parallel processing
3. Create caching system
4. Optimize memory usage

#### Phase 4: Integration and Testing
1. Integrate with existing training pipeline
2. Integrate with prediction pipeline
3. Add comprehensive testing
4. Performance benchmarking

### 10. Usage Examples

#### 10.1 Basic Training Data Loading
```python
from src.parquet_loader import ParquetDataLoader
from src.exceptions import ConfigError, DataNotFoundError

# Initialize loader - FAIL FAST if config missing
try:
    loader = ParquetDataLoader("config/parquet_loader_config.yaml")
except ConfigError as e:
    print(f"Configuration error: {e}")
    exit(1)

# Discover available data with range specification
available_files = loader.discover_files(
    symbol="SYMBOL",
    year_range=(2014, 2016),  # Multi-year range
    month_range=(1, 6),       # First half of each year
    timeframes=["1min"]
)

print(f"Found {len(available_files)} files for SYMBOL in range 2014-2016, months 1-6")

# Start audit session
session_id = loader.start_audit_session(
    symbol="SYMBOL",
    year_range=(2014, 2016),
    month_range=(1, 6)
)

# Load training data with idempotency
try:
    train_dataset = loader.load_training_data(
        symbol="SYMBOL",
        year_range=(2014, 2016),
        month_range=(1, 6),
        target_columns=["target_close", "target_volatility"],
        feature_columns=["feature_1", "feature_2", "feature_3"]
    )
    
    # End audit session
    loader.end_audit_session(session_id, "completed")
    
except DataNotFoundError as e:
    print(f"Data not found: {e}")
    loader.end_audit_session(session_id, "failed")
    exit(1)

# Use with Chronos trainer
trainer = ChronosTrainer(parquet_config_path="config/parquet_loader_config.yaml")
trainer.load_training_data_from_parquet(symbol="SYMBOL", year=2014)
```

#### 10.2 Real-time Prediction Loading
```python
from datetime import datetime

# Initialize loader - FAIL FAST
try:
    loader = ParquetDataLoader("config/parquet_loader_config.yaml")
except ConfigError as e:
    print(f"Configuration error: {e}")
    exit(1)

# Discover most recent data
current_year = datetime.now().year
current_month = datetime.now().month

prediction_files = loader.discover_files(
    symbol="SYMBOL",
    year=current_year,
    month=current_month,
    timeframes=["1min"]
)

if not prediction_files:
    # Fallback to previous month
    prev_month = current_month - 1 if current_month > 1 else 12
    prev_year = current_year if current_month > 1 else current_year - 1
    prediction_files = loader.discover_files(
        symbol="SYMBOL",
        year=prev_year,
        month=prev_month,
        timeframes=["1min"]
    )

# Load prediction context
try:
    prediction_dataset = loader.load_prediction_data(
        symbol="SYMBOL",
        year=prediction_files[-1].year,
        month=prediction_files[-1].month,
        context_length=100
    )
except DataNotFoundError as e:
    print(f"Data not found: {e}")
    exit(1)

# Use with Chronos loader
chronos_loader = ChronosLoader(parquet_config_path="config/parquet_loader_config.yaml")
context = chronos_loader.load_context_from_parquet(symbol="SYMBOL", year=current_year, month=current_month)
predictions = chronos_loader.predict(context)
```

### 11. Monitoring and Observability

#### 11.1 Data Quality Metrics
- Missing value rates by column
- Outlier detection rates
- Temporal gap analysis
- Schema compliance scores

#### 11.2 Performance Metrics
- Loading time per file size
- Memory usage patterns
- Cache hit rates
- Processing throughput

#### 11.3 Business Metrics
- Data freshness (time since last update)
- Feature completeness scores
- Prediction accuracy correlation with data quality

### 12. Enhanced Features

#### 12.1 Range Specification
The system supports flexible range specifications for both years and months:

```python
# Single year, all months
files = loader.discover_files(symbol="SYMBOL", year=2014)

# Year range, all months
files = loader.discover_files(symbol="SYMBOL", year_range=(2014, 2016))

# Single year, month range
files = loader.discover_files(symbol="SYMBOL", year=2014, month_range=(3, 6))

# Full range specification
files = loader.discover_files(symbol="SYMBOL", year_range=(2014, 2016), month_range=(1, 6))
```

#### 12.2 Idempotency
The system ensures idempotency by tracking processed files:

```python
# Check if file already processed
if loader.is_already_processed(file_info):
    logger.info("Skipping already processed file")
    continue

# Process file
process_file(file_info)

# Mark as processed
loader.mark_as_processed(file_info, "completed")
```

**State Tracking**: Uses `.processed_files.json` in the root directory to track:
- File path and metadata
- Processing timestamp
- File checksum for change detection
- Processing status

#### 12.3 Audit Logging (KISS)
Simple JSON-based audit logging for progress tracking:

```json
{
  "session_id": "SYMBOL_20240115_103000",
  "start_time": "2024-01-15T10:30:00Z",
  "end_time": "2024-01-15T10:45:00Z",
  "symbol": "SYMBOL",
  "year_range": [2014, 2016],
  "month_range": [1, 6],
  "files_discovered": 36,
  "files_processed": 32,
  "files_skipped": 4,
  "status": "completed"
}
```

**Audit Features**:
- Session-based tracking with unique IDs
- Progress counters (discovered, processed, skipped)
- Start/end timestamps
- Error status tracking
- Simple JSON format for easy parsing

### 13. Actual Data Structure

Based on the provided directory structure, the system handles data organized as:

```
/data/root/directory/
├── 2004/
│   ├── 01/
│   │   ├── SYMBOL_1min_h15_2004_01_769b531dbb2ff3cb.json
│   │   └── SYMBOL_1min_h15_2004_01_769b531dbb2ff3cb.parquet
│   ├── 02/
│   └── ...
├── 2005/
├── ...
└── 2024/
    ├── 01/
    ├── 02/
    └── ...
```

#### 12.1 File Naming Convention
- **Pattern**: `{SYMBOL}_{TIMEFRAME}_{HORIZON}_{YEAR}_{MONTH}_{HASH}.{EXTENSION}`
- **Example**: `SYMBOL_1min_h15_2014_03_769b531dbb2ff3cb.parquet`
- **Components**:
  - `SYMBOL`: Asset identifier (e.g., stock ticker, currency pair, etc.)
  - `TIMEFRAME`: Data frequency (1min, 5min, 1h, etc.)
  - `HORIZON`: Prediction horizon in minutes (h15 = 15 minutes)
  - `YEAR`: 4-digit year (2014, 2015, etc.)
  - `MONTH`: 2-digit month (01, 02, 03, etc.)
  - `HASH`: Unique identifier for the dataset
  - `EXTENSION`: File type (parquet, json)

#### 12.2 Directory Structure
- **Root**: Configurable via `data_paths.root_dir`
- **Year Directories**: 4-digit year format (2004, 2005, ..., 2024)
- **Month Directories**: 2-digit month format (01, 02, ..., 12)
- **Files**: Both `.parquet` and `.json` versions of the same data

#### 12.3 Fail-Fast Configuration Requirements
The system requires explicit configuration with no defaults:

```yaml
parquet_loader:
  data_paths:
    root_dir: "/path/to/your/data/directory"
    # NO DEFAULTS - MUST BE SPECIFIED
```

## Conclusion

This design provides a  framework for loading parquet data in the Chronos forecasting system. The modular architecture allows for incremental implementation while maintaining compatibility with existing training and prediction pipelines.doc 

Key design principles:
- **Fail-Fast Configuration**: No defaults, explicit configuration required
- **Directory-Aware Discovery**: Handles year/month directory structure
- **File Pattern Matching**: Robust parsing of standardized filenames
- **Range Specification**: Flexible year/month range support for efficient data selection
- **Idempotency**: Ensures no reprocessing of already processed data
- **Audit Logging**: Simple KISS approach to progress tracking and traceability
- **Error Handling**: Comprehensive exception hierarchy with clear error messages
- **Performance**: Chunked loading, parallel processing, and memory optimization
- **Integration**: Seamless integration with existing Chronos training and prediction pipelines
