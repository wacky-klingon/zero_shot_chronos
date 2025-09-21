"""
Configuration management for parquet loader.

Uses Pydantic for structured configuration with fail-fast validation.
No defaults are provided - all configuration must be explicit.
"""

from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple
from pydantic import BaseModel, Field, validator
import yaml
import logging

from .exceptions import ConfigError

logger = logging.getLogger(__name__)


class DataPathsConfig(BaseModel):
    """Data paths configuration - NO DEFAULTS, FAIL FAST."""

    root_dir: str = Field(
        ..., description="Root directory for parquet files - REQUIRED"
    )
    training_subdir: Optional[str] = Field(
        None, description="Training subdirectory (optional)"
    )
    prediction_subdir: Optional[str] = Field(
        None, description="Prediction subdirectory (optional)"
    )
    incremental_subdir: Optional[str] = Field(
        None, description="Incremental subdirectory (optional)"
    )

    @validator("root_dir")
    def validate_root_dir(cls, v):
        """Validate root directory exists and is accessible."""
        if not v:
            raise ValueError("root_dir is required and cannot be empty")

        root_path = Path(v)
        if not root_path.exists():
            raise ValueError(f"Root directory does not exist: {root_path}")

        if not root_path.is_dir():
            raise ValueError(f"Root path is not a directory: {root_path}")

        return str(root_path.absolute())


class FilePatternsConfig(BaseModel):
    """File naming patterns configuration."""

    parquet_extension: str = Field(".parquet", description="Parquet file extension")
    json_extension: str = Field(".json", description="JSON file extension")
    naming_regex: str = Field(
        r"([A-Z]+)_([0-9]+min)_h([0-9]+)_([0-9]{4})_([0-9]{2})_([a-f0-9]+)\.(parquet|json)$",
        description="Regex pattern for filename parsing",
    )


class DirectoryStructureConfig(BaseModel):
    """Directory structure configuration."""

    year_format: str = Field("YYYY", description="Year directory format")
    month_format: str = Field("MM", description="Month directory format")


class SchemaConfig(BaseModel):
    """Schema configuration for data validation."""

    datetime_column: str = Field("ds", description="Primary datetime column name")
    target_columns: List[str] = Field(..., description="Target columns for forecasting")
    feature_columns: Dict[str, List[str]] = Field(
        ..., description="Feature columns by category"
    )
    timestamp_columns: List[str] = Field(
        default_factory=list, description="Additional timestamp columns"
    )


class ProcessingConfig(BaseModel):
    """Data processing configuration."""

    null_handling: str = Field("forward_fill", description="Null handling strategy")
    outlier_detection: bool = Field(True, description="Enable outlier detection")
    outlier_method: str = Field("iqr", description="Outlier detection method")
    outlier_threshold: float = Field(3.0, description="Outlier detection threshold")
    optimize_dtypes: bool = Field(True, description="Optimize data types")
    use_categorical: bool = Field(True, description="Use categorical data types")
    chunk_size: int = Field(10000, description="Chunk size for processing")


class PerformanceConfig(BaseModel):
    """Performance configuration."""

    use_parallel: bool = Field(True, description="Enable parallel processing")
    n_workers: int = Field(4, description="Number of worker processes")
    memory_limit: str = Field("2GB", description="Memory limit for processing")
    cache_intermediate: bool = Field(True, description="Cache intermediate results")
    cache_dir: str = Field("data/cache/parquet", description="Cache directory")


class IdempotencyConfig(BaseModel):
    """Idempotency configuration."""

    enabled: bool = Field(True, description="Enable idempotency tracking")
    state_file: str = Field(".processed_files.json", description="State file name")
    checksum_algorithm: str = Field("md5", description="Checksum algorithm")


class AuditConfig(BaseModel):
    """Audit logging configuration."""

    enabled: bool = Field(True, description="Enable audit logging")
    log_dir: str = Field("audit_logs", description="Audit log directory")
    session_timeout_hours: int = Field(24, description="Session timeout in hours")
    max_log_files: int = Field(100, description="Maximum number of log files")


class ParquetLoaderConfig(BaseModel):
    """Main configuration class for parquet loader."""

    data_paths: DataPathsConfig = Field(..., description="Data paths configuration")
    file_patterns: FilePatternsConfig = Field(default_factory=FilePatternsConfig)
    directory_structure: DirectoryStructureConfig = Field(
        default_factory=DirectoryStructureConfig
    )
    schema: SchemaConfig = Field(..., description="Schema configuration")
    processing: ProcessingConfig = Field(default_factory=ProcessingConfig)
    performance: PerformanceConfig = Field(default_factory=PerformanceConfig)
    idempotency: IdempotencyConfig = Field(default_factory=IdempotencyConfig)
    audit: AuditConfig = Field(default_factory=AuditConfig)

    @classmethod
    def from_yaml(cls, config_path: str) -> "ParquetLoaderConfig":
        """Load configuration from YAML file with fail-fast validation."""
        config_file = Path(config_path)

        if not config_file.exists():
            raise ConfigError(f"Configuration file not found: {config_path}")

        try:
            with open(config_file, "r") as f:
                config_data = yaml.safe_load(f)

            if not config_data:
                raise ConfigError(f"Configuration file is empty: {config_path}")

            # Extract parquet_loader section
            parquet_config = config_data.get("parquet_loader")
            if not parquet_config:
                raise ConfigError(f"No 'parquet_loader' section found in {config_path}")

            return cls(**parquet_config)

        except yaml.YAMLError as e:
            raise ConfigError(f"Invalid YAML in configuration file {config_path}: {e}")
        except Exception as e:
            raise ConfigError(f"Failed to load configuration from {config_path}: {e}")

    def validate_config(self) -> None:
        """Validate configuration - FAIL FAST on any issues."""
        # Validate required fields
        if not self.data_paths.root_dir:
            raise ConfigError("root_dir is required and cannot be empty")

        # Validate root directory exists
        root_path = Path(self.data_paths.root_dir)
        if not root_path.exists():
            raise ConfigError(f"Root directory does not exist: {root_path}")

        if not root_path.is_dir():
            raise ConfigError(f"Root path is not a directory: {root_path}")

        # Validate schema configuration
        if not self.schema.target_columns:
            raise ConfigError("target_columns cannot be empty")

        if not self.schema.feature_columns:
            raise ConfigError("feature_columns cannot be empty")

        logger.info("Configuration validation passed")


def load_config(config_path: str) -> ParquetLoaderConfig:
    """Load and validate configuration from YAML file."""
    config = ParquetLoaderConfig.from_yaml(config_path)
    config.validate_config()
    return config
