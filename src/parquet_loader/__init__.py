"""
Parquet Data Loader for Chronos Forecasting System

This module provides functionality for loading parquet data with:
- Range specification for years/months
- Idempotency to avoid reprocessing
- Audit logging for progress tracking
- Fail-fast configuration validation
"""

from .loader import ParquetDataLoader
from .exceptions import (
    ParquetLoaderError,
    ConfigError,
    DataNotFoundError,
    InvalidFilenameError,
    DataQualityError,
    IncompleteContextError,
    SchemaValidationError,
)
from .config import ParquetLoaderConfig
from .file_discovery import FileDiscovery, ParquetFileInfo
from .idempotency import IdempotencyTracker
from .audit import AuditLogger

__version__ = "0.1.0"
__all__ = [
    "ParquetDataLoader",
    "ParquetLoaderError",
    "ConfigError",
    "DataNotFoundError",
    "InvalidFilenameError",
    "DataQualityError",
    "IncompleteContextError",
    "SchemaValidationError",
    "ParquetLoaderConfig",
    "FileDiscovery",
    "ParquetFileInfo",
    "IdempotencyTracker",
    "AuditLogger",
]
