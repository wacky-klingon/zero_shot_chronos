"""
Exception hierarchy for parquet loader.

All exceptions inherit from ParquetLoaderError to provide a common base
for error handling and to distinguish parquet loader errors from other exceptions.
"""


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
