"""
Main parquet data loader class.

Integrates all components (discovery, idempotency, audit) to provide
a unified interface for loading parquet data with range support.
"""

import logging
from pathlib import Path
from typing import List, Optional, Tuple, Dict, Any
from datetime import datetime

from .config import ParquetLoaderConfig, load_config
from .exceptions import ConfigError, DataNotFoundError
from .file_discovery import FileDiscovery, ParquetFileInfo
from .idempotency import IdempotencyTracker
from .audit import AuditLogger

logger = logging.getLogger(__name__)


class ParquetDataLoader:
    """Primary interface for loading parquet data with schema validation."""

    def __init__(self, config_path: str):
        """
        Initialize with config path - NO DEFAULTS, FAIL FAST.

        Args:
            config_path: Path to configuration YAML file

        Raises:
            ConfigError: If configuration is invalid or missing
        """
        self.config = load_config(config_path)
        self.file_discovery = FileDiscovery(self.config)
        self.idempotency_tracker = IdempotencyTracker(self.config)
        self.audit_logger = AuditLogger(self.config)

        logger.info(f"ParquetDataLoader initialized with config: {config_path}")

    def discover_files(
        self,
        symbol: str,
        year: Optional[int] = None,
        year_range: Optional[Tuple[int, int]] = None,
        month: Optional[int] = None,
        month_range: Optional[Tuple[int, int]] = None,
        timeframes: Optional[List[str]] = None,
    ) -> List[ParquetFileInfo]:
        """
        Discover parquet files matching criteria with range support.

        Args:
            symbol: Asset symbol to search for
            year: Single year to search (mutually exclusive with year_range)
            year_range: Tuple of (start_year, end_year) inclusive
            month: Single month to search (mutually exclusive with month_range)
            month_range: Tuple of (start_month, end_month) inclusive
            timeframes: List of timeframes to filter by (e.g., ["1min", "5min"])

        Returns:
            List of ParquetFileInfo objects sorted by year, month, symbol
        """
        return self.file_discovery.discover_files(
            symbol=symbol,
            year=year,
            year_range=year_range,
            month=month,
            month_range=month_range,
            timeframes=timeframes,
        )

    def is_already_processed(self, file_info: ParquetFileInfo) -> bool:
        """
        Check if file has already been processed (idempotency).

        Args:
            file_info: ParquetFileInfo object to check

        Returns:
            True if file has been processed, False otherwise
        """
        return self.idempotency_tracker.is_processed(file_info)

    def mark_as_processed(
        self, file_info: ParquetFileInfo, status: str = "completed"
    ) -> None:
        """
        Mark file as processed in idempotency tracker.

        Args:
            file_info: ParquetFileInfo object to mark as processed
            status: Processing status (completed, failed, etc.)
        """
        self.idempotency_tracker.mark_processed(file_info, status)

    def start_audit_session(
        self,
        symbol: str,
        year_range: Optional[Tuple[int, int]] = None,
        month_range: Optional[Tuple[int, int]] = None,
    ) -> str:
        """
        Start audit session and return session ID.

        Args:
            symbol: Asset symbol being processed
            year_range: Tuple of (start_year, end_year) or None
            month_range: Tuple of (start_month, end_month) or None

        Returns:
            Unique session ID
        """
        return self.audit_logger.start_session(symbol, year_range, month_range)

    def end_audit_session(self, session_id: str, status: str = "completed") -> None:
        """
        End audit session with final status.

        Args:
            session_id: Session ID to end
            status: Final status (completed, failed, cancelled)
        """
        self.audit_logger.end_session(session_id, status)

    def load_training_data(
        self,
        symbol: str,
        year: Optional[int] = None,
        year_range: Optional[Tuple[int, int]] = None,
        month: Optional[int] = None,
        month_range: Optional[Tuple[int, int]] = None,
        target_columns: Optional[List[str]] = None,
        feature_columns: Optional[List[str]] = None,
    ) -> Dict[str, Any]:
        """
        Load and prepare data for training from discovered files.

        Args:
            symbol: Asset symbol to load data for
            year: Single year to load (mutually exclusive with year_range)
            year_range: Tuple of (start_year, end_year) inclusive
            month: Single month to load (mutually exclusive with month_range)
            month_range: Tuple of (start_month, end_month) inclusive
            target_columns: List of target columns (uses config default if None)
            feature_columns: List of feature columns (uses config default if None)

        Returns:
            Dictionary containing loaded data and metadata
        """
        # Start audit session
        session_id = self.start_audit_session(symbol, year_range, month_range)

        try:
            # Discover files
            files = self.discover_files(
                symbol=symbol,
                year=year,
                year_range=year_range,
                month=month,
                month_range=month_range,
                timeframes=["1min"],  # Default to 1-minute data
            )

            if not files:
                raise DataNotFoundError(f"No files found for symbol={symbol}")

            # Update audit with discovered files
            self.audit_logger.update_session(session_id, files_discovered=len(files))

            # Process files with idempotency checks
            processed_data = []
            processed_count = 0
            skipped_count = 0
            failed_count = 0

            for file_info in files:
                try:
                    if self.is_already_processed(file_info):
                        logger.info(
                            f"Skipping already processed file: {file_info.file_path}"
                        )
                        skipped_count += 1
                        continue

                    # Load file data (placeholder - would implement actual loading)
                    file_data = self._load_file_data(
                        file_info, target_columns, feature_columns
                    )
                    processed_data.append(file_data)

                    # Mark as processed
                    self.mark_as_processed(file_info, "completed")
                    processed_count += 1

                except Exception as e:
                    logger.error(f"Failed to process file {file_info.file_path}: {e}")
                    self.idempotency_tracker.mark_failed(file_info, str(e))
                    self.audit_logger.log_error(
                        session_id, str(e), str(file_info.file_path)
                    )
                    failed_count += 1

            # Update audit with final counts
            self.audit_logger.update_session(
                session_id,
                files_processed=processed_count,
                files_skipped=skipped_count,
                files_failed=failed_count,
            )

            return {
                "data": processed_data,
                "files_processed": processed_count,
                "files_skipped": skipped_count,
                "files_failed": failed_count,
                "session_id": session_id,
            }

        except Exception as e:
            self.audit_logger.log_error(session_id, str(e))
            self.end_audit_session(session_id, "failed")
            raise
        finally:
            self.end_audit_session(session_id, "completed")

    def load_prediction_data(
        self,
        symbol: str,
        year: Optional[int] = None,
        year_range: Optional[Tuple[int, int]] = None,
        month: Optional[int] = None,
        month_range: Optional[Tuple[int, int]] = None,
        context_length: Optional[int] = None,
    ) -> Dict[str, Any]:
        """
        Load data for prediction/inference from discovered files.

        Args:
            symbol: Asset symbol to load data for
            year: Single year to load (mutually exclusive with year_range)
            year_range: Tuple of (start_year, end_year) inclusive
            month: Single month to load (mutually exclusive with month_range)
            month_range: Tuple of (start_month, end_month) inclusive
            context_length: Context length for prediction (uses config default if None)

        Returns:
            Dictionary containing prediction data and metadata
        """
        # Start audit session
        session_id = self.start_audit_session(symbol, year_range, month_range)

        try:
            # Discover most recent files
            files = self.discover_files(
                symbol=symbol,
                year=year,
                year_range=year_range,
                month=month,
                month_range=month_range,
                timeframes=["1min"],
            )

            if not files:
                raise DataNotFoundError(
                    f"No prediction files found for symbol={symbol}"
                )

            # Use most recent file for prediction
            latest_file = files[-1]

            # Load context data (placeholder - would implement actual loading)
            context_data = self._load_file_data(
                latest_file, context_length=context_length
            )

            self.audit_logger.update_session(
                session_id, files_discovered=len(files), files_processed=1
            )

            return {
                "context_data": context_data,
                "file_info": latest_file,
                "session_id": session_id,
            }

        except Exception as e:
            self.audit_logger.log_error(session_id, str(e))
            self.end_audit_session(session_id, "failed")
            raise
        finally:
            self.end_audit_session(session_id, "completed")

    def load_incremental_data(
        self, symbol: str, last_timestamp: datetime
    ) -> Dict[str, Any]:
        """
        Load only new data since last timestamp.

        Args:
            symbol: Asset symbol to load data for
            last_timestamp: Last processed timestamp

        Returns:
            Dictionary containing incremental data and metadata
        """
        # Start audit session
        session_id = self.start_audit_session(symbol)

        try:
            # Determine year/month range since last timestamp
            current_year = datetime.now().year
            current_month = datetime.now().month
            last_year = last_timestamp.year
            last_month = last_timestamp.month

            # Discover files since last timestamp
            files = []
            for year in range(last_year, current_year + 1):
                start_month = last_month if year == last_year else 1
                end_month = current_month if year == current_year else 12

                year_files = self.discover_files(
                    symbol=symbol,
                    year=year,
                    month_range=(start_month, end_month),
                    timeframes=["1min"],
                )
                files.extend(year_files)

            if not files:
                raise DataNotFoundError(
                    f"No new data found for symbol={symbol} since {last_timestamp}"
                )

            # Process incremental files
            processed_data = []
            processed_count = 0

            for file_info in files:
                if not self.is_already_processed(file_info):
                    file_data = self._load_file_data(file_info)
                    processed_data.append(file_data)
                    self.mark_as_processed(file_info, "completed")
                    processed_count += 1

            self.audit_logger.update_session(
                session_id, files_discovered=len(files), files_processed=processed_count
            )

            return {
                "data": processed_data,
                "files_processed": processed_count,
                "session_id": session_id,
            }

        except Exception as e:
            self.audit_logger.log_error(session_id, str(e))
            self.end_audit_session(session_id, "failed")
            raise
        finally:
            self.end_audit_session(session_id, "completed")

    def _load_file_data(
        self,
        file_info: ParquetFileInfo,
        target_columns: Optional[List[str]] = None,
        feature_columns: Optional[List[str]] = None,
        context_length: Optional[int] = None,
    ) -> Dict[str, Any]:
        """
        Load data from a single parquet file.

        This is a placeholder implementation that would be replaced with
        actual parquet file loading logic.

        Args:
            file_info: ParquetFileInfo object
            target_columns: Target columns to load
            feature_columns: Feature columns to load
            context_length: Context length for prediction data

        Returns:
            Dictionary containing loaded data
        """
        # Placeholder implementation
        logger.info(f"Loading data from {file_info.file_path}")

        return {
            "file_info": file_info,
            "data": f"Data from {file_info.file_path}",  # Placeholder
            "target_columns": target_columns or self.config.schema.target_columns,
            "feature_columns": feature_columns
            or list(self.config.schema.feature_columns.keys()),
            "context_length": context_length or 100,
        }

    def get_processing_stats(self) -> Dict[str, Any]:
        """Get processing statistics from idempotency tracker."""
        return self.idempotency_tracker.get_processing_stats()

    def get_audit_stats(self) -> Dict[str, Any]:
        """Get audit statistics."""
        return self.audit_logger.get_session_stats()

    def list_available_symbols(self) -> List[str]:
        """Get list of available symbols."""
        return self.file_discovery.get_available_symbols()

    def list_available_years(self) -> List[int]:
        """Get list of available years."""
        return self.file_discovery.get_available_years()

    def list_available_months(self, year: int) -> List[int]:
        """Get list of available months for a given year."""
        return self.file_discovery.get_available_months(year)
