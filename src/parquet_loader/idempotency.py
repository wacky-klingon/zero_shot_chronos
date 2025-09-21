"""
Idempotency tracking for parquet loader.

Tracks processed files to ensure no reprocessing of already processed data.
Uses JSON-based state file with file checksums for change detection.
"""

import json
import hashlib
from pathlib import Path
from typing import Dict, Any, Optional
from datetime import datetime
import logging

from .exceptions import ParquetLoaderError
from .config import ParquetLoaderConfig
from .file_discovery import ParquetFileInfo

logger = logging.getLogger(__name__)


class IdempotencyError(ParquetLoaderError):
    """Idempotency-related errors."""


class IdempotencyTracker:
    """Tracks processed files to ensure idempotency."""

    def __init__(self, config: ParquetLoaderConfig):
        """Initialize idempotency tracker with configuration."""
        self.config = config
        self.state_file = (
            Path(config.data_paths.root_dir) / config.idempotency.state_file
        )
        self.processed_files = self._load_state()

        logger.info(
            f"IdempotencyTracker initialized with state_file: {self.state_file}"
        )

    def _load_state(self) -> Dict[str, Any]:
        """Load processed files state from disk."""
        if not self.state_file.exists():
            logger.info("No existing state file found, starting fresh")
            return {"processed_files": {}}

        try:
            with open(self.state_file, "r") as f:
                state = json.load(f)

            # Ensure the structure is correct
            if "processed_files" not in state:
                state = {"processed_files": {}}

            logger.info(
                "Loaded state with %d processed files", len(state["processed_files"])
            )
            return state

        except (json.JSONDecodeError, IOError) as e:
            logger.warning("Failed to load state file %s: %s", self.state_file, e)
            return {"processed_files": {}}

    def _save_state(self) -> None:
        """Save processed files state to disk."""
        try:
            # Ensure parent directory exists
            self.state_file.parent.mkdir(parents=True, exist_ok=True)

            # Write state file atomically
            temp_file = self.state_file.with_suffix(".tmp")
            with open(temp_file, "w", encoding="utf-8") as f:
                json.dump(self.processed_files, f, indent=2)

            # Atomic move
            temp_file.replace(self.state_file)

            logger.debug("State saved to %s", self.state_file)

        except (IOError, OSError) as e:
            raise IdempotencyError(
                f"Failed to save state file {self.state_file}: {e}"
            ) from e

    def is_processed(self, file_info: ParquetFileInfo) -> bool:
        """
        Check if file has already been processed.

        Args:
            file_info: ParquetFileInfo object to check

        Returns:
            True if file has been processed, False otherwise
        """
        file_key = file_info.file_key

        if file_key not in self.processed_files["processed_files"]:
            return False

        # Check if file has changed by comparing checksums
        try:
            current_checksum = self._calculate_checksum(file_info.file_path)
            stored_checksum = self.processed_files["processed_files"][file_key][
                "checksum"
            ]

            if current_checksum != stored_checksum:
                logger.info("File %s has changed, will reprocess", file_info.file_path)
                return False

            return True

        except Exception as e:
            logger.warning("Failed to check file %s: %s", file_info.file_path, e)
            return False

    def mark_processed(
        self, file_info: ParquetFileInfo, status: str = "completed"
    ) -> None:
        """
        Mark file as processed.

        Args:
            file_info: ParquetFileInfo object to mark as processed
            status: Processing status (completed, failed, etc.)
        """
        file_key = file_info.file_key

        try:
            checksum = self._calculate_checksum(file_info.file_path)

            self.processed_files["processed_files"][file_key] = {
                "file_path": str(file_info.file_path),
                "processed_at": datetime.now().isoformat(),
                "checksum": checksum,
                "status": status,
                "symbol": file_info.symbol,
                "year": file_info.year,
                "month": file_info.month,
                "hash_id": file_info.hash_id,
            }

            self._save_state()
            logger.debug("Marked file as processed: %s", file_info.file_path)

        except Exception as e:
            raise IdempotencyError(
                f"Failed to mark file as processed {file_info.file_path}: {e}"
            ) from e

    def mark_failed(self, file_info: ParquetFileInfo, error_message: str = "") -> None:
        """
        Mark file as failed processing.

        Args:
            file_info: ParquetFileInfo object to mark as failed
            error_message: Error message describing the failure
        """
        file_key = file_info.file_key

        self.processed_files["processed_files"][file_key] = {
            "file_path": str(file_info.file_path),
            "processed_at": datetime.now().isoformat(),
            "checksum": "",  # No checksum for failed files
            "status": "failed",
            "error_message": error_message,
            "symbol": file_info.symbol,
            "year": file_info.year,
            "month": file_info.month,
            "hash_id": file_info.hash_id,
        }

        self._save_state()
        logger.warning(
            "Marked file as failed: %s - %s", file_info.file_path, error_message
        )

    def _calculate_checksum(self, file_path: Path) -> str:
        """
        Calculate file checksum for change detection.

        Args:
            file_path: Path to file to calculate checksum for

        Returns:
            Hexadecimal checksum string
        """
        algorithm = self.config.idempotency.checksum_algorithm.lower()

        if algorithm == "md5":
            hash_func = hashlib.md5()
        elif algorithm == "sha1":
            hash_func = hashlib.sha1()
        elif algorithm == "sha256":
            hash_func = hashlib.sha256()
        else:
            raise IdempotencyError(f"Unsupported checksum algorithm: {algorithm}")

        try:
            with open(file_path, "rb") as f:
                for chunk in iter(lambda: f.read(4096), b""):
                    hash_func.update(chunk)

            return hash_func.hexdigest()

        except (IOError, OSError) as e:
            raise IdempotencyError(
                f"Failed to calculate checksum for {file_path}: {e}"
            ) from e

    def get_processed_files(
        self, symbol: Optional[str] = None, status: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Get processed files with optional filtering.

        Args:
            symbol: Filter by symbol (optional)
            status: Filter by status (optional)

        Returns:
            Dictionary of processed files
        """
        processed = self.processed_files["processed_files"]

        if symbol or status:
            filtered = {}
            for key, info in processed.items():
                if symbol and info.get("symbol") != symbol:
                    continue
                if status and info.get("status") != status:
                    continue
                filtered[key] = info
            return filtered

        return processed

    def get_processing_stats(self) -> Dict[str, int]:
        """
        Get processing statistics.

        Returns:
            Dictionary with processing statistics
        """
        processed = self.processed_files["processed_files"]

        stats = {"total_files": len(processed), "completed": 0, "failed": 0, "other": 0}

        for info in processed.values():
            status = info.get("status", "unknown")
            if status == "completed":
                stats["completed"] += 1
            elif status == "failed":
                stats["failed"] += 1
            else:
                stats["other"] += 1

        return stats

    def clear_state(self) -> None:
        """Clear all processed file state."""
        self.processed_files = {"processed_files": {}}
        self._save_state()
        logger.info("Cleared all processed file state")

    def remove_file(self, file_info: ParquetFileInfo) -> None:
        """
        Remove file from processed state.

        Args:
            file_info: ParquetFileInfo object to remove
        """
        file_key = file_info.file_key

        if file_key in self.processed_files["processed_files"]:
            del self.processed_files["processed_files"][file_key]
            self._save_state()
            logger.info("Removed file from state: %s", file_info.file_path)
        else:
            logger.warning("File not found in state: %s", file_info.file_path)
