"""
Audit logging for parquet loader.

Simple JSON-based audit logging for progress tracking (KISS approach).
Tracks processing sessions with file counts and timestamps.
"""

import json
from pathlib import Path
from typing import Dict, Any, Optional, Tuple, List
from datetime import datetime
import logging

from .exceptions import ParquetLoaderError
from .config import ParquetLoaderConfig

logger = logging.getLogger(__name__)


class AuditError(ParquetLoaderError):
    """Audit logging related errors."""

    pass


class AuditLogger:
    """Simple audit logging for progress tracking (KISS approach)."""

    def __init__(self, config: ParquetLoaderConfig):
        """Initialize audit logger with configuration."""
        self.config = config
        self.audit_dir = Path(config.data_paths.root_dir) / config.audit.log_dir
        self.audit_dir.mkdir(exist_ok=True)

        logger.info(f"AuditLogger initialized with audit_dir: {self.audit_dir}")

    def start_session(
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
        session_id = f"{symbol}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

        session_data = {
            "session_id": session_id,
            "start_time": datetime.now().isoformat(),
            "symbol": symbol,
            "year_range": list(year_range) if year_range else None,
            "month_range": list(month_range) if month_range else None,
            "files_discovered": 0,
            "files_processed": 0,
            "files_skipped": 0,
            "files_failed": 0,
            "status": "running",
            "errors": [],
        }

        try:
            audit_file = self.audit_dir / f"{session_id}.json"
            with open(audit_file, "w") as f:
                json.dump(session_data, f, indent=2)

            logger.info(f"Started audit session: {session_id}")
            return session_id

        except (IOError, OSError) as e:
            raise AuditError(f"Failed to start audit session {session_id}: {e}")

    def update_session(self, session_id: str, **updates) -> None:
        """
        Update session with progress information.

        Args:
            session_id: Session ID to update
            **updates: Key-value pairs to update in session data
        """
        audit_file = self.audit_dir / f"{session_id}.json"

        if not audit_file.exists():
            logger.warning(f"Audit file not found for session: {session_id}")
            return

        try:
            # Load existing session data
            with open(audit_file, "r") as f:
                session_data = json.load(f)

            # Update with new data
            session_data.update(updates)

            # Add update timestamp
            session_data["last_updated"] = datetime.now().isoformat()

            # Save updated session data
            with open(audit_file, "w") as f:
                json.dump(session_data, f, indent=2)

            logger.debug(f"Updated audit session: {session_id}")

        except (json.JSONDecodeError, IOError, OSError) as e:
            raise AuditError(f"Failed to update audit session {session_id}: {e}")

    def end_session(self, session_id: str, status: str = "completed") -> None:
        """
        End audit session with final status.

        Args:
            session_id: Session ID to end
            status: Final status (completed, failed, cancelled)
        """
        try:
            self.update_session(
                session_id, end_time=datetime.now().isoformat(), status=status
            )

            logger.info(f"Ended audit session: {session_id} with status: {status}")

        except AuditError as e:
            logger.error(f"Failed to end audit session {session_id}: {e}")

    def log_error(
        self, session_id: str, error_message: str, file_path: Optional[str] = None
    ) -> None:
        """
        Log an error in the audit session.

        Args:
            session_id: Session ID to log error for
            error_message: Error message
            file_path: Optional file path where error occurred
        """
        error_entry = {
            "timestamp": datetime.now().isoformat(),
            "message": error_message,
            "file_path": file_path,
        }

        try:
            audit_file = self.audit_dir / f"{session_id}.json"

            if audit_file.exists():
                with open(audit_file, "r") as f:
                    session_data = json.load(f)

                if "errors" not in session_data:
                    session_data["errors"] = []

                session_data["errors"].append(error_entry)
                session_data["last_updated"] = datetime.now().isoformat()

                with open(audit_file, "w") as f:
                    json.dump(session_data, f, indent=2)

                logger.warning(f"Logged error in session {session_id}: {error_message}")

        except (json.JSONDecodeError, IOError, OSError) as e:
            logger.error(f"Failed to log error in session {session_id}: {e}")

    def get_session(self, session_id: str) -> Optional[Dict[str, Any]]:
        """
        Get session data by session ID.

        Args:
            session_id: Session ID to retrieve

        Returns:
            Session data dictionary or None if not found
        """
        audit_file = self.audit_dir / f"{session_id}.json"

        if not audit_file.exists():
            return None

        try:
            with open(audit_file, "r") as f:
                return json.load(f)

        except (json.JSONDecodeError, IOError, OSError) as e:
            logger.error(f"Failed to load session {session_id}: {e}")
            return None

    def list_sessions(
        self, symbol: Optional[str] = None, status: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """
        List audit sessions with optional filtering.

        Args:
            symbol: Filter by symbol (optional)
            status: Filter by status (optional)

        Returns:
            List of session data dictionaries
        """
        sessions = []

        try:
            for audit_file in self.audit_dir.glob("*.json"):
                try:
                    with open(audit_file, "r") as f:
                        session_data = json.load(f)

                    # Apply filters
                    if symbol and session_data.get("symbol") != symbol:
                        continue
                    if status and session_data.get("status") != status:
                        continue

                    sessions.append(session_data)

                except (json.JSONDecodeError, IOError, OSError):
                    continue

            # Sort by start time (newest first)
            sessions.sort(key=lambda x: x.get("start_time", ""), reverse=True)

        except OSError as e:
            logger.error(f"Failed to list sessions: {e}")

        return sessions

    def get_session_stats(self) -> Dict[str, Any]:
        """
        Get overall session statistics.

        Returns:
            Dictionary with session statistics
        """
        sessions = self.list_sessions()

        stats = {
            "total_sessions": len(sessions),
            "completed": 0,
            "failed": 0,
            "running": 0,
            "cancelled": 0,
            "total_files_discovered": 0,
            "total_files_processed": 0,
            "total_files_skipped": 0,
            "total_files_failed": 0,
        }

        for session in sessions:
            status = session.get("status", "unknown")
            if status in stats:
                stats[status] += 1

            stats["total_files_discovered"] += session.get("files_discovered", 0)
            stats["total_files_processed"] += session.get("files_processed", 0)
            stats["total_files_skipped"] += session.get("files_skipped", 0)
            stats["total_files_failed"] += session.get("files_failed", 0)

        return stats

    def cleanup_old_sessions(self, max_age_hours: Optional[int] = None) -> int:
        """
        Clean up old session files.

        Args:
            max_age_hours: Maximum age in hours (uses config default if None)

        Returns:
            Number of files cleaned up
        """
        if max_age_hours is None:
            max_age_hours = self.config.audit.session_timeout_hours

        cutoff_time = datetime.now().timestamp() - (max_age_hours * 3600)
        cleaned_count = 0

        try:
            for audit_file in self.audit_dir.glob("*.json"):
                if audit_file.stat().st_mtime < cutoff_time:
                    audit_file.unlink()
                    cleaned_count += 1
                    logger.info(f"Cleaned up old session file: {audit_file}")

        except OSError as e:
            logger.error(f"Failed to cleanup old sessions: {e}")

        return cleaned_count
