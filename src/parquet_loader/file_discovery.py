"""
File discovery for parquet files in year/month directory structure.

Supports range-based discovery with flexible year and month specifications.
Handles missing directories gracefully without failing.
"""

import re
from pathlib import Path
from typing import List, Optional, Tuple
from dataclasses import dataclass
import logging

from .exceptions import DataNotFoundError, InvalidFilenameError
from .config import ParquetLoaderConfig

logger = logging.getLogger(__name__)


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
        """Check if file is a parquet file."""
        return self.extension == "parquet"

    @property
    def is_json(self) -> bool:
        """Check if file is a JSON file."""
        return self.extension == "json"

    @property
    def file_key(self) -> str:
        """Generate unique key for this file."""
        return f"{self.symbol}_{self.year}_{self.month:02d}_{self.hash_id}"


class FileDiscovery:
    """Discovers and manages parquet files in the year/month directory structure."""

    def __init__(self, config: ParquetLoaderConfig):
        """Initialize file discovery with configuration."""
        self.config = config
        self.root_dir = Path(config.data_paths.root_dir)
        self.naming_regex = re.compile(config.file_patterns.naming_regex)

        logger.info(f"FileDiscovery initialized with root_dir: {self.root_dir}")

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

        Raises:
            ValueError: If neither year nor year_range is specified
            DataNotFoundError: If no files found and strict mode enabled
        """
        # Validate parameters
        if year is None and year_range is None:
            raise ValueError("Either 'year' or 'year_range' must be specified")

        if year is not None and year_range is not None:
            raise ValueError("Cannot specify both 'year' and 'year_range'")

        if month is not None and month_range is not None:
            raise ValueError("Cannot specify both 'month' and 'month_range'")

        files = []

        # Determine year range to search
        if year_range:
            years_to_search = range(year_range[0], year_range[1] + 1)
        else:
            years_to_search = [year]

        logger.info(
            f"Discovering files for symbol={symbol}, years={list(years_to_search)}"
        )

        for search_year in years_to_search:
            year_dir = self.root_dir / str(search_year)

            if not year_dir.exists():
                logger.warning(f"Year directory not found: {year_dir}")
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
                    logger.warning(f"Month directory not found: {month_dir}")
                    continue  # Skip missing months

                # Search for parquet files in this month directory
                for file_path in month_dir.glob("*.parquet"):
                    try:
                        file_info = self._parse_filename(file_path)
                        if self._matches_criteria(file_info, symbol, timeframes):
                            files.append(file_info)
                            logger.debug(f"Found file: {file_path}")
                    except InvalidFilenameError as e:
                        logger.warning(
                            f"Skipping file with invalid name: {file_path} - {e}"
                        )
                        continue

        # Sort files by year, month, symbol
        files.sort(key=lambda x: (x.year, x.month, x.symbol))

        logger.info(f"Discovered {len(files)} files for symbol={symbol}")
        return files

    def _parse_filename(self, file_path: Path) -> ParquetFileInfo:
        """Parse filename to extract metadata."""
        match = self.naming_regex.match(file_path.name)
        if not match:
            raise InvalidFilenameError(
                f"Filename does not match expected pattern: {file_path.name}"
            )

        symbol, timeframe, horizon, year, month, hash_id, extension = match.groups()

        return ParquetFileInfo(
            file_path=file_path,
            symbol=symbol,
            timeframe=timeframe,
            horizon=int(horizon),
            year=int(year),
            month=int(month),
            hash_id=hash_id,
            extension=extension,
        )

    def _matches_criteria(
        self,
        file_info: ParquetFileInfo,
        symbol: str,
        timeframes: Optional[List[str]] = None,
    ) -> bool:
        """Check if file matches discovery criteria."""
        if file_info.symbol != symbol:
            return False

        if timeframes and file_info.timeframe not in timeframes:
            return False

        return True

    def get_available_years(self) -> List[int]:
        """Get list of available years in the root directory."""
        years = []
        for item in self.root_dir.iterdir():
            if item.is_dir() and item.name.isdigit():
                try:
                    years.append(int(item.name))
                except ValueError:
                    continue
        return sorted(years)

    def get_available_months(self, year: int) -> List[int]:
        """Get list of available months for a given year."""
        year_dir = self.root_dir / str(year)
        if not year_dir.exists():
            return []

        months = []
        for item in year_dir.iterdir():
            if item.is_dir() and item.name.isdigit():
                try:
                    month_num = int(item.name)
                    if 1 <= month_num <= 12:
                        months.append(month_num)
                except ValueError:
                    continue
        return sorted(months)

    def get_available_symbols(
        self, year: Optional[int] = None, month: Optional[int] = None
    ) -> List[str]:
        """Get list of available symbols in the specified year/month."""
        symbols = set()

        if year:
            years_to_search = [year]
        else:
            years_to_search = self.get_available_years()

        for search_year in years_to_search:
            year_dir = self.root_dir / str(search_year)
            if not year_dir.exists():
                continue

            if month:
                months_to_search = [month]
            else:
                months_to_search = self.get_available_months(search_year)

            for search_month in months_to_search:
                month_dir = year_dir / f"{search_month:02d}"
                if not month_dir.exists():
                    continue

                for file_path in month_dir.glob("*.parquet"):
                    try:
                        file_info = self._parse_filename(file_path)
                        symbols.add(file_info.symbol)
                    except InvalidFilenameError:
                        continue

        return sorted(list(symbols))
