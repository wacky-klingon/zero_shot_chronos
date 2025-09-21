#!/usr/bin/env python3
"""
Chronos Integration Example

This example shows how to integrate the parquet loader with existing
Chronos training and prediction components.
"""

import sys
import logging
from pathlib import Path
from datetime import datetime

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

# Import after path modification
from parquet_loader import (  # noqa: E402
    ParquetDataLoader,
    ConfigError,
    DataNotFoundError,
)

# Setup logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


class ChronosTrainerWithParquet:
    """Enhanced ChronosTrainer with parquet loader integration."""

    def __init__(self, parquet_config_path: str = None):
        """Initialize trainer with optional parquet loader."""
        self.parquet_loader = None

        if parquet_config_path:
            try:
                self.parquet_loader = ParquetDataLoader(parquet_config_path)
                logger.info("Parquet loader initialized successfully")
            except ConfigError as e:
                logger.error(f"Failed to initialize parquet loader: {e}")
                raise
        else:
            logger.info("No parquet config provided, using dummy data")

    def load_training_data_from_parquet(
        self, symbol: str, year_range: tuple = None, month_range: tuple = None
    ) -> dict:
        """
        Load training data from parquet files.

        Args:
            symbol: Asset symbol to load
            year_range: Tuple of (start_year, end_year)
            month_range: Tuple of (start_month, end_month)

        Returns:
            Dictionary containing training data and metadata
        """
        if not self.parquet_loader:
            raise ValueError("Parquet loader not initialized")

        logger.info(
            f"Loading training data for {symbol} with ranges: year={year_range}, month={month_range}"
        )

        # Load data with range specification
        result = self.parquet_loader.load_training_data(
            symbol=symbol,
            year_range=year_range,
            month_range=month_range,
            target_columns=["target_close", "target_volatility"],
            feature_columns=["feature_1", "feature_2", "feature_3"],
        )

        logger.info(
            f"Loaded {result['files_processed']} files, skipped {result['files_skipped']}"
        )
        return result

    def train_model(self, data: dict) -> dict:
        """Train model with loaded data (placeholder)."""
        logger.info("Training model with loaded data...")

        # Placeholder training logic
        return {
            "model_id": "trained_model_123",
            "training_data_files": data.get("files_processed", 0),
            "session_id": data.get("session_id"),
        }


class ChronosLoaderWithParquet:
    """Enhanced ChronosLoader with parquet loader integration."""

    def __init__(self, parquet_config_path: str = None):
        """Initialize loader with optional parquet loader."""
        self.parquet_loader = None

        if parquet_config_path:
            try:
                self.parquet_loader = ParquetDataLoader(parquet_config_path)
                logger.info("Parquet loader initialized successfully")
            except ConfigError as e:
                logger.error(f"Failed to initialize parquet loader: {e}")
                raise
        else:
            logger.info("No parquet config provided")

    def load_context_from_parquet(
        self, symbol: str, year: int = None, month: int = None
    ) -> dict:
        """
        Load prediction context from parquet files.

        Args:
            symbol: Asset symbol to load
            year: Year to load (uses current year if None)
            month: Month to load (uses current month if None)

        Returns:
            Dictionary containing context data
        """
        if not self.parquet_loader:
            raise ValueError("Parquet loader not initialized")

        # Use current year/month if not specified
        if year is None:
            year = datetime.now().year
        if month is None:
            month = datetime.now().month

        logger.info(f"Loading prediction context for {symbol} from {year}-{month:02d}")

        # Load prediction data
        result = self.parquet_loader.load_prediction_data(
            symbol=symbol, year=year, month=month, context_length=100
        )

        logger.info(f"Loaded context data from {result['file_info'].file_path}")
        return result

    def predict(self, context_data: dict) -> dict:
        """Generate predictions (placeholder)."""
        logger.info("Generating predictions...")

        # Placeholder prediction logic
        return {
            "predictions": [1.0, 2.0, 3.0, 4.0, 5.0],
            "context_file": str(
                context_data.get("file_info", {}).get("file_path", "unknown")
            ),
            "session_id": context_data.get("session_id"),
        }


def example_training_workflow():
    """Example training workflow with parquet loader."""
    print("Training Workflow Example")
    print("-" * 30)

    try:
        # Initialize trainer with parquet loader
        trainer = ChronosTrainerWithParquet("config/parquet_loader_config.yaml")

        # Load training data with range specification
        training_data = trainer.load_training_data_from_parquet(
            symbol="SYMBOL",
            year_range=(2020, 2022),  # 3 years of data
            month_range=(1, 6),  # First half of each year
        )

        # Train model
        model_result = trainer.train_model(training_data)

        print(f"✓ Training completed: {model_result}")
        return True

    except Exception as e:
        print(f"✗ Training workflow failed: {e}")
        return False


def example_prediction_workflow():
    """Example prediction workflow with parquet loader."""
    print("\nPrediction Workflow Example")
    print("-" * 30)

    try:
        # Initialize loader with parquet loader
        loader = ChronosLoaderWithParquet("config/parquet_loader_config.yaml")

        # Load prediction context
        context_data = loader.load_context_from_parquet(
            symbol="SYMBOL", year=2024, month=1
        )

        # Generate predictions
        predictions = loader.predict(context_data)

        print(f"✓ Prediction completed: {predictions}")
        return True

    except Exception as e:
        print(f"✗ Prediction workflow failed: {e}")
        return False


def example_audit_and_stats():
    """Example of audit logging and statistics."""
    print("\nAudit and Statistics Example")
    print("-" * 30)

    try:
        # Initialize parquet loader
        loader = ParquetDataLoader("config/parquet_loader_config.yaml")

        # Get processing statistics
        processing_stats = loader.get_processing_stats()
        print(f"Processing stats: {processing_stats}")

        # Get audit statistics
        audit_stats = loader.get_audit_stats()
        print(f"Audit stats: {audit_stats}")

        # List available data
        symbols = loader.list_available_symbols()
        years = loader.list_available_years()
        print(f"Available symbols: {symbols}")
        print(f"Available years: {years}")

        print("✓ Audit and stats example completed")
        return True

    except Exception as e:
        print(f"✗ Audit and stats example failed: {e}")
        return False


def main():
    """Run all examples."""
    print("Chronos Parquet Loader Integration Examples")
    print("=" * 50)

    examples = [
        example_training_workflow,
        example_prediction_workflow,
        example_audit_and_stats,
    ]

    passed = 0
    total = len(examples)

    for example in examples:
        if example():
            passed += 1

    print("\n" + "=" * 50)
    print(f"Example Results: {passed}/{total} examples completed")

    if passed == total:
        print("✓ All examples completed successfully!")
        return 0
    else:
        print("✗ Some examples failed!")
        return 1


if __name__ == "__main__":
    sys.exit(main())
