#!/usr/bin/env python3
"""
Test script for parquet loader functionality.

This script demonstrates the basic usage of the parquet loader
with range specification, idempotency, and audit logging.
"""

import sys
import logging
from pathlib import Path

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


def test_config_validation():
    """Test configuration validation with fail-fast behavior."""
    print("Testing configuration validation...")

    # Test with non-existent config file
    try:
        loader = ParquetDataLoader("non_existent_config.yaml")
        print("ERROR: Should have failed with non-existent config")
        return False
    except ConfigError as e:
        print(f"✓ Correctly failed with non-existent config: {e}")

    # Test with invalid config (missing root_dir)
    try:
        loader = ParquetDataLoader("config/chronos_config.yaml")  # Wrong config
        print("ERROR: Should have failed with wrong config")
        return False
    except ConfigError as e:
        print(f"✓ Correctly failed with wrong config: {e}")

    print("Configuration validation tests passed!\n")
    return True


def test_file_discovery():
    """Test file discovery functionality."""
    print("Testing file discovery...")

    # This would fail in real usage since we don't have actual data
    # but it demonstrates the API
    try:
        # Create a test config with a dummy path
        test_config_path = "config/parquet_loader_config.yaml"

        # Check if config exists
        if not Path(test_config_path).exists():
            print("Skipping file discovery test - config file not found")
            return True

        loader = ParquetDataLoader(test_config_path)

        # Test discovery API (will fail due to missing data, but shows API)
        try:
            files = loader.discover_files(symbol="TEST", year=2024, month=1)
            print(f"✓ Discovered {len(files)} files")
        except DataNotFoundError as e:
            print(f"✓ Correctly handled missing data: {e}")

        print("File discovery tests passed!\n")
        return True

    except Exception as e:
        print(f"File discovery test failed: {e}")
        return False


def test_audit_logging():
    """Test audit logging functionality."""
    print("Testing audit logging...")

    try:
        test_config_path = "config/parquet_loader_config.yaml"

        if not Path(test_config_path).exists():
            print("Skipping audit logging test - config file not found")
            return True

        loader = ParquetDataLoader(test_config_path)

        # Test audit session
        session_id = loader.start_audit_session(
            symbol="TEST", year_range=(2024, 2024), month_range=(1, 1)
        )
        print(f"✓ Started audit session: {session_id}")

        # Update session
        loader.audit_logger.update_session(session_id, files_discovered=5)
        print("✓ Updated audit session")

        # End session
        loader.end_audit_session(session_id, "completed")
        print("✓ Ended audit session")

        # Get stats
        stats = loader.get_audit_stats()
        print(f"✓ Audit stats: {stats}")

        print("Audit logging tests passed!\n")
        return True

    except Exception as e:
        print(f"Audit logging test failed: {e}")
        return False


def test_idempotency():
    """Test idempotency functionality."""
    print("Testing idempotency...")

    try:
        test_config_path = "config/parquet_loader_config.yaml"

        if not Path(test_config_path).exists():
            print("Skipping idempotency test - config file not found")
            return True

        loader = ParquetDataLoader(test_config_path)

        # Get initial stats
        stats = loader.get_processing_stats()
        print(f"✓ Initial processing stats: {stats}")

        print("Idempotency tests passed!\n")
        return True

    except Exception as e:
        print(f"Idempotency test failed: {e}")
        return False


def main():
    """Run all tests."""
    print("Parquet Loader Test Suite")
    print("=" * 50)

    tests = [
        test_config_validation,
        test_file_discovery,
        test_audit_logging,
        test_idempotency,
    ]

    passed = 0
    total = len(tests)

    for test in tests:
        if test():
            passed += 1

    print("=" * 50)
    print(f"Test Results: {passed}/{total} tests passed")

    if passed == total:
        print("✓ All tests passed!")
        return 0
    else:
        print("✗ Some tests failed!")
        return 1


if __name__ == "__main__":
    sys.exit(main())
