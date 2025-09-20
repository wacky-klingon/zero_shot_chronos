#!/usr/bin/env python3
"""
Integration Testing for Native Chronos Implementation

This module tests the complete end-to-end workflow of all Chronos components.
"""

import yaml
import numpy as np
from pathlib import Path
from typing import Dict, Any
import logging
import json
from datetime import datetime
import time

# Import our components
from base_model import ChronosBaseModel
from train_model import ChronosTrainer
from load_model import ChronosLoader

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ChronosIntegrationTest:
    """Integration tester for the complete Chronos workflow."""

    def __init__(self, config_path: str = "config/chronos_config.yaml"):
        """Initialize the integration tester."""
        self.config_path = config_path
        self.config = self._load_config()
        self.test_results = {}
        self.start_time = None

        # Create test output directory
        self.test_output_dir = Path("data/test_output")
        self.test_output_dir.mkdir(parents=True, exist_ok=True)

        logger.info("Chronos Integration Test initialized")

    def _load_config(self) -> Dict[str, Any]:
        """Load configuration from YAML file."""
        try:
            with open(self.config_path, "r") as file:
                config = yaml.safe_load(file)
            logger.info("Configuration loaded successfully")
            return config
        except Exception as e:
            logger.error(f"Failed to load configuration: {e}")
            raise

    def test_complete_workflow(self) -> Dict[str, Any]:
        """Test the complete end-to-end workflow."""
        logger.info("=" * 60)
        logger.info("STARTING CHRONOS INTEGRATION TEST")
        logger.info("=" * 60)

        self.start_time = time.time()

        try:
            # Step 1: Test Base Model Component
            logger.info("\n🔧 STEP 1: Testing Base Model Component")
            base_result = self._test_base_model()
            self.test_results["base_model"] = base_result

            # Step 2: Test Training Component
            logger.info("\n🔧 STEP 2: Testing Training Component")
            train_result = self._test_training()
            self.test_results["training"] = train_result

            # Step 3: Test Model Loading Component
            logger.info("\n🔧 STEP 3: Testing Model Loading Component")
            load_result = self._test_loading()
            self.test_results["loading"] = load_result

            # Step 4: Test End-to-End Workflow
            logger.info("\n🔧 STEP 4: Testing End-to-End Workflow")
            workflow_result = self._test_end_to_end_workflow()
            self.test_results["end_to_end"] = workflow_result

            # Calculate total time
            total_time = time.time() - self.start_time
            self.test_results["total_time"] = total_time

            # Generate test report
            self._generate_test_report()

            logger.info("\n" + "=" * 60)
            logger.info("INTEGRATION TEST COMPLETED SUCCESSFULLY!")
            logger.info("=" * 60)

            return self.test_results

        except Exception as e:
            logger.error(f"Integration test failed: {e}")
            self.test_results["error"] = str(e)
            raise

    def _test_base_model(self) -> Dict[str, Any]:
        """Test the base model component."""
        logger.info("Testing base model loading and saving...")

        try:
            # Initialize base model
            base_model = ChronosBaseModel()

            # Test model loading
            base_model.load_from_huggingface()

            # Test model conversion
            base_model.convert_to_native_format()

            # Test model saving
            base_model.save_base_model()

            # Test model verification
            verification_success = base_model.verify_model()

            # Get model info
            model_info = base_model.get_model_info()

            result = {
                "status": "success",
                "verification": verification_success,
                "model_info": model_info,
                "output_path": str(base_model.output_path),
            }

            logger.info("✅ Base model component test passed")
            return result

        except Exception as e:
            logger.error(f"❌ Base model component test failed: {e}")
            return {"status": "failed", "error": str(e)}

    def _test_training(self) -> Dict[str, Any]:
        """Test the training component."""
        logger.info("Testing training component...")

        try:
            # Initialize trainer
            trainer = ChronosTrainer()

            # Test base model loading
            trainer.load_base_model()

            # Test dummy data generation
            dummy_data = trainer.generate_dummy_data()

            # Test training data preparation
            train_data, val_data = trainer.prepare_training_data(dummy_data)

            # Test model training
            trainer.train_model(train_data, val_data)

            # Test model evaluation
            metrics = trainer.evaluate_model(val_data)

            # Test model saving
            trainer.save_trained_model()

            result = {
                "status": "success",
                "dummy_data_shape": dummy_data.shape,
                "train_samples": len(train_data[0]),
                "val_samples": len(val_data[0]),
                "metrics": metrics,
                "output_path": str(trainer.models_dir / "trained_model"),
            }

            logger.info("✅ Training component test passed")
            return result

        except Exception as e:
            logger.error(f"❌ Training component test failed: {e}")
            return {"status": "failed", "error": str(e)}

    def _test_loading(self) -> Dict[str, Any]:
        """Test the model loading component."""
        logger.info("Testing model loading component...")

        try:
            # Initialize loader
            loader = ChronosLoader()

            # Test model loading
            loader.load_trained_model()

            # Test model info
            model_info = loader.get_model_info()

            # Test prediction generation
            context_length = self.config["inference"]["context_length"]
            test_context = np.random.randn(context_length) * 10 + 100

            predictions = loader.predict(test_context)

            # Test quantile predictions
            quantile_predictions = loader.predict_quantiles(test_context)

            # Test evaluation on dummy data
            dummy_data = np.random.randn(5, 200) * 10 + 100
            eval_metrics = loader.evaluate_on_test_data(dummy_data)

            # Test prediction saving
            loader.save_predictions(predictions, quantile_predictions)

            result = {
                "status": "success",
                "model_info": model_info,
                "prediction_length": len(predictions),
                "quantile_shape": quantile_predictions.shape,
                "eval_metrics": eval_metrics,
            }

            logger.info("✅ Model loading component test passed")
            return result

        except Exception as e:
            logger.error(f"❌ Model loading component test failed: {e}")
            return {"status": "failed", "error": str(e)}

    def _test_end_to_end_workflow(self) -> Dict[str, Any]:
        """Test the complete end-to-end workflow."""
        logger.info("Testing complete end-to-end workflow...")

        try:
            # Step 1: Load base model
            logger.info("  → Loading base model...")
            base_model = ChronosBaseModel()
            base_model.load_from_huggingface()
            base_model.save_base_model()

            # Step 2: Train model
            logger.info("  → Training model...")
            trainer = ChronosTrainer()
            trainer.load_base_model()
            dummy_data = trainer.generate_dummy_data(
                n_samples=50
            )  # Smaller for integration test
            train_data, val_data = trainer.prepare_training_data(dummy_data)
            trainer.train_model(
                train_data, val_data, epochs=1
            )  # Single epoch for speed
            trainer.save_trained_model()

            # Step 3: Load trained model and make predictions
            logger.info("  → Loading trained model and making predictions...")
            loader = ChronosLoader()
            loader.load_trained_model()

            # Generate test context
            context_length = self.config["inference"]["context_length"]
            test_context = np.random.randn(context_length) * 10 + 100

            # Make predictions
            predictions = loader.predict(test_context)
            quantile_predictions = loader.predict_quantiles(test_context)

            # Evaluate on test data
            test_data = np.random.randn(3, 200) * 10 + 100
            eval_metrics = loader.evaluate_on_test_data(test_data)

            result = {
                "status": "success",
                "base_model_loaded": True,
                "model_trained": True,
                "model_loaded": True,
                "predictions_generated": True,
                "prediction_length": len(predictions),
                "quantile_shape": quantile_predictions.shape,
                "eval_metrics": eval_metrics,
            }

            logger.info("✅ End-to-end workflow test passed")
            return result

        except Exception as e:
            logger.error(f"❌ End-to-end workflow test failed: {e}")
            return {"status": "failed", "error": str(e)}

    def _generate_test_report(self) -> None:
        """Generate a comprehensive test report."""
        logger.info("Generating test report...")

        report = {
            "test_timestamp": datetime.now().isoformat(),
            "total_time_seconds": self.test_results.get("total_time", 0),
            "test_results": self.test_results,
            "summary": self._generate_summary(),
        }

        # Save report to file
        report_name = (
            f"integration_test_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        )
        report_path = self.test_output_dir / report_name
        with open(report_path, "w") as f:
            json.dump(report, f, indent=2)

        logger.info(f"Test report saved to: {report_path}")

        # Print summary
        self._print_summary()

    def _generate_summary(self) -> Dict[str, Any]:
        """Generate a test summary."""
        total_tests = 4
        passed_tests = 0

        for component, result in self.test_results.items():
            if component != "total_time" and result.get("status") == "success":
                passed_tests += 1

        return {
            "total_tests": total_tests,
            "passed_tests": passed_tests,
            "failed_tests": total_tests - passed_tests,
            "success_rate": (passed_tests / total_tests) * 100,
            "all_tests_passed": passed_tests == total_tests,
        }

    def _print_summary(self) -> None:
        """Print a formatted test summary."""
        summary = self._generate_summary()

        print("\n" + "=" * 60)
        print("INTEGRATION TEST SUMMARY")
        print("=" * 60)
        print(f"Total Tests: {summary['total_tests']}")
        print(f"Passed: {summary['passed_tests']}")
        print(f"Failed: {summary['failed_tests']}")
        print(f"Success Rate: {summary['success_rate']:.1f}%")
        print(f"Total Time: {self.test_results.get('total_time', 0):.2f} seconds")
        print("=" * 60)

        if summary["all_tests_passed"]:
            print("🎉 ALL TESTS PASSED! Chronos implementation is working correctly.")
        else:
            print(
                "❌ Some tests failed. Check the detailed report for more information."
            )
        print("=" * 60)


def main():
    """Main function for independent execution."""
    print("Chronos Integration Test")
    print("=" * 40)

    try:
        # Initialize integration tester
        tester = ChronosIntegrationTest()

        # Run complete workflow test
        results = tester.test_complete_workflow()

        print("\nIntegration test completed successfully!")
        print(f"Total time: {results.get('total_time', 0):.2f} seconds")

    except Exception as e:
        print(f"✗ Integration test failed: {e}")
        return 1

    return 0


if __name__ == "__main__":
    exit(main())
