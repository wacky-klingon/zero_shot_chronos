#!/usr/bin/env python3
"""
Model Loading Component for Native Chronos Implementation

This module handles loading trained Chronos models and running inference.
"""

import yaml
import torch
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, Optional, Any, List
import logging
from chronos import ChronosBoltPipeline
import json
from datetime import datetime

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ChronosLoader:
    """Loader for trained Chronos models with inference capabilities."""

    def __init__(
        self,
        model_path: str = "data/models/trained_model",
        config_path: str = "config/chronos_config.yaml",
    ):
        """Initialize the loader with model path and configuration."""
        self.model_path = Path(model_path)
        self.config_path = config_path
        self.config = self._load_config()
        self.model = None
        self.tokenizer = None
        self.model_metadata = {}

        # Create output directories
        self.predictions_dir = Path(self.config["paths"]["predictions_dir"])
        self.predictions_dir.mkdir(parents=True, exist_ok=True)

        logger.info(f"ChronosLoader initialized with model: {model_path}")

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

    def load_trained_model(self) -> None:
        """Load the trained Chronos model from disk."""
        try:
            logger.info(f"Loading trained Chronos model from: {self.model_path}")

            # Load ChronosBoltPipeline from the saved trained model
            self.model = ChronosBoltPipeline.from_pretrained(
                str(self.model_path), torch_dtype=torch.float32, device_map="cpu"
            )

            # Load model metadata
            metadata_path = self.model_path / "model_metadata.json"
            if metadata_path.exists():
                with open(metadata_path, "r") as f:
                    self.model_metadata = json.load(f)
                logger.info("Model metadata loaded successfully")
            else:
                logger.warning("No model metadata found")

            logger.info("Trained Chronos model loaded successfully")

        except Exception as e:
            logger.error(f"Failed to load trained Chronos model: {e}")
            raise

    def predict(
        self, context: np.ndarray, prediction_length: Optional[int] = None
    ) -> np.ndarray:
        """Generate predictions using the trained Chronos model."""
        if self.model is None:
            raise ValueError("Chronos model must be loaded before making predictions")

        if prediction_length is None:
            prediction_length = self.config["inference"]["prediction_length"]

        try:
            logger.info(
                f"Generating Chronos predictions for context length: {len(context)}"
            )

            # Convert context to tensor
            context_tensor = torch.tensor(context, dtype=torch.float32).unsqueeze(0)

            # Generate predictions using ChronosBoltPipeline
            with torch.no_grad():
                predictions = self.model.predict(
                    context_tensor, prediction_length=prediction_length
                )

                # Extract median prediction (0.5 quantile) - index 4 out of 9 quantiles
                median_pred = predictions[:, 4, :].squeeze(0).cpu().numpy()

            logger.info(f"Chronos predictions generated: {len(median_pred)} steps")
            return median_pred

        except Exception as e:
            logger.error(f"Failed to generate Chronos predictions: {e}")
            raise

    def predict_quantiles(
        self,
        context: np.ndarray,
        quantiles: Optional[List[float]] = None,
        prediction_length: Optional[int] = None,
    ) -> np.ndarray:
        """Generate quantile predictions using ChronosBoltPipeline."""
        if quantiles is None:
            quantiles = self.config["inference"]["quantiles"]

        if prediction_length is None:
            prediction_length = self.config["inference"]["prediction_length"]

        try:
            logger.info(f"Generating Chronos quantile predictions: {quantiles}")

            # Convert context to tensor
            context_tensor = torch.tensor(context, dtype=torch.float32).unsqueeze(0)

            # Generate quantile predictions using ChronosBoltPipeline
            with torch.no_grad():
                predictions = self.model.predict(
                    context_tensor, prediction_length=prediction_length
                )

                # Extract specific quantiles from Chronos output
                # ChronosBoltPipeline outputs shape: [batch, quantiles, prediction_length]
                # We need to map requested quantiles to Chronos quantiles
                chronos_quantiles = (
                    self.model.quantiles
                )  # [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]

                quantile_predictions = []
                for q in quantiles:
                    # Find closest Chronos quantile
                    closest_idx = np.argmin(np.abs(np.array(chronos_quantiles) - q))
                    q_pred = predictions[0, closest_idx, :].cpu().numpy()
                    quantile_predictions.append(q_pred)

                quantile_predictions = np.array(quantile_predictions)

            logger.info(
                f"Chronos quantile predictions generated: {quantile_predictions.shape}"
            )
            return quantile_predictions

        except Exception as e:
            logger.error(f"Failed to generate Chronos quantile predictions: {e}")
            raise

    def evaluate_on_test_data(self, test_data: np.ndarray) -> Dict[str, float]:
        """Evaluate the Chronos model on test data."""
        if self.model is None:
            raise ValueError("Chronos model must be loaded before evaluation")

        try:
            logger.info("Evaluating Chronos model on test data...")

            context_length = self.config["inference"]["context_length"]
            prediction_length = self.config["inference"]["prediction_length"]

            # Create test contexts and targets
            contexts = []
            targets = []

            for series in test_data:
                for i in range(len(series) - context_length - prediction_length + 1):
                    context = series[i : i + context_length]
                    target = series[
                        i + context_length : i + context_length + prediction_length
                    ]

                    contexts.append(context)
                    targets.append(target)

            contexts = np.array(contexts)
            targets = np.array(targets)

            # Generate predictions using ChronosBoltPipeline
            predictions = []
            for context in contexts:
                pred = self.predict(context, prediction_length)
                predictions.append(pred)

            predictions = np.array(predictions)

            # Calculate metrics
            mse = np.mean((predictions - targets) ** 2)
            mae = np.mean(np.abs(predictions - targets))
            mape = np.mean(np.abs((targets - predictions) / (targets + 1e-8))) * 100

            metrics = {
                "mse": float(mse),
                "mae": float(mae),
                "mape": float(mape),
                "n_samples": len(contexts),
            }

            logger.info("Chronos model evaluation completed:")
            logger.info(f"  MSE: {mse:.4f}")
            logger.info(f"  MAE: {mae:.4f}")
            logger.info(f"  MAPE: {mape:.2f}%")
            logger.info(f"  Samples: {len(contexts)}")

            return metrics

        except Exception as e:
            logger.error(f"Failed to evaluate Chronos model: {e}")
            raise

    def save_predictions(
        self,
        predictions: np.ndarray,
        quantile_predictions: Optional[np.ndarray] = None,
        output_path: Optional[str] = None,
    ) -> None:
        """Save predictions to file."""
        if output_path is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_path = self.predictions_dir / f"predictions_{timestamp}.csv"

        output_path = Path(output_path)

        try:
            logger.info(f"Saving predictions to: {output_path}")

            # Create DataFrame with predictions
            pred_df = pd.DataFrame(
                {"step": range(1, len(predictions) + 1), "prediction": predictions}
            )

            # Add quantile predictions if available
            if quantile_predictions is not None:
                quantiles = self.config["inference"]["quantiles"]
                for i, q in enumerate(quantiles):
                    pred_df[f"quantile_{q}"] = quantile_predictions[i]

            # Save to CSV
            pred_df.to_csv(output_path, index=False)

            logger.info(f"Predictions saved successfully to: {output_path}")

        except Exception as e:
            logger.error(f"Failed to save predictions: {e}")
            raise

    def get_model_info(self) -> Dict[str, Any]:
        """Get information about the loaded Chronos model."""
        if self.model is None:
            return {"status": "no_model_loaded"}

        return {
            "status": "loaded",
            "model_path": str(self.model_path),
            "model_metadata": self.model_metadata,
            "config": self.config["inference"],
            "quantiles": self.model.quantiles,
            "context_length": self.model.default_context_length,
            "pipeline_type": "ChronosBoltPipeline",
        }

    def generate_sample_predictions(self) -> None:
        """Generate sample predictions using dummy data."""
        try:
            logger.info("Generating sample predictions...")

            # Generate sample context data
            context_length = self.config["inference"]["context_length"]
            context = np.random.randn(context_length) * 10 + 100  # Sample time series

            # Generate predictions
            predictions = self.predict(context)

            # Generate quantile predictions
            quantile_predictions = self.predict_quantiles(context)

            # Save predictions
            self.save_predictions(predictions, quantile_predictions)

            logger.info("Sample predictions generated successfully")

        except Exception as e:
            logger.error(f"Failed to generate sample predictions: {e}")
            raise


def main():
    """Main function for independent execution."""
    print("Chronos Model Loading Component")
    print("=" * 40)

    try:
        # Initialize loader
        loader = ChronosLoader()

        # Load trained model
        print("Loading trained model...")
        loader.load_trained_model()

        # Display model info
        model_info = loader.get_model_info()
        print(f"Model loaded: {model_info['status']}")
        print(f"Model path: {model_info['model_path']}")

        # Generate sample predictions
        print("Generating sample predictions...")
        loader.generate_sample_predictions()

        # Test with dummy data
        print("Testing with dummy data...")
        dummy_data = np.random.randn(5, 200) * 10 + 100  # 5 series of 200 points
        metrics = loader.evaluate_on_test_data(dummy_data)

        print(f"\nEvaluation metrics: {metrics}")
        print("✓ Model loading component completed successfully!")

    except Exception as e:
        print(f"✗ Error: {e}")
        return 1

    return 0


if __name__ == "__main__":
    exit(main())
