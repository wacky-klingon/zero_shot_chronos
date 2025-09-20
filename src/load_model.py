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
from transformers import AutoModelForCausalLM, AutoTokenizer
import json
from datetime import datetime

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ChronosLoader:
    """Loader for trained Chronos models with inference capabilities."""
    
    def __init__(self, model_path: str = "data/models/trained_model", config_path: str = "config/chronos_config.yaml"):
        """Initialize the loader with model path and configuration."""
        self.model_path = Path(model_path)
        self.config_path = config_path
        self.config = self._load_config()
        self.model = None
        self.tokenizer = None
        self.model_metadata = {}
        
        # Create output directories
        self.predictions_dir = Path(self.config['paths']['predictions_dir'])
        self.predictions_dir.mkdir(parents=True, exist_ok=True)
        
        logger.info(f"ChronosLoader initialized with model: {model_path}")
    
    def _load_config(self) -> Dict[str, Any]:
        """Load configuration from YAML file."""
        try:
            with open(self.config_path, 'r') as file:
                config = yaml.safe_load(file)
            logger.info("Configuration loaded successfully")
            return config
        except Exception as e:
            logger.error(f"Failed to load configuration: {e}")
            raise
    
    def load_trained_model(self) -> None:
        """Load the trained model from disk."""
        try:
            logger.info(f"Loading trained model from: {self.model_path}")
            
            # Load model and tokenizer
            self.model = AutoModelForCausalLM.from_pretrained(
                str(self.model_path),
                torch_dtype=torch.float32,
                device_map="cpu"
            )
            
            self.tokenizer = AutoTokenizer.from_pretrained(str(self.model_path))
            
            # Load model metadata
            metadata_path = self.model_path / 'model_metadata.json'
            if metadata_path.exists():
                with open(metadata_path, 'r') as f:
                    self.model_metadata = json.load(f)
                logger.info("Model metadata loaded successfully")
            else:
                logger.warning("No model metadata found")
            
            # Set model to evaluation mode
            self.model.eval()
            
            logger.info("Trained model loaded successfully")
            
        except Exception as e:
            logger.error(f"Failed to load trained model: {e}")
            raise
    
    def predict(self, context: np.ndarray, prediction_length: Optional[int] = None) -> np.ndarray:
        """Generate predictions using the trained model."""
        if self.model is None:
            raise ValueError("Model must be loaded before making predictions")
        
        if prediction_length is None:
            prediction_length = self.config['inference']['prediction_length']
        
        try:
            logger.info(f"Generating predictions for context length: {len(context)}")
            
            # Convert context to tensor
            context_tensor = torch.tensor(context, dtype=torch.float32).unsqueeze(0)
            
            # Generate predictions
            with torch.no_grad():
                # For this simplified implementation, we'll use the model to predict
                # In a real Chronos implementation, this would be more sophisticated
                outputs = self.model(context_tensor.long())
                
                # Extract predictions (simplified)
                predictions = outputs.logits.squeeze(0).cpu().numpy()
                
                # Truncate to prediction length
                if len(predictions) > prediction_length:
                    predictions = predictions[:prediction_length]
                elif len(predictions) < prediction_length:
                    # Pad with last value if needed
                    padding = np.full(prediction_length - len(predictions), predictions[-1])
                    predictions = np.concatenate([predictions, padding])
            
            logger.info(f"Predictions generated: {len(predictions)} steps")
            return predictions
            
        except Exception as e:
            logger.error(f"Failed to generate predictions: {e}")
            raise
    
    def predict_quantiles(self, context: np.ndarray, 
                         quantiles: Optional[List[float]] = None,
                         prediction_length: Optional[int] = None) -> np.ndarray:
        """Generate quantile predictions for uncertainty estimation."""
        if quantiles is None:
            quantiles = self.config['inference']['quantiles']
        
        if prediction_length is None:
            prediction_length = self.config['inference']['prediction_length']
        
        try:
            logger.info(f"Generating quantile predictions: {quantiles}")
            
            # Generate multiple predictions for uncertainty estimation
            predictions = []
            n_samples = 10  # Number of samples for uncertainty estimation
            
            for _ in range(n_samples):
                pred = self.predict(context, prediction_length)
                predictions.append(pred)
            
            predictions = np.array(predictions)
            
            # Calculate quantiles
            quantile_predictions = []
            for q in quantiles:
                q_pred = np.quantile(predictions, q, axis=0)
                quantile_predictions.append(q_pred)
            
            quantile_predictions = np.array(quantile_predictions)
            
            logger.info(f"Quantile predictions generated: {quantile_predictions.shape}")
            return quantile_predictions
            
        except Exception as e:
            logger.error(f"Failed to generate quantile predictions: {e}")
            raise
    
    def evaluate_on_test_data(self, test_data: np.ndarray) -> Dict[str, float]:
        """Evaluate the model on test data."""
        if self.model is None:
            raise ValueError("Model must be loaded before evaluation")
        
        try:
            logger.info("Evaluating model on test data...")
            
            context_length = self.config['inference']['context_length']
            prediction_length = self.config['inference']['prediction_length']
            
            # Create test contexts and targets
            contexts = []
            targets = []
            
            for series in test_data:
                for i in range(len(series) - context_length - prediction_length + 1):
                    context = series[i:i + context_length]
                    target = series[i + context_length:i + context_length + prediction_length]
                    
                    contexts.append(context)
                    targets.append(target)
            
            contexts = np.array(contexts)
            targets = np.array(targets)
            
            # Generate predictions
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
                'mse': float(mse),
                'mae': float(mae),
                'mape': float(mape),
                'n_samples': len(contexts)
            }
            
            logger.info(f"Evaluation completed:")
            logger.info(f"  MSE: {mse:.4f}")
            logger.info(f"  MAE: {mae:.4f}")
            logger.info(f"  MAPE: {mape:.2f}%")
            logger.info(f"  Samples: {len(contexts)}")
            
            return metrics
            
        except Exception as e:
            logger.error(f"Failed to evaluate model: {e}")
            raise
    
    def save_predictions(self, predictions: np.ndarray, 
                        quantile_predictions: Optional[np.ndarray] = None,
                        output_path: Optional[str] = None) -> None:
        """Save predictions to file."""
        if output_path is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_path = self.predictions_dir / f"predictions_{timestamp}.csv"
        
        output_path = Path(output_path)
        
        try:
            logger.info(f"Saving predictions to: {output_path}")
            
            # Create DataFrame with predictions
            pred_df = pd.DataFrame({
                'step': range(1, len(predictions) + 1),
                'prediction': predictions
            })
            
            # Add quantile predictions if available
            if quantile_predictions is not None:
                quantiles = self.config['inference']['quantiles']
                for i, q in enumerate(quantiles):
                    pred_df[f'quantile_{q}'] = quantile_predictions[i]
            
            # Save to CSV
            pred_df.to_csv(output_path, index=False)
            
            logger.info(f"Predictions saved successfully to: {output_path}")
            
        except Exception as e:
            logger.error(f"Failed to save predictions: {e}")
            raise
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get information about the loaded model."""
        if self.model is None:
            return {'status': 'no_model_loaded'}
        
        return {
            'status': 'loaded',
            'model_path': str(self.model_path),
            'model_metadata': self.model_metadata,
            'config': self.config['inference']
        }
    
    def generate_sample_predictions(self) -> None:
        """Generate sample predictions using dummy data."""
        try:
            logger.info("Generating sample predictions...")
            
            # Generate sample context data
            context_length = self.config['inference']['context_length']
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
