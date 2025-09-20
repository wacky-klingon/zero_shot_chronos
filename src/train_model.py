#!/usr/bin/env python3
"""
Training Component for Native Chronos Implementation

This module handles training Chronos models on dummy data and saving trained models.
"""

import yaml
import torch
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, Optional, Any, Tuple, List
import logging
from transformers import AutoModelForCausalLM, AutoTokenizer
import json
from datetime import datetime

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ChronosTrainer:
    """Trainer for Chronos models with dummy data generation."""
    
    def __init__(self, base_model_path: str = "data/models/base", config_path: str = "config/chronos_config.yaml"):
        """Initialize the trainer with base model path and configuration."""
        self.base_model_path = Path(base_model_path)
        self.config_path = config_path
        self.config = self._load_config()
        self.model = None
        self.tokenizer = None
        self.training_history = []
        
        # Create output directories
        self.models_dir = Path(self.config['paths']['models_dir'])
        self.models_dir.mkdir(parents=True, exist_ok=True)
        
        self.dummy_data_dir = Path(self.config['paths']['dummy_data_dir'])
        self.dummy_data_dir.mkdir(parents=True, exist_ok=True)
        
        logger.info(f"ChronosTrainer initialized with base model: {base_model_path}")
    
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
    
    def load_base_model(self) -> None:
        """Load the base model for training."""
        try:
            logger.info(f"Loading base model from: {self.base_model_path}")
            
            # Load model and tokenizer
            self.model = AutoModelForCausalLM.from_pretrained(
                str(self.base_model_path),
                torch_dtype=torch.float32,
                device_map="cpu"
            )
            
            self.tokenizer = AutoTokenizer.from_pretrained(str(self.base_model_path))
            
            logger.info("Base model loaded successfully")
            
        except Exception as e:
            logger.error(f"Failed to load base model: {e}")
            raise
    
    def generate_dummy_data(self, n_samples: Optional[int] = None) -> np.ndarray:
        """Generate synthetic time series data for training."""
        if n_samples is None:
            n_samples = self.config['training']['dummy_data']['n_samples']
        
        n_series = self.config['training']['dummy_data']['n_series']
        length = self.config['training']['dummy_data']['length']
        patterns = self.config['training']['dummy_data']['patterns']
        
        logger.info(f"Generating {n_samples} dummy time series samples...")
        
        all_series = []
        
        for i in range(n_samples):
            # Generate base time series
            t = np.arange(length)
            series = np.zeros(length)
            
            # Add trend pattern
            if "trend" in patterns:
                trend = np.linspace(0, 10, length) + np.random.normal(0, 0.5, length)
                series += trend
            
            # Add seasonal pattern
            if "seasonal" in patterns:
                seasonal = 5 * np.sin(2 * np.pi * t / 24) + 2 * np.sin(2 * np.pi * t / 7)
                series += seasonal
            
            # Add noise
            if "noise" in patterns:
                noise = np.random.normal(0, 1, length)
                series += noise
            
            # Add some random walk component
            random_walk = np.cumsum(np.random.normal(0, 0.1, length))
            series += random_walk
            
            all_series.append(series)
        
        # Convert to numpy array
        dummy_data = np.array(all_series)
        
        # Save dummy data
        dummy_data_path = self.dummy_data_dir / f"dummy_data_{n_samples}_samples.npy"
        np.save(dummy_data_path, dummy_data)
        
        logger.info(f"Dummy data generated and saved to: {dummy_data_path}")
        logger.info(f"Data shape: {dummy_data.shape}")
        
        return dummy_data
    
    def prepare_training_data(self, data: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Prepare data for Chronos training format."""
        logger.info("Preparing training data...")
        
        # For Chronos, we need to create context-target pairs
        context_length = self.config['inference']['context_length']
        prediction_length = self.config['inference']['prediction_length']
        
        contexts = []
        targets = []
        
        for series in data:
            # Create sliding windows
            for i in range(len(series) - context_length - prediction_length + 1):
                context = series[i:i + context_length]
                target = series[i + context_length:i + context_length + prediction_length]
                
                contexts.append(context)
                targets.append(target)
        
        contexts = np.array(contexts)
        targets = np.array(targets)
        
        # Split into train/validation
        val_split = self.config['training']['validation_split']
        n_val = int(len(contexts) * val_split)
        n_train = len(contexts) - n_val
        
        # Shuffle indices
        indices = np.random.permutation(len(contexts))
        
        train_indices = indices[:n_train]
        val_indices = indices[n_train:]
        
        train_contexts = contexts[train_indices]
        train_targets = targets[train_indices]
        val_contexts = contexts[val_indices]
        val_targets = targets[val_indices]
        
        logger.info(f"Training data prepared:")
        logger.info(f"  Train: {len(train_contexts)} samples")
        logger.info(f"  Validation: {len(val_contexts)} samples")
        
        return (train_contexts, train_targets), (val_contexts, val_targets)
    
    def train_model(self, train_data: Tuple[np.ndarray, np.ndarray], 
                   val_data: Tuple[np.ndarray, np.ndarray], 
                   epochs: Optional[int] = None) -> None:
        """Train the Chronos model on dummy data."""
        if self.model is None:
            raise ValueError("Base model must be loaded before training")
        
        if epochs is None:
            epochs = self.config['training']['epochs']
        
        logger.info(f"Starting training for {epochs} epochs...")
        
        # Set model to training mode
        self.model.train()
        
        # Training parameters
        learning_rate = self.config['training']['learning_rate']
        batch_size = self.config['training']['batch_size']
        
        # Setup optimizer
        optimizer = torch.optim.AdamW(self.model.parameters(), lr=learning_rate)
        
        train_contexts, train_targets = train_data
        val_contexts, val_targets = val_data
        
        # Convert to tensors
        train_contexts = torch.tensor(train_contexts, dtype=torch.float32)
        train_targets = torch.tensor(train_targets, dtype=torch.float32)
        val_contexts = torch.tensor(val_contexts, dtype=torch.float32)
        val_targets = torch.tensor(val_targets, dtype=torch.float32)
        
        # Training loop
        for epoch in range(epochs):
            # Training phase
            train_loss = 0.0
            n_batches = 0
            
            for i in range(0, len(train_contexts), batch_size):
                batch_contexts = train_contexts[i:i + batch_size]
                batch_targets = train_targets[i:i + batch_size]
                
                optimizer.zero_grad()
                
                # Forward pass (simplified - actual Chronos training would be more complex)
                outputs = self.model(batch_contexts.long())
                loss = torch.nn.functional.mse_loss(outputs.logits, batch_targets)
                
                loss.backward()
                optimizer.step()
                
                train_loss += loss.item()
                n_batches += 1
            
            avg_train_loss = train_loss / n_batches if n_batches > 0 else 0
            
            # Validation phase
            self.model.eval()
            val_loss = 0.0
            n_val_batches = 0
            
            with torch.no_grad():
                for i in range(0, len(val_contexts), batch_size):
                    batch_contexts = val_contexts[i:i + batch_size]
                    batch_targets = val_targets[i:i + batch_size]
                    
                    outputs = self.model(batch_contexts.long())
                    loss = torch.nn.functional.mse_loss(outputs.logits, batch_targets)
                    
                    val_loss += loss.item()
                    n_val_batches += 1
            
            avg_val_loss = val_loss / n_val_batches if n_val_batches > 0 else 0
            
            # Log training progress
            epoch_info = {
                'epoch': epoch + 1,
                'train_loss': avg_train_loss,
                'val_loss': avg_val_loss,
                'timestamp': datetime.now().isoformat()
            }
            self.training_history.append(epoch_info)
            
            logger.info(f"Epoch {epoch + 1}/{epochs}: "
                       f"Train Loss: {avg_train_loss:.4f}, "
                       f"Val Loss: {avg_val_loss:.4f}")
            
            self.model.train()
        
        logger.info("Training completed successfully")
    
    def save_trained_model(self, output_path: Optional[str] = None) -> None:
        """Save the trained model."""
        if self.model is None:
            raise ValueError("No model to save")
        
        if output_path is None:
            output_path = self.models_dir / "trained_model"
        
        output_path = Path(output_path)
        output_path.mkdir(parents=True, exist_ok=True)
        
        try:
            logger.info(f"Saving trained model to: {output_path}")
            
            # Save model and tokenizer
            self.model.save_pretrained(str(output_path))
            self.tokenizer.save_pretrained(str(output_path))
            
            # Save training history
            history_path = output_path / 'training_history.json'
            with open(history_path, 'w') as f:
                json.dump(self.training_history, f, indent=2)
            
            # Save model metadata
            metadata = {
                'model_type': 'chronos-trained',
                'base_model_path': str(self.base_model_path),
                'training_config': self.config['training'],
                'training_history': self.training_history,
                'saved_at': datetime.now().isoformat(),
                'version': '1.0'
            }
            
            metadata_path = output_path / 'model_metadata.json'
            with open(metadata_path, 'w') as f:
                json.dump(metadata, f, indent=2)
            
            logger.info(f"Trained model saved successfully to: {output_path}")
            
        except Exception as e:
            logger.error(f"Failed to save trained model: {e}")
            raise
    
    def evaluate_model(self, test_data: Tuple[np.ndarray, np.ndarray]) -> Dict[str, float]:
        """Evaluate the trained model on test data."""
        if self.model is None:
            raise ValueError("No model to evaluate")
        
        self.model.eval()
        
        test_contexts, test_targets = test_data
        test_contexts = torch.tensor(test_contexts, dtype=torch.float32)
        test_targets = torch.tensor(test_targets, dtype=torch.float32)
        
        predictions = []
        total_loss = 0.0
        n_batches = 0
        
        with torch.no_grad():
            for i in range(0, len(test_contexts), 32):  # Batch size of 32
                batch_contexts = test_contexts[i:i + 32]
                batch_targets = test_targets[i:i + 32]
                
                outputs = self.model(batch_contexts.long())
                loss = torch.nn.functional.mse_loss(outputs.logits, batch_targets)
                
                total_loss += loss.item()
                n_batches += 1
                
                predictions.append(outputs.logits.cpu().numpy())
        
        avg_loss = total_loss / n_batches if n_batches > 0 else 0
        
        # Calculate additional metrics
        predictions = np.concatenate(predictions, axis=0)
        mse = np.mean((predictions - test_targets.numpy()) ** 2)
        mae = np.mean(np.abs(predictions - test_targets.numpy()))
        
        metrics = {
            'mse': float(mse),
            'mae': float(mae),
            'avg_loss': float(avg_loss)
        }
        
        logger.info(f"Model evaluation completed:")
        logger.info(f"  MSE: {mse:.4f}")
        logger.info(f"  MAE: {mae:.4f}")
        logger.info(f"  Avg Loss: {avg_loss:.4f}")
        
        return metrics


def main():
    """Main function for independent execution."""
    print("Chronos Training Component")
    print("=" * 40)
    
    try:
        # Initialize trainer
        trainer = ChronosTrainer()
        
        # Load base model
        print("Loading base model...")
        trainer.load_base_model()
        
        # Generate dummy data
        print("Generating dummy data...")
        dummy_data = trainer.generate_dummy_data()
        
        # Prepare training data
        print("Preparing training data...")
        train_data, val_data = trainer.prepare_training_data(dummy_data)
        
        # Train model
        print("Training model...")
        trainer.train_model(train_data, val_data)
        
        # Evaluate model
        print("Evaluating model...")
        metrics = trainer.evaluate_model(val_data)
        
        # Save trained model
        print("Saving trained model...")
        trainer.save_trained_model()
        
        print(f"\nTraining completed successfully!")
        print(f"Final metrics: {metrics}")
        
    except Exception as e:
        print(f"✗ Error: {e}")
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main())
