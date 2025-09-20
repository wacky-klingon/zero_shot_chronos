#!/usr/bin/env python3
"""
Base Model Component for Native Chronos Implementation

This module handles loading base Chronos models from Hugging Face and converting
them to our native format for local use.
"""

import yaml
import torch
from pathlib import Path
from typing import Dict, Optional, Any
import logging
import sys
import json

from chronos import ChronosBoltPipeline

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ChronosBaseModel:
    """Base model loader for Chronos models from Hugging Face."""
    
    def __init__(self, config_path: str = "config/chronos_config.yaml"):
        """Initialize the base model loader with configuration."""
        self.config_path = config_path
        self.config = self._load_config()
        self.model = None
        self.tokenizer = None
        self.model_info = {}
        
        # Create output directory
        self.output_path = Path(self.config['base_model']['output_path'])
        self.output_path.mkdir(parents=True, exist_ok=True)
        
        logger.info(f"ChronosBaseModel initialized with config: {config_path}")
    
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
    
    def load_from_huggingface(self) -> None:
        """Load Chronos model from Hugging Face using ChronosBoltPipeline."""
        model_name = self.config['base_model']['model_name']
        
        try:
            logger.info(f"Loading Chronos model from Hugging Face: {model_name}")
            
            # Load ChronosBoltPipeline directly
            self.model = ChronosBoltPipeline.from_pretrained(
                model_name,
                torch_dtype=torch.float32,  # Use float32 for CPU compatibility
                device_map="cpu"
            )
            
            # Store model information
            self.model_info = {
                'model_name': model_name,
                'model_type': 'chronos-bolt-pipeline',
                'device': 'cpu',
                'dtype': 'float32',
                'loaded_from': 'huggingface',
                'quantiles': self.model.quantiles,
                'context_length': self.model.default_context_length
            }
            
            logger.info("Chronos model loaded successfully from Hugging Face")
            
        except Exception as e:
            logger.error(f"Failed to load Chronos model from Hugging Face: {e}")
            raise
    
    def convert_to_native_format(self) -> None:
        """Convert model to our native format if needed."""
        if self.model is None:
            raise ValueError("Model must be loaded before conversion")
        
        try:
            logger.info("Converting Chronos model to native format...")
            
            # ChronosBoltPipeline is already in the correct format for time series forecasting
            # No conversion needed - it's designed for this purpose
            logger.info("Chronos model conversion completed (already in native format)")
            
        except Exception as e:
            logger.error(f"Failed to convert Chronos model: {e}")
            raise
    
    def save_base_model(self, output_path: Optional[str] = None) -> None:
        """Save the base Chronos model to local storage."""
        if self.model is None:
            raise ValueError("Model must be loaded before saving")
        
        if output_path is None:
            output_path = self.output_path
        
        output_path = Path(output_path)
        output_path.mkdir(parents=True, exist_ok=True)
        
        try:
            logger.info(f"Saving Chronos base model to: {output_path}")
            
            # Save the ChronosBoltPipeline model
            self.model.inner_model.save_pretrained(str(output_path))
            
            # Save model metadata
            metadata = {
                'model_info': self.model_info,
                'config': self.config['base_model'],
                'saved_at': str(Path.cwd()),
                'version': '1.0',
                'pipeline_type': 'ChronosBoltPipeline'
            }
            
            metadata_path = output_path / 'model_metadata.json'
            with open(metadata_path, 'w') as f:
                json.dump(metadata, f, indent=2)
            
            logger.info(f"Chronos base model saved successfully to: {output_path}")
            
        except Exception as e:
            logger.error(f"Failed to save Chronos base model: {e}")
            raise
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get information about the loaded model."""
        if self.model is None:
            return {'status': 'no_model_loaded'}
        
        return {
            'status': 'loaded',
            'model_info': self.model_info,
            'config': self.config['base_model'],
            'output_path': str(self.output_path)
        }
    
    def verify_model(self) -> bool:
        """Verify that the Chronos model can be loaded and used."""
        try:
            if self.model is None:
                logger.error("No Chronos model loaded")
                return False
            
            # Test Chronos model with a simple time series input
            test_context = torch.tensor([[1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0]], dtype=torch.float32)
            
            with torch.no_grad():
                # Test prediction with ChronosBoltPipeline
                predictions = self.model.predict(test_context, prediction_length=5)
                logger.info(f"Test prediction shape: {predictions.shape}")
            
            logger.info("Chronos model verification successful")
            return True
            
        except Exception as e:
            logger.error(f"Chronos model verification failed: {e}")
            return False


def main():
    """Main function for independent execution."""
    print("Chronos Base Model Component")
    print("=" * 40)
    
    try:
        # Initialize base model loader
        base_model = ChronosBaseModel()
        
        # Load model from Hugging Face
        print("Loading model from Hugging Face...")
        base_model.load_from_huggingface()
        
        # Convert to native format
        print("Converting to native format...")
        base_model.convert_to_native_format()
        
        # Save base model
        print("Saving base model...")
        base_model.save_base_model()
        
        # Verify model
        print("Verifying model...")
        if base_model.verify_model():
            print("✓ Model verification successful")
        else:
            print("✗ Model verification failed")
        
        # Display model info
        model_info = base_model.get_model_info()
        print(f"\nModel Information:")
        print(f"  Status: {model_info['status']}")
        print(f"  Model: {model_info['model_info']['model_name']}")
        print(f"  Output Path: {model_info['output_path']}")
        
        print("\n✓ Base model component completed successfully!")
        
    except Exception as e:
        print(f"✗ Error: {e}")
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main())
