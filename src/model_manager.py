#!/usr/bin/env python3
"""
Model management utility for Chronos models.

This module provides utilities for managing different model versions,
switching between models, and listing available models.
"""

import os
import sys
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import yaml
import shutil


class ModelManager:
    """Manages Chronos model versions and configurations."""
    
    def __init__(self, config_path: str = "config/settings.yaml") -> None:
        """
        Initialize the model manager.
        
        Args:
            config_path: Path to the YAML configuration file
        """
        with open(config_path, 'r') as file:
            self.config = yaml.safe_load(file)
        
        self.model_dir = Path(self.config['data']['model_dir'])
        self.current_model_path = self.config['model']['model_path']
    
    def list_available_models(self) -> Dict[str, List[str]]:
        """
        List all available models and their versions.
        
        Returns:
            Dictionary mapping model types to their available versions
        """
        models = {}
        
        if not self.model_dir.exists():
            return models
        
        for model_type_dir in self.model_dir.iterdir():
            if model_type_dir.is_dir():
                model_type = model_type_dir.name
                versions = []
                
                for version_dir in model_type_dir.iterdir():
                    if version_dir.is_dir() and self._is_valid_model_dir(version_dir):
                        versions.append(version_dir.name)
                
                if versions:
                    models[model_type] = sorted(versions)
        
        return models
    
    def get_current_model_info(self) -> Tuple[str, str]:
        """
        Get current model type and version.
        
        Returns:
            Tuple of (model_type, version)
        """
        model_path = Path(self.current_model_path)
        if len(model_path.parts) >= 3:
            return model_path.parts[-2], model_path.parts[-1]
        return "unknown", "unknown"
    
    def switch_model(self, model_type: str, version: str) -> bool:
        """
        Switch to a different model version.
        
        Args:
            model_type: Type of model (e.g., 'chronos-bolt-base')
            version: Version of the model (e.g., 'v1.0')
            
        Returns:
            True if successful, False otherwise
        """
        new_model_path = self.model_dir / model_type / version
        
        if not new_model_path.exists():
            print(f"Model path does not exist: {new_model_path}")
            return False
        
        if not self._is_valid_model_dir(new_model_path):
            print(f"Invalid model directory: {new_model_path}")
            return False
        
        # Update configuration
        self.config['model']['model_path'] = str(new_model_path)
        self.config['model']['model_type'] = model_type
        self.config['model']['version'] = version
        
        # Save updated configuration
        with open('config/settings.yaml', 'w') as file:
            yaml.dump(self.config, file, default_flow_style=False)
        
        print(f"Switched to {model_type} version {version}")
        print(f"Model path: {new_model_path}")
        return True
    
    def create_model_backup(self, model_type: str, version: str, 
                           backup_name: str) -> bool:
        """
        Create a backup of a model version.
        
        Args:
            model_type: Type of model to backup
            version: Version to backup
            backup_name: Name for the backup
            
        Returns:
            True if successful, False otherwise
        """
        source_path = self.model_dir / model_type / version
        backup_path = self.model_dir / model_type / backup_name
        
        if not source_path.exists():
            print(f"Source model does not exist: {source_path}")
            return False
        
        if backup_path.exists():
            print(f"Backup already exists: {backup_path}")
            return False
        
        try:
            shutil.copytree(source_path, backup_path)
            print(f"Created backup: {backup_path}")
            return True
        except Exception as e:
            print(f"Error creating backup: {e}")
            return False
    
    def remove_model_version(self, model_type: str, version: str) -> bool:
        """
        Remove a model version.
        
        Args:
            model_type: Type of model to remove
            version: Version to remove
            
        Returns:
            True if successful, False otherwise
        """
        model_path = self.model_dir / model_type / version
        
        if not model_path.exists():
            print(f"Model does not exist: {model_path}")
            return False
        
        # Check if this is the current model
        current_type, current_version = self.get_current_model_info()
        if model_type == current_type and version == current_version:
            print("Cannot remove currently active model")
            return False
        
        try:
            shutil.rmtree(model_path)
            print(f"Removed model: {model_path}")
            return True
        except Exception as e:
            print(f"Error removing model: {e}")
            return False
    
    def _is_valid_model_dir(self, model_dir: Path) -> bool:
        """
        Check if a directory contains a valid model.
        
        Args:
            model_dir: Directory to check
            
        Returns:
            True if valid model directory, False otherwise
        """
        required_files = ['config.json', 'model.safetensors']
        return all((model_dir / file).exists() for file in required_files)
    
    def get_model_info(self, model_type: str, version: str) -> Optional[Dict]:
        """
        Get information about a specific model version.
        
        Args:
            model_type: Type of model
            version: Version of model
            
        Returns:
            Dictionary with model information or None if not found
        """
        model_path = self.model_dir / model_type / version
        
        if not model_path.exists():
            return None
        
        config_file = model_path / 'config.json'
        if not config_file.exists():
            return None
        
        try:
            import json
            with open(config_file, 'r') as f:
                config = json.load(f)
            
            return {
                'model_type': model_type,
                'version': version,
                'path': str(model_path),
                'config': config,
                'size_mb': self._get_directory_size(model_path)
            }
        except Exception as e:
            print(f"Error reading model info: {e}")
            return None
    
    def _get_directory_size(self, directory: Path) -> float:
        """Get directory size in MB."""
        total_size = 0
        for file_path in directory.rglob('*'):
            if file_path.is_file():
                total_size += file_path.stat().st_size
        return total_size / (1024 * 1024)  # Convert to MB


def main():
    """Main execution function for model management CLI."""
    manager = ModelManager()
    
    print("Chronos Model Manager")
    print("=" * 30)
    
    # List available models
    models = manager.list_available_models()
    
    if not models:
        print("No models found.")
        return
    
    print("\nAvailable models:")
    for model_type, versions in models.items():
        print(f"\n{model_type}:")
        for version in versions:
            info = manager.get_model_info(model_type, version)
            size_str = f" ({info['size_mb']:.1f} MB)" if info else ""
            current = " (CURRENT)" if (model_type, version) == manager.get_current_model_info() else ""
            print(f"  - {version}{size_str}{current}")
    
    # Show current model
    current_type, current_version = manager.get_current_model_info()
    print(f"\nCurrent model: {current_type} v{current_version}")


if __name__ == "__main__":
    main()
