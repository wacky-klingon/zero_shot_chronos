#!/usr/bin/env python3
"""
Script to download the Chronos-Bolt-Base model from Hugging Face.

This script downloads the amazon/chronos-bolt-base model and saves it
to the data/model/ directory for local use.
"""

import os
import sys
from pathlib import Path
from huggingface_hub import snapshot_download
import shutil


def download_chronos_model():
    """Download the Chronos-Bolt-Base model from Hugging Face."""
    
    # Set up paths
    model_name = "amazon/chronos-bolt-base"
    model_dir = Path("data/model")
    temp_dir = Path("data/model_temp")
    
    # Create model directory if it doesn't exist
    model_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"Downloading {model_name} model...")
    print(f"Target directory: {model_dir.absolute()}")
    
    try:
        # Download the entire model repository
        print("Downloading model repository...")
        downloaded_path = snapshot_download(
            repo_id=model_name,
            local_dir=str(temp_dir),
            local_dir_use_symlinks=False
        )
        
        print(f"Model downloaded to temporary directory: {downloaded_path}")
        
        # Move files from temp directory to final model directory
        print("Moving files to final location...")
        if temp_dir.exists():
            # Remove existing files in model_dir if any
            if model_dir.exists():
                shutil.rmtree(model_dir)
            
            # Move the downloaded content
            shutil.move(str(temp_dir), str(model_dir))
            print(f"Model successfully moved to: {model_dir.absolute()}")
        
        # List downloaded files
        print("\nDownloaded files:")
        for file_path in model_dir.rglob("*"):
            if file_path.is_file():
                print(f"  - {file_path.relative_to(model_dir)}")
        
        return True
        
    except Exception as e:
        print(f"Error downloading model: {e}")
        # Clean up temp directory if it exists
        if temp_dir.exists():
            shutil.rmtree(temp_dir)
        return False


def main():
    """Main execution function."""
    print("Chronos-Bolt-Base Model Downloader")
    print("=" * 40)
    
    # Check if model already exists
    model_dir = Path("data/model")
    if model_dir.exists() and any(model_dir.iterdir()):
        print(f"Model directory {model_dir} already contains files.")
        response = input("Do you want to re-download? (y/N): ").strip().lower()
        if response not in ['y', 'yes']:
            print("Download cancelled.")
            return
    
    # Clean up any existing temp directory
    temp_dir = Path("data/model_temp")
    if temp_dir.exists():
        shutil.rmtree(temp_dir)
    
    # Download the model
    success = download_chronos_model()
    
    if success:
        print("\n" + "=" * 40)
        print("Download completed successfully!")
        print("The model is now available in data/model/")
        print("\nNext steps:")
        print("1. Update config/settings.yaml to use the local model")
        print("2. Run your forecasting script")
    else:
        print("\n" + "=" * 40)
        print("Download failed. Please check the error messages above.")
        sys.exit(1)


if __name__ == "__main__":
    main()
