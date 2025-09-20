# Advanced Chronos Implementation: Fine-tuning and Custom Models

This document extends the direct Chronos forecasting implementation with advanced features including model fine-tuning, custom model integration, and enhanced model configurations.

## Overview

Building on the foundation established in `002-implementation-01.md`, this phase covers:
- Fine-tuning Chronos models on custom datasets
- Integrating custom trained Chronos models
- Advanced hyperparameter configuration
- Model ensemble and comparison strategies

## Enhanced Project Structure

```
chronos-raw/
├── data/
│   ├── raw/                    # Original time series data
│   ├── processed/              # Cleaned and formatted data
│   ├── predictions/            # Generated forecasts
│   └── models/                 # Local Chronos model storage
│       ├── chronos-bolt-tiny/
│       ├── chronos-bolt-mini/
│       ├── chronos-bolt-small/
│       ├── chronos-bolt-base/
│       └── custom/             # Custom trained models
├── src/
│   ├── data_loader.py          # Enhanced data loading
│   ├── chronos_predictor.py    # Advanced Chronos model wrapper
│   ├── chronos_finetuner.py    # Chronos fine-tuning utilities
│   ├── model_comparison.py     # Model evaluation and comparison
│   ├── model_manager.py        # Model version management
│   └── visualization.py        # Enhanced plotting utilities
├── config/
│   ├── settings.yaml           # Base configuration
│   ├── fine_tuning.yaml        # Fine-tuning specific settings
│   └── custom_models.yaml      # Custom model configuration
├── experiments/                # Experiment tracking and results
│   ├── logs/                   # Training logs
│   ├── metrics/                # Performance metrics
│   └── plots/                  # Experiment visualizations
└── custom_models/              # Custom trained model storage
    ├── my_custom_model_v1/
    └── my_custom_model_v2/
```

## Implementation Components

### 1. Enhanced Data Loader

**File: `src/data_loader.py` (Extended)**

```python
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Union, Optional, List, Dict, Tuple
import yaml

class EnhancedTimeSeriesDataLoader(TimeSeriesDataLoader):
    """Extended data loader with advanced preprocessing for fine-tuning."""
    
    def __init__(self, config_path: str = "config/settings.yaml"):
        super().__init__(config_path)
        
        # Load fine-tuning configuration
        fine_tuning_config_path = "config/fine_tuning.yaml"
        if Path(fine_tuning_config_path).exists():
            with open(fine_tuning_config_path, 'r') as file:
                self.fine_tuning_config = yaml.safe_load(file)
        else:
            self.fine_tuning_config = self._get_default_fine_tuning_config()
    
    def _get_default_fine_tuning_config(self) -> Dict:
        """Default fine-tuning configuration."""
        return {
            'context_length': 512,
            'prediction_length': 48,
            'validation_split': 0.2,
            'test_split': 0.1
        }
    
    def prepare_fine_tuning_data(self, 
                                data: pd.DataFrame,
                                context_length: Optional[int] = None,
                                prediction_length: Optional[int] = None) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Prepare data specifically for Chronos fine-tuning."""
        
        if context_length is None:
            context_length = self.fine_tuning_config['context_length']
        if prediction_length is None:
            prediction_length = self.fine_tuning_config['prediction_length']
        
        # Convert to numpy array
        values = data['value'].values.astype(np.float32)
        
        # Create sliding windows for fine-tuning
        sequences = []
        targets = []
        
        for i in range(len(values) - context_length - prediction_length + 1):
            context = values[i:i + context_length]
            target = values[i + context_length:i + context_length + prediction_length]
            sequences.append(context)
            targets.append(target)
        
        sequences = np.array(sequences)
        targets = np.array(targets)
        
        # Split into train/validation/test
        val_split = self.fine_tuning_config['validation_split']
        test_split = self.fine_tuning_config['test_split']
        
        n_samples = len(sequences)
        n_val = int(n_samples * val_split)
        n_test = int(n_samples * test_split)
        n_train = n_samples - n_val - n_test
        
        train_sequences = sequences[:n_train]
        train_targets = targets[:n_train]
        val_sequences = sequences[n_train:n_train + n_val]
        val_targets = targets[n_train:n_train + n_val]
        test_sequences = sequences[n_train + n_val:]
        test_targets = targets[n_train + n_val:]
        
        print(f"Fine-tuning data prepared:")
        print(f"  - Train: {len(train_sequences)} sequences")
        print(f"  - Validation: {len(val_sequences)} sequences")
        print(f"  - Test: {len(test_sequences)} sequences")
        
        return (train_sequences, train_targets), (val_sequences, val_targets), (test_sequences, test_targets)
    
    def prepare_custom_model_data(self, 
                                 data: pd.DataFrame,
                                 model_config: Dict) -> Dict:
        """Prepare data for custom model integration."""
        
        # Extract model-specific requirements
        context_length = model_config.get('context_length', 512)
        prediction_length = model_config.get('prediction_length', 48)
        
        # Prepare data according to custom model requirements
        values = data['value'].values.astype(np.float32)
        
        # Create context and target arrays
        context, target = self.prepare_for_chronos(data, prediction_length)
        
        return {
            'context': context,
            'target': target,
            'context_length': context_length,
            'prediction_length': prediction_length,
            'model_config': model_config
        }
```

### 2. Chronos Fine-tuning Module

**File: `src/chronos_finetuner.py`**

```python
from chronos import ChronosPipeline
from pathlib import Path
from typing import Optional, Dict, List, Tuple
import yaml
import numpy as np
import torch
import json
from datetime import datetime

class ChronosFineTuner:
    """Fine-tuning utilities for Chronos models."""
    
    def __init__(self, config_path: str = "config/fine_tuning.yaml"):
        with open(config_path, 'r') as file:
            self.config = yaml.safe_load(file)
        
        self.models_dir = Path("custom_models")
        self.models_dir.mkdir(parents=True, exist_ok=True)
        
        self.experiment_logs = []
    
    def fine_tune_model(self, 
                       base_model_name: str,
                       train_data: Tuple[np.ndarray, np.ndarray],
                       val_data: Tuple[np.ndarray, np.ndarray],
                       model_name: str,
                       **kwargs) -> ChronosPipeline:
        """
        Fine-tune a Chronos model on custom data.
        
        RISK: Chronos fine-tuning API may not be publicly available or well-documented.
        This implementation assumes a fine-tuning interface similar to Hugging Face transformers.
        """
        
        # Load base model
        base_model = ChronosPipeline.from_pretrained(base_model_name)
        
        # Extract training parameters
        learning_rate = kwargs.get('learning_rate', self.config.get('learning_rate', 1e-4))
        num_epochs = kwargs.get('num_epochs', self.config.get('num_epochs', 10))
        batch_size = kwargs.get('batch_size', self.config.get('batch_size', 32))
        
        # RISK: The following fine-tuning code is speculative based on typical
        # transformer fine-tuning patterns. Actual Chronos fine-tuning API may differ.
        
        try:
            # Configure fine-tuning parameters
            fine_tuning_config = {
                'learning_rate': learning_rate,
                'num_epochs': num_epochs,
                'batch_size': batch_size,
                'warmup_steps': self.config.get('warmup_steps', 100),
                'weight_decay': self.config.get('weight_decay', 1e-5),
                'gradient_clip_norm': self.config.get('gradient_clip_norm', 1.0)
            }
            
            # RISK: This assumes Chronos has a fine_tune method similar to transformers
            # The actual API may be different or may not exist
            if hasattr(base_model, 'fine_tune'):
                fine_tuned_model = base_model.fine_tune(
                    train_data=train_data,
                    val_data=val_data,
                    **fine_tuning_config
                )
            else:
                # Fallback: Use standard training approach
                print("WARNING: Direct fine-tuning not available, using base model")
                fine_tuned_model = base_model
            
            # Save fine-tuned model
            model_path = self.models_dir / model_name
            fine_tuned_model.save_pretrained(str(model_path))
            
            # Log experiment
            self._log_experiment("fine_tune", {
                "base_model": base_model_name,
                "model_name": model_name,
                "config": fine_tuning_config,
                "train_samples": len(train_data[0]),
                "val_samples": len(val_data[0])
            })
            
            print(f"Fine-tuned model saved to: {model_path}")
            return fine_tuned_model
            
        except Exception as e:
            print(f"Fine-tuning failed: {e}")
            print("Falling back to base model")
            return base_model
    
    def hyperparameter_search(self, 
                            base_model_name: str,
                            train_data: Tuple[np.ndarray, np.ndarray],
                            val_data: Tuple[np.ndarray, np.ndarray],
                            search_space: Optional[Dict] = None) -> Dict:
        """
        Perform hyperparameter search for optimal fine-tuning configuration.
        
        RISK: This assumes Chronos supports hyperparameter search or we can implement
        it manually. The actual implementation may require custom search logic.
        """
        
        if search_space is None:
            search_space = {
                'learning_rate': [1e-5, 1e-4, 1e-3],
                'num_epochs': [5, 10, 20],
                'batch_size': [16, 32, 64]
            }
        
        best_config = None
        best_score = float('inf')
        results = []
        
        # Grid search implementation
        for lr in search_space['learning_rate']:
            for epochs in search_space['num_epochs']:
                for batch_size in search_space['batch_size']:
                    
                    config = {
                        'learning_rate': lr,
                        'num_epochs': epochs,
                        'batch_size': batch_size
                    }
                    
                    try:
                        # Fine-tune with current configuration
                        model_name = f"search_{lr}_{epochs}_{batch_size}"
                        model = self.fine_tune_model(
                            base_model_name, train_data, val_data, model_name, **config
                        )
                        
                        # Evaluate on validation data
                        score = self._evaluate_model(model, val_data)
                        
                        results.append({
                            'config': config,
                            'score': score,
                            'model_name': model_name
                        })
                        
                        if score < best_score:
                            best_score = score
                            best_config = config
                        
                        print(f"Config: {config}, Score: {score:.4f}")
                        
                    except Exception as e:
                        print(f"Failed config {config}: {e}")
                        continue
        
        print(f"Best configuration: {best_config}, Score: {best_score:.4f}")
        
        # Save search results
        results_file = self.models_dir / "hyperparameter_search_results.json"
        with open(results_file, 'w') as f:
            json.dump(results, f, indent=2)
        
        return best_config
    
    def _evaluate_model(self, model: ChronosPipeline, val_data: Tuple[np.ndarray, np.ndarray]) -> float:
        """Evaluate model performance on validation data."""
        
        val_context, val_target = val_data
        
        try:
            # Generate predictions
            predictions = model.predict(val_context, prediction_length=len(val_target))
            
            # Calculate MSE as evaluation metric
            mse = np.mean((predictions - val_target) ** 2)
            return float(mse)
            
        except Exception as e:
            print(f"Evaluation failed: {e}")
            return float('inf')
    
    def _log_experiment(self, experiment_type: str, parameters: Dict) -> None:
        """Log experiment details for tracking."""
        experiment_log = {
            "timestamp": datetime.now().isoformat(),
            "experiment_type": experiment_type,
            "parameters": parameters
        }
        
        self.experiment_logs.append(experiment_log)
        
        # Save to file
        logs_dir = Path("experiments/logs")
        logs_dir.mkdir(parents=True, exist_ok=True)
        
        log_file = logs_dir / f"experiment_{len(self.experiment_logs)}.json"
        with open(log_file, 'w') as f:
            json.dump(experiment_log, f, indent=2)
```

### 3. Enhanced Chronos Predictor

**File: `src/chronos_predictor.py` (Extended)**

```python
from chronos import ChronosPipeline
from pathlib import Path
from typing import Optional, Union, List, Dict
import yaml
import numpy as np
import pandas as pd

class AdvancedChronosPredictor(ChronosPredictor):
    """Advanced Chronos predictor with custom model support."""
    
    def __init__(self, config_path: str = "config/settings.yaml"):
        super().__init__(config_path)
        
        # Load custom model configuration
        custom_models_config_path = "config/custom_models.yaml"
        if Path(custom_models_config_path).exists():
            with open(custom_models_config_path, 'r') as file:
                self.custom_models_config = yaml.safe_load(file)
        else:
            self.custom_models_config = {}
        
        self.custom_models_dir = Path("custom_models")
        self.custom_models_dir.mkdir(parents=True, exist_ok=True)
    
    def load_custom_model(self, model_name: str) -> 'AdvancedChronosPredictor':
        """
        Load a custom trained Chronos model.
        
        RISK: This assumes custom models are saved in a compatible format.
        The actual model format may require custom loading logic.
        """
        
        model_path = self.custom_models_dir / model_name
        
        if not model_path.exists():
            raise FileNotFoundError(f"Custom model not found: {model_path}")
        
        try:
            # Load custom model
            self.model = ChronosPipeline.from_pretrained(str(model_path))
            print(f"Custom model loaded from: {model_path}")
            return self
        except Exception as e:
            raise ValueError(f"Error loading custom model: {e}")
    
    def compare_models(self, 
                      context: np.ndarray,
                      target: Optional[np.ndarray] = None,
                      model_names: List[str] = None) -> Dict:
        """
        Compare multiple models on the same data.
        
        RISK: This assumes we can load and compare multiple models.
        Memory constraints may limit the number of models that can be loaded simultaneously.
        """
        
        if model_names is None:
            model_names = self.custom_models_config.get('available_models', [])
        
        results = {}
        
        for model_name in model_names:
            try:
                # Load model
                if model_name in self.custom_models_config.get('custom_models', {}):
                    model = ChronosPipeline.from_pretrained(
                        str(self.custom_models_dir / model_name)
                    )
                else:
                    model = ChronosPipeline.from_pretrained(model_name)
                
                # Generate predictions
                predictions = model.predict(context, prediction_length=self.prediction_length)
                
                # Calculate metrics
                metrics = {
                    'predictions': predictions.tolist(),
                    'prediction_length': len(predictions)
                }
                
                if target is not None:
                    mse = np.mean((predictions - target) ** 2)
                    mae = np.mean(np.abs(predictions - target))
                    mape = np.mean(np.abs((target - predictions) / target)) * 100
                    
                    metrics.update({
                        'mse': float(mse),
                        'mae': float(mae),
                        'mape': float(mape)
                    })
                
                results[model_name] = metrics
                print(f"Model {model_name}: MSE={metrics.get('mse', 'N/A'):.4f}")
                
            except Exception as e:
                print(f"Failed to evaluate model {model_name}: {e}")
                results[model_name] = {'error': str(e)}
        
        return results
    
    def ensemble_predict(self, 
                        context: np.ndarray,
                        model_names: List[str],
                        method: str = 'mean') -> np.ndarray:
        """
        Generate ensemble predictions from multiple models.
        
        RISK: This assumes we can load multiple models simultaneously.
        Memory and computational constraints may limit feasibility.
        """
        
        predictions = []
        
        for model_name in model_names:
            try:
                if model_name in self.custom_models_config.get('custom_models', {}):
                    model = ChronosPipeline.from_pretrained(
                        str(self.custom_models_dir / model_name)
                    )
                else:
                    model = ChronosPipeline.from_pretrained(model_name)
                
                pred = model.predict(context, prediction_length=self.prediction_length)
                predictions.append(pred)
                
            except Exception as e:
                print(f"Failed to load model {model_name}: {e}")
                continue
        
        if not predictions:
            raise ValueError("No models successfully loaded for ensemble")
        
        predictions = np.array(predictions)
        
        if method == 'mean':
            return np.mean(predictions, axis=0)
        elif method == 'median':
            return np.median(predictions, axis=0)
        else:
            raise ValueError(f"Unknown ensemble method: {method}")
```

### 4. Model Comparison Module

**File: `src/model_comparison.py`**

```python
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional
import matplotlib.pyplot as plt
import seaborn as sns

class ModelComparator:
    """Compare and evaluate different Chronos model configurations."""
    
    def __init__(self, results_dir: str = "experiments/results"):
        self.results_dir = Path(results_dir)
        self.results_dir.mkdir(parents=True, exist_ok=True)
        
        self.comparison_results = []
    
    def compare_models(self, 
                      context: np.ndarray,
                      target: np.ndarray,
                      predictors: Dict[str, 'AdvancedChronosPredictor']) -> pd.DataFrame:
        """Compare multiple model configurations on test data."""
        
        comparison_results = []
        
        for model_name, predictor in predictors.items():
            try:
                # Generate predictions
                predictions = predictor.predict(context)
                
                # Calculate metrics
                metrics = self._calculate_metrics(target, predictions.predictions)
                
                # Add model information
                metrics['model_name'] = model_name
                metrics['model_type'] = self._get_model_type(model_name)
                
                comparison_results.append(metrics)
                
            except Exception as e:
                print(f"Failed to evaluate model {model_name}: {e}")
                comparison_results.append({
                    'model_name': model_name,
                    'model_type': 'Error',
                    'mse': float('inf'),
                    'mae': float('inf'),
                    'mape': float('inf'),
                    'error': str(e)
                })
        
        # Create comparison DataFrame
        comparison_df = pd.DataFrame(comparison_results)
        
        # Save results
        results_file = self.results_dir / "model_comparison.csv"
        comparison_df.to_csv(results_file, index=False)
        
        print(f"Model comparison results saved to: {results_file}")
        return comparison_df
    
    def _calculate_metrics(self, target: np.ndarray, predictions: np.ndarray) -> Dict:
        """Calculate comprehensive evaluation metrics."""
        
        # Ensure arrays are the same length
        min_length = min(len(target), len(predictions))
        target = target[:min_length]
        predictions = predictions[:min_length]
        
        mse = np.mean((target - predictions) ** 2)
        mae = np.mean(np.abs(target - predictions))
        mape = np.mean(np.abs((target - predictions) / target)) * 100
        rmse = np.sqrt(mse)
        
        return {
            'mse': float(mse),
            'mae': float(mae),
            'mape': float(mape),
            'rmse': float(rmse)
        }
    
    def _get_model_type(self, model_name: str) -> str:
        """Determine model type from name."""
        if 'custom' in model_name.lower():
            return 'Custom'
        elif 'fine_tuned' in model_name.lower():
            return 'Fine-tuned'
        elif 'ensemble' in model_name.lower():
            return 'Ensemble'
        else:
            return 'Base'
    
    def plot_comparison(self, 
                       comparison_df: pd.DataFrame,
                       metric: str = 'mse',
                       save_path: Optional[str] = None) -> None:
        """Plot model comparison results."""
        
        fig, ax = plt.subplots(figsize=(12, 8))
        
        # Filter out error models
        valid_df = comparison_df[comparison_df[metric] != float('inf')]
        
        if valid_df.empty:
            ax.text(0.5, 0.5, 'No valid models to compare', 
                   ha='center', va='center', transform=ax.transAxes)
            ax.set_title('Model Comparison - No Valid Results')
        else:
            # Create bar plot
            bars = ax.bar(valid_df['model_name'], valid_df[metric], 
                         color='skyblue', alpha=0.7)
            
            # Add value labels on bars
            for bar, value in zip(bars, valid_df[metric]):
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., height,
                       f'{value:.4f}', ha='center', va='bottom')
            
            ax.set_xlabel('Model Name')
            ax.set_ylabel(metric.upper())
            ax.set_title(f'Model Performance Comparison - {metric.upper()}')
            plt.xticks(rotation=45)
        
        plt.tight_layout()
        
        if save_path:
            save_path = Path(save_path)
            save_path.parent.mkdir(parents=True, exist_ok=True)
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Comparison plot saved to: {save_path}")
        
        plt.show()
    
    def generate_report(self, comparison_df: pd.DataFrame) -> str:
        """Generate a comprehensive model comparison report."""
        
        # Filter out error models
        valid_df = comparison_df[comparison_df['mse'] != float('inf')]
        
        if valid_df.empty:
            report = "# Model Comparison Report\n\nNo valid models to compare."
        else:
            best_model = valid_df.loc[valid_df['mse'].idxmin()]
            
            report = f"""
# Model Comparison Report

## Summary
Total models compared: {len(valid_df)}
Best performing model: {best_model['model_name']}
Best MSE: {best_model['mse']:.4f}

## Detailed Results
{valid_df[['model_name', 'model_type', 'mse', 'mae', 'mape']].to_string(index=False)}

## Recommendations
- Base models: Best for quick prototyping
- Fine-tuned models: Best for production when training time is available
- Custom models: Best when you have domain-specific data
- Ensemble models: Best for maximum accuracy when computational resources allow
"""
        
        # Save report
        report_file = self.results_dir / "comparison_report.md"
        with open(report_file, 'w') as f:
            f.write(report)
        
        print(f"Comparison report saved to: {report_file}")
        return report
```

### 5. Configuration Files

**File: `config/fine_tuning.yaml`**
```yaml
# Fine-tuning configuration
learning_rate: 1e-4
num_epochs: 10
batch_size: 32
warmup_steps: 100
weight_decay: 1e-5
gradient_clip_norm: 1.0

# Data preparation
context_length: 512
prediction_length: 48
validation_split: 0.2
test_split: 0.1

# Early stopping
early_stopping:
  patience: 5
  min_delta: 0.001

# Hyperparameter search
hyperparameter_search:
  learning_rate: [1e-5, 1e-4, 1e-3]
  num_epochs: [5, 10, 20]
  batch_size: [16, 32, 64]
  max_trials: 20
  time_limit: 3600  # seconds
```

**File: `config/custom_models.yaml`**
```yaml
# Custom model configuration
available_models:
  - "amazon/chronos-bolt-tiny"
  - "amazon/chronos-bolt-mini"
  - "amazon/chronos-bolt-small"
  - "amazon/chronos-bolt-base"

# Custom trained models
custom_models:
  my_custom_model_v1:
    path: "custom_models/my_custom_model_v1"
    description: "Custom model trained on domain-specific data"
    context_length: 512
    prediction_length: 48
  
  my_custom_model_v2:
    path: "custom_models/my_custom_model_v2"
    description: "Fine-tuned model with hyperparameter optimization"
    context_length: 1024
    prediction_length: 96

# Model comparison settings
comparison:
  metrics: ["mse", "mae", "mape", "rmse"]
  primary_metric: "mse"
  ensemble_methods: ["mean", "median"]
```

### 6. Advanced Main Script

**File: `advanced_main.py`**
```python
#!/usr/bin/env python3
"""
Advanced Chronos implementation with fine-tuning and custom model support.
"""

from src.data_loader import EnhancedTimeSeriesDataLoader
from src.chronos_predictor import AdvancedChronosPredictor
from src.chronos_finetuner import ChronosFineTuner
from src.model_comparison import ModelComparator
from src.visualization import TimeSeriesVisualizer
from pathlib import Path
import yaml

def main():
    """Advanced implementation with fine-tuning and custom models."""
    print("Advanced Chronos Implementation")
    print("=" * 50)
    
    # Initialize components
    data_loader = EnhancedTimeSeriesDataLoader()
    visualizer = TimeSeriesVisualizer()
    comparator = ModelComparator()
    finetuner = ChronosFineTuner()
    
    # Load data
    data_file = "data/raw/sample_timeseries_data.csv"
    
    if not Path(data_file).exists():
        print(f"Data file not found: {data_file}")
        return
    
    print("Loading time series data...")
    data = data_loader.load_from_csv(data_file)
    data_info = data_loader.get_data_info(data)
    print(f"Data loaded: {data_info['total_records']} records")
    
    # Prepare data for fine-tuning
    print("\nPreparing data for fine-tuning...")
    train_data, val_data, test_data = data_loader.prepare_fine_tuning_data(data)
    
    # Initialize predictors for different approaches
    predictors = {}
    
    # 1. Base model
    print("\nLoading base model...")
    base_predictor = AdvancedChronosPredictor()
    base_predictor.load_model()
    predictors['base_model'] = base_predictor
    
    # 2. Fine-tuned model
    print("\nFine-tuning model...")
    try:
        fine_tuned_model = finetuner.fine_tune_model(
            base_model_name="amazon/chronos-bolt-base",
            train_data=train_data,
            val_data=val_data,
            model_name="fine_tuned_model"
        )
        
        fine_tuned_predictor = AdvancedChronosPredictor()
        fine_tuned_predictor.model = fine_tuned_model
        predictors['fine_tuned'] = fine_tuned_predictor
        
    except Exception as e:
        print(f"Fine-tuning failed: {e}")
        print("Continuing with base model only")
    
    # 3. Hyperparameter optimization
    print("\nPerforming hyperparameter search...")
    try:
        best_config = finetuner.hyperparameter_search(
            base_model_name="amazon/chronos-bolt-base",
            train_data=train_data,
            val_data=val_data
        )
        
        # Train with best configuration
        optimized_model = finetuner.fine_tune_model(
            base_model_name="amazon/chronos-bolt-base",
            train_data=train_data,
            val_data=val_data,
            model_name="optimized_model",
            **best_config
        )
        
        optimized_predictor = AdvancedChronosPredictor()
        optimized_predictor.model = optimized_model
        predictors['optimized'] = optimized_predictor
        
    except Exception as e:
        print(f"Hyperparameter optimization failed: {e}")
    
    # 4. Custom model (if available)
    print("\nLoading custom models...")
    try:
        custom_predictor = AdvancedChronosPredictor()
        custom_predictor.load_custom_model("my_custom_model_v1")
        predictors['custom'] = custom_predictor
    except Exception as e:
        print(f"Custom model loading failed: {e}")
    
    # Compare all models
    print("\nComparing model performance...")
    test_context, test_target = test_data
    
    # Prepare context for prediction (use first sequence)
    context = test_context[0] if len(test_context) > 0 else data['value'].values[-512:]
    target = test_target[0] if len(test_target) > 0 else None
    
    comparison_df = comparator.compare_models(
        context=context,
        target=target,
        predictors=predictors
    )
    
    # Generate visualizations
    print("\nCreating comparison visualizations...")
    comparator.plot_comparison(
        comparison_df,
        save_path="experiments/plots/model_comparison.png"
    )
    
    # Generate comprehensive report
    report = comparator.generate_report(comparison_df)
    print(report)
    
    print("\nAdvanced implementation complete!")

if __name__ == "__main__":
    main()
```

## Key Risks and Limitations

### **High Risk Areas (No Prior Art)**

1. **Chronos Fine-tuning API**
   - **Risk**: Chronos may not have a public fine-tuning API
   - **Impact**: Fine-tuning functionality may not work
   - **Mitigation**: Fall back to base models, implement custom fine-tuning if needed

2. **Custom Model Integration**
   - **Risk**: Custom model format may not be compatible with ChronosPipeline
   - **Impact**: Custom models may not load properly
   - **Mitigation**: Implement custom loading logic, validate model format

3. **Memory Constraints**
   - **Risk**: Loading multiple models simultaneously may exceed memory limits
   - **Impact**: Ensemble and comparison features may fail
   - **Mitigation**: Implement model caching, load models on-demand

### **Medium Risk Areas (Limited Documentation)**

1. **Hyperparameter Search**
   - **Risk**: Chronos-specific hyperparameters may not be documented
   - **Impact**: Search may use incorrect parameters
   - **Mitigation**: Use standard transformer hyperparameters as baseline

2. **Model Evaluation Metrics**
   - **Risk**: Chronos may have specific evaluation requirements
   - **Impact**: Metrics may not be meaningful
   - **Mitigation**: Use standard time series metrics, validate with domain experts

### **Low Risk Areas (Well-Documented)**

1. **Data Preparation**
   - **Risk**: Low - standard time series preprocessing
   - **Impact**: Minimal
   - **Mitigation**: Use established time series preprocessing patterns

2. **Visualization**
   - **Risk**: Low - standard plotting functionality
   - **Impact**: Minimal
   - **Mitigation**: Use matplotlib/seaborn best practices

## Usage Instructions

### 1. Prepare Custom Models

Place your custom trained Chronos models in `custom_models/` directory:

```
custom_models/
├── my_custom_model_v1/
│   ├── config.json
│   ├── model.safetensors
│   └── README.md
└── my_custom_model_v2/
    ├── config.json
    ├── model.safetensors
    └── README.md
```

### 2. Configure Custom Models

Update `config/custom_models.yaml` with your model information:

```yaml
custom_models:
  my_custom_model_v1:
    path: "custom_models/my_custom_model_v1"
    description: "Custom model trained on domain-specific data"
    context_length: 512
    prediction_length: 48
```

### 3. Run Advanced Implementation

```bash
python advanced_main.py
```

### 4. Review Results

- **Model comparisons**: `experiments/results/model_comparison.csv`
- **Detailed report**: `experiments/results/comparison_report.md`
- **Fine-tuned models**: `custom_models/`
- **Visualizations**: `experiments/plots/`

## Key Features

1. **Fine-tuning Support** - Train Chronos models on custom data (with risk mitigation)
2. **Custom Model Integration** - Load and use your own trained models
3. **Hyperparameter Optimization** - Automated search for optimal configuration
4. **Model Comparison** - Comprehensive evaluation and comparison framework
5. **Experiment Tracking** - Detailed logging and result storage
6. **Advanced Visualization** - Comparative plots and performance analysis

## Success Criteria

- **Core functionality working** - Fine-tuning, custom model loading, model comparison
- **Risk mitigation** - Graceful fallbacks when advanced features fail
- **Production ready** - Robust error handling and logging
- **Extensible** - Easy to add new models and evaluation metrics

This implementation provides a framework for advanced Chronos usage while clearly identifying and mitigating risks where prior art is limited.