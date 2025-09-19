# Advanced Chronos Implementation: Fine-tuning and Covariates

This document extends the zero-shot forecasting implementation with advanced features including model fine-tuning, covariate integration, and enhanced model configurations.

## Overview

Building on the foundation established in `002-implementation-01.md`, this phase covers:
- Fine-tuning Chronos models on custom datasets
- Incorporating covariates and static features
- Advanced hyperparameter configuration
- Model ensemble and comparison strategies

## Enhanced Project Structure

```
chronos-raw/
├── data/
│   ├── raw/                    # Original time series data
│   ├── processed/              # Cleaned and formatted data
│   ├── predictions/            # Generated forecasts
│   └── covariates/             # External covariate data
├── src/
│   ├── data_loader.py          # Enhanced data loading with covariates
│   ├── chronos_predictor.py    # Advanced Chronos model wrapper
│   ├── visualization.py        # Enhanced plotting utilities
│   ├── model_comparison.py     # Model evaluation and comparison
│   └── hyperparameter_tuning.py # Automated hyperparameter optimization
├── config/
│   ├── settings.yaml           # Base configuration
│   ├── fine_tuning.yaml        # Fine-tuning specific settings
│   └── covariates.yaml         # Covariate configuration
├── models/                     # Saved model artifacts
│   ├── zero_shot/              # Zero-shot model checkpoints
│   ├── fine_tuned/             # Fine-tuned model checkpoints
│   └── ensemble/               # Ensemble model configurations
└── experiments/                # Experiment tracking and results
    ├── logs/                   # Training logs
    ├── metrics/                # Performance metrics
    └── plots/                  # Experiment visualizations
```

## Implementation Components

### 1. Enhanced Data Loader with Covariates

**File: `src/data_loader.py` (Extended)**

```python
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Union, Optional, List, Dict
from autogluon.timeseries import TimeSeriesDataFrame
import tomli

class EnhancedTimeSeriesDataLoader(TimeSeriesDataLoader):
    """Extended data loader with covariate support."""
    
    def __init__(self, config_path: str = "config/settings.toml"):
        super().__init__(config_path)
        
        # Load covariate configuration
        covariate_config_path = "config/covariates.toml"
        if Path(covariate_config_path).exists():
            with open(covariate_config_path, 'rb') as file:
                self.covariate_config = tomli.load(file)
        else:
            self.covariate_config = {}
    
    def load_with_covariates(self, 
                           data_file: Union[str, Path],
                           covariate_files: Optional[Dict[str, str]] = None) -> TimeSeriesDataFrame:
        """Load time series data with covariate information."""
        
        # Load main time series data
        main_data = self.load_from_csv(data_file)
        
        if covariate_files is None:
            covariate_files = self.covariate_config.get('covariate_files', {})
        
        # Load and merge covariate data
        for covariate_name, file_path in covariate_files.items():
            if Path(file_path).exists():
                covariate_data = pd.read_csv(file_path)
                covariate_data['timestamp'] = pd.to_datetime(covariate_data['timestamp'])
                
                # Merge covariate data
                main_data = self._merge_covariate_data(main_data, covariate_data, covariate_name)
                print(f"Loaded covariate: {covariate_name}")
            else:
                print(f"Warning: Covariate file not found: {file_path}")
        
        return main_data
    
    def _merge_covariate_data(self, 
                            ts_data: TimeSeriesDataFrame, 
                            covariate_data: pd.DataFrame,
                            covariate_name: str) -> TimeSeriesDataFrame:
        """Merge covariate data with time series data."""
        # Implementation for merging covariate data
        # This would handle the complex merging logic for time series with covariates
        return ts_data
    
    def prepare_fine_tuning_data(self, 
                                data: TimeSeriesDataFrame,
                                validation_split: float = 0.2) -> tuple:
        """Prepare data specifically for fine-tuning with train/validation split."""
        # Split data for fine-tuning
        train_size = int(len(data) * (1 - validation_split))
        train_data = data.iloc[:train_size]
        val_data = data.iloc[train_size:]
        
        return train_data, val_data
```

### 2. Advanced Chronos Predictor

**File: `src/chronos_predictor.py` (Extended)**

```python
from autogluon.timeseries import TimeSeriesPredictor
from pathlib import Path
from typing import Optional, Dict, List, Union
import tomli
import json
from datetime import datetime

class AdvancedChronosPredictor(ChronosPredictor):
    """Advanced Chronos predictor with fine-tuning and covariate support."""
    
    def __init__(self, config_path: str = "config/settings.toml"):
        super().__init__(config_path)
        
        # Load fine-tuning configuration
        fine_tuning_config_path = "config/fine_tuning.toml"
        if Path(fine_tuning_config_path).exists():
            with open(fine_tuning_config_path, 'rb') as file:
                self.fine_tuning_config = tomli.load(file)
        else:
            self.fine_tuning_config = self._get_default_fine_tuning_config()
        
        self.models_dir = Path("models")
        self.models_dir.mkdir(parents=True, exist_ok=True)
        
        self.experiment_logs = []
    
    def _get_default_fine_tuning_config(self) -> Dict:
        """Default fine-tuning configuration."""
        return {
            'fine_tune_lr': 1e-4,
            'fine_tune_steps': 2000,
            'fine_tune_batch_size': 32,
            'fine_tune_epochs': 10,
            'early_stopping_patience': 5
        }
    
    def fit_with_fine_tuning(self, 
                           train_data,
                           model_variants: Optional[List[Dict]] = None) -> 'AdvancedChronosPredictor':
        """Fit multiple Chronos model variants including fine-tuned versions."""
        
        if model_variants is None:
            model_variants = [
                {
                    "model_path": self.model_preset,
                    "ag_args": {"name_suffix": "ZeroShot"}
                },
                {
                    "model_path": self.model_preset,
                    "fine_tune": True,
                    "fine_tune_lr": self.fine_tuning_config['fine_tune_lr'],
                    "fine_tune_steps": self.fine_tuning_config['fine_tune_steps'],
                    "ag_args": {"name_suffix": "FineTuned"}
                }
            ]
        
        # Configure hyperparameters
        hyperparameters = {"Chronos": model_variants}
        
        # Fit predictor with multiple model variants
        self.predictor = TimeSeriesPredictor(
            prediction_length=self.prediction_length
        ).fit(
            train_data=train_data,
            hyperparameters=hyperparameters,
            time_limit=self.fine_tuning_config.get('time_limit', 300),
            enable_ensemble=False
        )
        
        # Log experiment
        self._log_experiment("fit_with_fine_tuning", {
            "model_variants": len(model_variants),
            "fine_tuning_config": self.fine_tuning_config
        })
        
        print(f"Fitted {len(model_variants)} Chronos model variants")
        return self
    
    def fit_with_covariates(self, 
                          train_data,
                          known_covariates_names: List[str],
                          target: str = "value") -> 'AdvancedChronosPredictor':
        """Fit Chronos model with covariate regressor."""
        
        model_variants = [
            # Zero-shot model without covariates
            {
                "model_path": self.model_preset,
                "ag_args": {"name_suffix": "ZeroShot"}
            },
            # Chronos with covariate regressor
            {
                "model_path": self.model_preset,
                "covariate_regressor": "CAT",  # CatBoost regressor
                "target_scaler": "standard",
                "ag_args": {"name_suffix": "WithRegressor"}
            }
        ]
        
        # Configure predictor with covariates
        self.predictor = TimeSeriesPredictor(
            prediction_length=self.prediction_length,
            target=target,
            known_covariates_names=known_covariates_names
        ).fit(
            train_data,
            hyperparameters={"Chronos": model_variants},
            enable_ensemble=False,
            time_limit=self.fine_tuning_config.get('time_limit', 300)
        )
        
        # Log experiment
        self._log_experiment("fit_with_covariates", {
            "known_covariates": known_covariates_names,
            "target": target
        })
        
        print(f"Fitted Chronos model with covariates: {known_covariates_names}")
        return self
    
    def hyperparameter_search(self, 
                            train_data,
                            val_data,
                            search_space: Optional[Dict] = None) -> Dict:
        """Perform hyperparameter search for optimal model configuration."""
        
        if search_space is None:
            search_space = {
                "fine_tune_lr": [1e-5, 1e-4, 1e-3],
                "fine_tune_steps": [1000, 2000, 5000],
                "fine_tune_batch_size": [16, 32, 64]
            }
        
        best_config = None
        best_score = float('inf')
        
        # Grid search implementation
        for lr in search_space["fine_tune_lr"]:
            for steps in search_space["fine_tune_steps"]:
                for batch_size in search_space["fine_tune_batch_size"]:
                    
                    config = {
                        "fine_tune_lr": lr,
                        "fine_tune_steps": steps,
                        "fine_tune_batch_size": batch_size
                    }
                    
                    # Fit model with current configuration
                    model_variants = [{
                        "model_path": self.model_preset,
                        "fine_tune": True,
                        **config,
                        "ag_args": {"name_suffix": f"Tuned_{lr}_{steps}_{batch_size}"}
                    }]
                    
                    temp_predictor = TimeSeriesPredictor(
                        prediction_length=self.prediction_length
                    ).fit(
                        train_data,
                        hyperparameters={"Chronos": model_variants},
                        time_limit=60
                    )
                    
                    # Evaluate on validation data
                    predictions = temp_predictor.predict(val_data)
                    score = self._calculate_validation_score(val_data, predictions)
                    
                    if score < best_score:
                        best_score = score
                        best_config = config
                    
                    print(f"Config: {config}, Score: {score:.4f}")
        
        print(f"Best configuration: {best_config}, Score: {best_score:.4f}")
        return best_config
    
    def _calculate_validation_score(self, val_data, predictions) -> float:
        """Calculate validation score for hyperparameter tuning."""
        # Implementation would calculate appropriate metric (e.g., WQL, MAE)
        # For now, return a placeholder
        return 0.0
    
    def _log_experiment(self, experiment_type: str, parameters: Dict) -> None:
        """Log experiment details for tracking."""
        experiment_log = {
            "timestamp": datetime.now().isoformat(),
            "experiment_type": experiment_type,
            "parameters": parameters,
            "model_preset": self.model_preset,
            "prediction_length": self.prediction_length
        }
        
        self.experiment_logs.append(experiment_log)
        
        # Save to file
        logs_dir = Path("experiments/logs")
        logs_dir.mkdir(parents=True, exist_ok=True)
        
        log_file = logs_dir / f"experiment_{len(self.experiment_logs)}.json"
        with open(log_file, 'w') as f:
            json.dump(experiment_log, f, indent=2)
    
    def save_model(self, model_name: str) -> None:
        """Save the trained model for later use."""
        if self.predictor is None:
            raise ValueError("No model to save")
        
        model_dir = self.models_dir / model_name
        model_dir.mkdir(parents=True, exist_ok=True)
        
        # Save model artifacts
        self.predictor.save(model_dir)
        print(f"Model saved to: {model_dir}")
    
    def load_model(self, model_path: Union[str, Path]) -> 'AdvancedChronosPredictor':
        """Load a previously saved model."""
        model_path = Path(model_path)
        self.predictor = TimeSeriesPredictor.load(model_path)
        print(f"Model loaded from: {model_path}")
        return self
```

### 3. Model Comparison and Evaluation

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
                      test_data,
                      predictors: Dict[str, 'AdvancedChronosPredictor']) -> pd.DataFrame:
        """Compare multiple model configurations on test data."""
        
        comparison_results = []
        
        for model_name, predictor in predictors.items():
            # Generate predictions
            predictions = predictor.predict(test_data)
            
            # Calculate metrics
            metrics = self._calculate_metrics(test_data, predictions)
            
            # Add model information
            metrics['model_name'] = model_name
            metrics['model_type'] = self._get_model_type(model_name)
            
            comparison_results.append(metrics)
        
        # Create comparison DataFrame
        comparison_df = pd.DataFrame(comparison_results)
        
        # Save results
        results_file = self.results_dir / "model_comparison.csv"
        comparison_df.to_csv(results_file, index=False)
        
        print(f"Model comparison results saved to: {results_file}")
        return comparison_df
    
    def _calculate_metrics(self, test_data, predictions) -> Dict:
        """Calculate comprehensive evaluation metrics."""
        # Implementation would calculate various time series metrics
        # such as WQL, MAE, RMSE, MAPE, etc.
        
        metrics = {
            'wql_score': 0.0,  # Placeholder
            'mae': 0.0,
            'rmse': 0.0,
            'mape': 0.0
        }
        
        return metrics
    
    def _get_model_type(self, model_name: str) -> str:
        """Determine model type from name."""
        if 'ZeroShot' in model_name:
            return 'Zero-shot'
        elif 'FineTuned' in model_name:
            return 'Fine-tuned'
        elif 'WithRegressor' in model_name:
            return 'With Covariates'
        else:
            return 'Unknown'
    
    def plot_comparison(self, 
                       comparison_df: pd.DataFrame,
                       metric: str = 'wql_score',
                       save_path: Optional[str] = None) -> None:
        """Plot model comparison results."""
        
        fig, ax = plt.subplots(figsize=(12, 8))
        
        # Create grouped bar plot
        model_types = comparison_df['model_type'].unique()
        x_pos = np.arange(len(model_types))
        
        for i, model_type in enumerate(model_types):
            type_data = comparison_df[comparison_df['model_type'] == model_type]
            ax.bar(x_pos[i], type_data[metric].mean(), 
                  label=model_type, alpha=0.7)
        
        ax.set_xlabel('Model Type')
        ax.set_ylabel(metric.upper())
        ax.set_title('Model Performance Comparison')
        ax.set_xticks(x_pos)
        ax.set_xticklabels(model_types)
        ax.legend()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Comparison plot saved to: {save_path}")
        
        plt.show()
    
    def generate_report(self, comparison_df: pd.DataFrame) -> str:
        """Generate a comprehensive model comparison report."""
        
        report = f"""
# Model Comparison Report

## Summary
Total models compared: {len(comparison_df)}
Best performing model: {comparison_df.loc[comparison_df['wql_score'].idxmax(), 'model_name']}

## Detailed Results
{comparison_df.to_string(index=False)}

## Recommendations
- Zero-shot models: Best for quick prototyping
- Fine-tuned models: Best for production when training time is available
- Covariate models: Best when external features are available and relevant
"""
        
        # Save report
        report_file = self.results_dir / "comparison_report.md"
        with open(report_file, 'w') as f:
            f.write(report)
        
        print(f"Comparison report saved to: {report_file}")
        return report
```

### 4. Configuration Files

**File: `config/fine_tuning.toml`**
```toml
# Fine-tuning configuration
[fine_tuning]
learning_rate = 1e-4
steps = 2000
batch_size = 32
epochs = 10

[fine_tuning.early_stopping]
patience = 5
min_delta = 0.001

# Advanced options
weight_decay = 1e-5
warmup_steps = 100
gradient_clip_norm = 1.0

# Model variants to train
[[model_variants]]
name = "zero_shot"
[model_variants.config]
model_path = "bolt_small"
[model_variants.config.ag_args]
name_suffix = "ZeroShot"

[[model_variants]]
name = "fine_tuned"
[model_variants.config]
model_path = "bolt_small"
fine_tune = true
fine_tune_lr = 1e-4
fine_tune_steps = 2000
[model_variants.config.ag_args]
name_suffix = "FineTuned"

[[model_variants]]
name = "with_covariates"
[model_variants.config]
model_path = "bolt_small"
covariate_regressor = "CAT"
target_scaler = "standard"
[model_variants.config.ag_args]
name_suffix = "WithRegressor"

# Hyperparameter search space
[hyperparameter_search]
fine_tune_lr = [1e-5, 1e-4, 1e-3]
fine_tune_steps = [1000, 2000, 5000]
fine_tune_batch_size = [16, 32, 64]

# Search strategy
search_strategy = "grid"  # or "random", "bayesian"
max_trials = 20
time_limit = 3600  # seconds
```

**File: `config/covariates.toml`**
```toml
# Covariate configuration
[covariates]
# Known covariates (available at prediction time)
known_covariates = [
    "scaled_price",
    "promotion_email", 
    "promotion_homepage",
    "holiday_flag",
    "seasonal_indicator"
]

# Static features (constant per time series)
static_features = [
    "store_id",
    "category",
    "region"
]

# File paths for covariate data
[covariates.covariate_files]
price_data = "data/covariates/price_data.csv"
promotion_data = "data/covariates/promotion_data.csv"
holiday_data = "data/covariates/holiday_data.csv"
seasonal_data = "data/covariates/seasonal_data.csv"

# Covariate preprocessing
[preprocessing.scaling]
method = "standard"  # or "minmax", "robust"
features = ["scaled_price"]

[preprocessing.encoding]
categorical_features = ["store_id", "category", "region"]
method = "onehot"  # or "label", "target"

[preprocessing.imputation]
method = "forward_fill"  # or "backward_fill", "interpolate", "mean"
features = ["promotion_email", "promotion_homepage"]
```

### 5. Advanced Main Script

**File: `advanced_main.py`**
```python
#!/usr/bin/env python3
"""
Advanced Chronos implementation with fine-tuning and covariate support.
"""

from src.data_loader import EnhancedTimeSeriesDataLoader
from src.chronos_predictor import AdvancedChronosPredictor
from src.model_comparison import ModelComparator
from src.visualization import TimeSeriesVisualizer
from pathlib import Path
import tomli

def main():
    """Advanced implementation with multiple model variants."""
    print("Advanced Chronos Implementation")
    print("=" * 50)
    
    # Initialize components
    data_loader = EnhancedTimeSeriesDataLoader()
    visualizer = TimeSeriesVisualizer()
    comparator = ModelComparator()
    
    # Load data with covariates
    data_file = "data/raw/your_timeseries_data.csv"
    covariate_files = {
        "price_data": "data/covariates/price_data.csv",
        "promotion_data": "data/covariates/promotion_data.csv"
    }
    
    print("Loading time series data with covariates...")
    data = data_loader.load_with_covariates(data_file, covariate_files)
    
    # Prepare data splits
    train_data, val_data = data_loader.prepare_fine_tuning_data(data)
    _, test_data = data_loader.train_test_split(data, 48)
    
    # Initialize predictors for different approaches
    predictors = {}
    
    # 1. Zero-shot and Fine-tuned models
    print("Training zero-shot and fine-tuned models...")
    predictor_ft = AdvancedChronosPredictor()
    predictor_ft.fit_with_fine_tuning(train_data)
    predictors['zero_shot_fine_tuned'] = predictor_ft
    
    # 2. Model with covariates
    print("Training model with covariates...")
    predictor_cov = AdvancedChronosPredictor()
    known_covariates = ["scaled_price", "promotion_email", "promotion_homepage"]
    predictor_cov.fit_with_covariates(train_data, known_covariates)
    predictors['with_covariates'] = predictor_cov
    
    # 3. Hyperparameter optimization
    print("Performing hyperparameter search...")
    predictor_hp = AdvancedChronosPredictor()
    best_config = predictor_hp.hyperparameter_search(train_data, val_data)
    
    # Train with best configuration
    model_variants = [{
        "model_path": "bolt_small",
        "fine_tune": True,
        **best_config,
        "ag_args": {"name_suffix": "Optimized"}
    }]
    
    predictor_hp.predictor = AdvancedChronosPredictor().fit(
        train_data,
        hyperparameters={"Chronos": model_variants}
    )
    predictors['hyperparameter_optimized'] = predictor_hp
    
    # Compare all models
    print("Comparing model performance...")
    comparison_df = comparator.compare_models(test_data, predictors)
    
    # Generate visualizations
    print("Creating comparison visualizations...")
    comparator.plot_comparison(
        comparison_df,
        save_path="experiments/plots/model_comparison.png"
    )
    
    # Generate comprehensive report
    report = comparator.generate_report(comparison_df)
    print(report)
    
    # Save models
    for name, predictor in predictors.items():
        predictor.save_model(f"chronos_{name}")
    
    print("Advanced implementation complete!")

if __name__ == "__main__":
    main()
```

## Usage Instructions

### 1. Prepare Covariate Data

Create covariate data files in `data/covariates/`:

**File: `data/covariates/price_data.csv`**
```csv
timestamp,scaled_price
2020-01-01,1.2
2020-01-02,1.15
2020-01-03,1.3
...
```

**File: `data/covariates/promotion_data.csv`**
```csv
timestamp,promotion_email,promotion_homepage
2020-01-01,0,1
2020-01-02,1,0
2020-01-03,0,0
...
```

### 2. Run Advanced Implementation

```bash
python advanced_main.py
```

### 3. Review Results

- **Model comparisons**: `experiments/results/model_comparison.csv`
- **Detailed report**: `experiments/results/comparison_report.md`
- **Saved models**: `models/chronos_*/`
- **Visualizations**: `experiments/plots/`

## Key Features

1. **Multiple Model Variants**: Zero-shot, fine-tuned, and covariate-enhanced models
2. **Hyperparameter Optimization**: Automated search for optimal configuration
3. **Comprehensive Evaluation**: Multiple metrics and comparison frameworks
4. **Experiment Tracking**: Detailed logging and result storage
5. **Model Persistence**: Save and load trained models
6. **Advanced Visualization**: Comparative plots and performance analysis

This implementation provides a complete framework for advanced time series forecasting with Chronos models, supporting both research and production use cases.
