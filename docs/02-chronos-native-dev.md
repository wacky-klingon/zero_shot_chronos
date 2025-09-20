# Native Chronos Development Design

## Overview

This document outlines the design for a native Chronos time series forecasting implementation that uses the chronos-forecasting workspace as the source of truth for Chronos model usage patterns. The implementation will be modular with independent, runnable components.

## Design Principles

1. **Native Chronos Integration** - Use Chronos directly without AutoGluon wrapper
2. **Modular Architecture** - Each component can be run independently
3. **Reference Implementation** - Use chronos-forecasting workspace as source of truth
4. **Clear Separation** - Base model, training, and loading are separate concerns
5. **Dummy Data First** - Start with synthetic data for validation

## Workspace Structure

```
chronos-raw/
├── docs/
│   └── 02-chronos-native-dev.md    # This design document
├── src/
│   ├── base_model.py               # Step 2: Base model from Hugging Face
│   ├── train_model.py              # Step 3: Training with dummy data
│   └── load_model.py               # Step 4: Load saved model
├── data/
│   ├── dummy/                      # Synthetic training data
│   └── models/                     # Saved model storage
└── config/
    └── chronos_config.yaml         # Chronos-specific configuration
```

## Component Design

### Step 0: Reference Implementation Analysis

**Source**: `chronos-forecasting` workspace
**Purpose**: Understand native Chronos usage patterns

**Analysis Tasks**:
- Review Chronos model loading patterns
- Identify training data preparation methods
- Understand model saving/loading conventions
- Document Chronos API usage

**Deliverable**: Reference implementation notes and code patterns

### Step 2: Base Model Component (`src/base_model.py`)

**Purpose**: Load base Chronos model from Hugging Face and convert to our format

**Class**: `ChronosBaseModel`

**Key Methods**:
```python
class ChronosBaseModel:
    def __init__(self, model_name: str = "amazon/chronos-bolt-base")
    def load_from_huggingface(self) -> None
    def convert_to_native_format(self) -> None
    def save_base_model(self, output_path: str) -> None
    def get_model_info(self) -> dict
```

**Functionality**:
- Load Chronos model from Hugging Face
- Convert to our native format (if needed)
- Save in standardized location
- Provide model metadata

**Dependencies**:
- `transformers` for Hugging Face integration
- `torch` for model operations
- `huggingface_hub` for model downloading

**Configuration**:
```yaml
# config/chronos_config.yaml
base_model:
  model_name: "amazon/chronos-bolt-base"
  output_path: "data/models/base"
  convert_format: true
  save_metadata: true
```

**Independent Execution**:
```bash
python src/base_model.py
```

### Step 3: Training Component (`src/train_model.py`)

**Purpose**: Train Chronos model on dummy data and save trained model

**Class**: `ChronosTrainer`

**Key Methods**:
```python
class ChronosTrainer:
    def __init__(self, base_model_path: str)
    def generate_dummy_data(self, n_samples: int = 1000) -> np.ndarray
    def prepare_training_data(self, data: np.ndarray) -> tuple
    def train_model(self, train_data: tuple, epochs: int = 10) -> None
    def save_trained_model(self, output_path: str) -> None
    def evaluate_model(self, test_data: tuple) -> dict
```

**Functionality**:
- Generate synthetic time series data
- Prepare data in Chronos training format
- Train model on dummy data
- Save trained model
- Evaluate training performance

**Dummy Data Generation**:
- Multiple time series patterns (trend, seasonality, noise)
- Configurable length and complexity
- Realistic time series characteristics

**Training Configuration**:
```yaml
# config/chronos_config.yaml
training:
  epochs: 10
  batch_size: 32
  learning_rate: 1e-4
  validation_split: 0.2
  dummy_data:
    n_samples: 1000
    n_series: 10
    length: 200
    patterns: ["trend", "seasonal", "noise"]
```

**Independent Execution**:
```bash
python src/train_model.py
```

### Step 4: Model Loading Component (`src/load_model.py`)

**Purpose**: Load saved trained model and demonstrate inference

**Class**: `ChronosLoader`

**Key Methods**:
```python
class ChronosLoader:
    def __init__(self, model_path: str)
    def load_trained_model(self) -> None
    def predict(self, context: np.ndarray, prediction_length: int = 48) -> np.ndarray
    def evaluate_on_test_data(self, test_data: np.ndarray) -> dict
    def save_predictions(self, predictions: np.ndarray, output_path: str) -> None
```

**Functionality**:
- Load trained model from disk
- Generate predictions on new data
- Evaluate model performance
- Save prediction results

**Inference Configuration**:
```yaml
# config/chronos_config.yaml
inference:
  prediction_length: 48
  context_length: 200
  quantiles: [0.1, 0.5, 0.9]
  output_path: "data/predictions"
```

**Independent Execution**:
```bash
python src/load_model.py
```

## Data Flow

```
Hugging Face Model
        ↓
   Base Model (Step 2)
        ↓
   Dummy Data Generation
        ↓
   Model Training (Step 3)
        ↓
   Trained Model Storage
        ↓
   Model Loading (Step 4)
        ↓
   Inference Results
```

## Implementation Strategy

### Phase 1: Reference Analysis
1. Analyze chronos-forecasting workspace
2. Document Chronos usage patterns
3. Identify key API methods and data formats

### Phase 2: Base Model Implementation
1. Implement `ChronosBaseModel` class
2. Add Hugging Face integration
3. Test model loading and conversion
4. Validate base model functionality

### Phase 3: Training Implementation
1. Implement `ChronosTrainer` class
2. Add dummy data generation
3. Implement training loop
4. Test model saving

### Phase 4: Loading Implementation
1. Implement `ChronosLoader` class
2. Add inference functionality
3. Test end-to-end workflow
4. Validate predictions

## Configuration Management

**File**: `config/chronos_config.yaml`

```yaml
# Chronos Native Configuration
chronos:
  version: "0.1.0"
  device: "cpu"  # or "cuda" if available

base_model:
  model_name: "amazon/chronos-bolt-base"
  output_path: "data/models/base"
  convert_format: true
  save_metadata: true

training:
  epochs: 10
  batch_size: 32
  learning_rate: 1e-4
  validation_split: 0.2
  dummy_data:
    n_samples: 1000
    n_series: 10
    length: 200
    patterns: ["trend", "seasonal", "noise"]

inference:
  prediction_length: 48
  context_length: 200
  quantiles: [0.1, 0.5, 0.9]
  output_path: "data/predictions"

paths:
  data_dir: "data"
  models_dir: "data/models"
  predictions_dir: "data/predictions"
  dummy_data_dir: "data/dummy"
```

## Dependencies

**Core Dependencies**:
- `chronos>=0.1.0` - Main Chronos library
- `transformers>=4.30.0` - Hugging Face transformers
- `torch>=2.0.0` - PyTorch backend
- `huggingface_hub>=0.16.0` - Model downloading
- `numpy>=1.21.0` - Numerical operations
- `pandas>=1.5.0` - Data manipulation
- `pyyaml>=6.0` - Configuration management

**Optional Dependencies**:
- `matplotlib>=3.5.0` - Visualization
- `seaborn>=0.11.0` - Enhanced plotting

## Success Criteria

### Step 2 Success:
- [ ] Base model loads from Hugging Face
- [ ] Model converts to native format
- [ ] Model saves successfully
- [ ] Model metadata is accessible

### Step 3 Success:
- [ ] Dummy data generates correctly
- [ ] Training loop runs without errors
- [ ] Model trains on dummy data
- [ ] Trained model saves successfully

### Step 4 Success:
- [ ] Trained model loads from disk
- [ ] Inference generates predictions
- [ ] Predictions are reasonable
- [ ] End-to-end workflow completes

## Risk Mitigation

### High Risk Areas:
1. **Chronos API Changes** - Reference implementation may use different API
2. **Model Format Compatibility** - Native format conversion may fail
3. **Training Data Format** - Chronos training data requirements may be complex

### Mitigation Strategies:
1. **Reference Analysis** - Thoroughly analyze chronos-forecasting workspace
2. **Incremental Testing** - Test each component independently
3. **Fallback Options** - Keep Hugging Face format as backup
4. **Documentation** - Document all API usage patterns

## Next Steps

1. **Analyze chronos-forecasting workspace** - Understand native Chronos usage
2. **Implement base model component** - Start with model loading
3. **Implement training component** - Add dummy data and training
4. **Implement loading component** - Complete the workflow
5. **Integration testing** - Test all components together

## Development Checklist

### Phase 0: Reference Analysis
- [ ] **Analyze chronos-forecasting workspace**
  - [ ] Review Chronos model loading patterns
  - [ ] Identify training data preparation methods
  - [ ] Understand model saving/loading conventions
  - [ ] Document Chronos API usage patterns
  - [ ] Note any version-specific requirements
  - [ ] Create reference implementation notes

### Phase 1: Project Setup
- [ ] **Create project structure**
  - [ ] Create `src/` directory
  - [ ] Create `data/dummy/` directory
  - [ ] Create `data/models/` directory
  - [ ] Create `config/` directory
  - [ ] Create `data/predictions/` directory

- [ ] **Setup configuration**
  - [ ] Create `config/chronos_config.yaml`
  - [ ] Define base model configuration
  - [ ] Define training configuration
  - [ ] Define inference configuration
  - [ ] Define path configurations

- [ ] **Verify dependencies**
  - [ ] Install `chronos>=0.1.0`
  - [ ] Install `transformers>=4.30.0`
  - [ ] Install `torch>=2.0.0`
  - [ ] Install `huggingface_hub>=0.16.0`
  - [ ] Install `numpy>=1.21.0`
  - [ ] Install `pandas>=1.5.0`
  - [ ] Install `pyyaml>=6.0`

### Phase 2: Base Model Component (`src/base_model.py`)
- [ ] **Implement ChronosBaseModel class**
  - [ ] Create `__init__` method with configuration loading
  - [ ] Implement `load_from_huggingface()` method
  - [ ] Implement `convert_to_native_format()` method
  - [ ] Implement `save_base_model()` method
  - [ ] Implement `get_model_info()` method
  - [ ] Add error handling and logging

- [ ] **Test base model functionality**
  - [ ] Test model loading from Hugging Face
  - [ ] Test model conversion (if needed)
  - [ ] Test model saving
  - [ ] Test model metadata access
  - [ ] Verify model can be loaded back

- [ ] **Independent execution test**
  - [ ] Run `python src/base_model.py` successfully
  - [ ] Verify output files are created
  - [ ] Check model metadata is correct

### Phase 3: Training Component (`src/train_model.py`)
- [ ] **Implement ChronosTrainer class**
  - [ ] Create `__init__` method with base model path
  - [ ] Implement `generate_dummy_data()` method
  - [ ] Implement `prepare_training_data()` method
  - [ ] Implement `train_model()` method
  - [ ] Implement `save_trained_model()` method
  - [ ] Implement `evaluate_model()` method
  - [ ] Add error handling and logging

- [ ] **Implement dummy data generation**
  - [ ] Create trend patterns
  - [ ] Create seasonal patterns
  - [ ] Create noise patterns
  - [ ] Combine patterns realistically
  - [ ] Generate multiple time series
  - [ ] Save dummy data to files

- [ ] **Test training functionality**
  - [ ] Test dummy data generation
  - [ ] Test training data preparation
  - [ ] Test training loop execution
  - [ ] Test model saving
  - [ ] Test model evaluation

- [ ] **Independent execution test**
  - [ ] Run `python src/train_model.py` successfully
  - [ ] Verify trained model is saved
  - [ ] Check training metrics are reasonable

### Phase 4: Model Loading Component (`src/load_model.py`)
- [ ] **Implement ChronosLoader class**
  - [ ] Create `__init__` method with model path
  - [ ] Implement `load_trained_model()` method
  - [ ] Implement `predict()` method
  - [ ] Implement `evaluate_on_test_data()` method
  - [ ] Implement `save_predictions()` method
  - [ ] Add error handling and logging

- [ ] **Test inference functionality**
  - [ ] Test model loading from disk
  - [ ] Test prediction generation
  - [ ] Test evaluation on test data
  - [ ] Test prediction saving
  - [ ] Verify predictions are reasonable

- [ ] **Independent execution test**
  - [ ] Run `python src/load_model.py` successfully
  - [ ] Verify predictions are generated
  - [ ] Check prediction files are created

### Phase 5: Integration Testing
- [ ] **End-to-end workflow test**
  - [ ] Run all components in sequence
  - [ ] Verify data flows correctly between components
  - [ ] Test with different configuration settings
  - [ ] Verify all output files are created

- [ ] **Error handling test**
  - [ ] Test with missing base model
  - [ ] Test with missing trained model
  - [ ] Test with invalid configuration
  - [ ] Test with corrupted data files

- [ ] **Performance testing**
  - [ ] Measure model loading time
  - [ ] Measure training time
  - [ ] Measure inference time
  - [ ] Check memory usage

### Phase 6: Documentation and Cleanup
- [ ] **Code documentation**
  - [ ] Add docstrings to all methods
  - [ ] Add inline comments for complex logic
  - [ ] Create README for each component
  - [ ] Document configuration options

- [ ] **Code quality**
  - [ ] Run code formatting (black, isort)
  - [ ] Run linting (flake8, mypy)
  - [ ] Fix any code quality issues
  - [ ] Optimize performance where needed

- [ ] **Final validation**
  - [ ] Test all components independently
  - [ ] Test complete workflow
  - [ ] Verify all success criteria are met
  - [ ] Create final test report

## Success Criteria Validation

### Step 2 Success Criteria:
- [ ] Base model loads from Hugging Face ✓
- [ ] Model converts to native format ✓
- [ ] Model saves successfully ✓
- [ ] Model metadata is accessible ✓

### Step 3 Success Criteria:
- [ ] Dummy data generates correctly ✓
- [ ] Training loop runs without errors ✓
- [ ] Model trains on dummy data ✓
- [ ] Trained model saves successfully ✓

### Step 4 Success Criteria:
- [ ] Trained model loads from disk ✓
- [ ] Inference generates predictions ✓
- [ ] Predictions are reasonable ✓
- [ ] End-to-end workflow completes ✓

## Notes

- Each component is designed to be run independently
- Configuration is centralized in YAML format
- Dummy data generation is configurable and realistic
- Model storage follows a clear directory structure
- Error handling and logging are built into each component
- The design prioritizes simplicity and modularity over complexity

