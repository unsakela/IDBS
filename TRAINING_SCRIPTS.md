# IDBS Training Scripts Documentation

This document provides detailed information about all training scripts and core model components in the Intelligent Driver Behavior Sensing (IDBS) system.

## 📁 Training Scripts Directory Structure

```
IDBS/
├── 📁 training/                          # Main training scripts
│   ├── 🐍 train_tcn.py                  # TCN model training
│   ├── 🐍 train_vision.py              # Vision model training
│   └── 🐍 train_fusion.py              # Fusion model training
├── 📁 models/                            # Core model implementations
│   ├── 📁 fusion/
│   │   └── 🐍 cross_attention.py        # Cross-modal attention model
│   └── 📁 sensor/
│       └── 🐍 tcn.py                   # Temporal Convolutional Network
└── 🐍 [Additional training scripts]     # Alternative training approaches
    ├── 🐍 direct_train_tcn.py           # Direct TCN training
    ├── 🐍 direct_train_vision.py        # Direct vision training
    └── 🐍 direct_train_vision_full.py  # Full dataset vision training
```

---

## 🧠 Core Model Components

### 1. Cross-Modal Attention (`models/fusion/cross_attention.py`)

**Purpose**: Implements bidirectional cross-modal attention for multimodal fusion.

**Key Features**:
- Multi-head self-attention mechanism
- Bidirectional cross-modal processing
- Multiple fusion strategies (concat, add, max)
- Positional encoding for sequence data
- Layer normalization and dropout

**Architecture**:
```python
class CrossModalAttention(nn.Module):
    def __init__(self, d_model=128, num_heads=8, dropout=0.1, fusion_strategy='concat'):
        # Multi-head attention layers
        # Positional encoding
        # Fusion strategies
        # Output projection
```

**Key Methods**:
- `forward(modality1, modality2)`: Main forward pass with bidirectional attention
- `_create_positional_encoding(seq_len, d_model)`: Positional encoding for sequences
- `_apply_fusion_strategy(fused1, fused2, strategy)`: Fusion of modalities

**Usage**:
```python
attention = CrossModalAttention(d_model=128, num_heads=8, fusion_strategy='concat')
output, attention_weights = attention(modality1, modality2)
```

---

### 2. Temporal Convolutional Network (`models/sensor/tcn.py`)

**Purpose**: Processes temporal physiological signals using dilated causal convolutions.

**Key Features**:
- Dilated causal convolutions for long-range dependencies
- Residual connections with weight normalization
- Temporal causality preservation
- Variable sequence length support
- Efficient parallel computation

**Architecture**:
```python
class TCN(nn.Module):
    def __init__(self, input_size=64, num_channels=[64,128,256], kernel_size=3, dropout=0.2):
        # TemporalBlock layers with dilations
        # Residual connections
        # Output projection
```

**Components**:
- `TemporalBlock`: Basic TCN building block
- `TemporalConv`: Dilated causal convolution layer
- Residual connections with optional projections

**Usage**:
```python
tcn = TCN(input_size=64, num_channels=[64,128,256], kernel_size=3)
output = tcn(input_sequence)  # Shape: (batch_size, num_classes)
```

---

## 🏋️ Training Scripts

### 1. TCN Training (`training/train_tcn.py`)

**Purpose**: Trains Temporal Convolutional Network on WESAD physiological data.

**Features**:
- MLflow experiment tracking
- AdamW optimizer with cosine learning rate scheduling
- Early stopping based on validation F1-score
- Comprehensive logging and checkpointing
- Confusion matrix visualization

**Key Classes**:
```python
class TCNTrainer:
    def __init__(self, input_size=64, num_channels=[64,128,256], learning_rate=1e-3):
        # Model initialization
        # Optimizer and scheduler setup
        # Training history tracking
    
    def train_epoch(self, train_loader):
        # Single epoch training loop
        # Loss computation and backpropagation
        # Progress tracking
    
    def validate_epoch(self, val_loader):
        # Validation loop
        # Metrics computation
        # Best model tracking
```

**Usage**:
```bash
python training/train_tcn.py --epochs 30 --batch_size 64 --lr 1e-3
```

**Arguments**:
- `--epochs`: Number of training epochs (default: 30)
- `--batch_size`: Batch size (default: 64)
- `--lr`: Learning rate (default: 1e-3)
- `--data_path`: WESAD data path
- `--checkpoint_dir`: Checkpoint save directory
- `--experiment_name`: MLflow experiment name

---

### 2. Vision Training (`training/train_vision.py`)

**Purpose**: Fine-tunes EfficientNetV2-S on State Farm distracted driver dataset.

**Features**:
- Progressive unfreezing strategy (5 epochs frozen, then fine-tune)
- Data augmentation (crop, flip, rotation, color jitter)
- ImageNet pretrained backbone
- Class imbalance handling
- Detailed performance metrics

**Key Classes**:
```python
class EfficientNetV2Trainer:
    def __init__(self, num_classes=10, freeze_epochs=5, learning_rate=1e-3):
        # EfficientNetV2-S model initialization
        # Progressive unfreezing setup
        # Training configuration
    
    def freeze_backbone(self, freeze=True):
        # Freeze/unfreeze backbone parameters
        # Progressive training strategy
    
    def train_epoch(self, train_loader, epoch):
        # Training with progressive unfreezing
        # Learning rate adjustment after unfreezing
```

**Usage**:
```bash
python training/train_vision.py --epochs 20 --batch_size 32 --freeze_epochs 5
```

**Arguments**:
- `--epochs`: Number of training epochs (default: 20)
- `--batch_size`: Batch size (default: 32)
- `--freeze_epochs`: Epochs to freeze backbone (default: 5)
- `--lr`: Learning rate (default: 1e-3)
- `--data_dir`: State Farm data directory

---

### 3. Fusion Training (`training/train_fusion.py`)

**Purpose**: Trains multimodal fusion model combining TCN and vision features.

**Features**:
- Loads pretrained TCN and vision checkpoints
- Cross-modal attention fusion
- Progressive encoder unfreezing (10 epochs)
- Multi-task learning (behavior + risk classification)
- Combined loss functions

**Key Classes**:
```python
class CrossModalAttentionFusion(nn.Module):
    def __init__(self, tcn_input_size=64, vision_feature_dim=1280, num_behavior_classes=10):
        # TCN encoder
        # Vision encoder (EfficientNetV2-S)
        # Cross-modal attention
        # Fusion layers
        # Dual classification heads
    
    def forward(self, sensor_features, images):
        # Multimodal forward pass
        # Returns behavior_logits, risk_logits

class FusionTrainer:
    def load_pretrained_models(self, tcn_path, vision_path):
        # Load pretrained checkpoints
        # Initialize fusion model
    
    def train_epoch(self, train_loader, epoch):
        # Progressive unfreezing training
        # Combined loss computation
```

**Usage**:
```bash
python training/train_fusion.py --epochs 40 --batch_size 16 --freeze_epochs 10
```

**Arguments**:
- `--epochs`: Number of training epochs (default: 40)
- `--batch_size`: Batch size (default: 16)
- `--freeze_epochs`: Epochs to freeze encoders (default: 10)
- `--tcn_checkpoint`: TCN checkpoint path
- `--vision_checkpoint`: Vision checkpoint path

---

## 🚀 Alternative Training Scripts

### 1. Direct TCN Training (`direct_train_tcn.py`)

**Purpose**: Simplified TCN training for small datasets or quick testing.

**Features**:
- Direct data loading without complex data loaders
- Manual train/val/test splits
- Simple training loop
- Basic checkpointing

**Key Functions**:
```python
def direct_train_tcn():
    # Load processed WESAD data
    # Manual data splitting
    # Simple training loop
    # Model saving
```

**Usage**:
```bash
python direct_train_tcn.py
```

---

### 2. Direct Vision Training (`direct_train_vision.py`)

**Purpose**: Simplified vision training using processed data.

**Features**:
- Direct tensor loading
- Simple training loop
- Basic evaluation
- Minimal dependencies

**Key Functions**:
```python
def direct_train_vision():
    # Load processed State Farm tensors
    # Initialize EfficientNetV2-S
    # Training and evaluation
```

**Usage**:
```bash
python direct_train_vision.py
```

---

### 3. Full Dataset Vision Training (`direct_train_vision_full.py`)

**Purpose**: Train vision model on full State Farm dataset from raw data.

**Features**:
- Custom Dataset class for raw image loading
- Proper train/val/test splits with stratification
- Data augmentation and normalization
- Class distribution analysis
- Scalable to large datasets

**Key Classes**:
```python
class StateFarmDataset(Dataset):
    def __init__(self, image_paths, labels, transform=None):
        # Custom dataset implementation
    
    def __getitem__(self, idx):
        # Load and transform images
        # Return image, label pair

def load_statefarm_dataset():
    # Scan raw data directory
    # Collect image paths and labels
    # Class distribution analysis

def create_data_splits(image_paths, labels):
    # Stratified train/val/test splits
    # Maintain class balance
```

**Usage**:
```bash
python direct_train_vision_full.py
```

---

## 📊 Training Pipeline Overview

### Standard Training Flow

1. **Data Preprocessing**
   ```bash
   python data/preprocessing/wesad_preprocess.py
   python data/preprocessing/statefarm_preprocess.py
   ```

2. **Individual Model Training**
   ```bash
   python training/train_tcn.py --epochs 30 --batch_size 64
   python training/train_vision.py --epochs 20 --batch_size 32
   ```

3. **Fusion Model Training**
   ```bash
   python training/train_fusion.py --epochs 40 --batch_size 16
   ```

4. **Model Evaluation**
   ```bash
   python evaluation/evaluate.py
   ```

### Alternative Training Flow

```bash
# Direct training for small datasets
python direct_train_tcn.py
python direct_train_vision.py

# Full dataset training
python direct_train_vision_full.py
```

---

## 🔧 Configuration and Customization

### Common Parameters

**TCN Training**:
- `input_size`: Feature dimension (default: 64)
- `num_channels`: Channel sizes (default: [64,128,256])
- `kernel_size`: Convolution kernel size (default: 3)
- `dropout`: Dropout rate (default: 0.2)

**Vision Training**:
- `num_classes`: Number of output classes (default: 10)
- `freeze_epochs`: Backbone freezing epochs (default: 5)
- `image_size`: Input image size (default: 224x224)

**Fusion Training**:
- `fusion_dim`: Fusion layer dimension (default: 512)
- `num_heads`: Attention heads (default: 8)
- `freeze_epochs`: Encoder freezing epochs (default: 10)

### Model Checkpoints

All training scripts save checkpoints to `checkpoints/`:
- `tcn_best.pt`: Best TCN model
- `vision_best.pt`: Best vision model
- `fusion_best.pt`: Best fusion model

### MLflow Tracking

Training scripts use MLflow for experiment tracking:
- Automatic parameter logging
- Metric tracking (loss, accuracy, F1)
- Model artifact storage
- Performance visualization

---

## 🐛 Troubleshooting

### Common Issues

1. **Memory Issues**
   - Reduce batch size
   - Use gradient accumulation
   - Enable mixed precision training

2. **Convergence Issues**
   - Adjust learning rate
   - Increase training epochs
   - Check data quality

3. **Data Loading Issues**
   - Verify data paths
   - Check file permissions
   - Ensure proper data format

### Debug Mode

Add `--debug` flag to enable detailed logging:
```bash
python training/train_tcn.py --debug
```

---

## 📈 Performance Metrics

### Evaluation Metrics

All training scripts track:
- **Accuracy**: Overall classification accuracy
- **F1-Score**: Macro and weighted F1 scores
- **Precision/Recall**: Per-class metrics
- **Confusion Matrix**: Detailed error analysis
- **ROC Curves**: Multi-class ROC analysis

### Best Practices

1. **Hyperparameter Tuning**
   - Use learning rate scheduling
   - Experiment with dropout rates
   - Optimize batch sizes

2. **Data Augmentation**
   - Apply appropriate augmentations
   - Maintain class balance
   - Validate augmentation impact

3. **Model Selection**
   - Monitor validation performance
   - Use early stopping
   - Save best checkpoints

---

## 🤝 Contributing

### Adding New Training Scripts

1. Follow existing naming conventions
2. Include comprehensive docstrings
3. Add MLflow integration
4. Implement proper argument parsing
5. Include error handling and logging

### Code Style

- Use type hints for all functions
- Include detailed docstrings
- Follow PEP 8 formatting
- Add inline comments for complex logic

---

This documentation provides a comprehensive overview of all training scripts and model components in the IDBS system. Each script is designed to be modular, configurable, and production-ready with proper logging and experiment tracking.
