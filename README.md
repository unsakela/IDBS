# Intelligent Driver Behavior Sensing (IDBS) System

A comprehensive multimodal fusion system for driver behavior analysis using physiological signals (WESAD dataset) and visual data (State Farm distracted driver dataset).

## 📁 Project Directory Structure

```
IDBS/
├── 📄 README.md                           # This file
├── 📄 requirements.txt                    # Python dependencies
├── 📄 .windsurfrules                      # Model architecture documentation
│
├── 📁 data/                              # Data processing and storage
│   ├── 📁 preprocessing/                 # Data preprocessing scripts
│   │   ├── 🐍 wesad_preprocess.py       # WESAD physiological signal processing
│   │   └── 🐍 statefarm_preprocess.py   # State Farm image preprocessing
│   ├── 📁 loaders/                       # PyTorch Dataset and DataLoader classes
│   │   ├── 🐍 wesad_loader.py           # WESAD dataset loader
│   │   └── 🐍 statefarm_loader.py       # State Farm dataset loader
│   ├── 📁 raw/                           # Raw datasets
│   │   ├── 📁 wesad/                    # WESAD dataset (S2, S3, S4 subjects)
│   │   │   ├── 📁 S2/                   # Subject 2 data
│   │   │   │   └── 📄 S2.pkl           # Physiological signals
│   │   │   ├── 📁 S3/                   # Subject 3 data
│   │   │   │   └── 📄 S3.pkl
│   │   │   └── 📁 S4/                   # Subject 4 data
│   │   │       └── 📄 S4.pkl
│   │   └── 📁 statefarm/                # State Farm dataset
│   │       └── 📁 imgs/                 # Images folder
│   │           └── 📁 train/            # Training images
│   │               ├── 📁 c0/          # Class 0: safe driving
│   │               ├── 📁 c1/          # Class 1: texting
│   │               ├── 📁 c2/          # Class 2: talking
│   │               ├── 📁 c3/          # Class 3: operating radio
│   │               ├── 📁 c4/          # Class 4: drinking
│   │               ├── 📁 c5/          # Class 5: reaching behind
│   │               ├── 📁 c6/          # Class 6: hair/makeup
│   │               ├── 📁 c7/          # Class 7: talking to passenger
│   │               ├── 📁 c8/          # Class 8: reaching front
│   │               └── 📁 c9/          # Class 9: navigation
│   └── 📁 processed/                     # Processed datasets
│       ├── 📁 wesad/                    # Processed WESAD data
│       │   ├── 📄 wesad_processed.pkl  # Features and labels
│       │   └── 📄 wesad_metadata.csv   # Dataset metadata
│       └── 📁 statefarm/               # Processed State Farm data
│           ├── 📁 train/               # Training split
│           │   ├── 📄 images.pt        # Image tensors
│           │   └── 📄 labels.pt        # Label tensors
│           ├── 📁 val/                 # Validation split
│           │   ├── 📄 images.pt
│           │   └── 📄 labels.pt
│           ├── 📁 test/                # Test split
│           │   ├── 📄 images.pt
│           │   └── 📄 labels.pt
│           └── 📄 dataset_info.pkl     # Dataset information
│
├── 📁 models/                            # Neural network models
│   ├── 📁 fusion/                        # Multimodal fusion models
│   │   └── 🐍 cross_attention.py        # Bidirectional cross-modal attention
│   └── 📁 sensor/                        # Sensor processing models
│       └── 🐍 tcn.py                   # Temporal Convolutional Network
│
├── 📁 training/                          # Training scripts
│   ├── 🐍 train_tcn.py                  # TCN training with MLflow logging
│   ├── 🐍 train_vision.py              # EfficientNetV2-S training
│   └── 🐍 train_fusion.py              # Multimodal fusion training
│
├── 📁 evaluation/                        # Model evaluation
│   └── 🐍 evaluate.py                   # Comprehensive evaluation script
│
├── 📁 scripts/                           # Utility scripts
│   ├── 🐍 download_datasets.py         # Dataset download utilities
│   └── 🐍 process_datasets.py          # Dataset processing utilities
│
├── 📁 notebooks/                        # Jupyter notebooks (empty)
│   └── 📄 .gitkeep
│
├── 📁 checkpoints/                       # Model checkpoints
│   ├── 🤖 tcn_best.pt                  # Best TCN model
│   ├── 🤖 vision_best.pt               # Best vision model
│   └── 🤖 fusion_best.pt               # Best fusion model
│
├── 📁 results/                           # Evaluation results
│   ├── 📄 evaluation_results.json      # Detailed evaluation metrics
│   ├── 📄 summary_report.csv           # Performance summary
│   ├── 🖼️ confusion_matrix_tcn.png     # TCN confusion matrix
│   ├── 🖼️ confusion_matrix_vision.png  # Vision confusion matrix
│   ├── 🖼️ confusion_matrix_fusion.png  # Fusion confusion matrix
│   ├── 🖼️ roc_curves_tcn.png           # TCN ROC curves
│   ├── 🖼️ roc_curves_vision.png        # Vision ROC curves
│   └── 🖼️ roc_curves_fusion.png        # Fusion ROC curves
│
└── 🐍 [Various utility scripts]         # Additional helper scripts
    ├── 🐍 test_pipeline.py              # Pipeline validation
    ├── 🐍 examine_wesad.py              # WESAD data structure examination
    ├── 🐍 check_statefarm_structure.py  # State Farm data structure check
    ├── 🐍 check_processed_wesad.py      # Processed WESAD data check
    ├── 🐍 direct_train_tcn.py           # Direct TCN training
    ├── 🐍 direct_train_vision.py        # Direct vision training
    └── 🐍 direct_train_vision_full.py    # Full dataset vision training
```

## 🏗️ System Architecture

### Core Components

1. **Temporal Convolutional Network (TCN)**
   - Processes physiological signals from WESAD dataset
   - Dilated causal convolutions with residual connections
   - Captures long-range temporal dependencies

2. **EfficientNetV2-S Vision Model**
   - Processes State Farm distracted driver images
   - Fine-tuned pretrained backbone
   - Progressive unfreezing strategy

3. **Cross-Modal Attention Fusion**
   - Bidirectional cross-attention mechanism
   - Learns inter-modal relationships
   - Adaptive feature weighting

4. **Multimodal Integration**
   - Combines temporal and visual features
   - End-to-end trainable fusion architecture
   - Multi-task learning (behavior + risk classification)

## 📊 Datasets

### WESAD Dataset
- **Source**: Wearable Stress and Affect Detection (WESAD) study
- **Subjects**: S2, S3, S4 (3 subjects)
- **Signals**: ECG, EDA, EMG, Temperature, Accelerometer (chest + wrist)
- **Conditions**: Baseline (0), Stress (1)
- **Features**: 64-dimensional feature vectors per window
- **Total Samples**: 6 (3 baseline, 3 stress)

### State Farm Dataset
- **Source**: State Farm Distracted Driver Detection
- **Classes**: 10 driving behaviors (c0-c9)
- **Images**: RGB, resized to 224×224
- **Distribution**: ~22,000 images (full dataset) / 100 images (current dummy data)
- **Classes**:
  - c0: Safe driving
  - c1: Texting - right
  - c2: Talking on the phone - right
  - c3: Texting - left
  - c4: Talking on the phone - left
  - c5: Operating the radio
  - c6: Drinking
  - c7: Reaching behind
  - c8: Hair and makeup
  - c9: Talking to passenger

## 🚀 Quick Start

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Data Preprocessing
```bash
# Process WESAD physiological signals
python data/preprocessing/wesad_preprocess.py

# Process State Farm images
python data/preprocessing/statefarm_preprocess.py
```

### 3. Individual Model Training
```bash
# Train TCN on WESAD data
python training/train_tcn.py --epochs 30 --batch_size 64

# Train EfficientNetV2-S on State Farm data
python training/train_vision.py --epochs 20 --batch_size 32
```

### 4. Fusion Model Training
```bash
# Train multimodal fusion model
python training/train_fusion.py --epochs 40 --batch_size 16
```

### 5. Model Evaluation
```bash
# Evaluate all models
python evaluation/evaluate.py
```

### 6. Full Dataset Training (Alternative)
```bash
# Train vision model on full State Farm dataset
python direct_train_vision_full.py
```

## 📈 Model Performance

### Training Pipeline
- **TCN**: AdamW optimizer, cosine LR scheduling, MLflow logging
- **Vision**: Progressive unfreezing (5 epochs frozen), data augmentation
- **Fusion**: 10 epochs encoder freezing, combined loss functions

### Evaluation Metrics
- Accuracy, F1-score (macro/weighted)
- Confusion matrices per model
- ROC curves for multi-class classification
- Per-class precision/recall analysis

## 🔧 Technical Details

### Model Specifications
- **TCN**: Input size 64, channels [64,128,256], kernel size 3
- **Vision**: EfficientNetV2-S backbone, 10-class output
- **Fusion**: Cross-modal attention with 8 heads, 512-dim fusion

### Data Processing
- **WESAD**: Bandpass filtering (0.5-40Hz), sliding windows (500 samples, 50% overlap)
- **State Farm**: ImageNet normalization, augmentations (flip, rotate, color jitter)
- **Splits**: 80/10/10 train/val/test with stratification

### Hardware Requirements
- **CPU**: Recommended for preprocessing and small datasets
- **GPU**: CUDA-enabled for faster training (optional but recommended)
- **Memory**: 8GB+ RAM for full dataset processing

## 🐛 Troubleshooting

### Common Issues

1. **Small Dataset Size**
   - Current WESAD dataset: 6 samples (use for demonstration)
   - Solution: Use full WESAD dataset with more subjects

2. **State Farm Dataset Size**
   - Current: 100 images (dummy data)
   - Full dataset: ~22,000 images
   - Solution: Download full State Farm dataset

3. **Memory Issues**
   - Reduce batch size for large datasets
   - Use CPU training if GPU memory insufficient

4. **Missing Dependencies**
   ```bash
   pip install --upgrade pip
   pip install -r requirements.txt
   ```

### Data Loading Issues

1. **WESAD Loading Error**
   - Check pickle file integrity
   - Verify data structure matches expected format

2. **State Farm Loading Error**
   - Verify folder structure: `data/raw/statefarm/imgs/train/c0/...`
   - Check image file formats (.jpg, .jpeg, .png)

## 📝 Development Notes

### Code Structure
- **Type Hints**: All functions properly typed
- **Documentation**: Docstrings on all public methods
- **Tensor Shapes**: Comments showing (B, T, C) format
- **Error Handling**: Comprehensive error checking

### Best Practices
- **Modularity**: Separate processing for each modality
- **Scalability**: Support for variable dataset sizes
- **Interpretability**: Attention weights for modality importance
- **Reproducibility**: Fixed random seeds, MLflow logging

## 📚 References

1. **WESAD Dataset**: Schmidt et al. "Wearable Stress and Affect Detection (WESAD) Dataset"
2. **State Farm Dataset**: Kaggle Distracted Driver Detection Competition
3. **TCN**: Bai et al. "An Empirical Evaluation of Generic Convolutional and Recurrent Networks for Sequence Modeling"
4. **Cross-Attention**: Vaswani et al. "Attention Is All You Need"
5. **EfficientNetV2**: Tan & Le. "EfficientNetV2: Smaller Models and Faster Training"

## 🤝 Contributing

1. Fork the repository
2. Create feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to branch (`git push origin feature/AmazingFeature`)
5. Open Pull Request

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 📞 Contact

For questions or issues, please open an issue on GitHub or contact the development team.

---

**Note**: This project is designed for research and educational purposes. The current implementation uses dummy/small datasets for demonstration. For production use, please ensure you have the complete datasets and appropriate computational resources.
