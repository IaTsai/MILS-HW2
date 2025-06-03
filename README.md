# Unified Multi-Task Learning Project

This project implements a unified multi-task learning model that simultaneously performs object detection, semantic segmentation, and image classification.

## Project Structure

```
unified_multitask/
├── colab.ipynb           # Main training notebook for Google Colab
├── report.md             # Project report template
├── README.md             # This file
├── requirements.txt      # Python dependencies
├── scripts/              # Utility scripts
│   ├── download_data.py  # Download datasets
│   ├── verify_data.py    # Verify dataset integrity
│   ├── eval.py          # Evaluation script
│   └── phase1_check.py  # Phase 1 requirements checker
├── src/                 # Source code
│   ├── models/          # Model architectures
│   ├── datasets/        # Dataset loaders
│   ├── utils/           # Utilities and metrics
│   └── losses/          # Loss functions
└── data/                # Dataset directory (to be created)
```

## Setup Instructions

1. **Install Dependencies**
   ```bash
   pip install -r requirements.txt
   ```

2. **Download Datasets**
   ```bash
   python scripts/download_data.py --data-dir ./data
   ```

3. **Verify Datasets**
   ```bash
   python scripts/verify_data.py --data-dir ./data
   ```

4. **Check Phase 1 Requirements**
   ```bash
   cd scripts
   python phase1_check.py
   ```

## Datasets

- **Mini COCO**: Subset of COCO dataset for object detection
- **Mini VOC**: Subset of Pascal VOC for semantic segmentation  
- **Imagenette-160**: Subset of ImageNet for image classification

## Model Architecture

The unified model consists of:
- **Backbone**: ResNet-50 (shared across all tasks)
- **Neck**: Feature Pyramid Network (FPN)
- **Heads**: Task-specific heads for detection, segmentation, and classification

## Training

Open `colab.ipynb` in Google Colab and follow the instructions to:
1. Mount Google Drive
2. Install dependencies
3. Download datasets
4. Train the unified model
5. Evaluate performance

## Evaluation

Run evaluation on the trained model:
```bash
python scripts/eval.py --model-path path/to/checkpoint.pth --task all
```

## Tasks and Metrics

1. **Object Detection**
   - Dataset: Mini COCO
   - Metric: mAP (mean Average Precision)

2. **Semantic Segmentation**
   - Dataset: Mini VOC
   - Metric: mIoU (mean Intersection over Union)

3. **Image Classification**
   - Dataset: Imagenette-160
   - Metric: Top-1 Accuracy

## Phase Requirements

### Phase 1: Setup and Data Preparation
- ✅ Create project structure
- ✅ Implement dataset loaders
- ✅ Set up model architecture
- ✅ Prepare training notebook

### Phase 2: Model Implementation
- [ ] Implement unified model
- [ ] Implement multi-task loss
- [ ] Add task balancing

### Phase 3: Training and Evaluation
- [ ] Train model on all tasks
- [ ] Evaluate performance
- [ ] Generate visualizations
- [ ] Complete report

## Authors

[Your Name]

## License

This project is for educational purposes.