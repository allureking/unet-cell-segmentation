# Cell Counting in Microscopy Images — U-Net & Attention U-Net

<div align="center">

**Deep learning pipeline for automated cell counting in fluorescence microscopy images**

[![Python](https://img.shields.io/badge/Python-3.9+-3776AB?logo=python&logoColor=white)](https://python.org)
[![PyTorch](https://img.shields.io/badge/PyTorch-1.13+-EE4C2C?logo=pytorch&logoColor=white)](https://pytorch.org)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

</div>

---

## Overview

This project implements a complete deep learning pipeline for **counting cells in microscopy images** using semantic segmentation. Given fluorescence microscopy images, the model predicts a binary segmentation mask highlighting cell regions, then counts cells via connected-component labeling.

Built for the [Kaggle "Counting Cells in Microscopy Images" competition](https://www.kaggle.com/competitions/counting-cells-in-microscopy-images-2023), achieving a **MAE of 2.186** on the private leaderboard.

### Key Features

- **Two model architectures**: Standard U-Net and Attention U-Net with learned spatial focus
- **5 loss functions**: BCE, Dice, BCE+Dice, Focal, and Tversky loss
- **Rich data augmentation**: Flips, rotations, elastic deformation, noise injection, brightness/contrast
- **Post-processing**: Morphological operations and watershed segmentation for touching cells
- **Full evaluation suite**: Dice coefficient, IoU, pixel accuracy, precision/recall, cell count MAE
- **Visualization toolkit**: Prediction overlays, attention maps, error maps, training curves

## Architecture

### Standard U-Net

Based on [Ronneberger et al. (2015)](https://arxiv.org/abs/1505.04597), adapted for cell counting:

```
Input (1×128×128)
    │
    ├── Encoder 1: 1 → 64    ─────────────────────── Skip ──┐
    │   MaxPool                                              │
    ├── Encoder 2: 64 → 128   ──────────────── Skip ──┐     │
    │   MaxPool                                        │     │
    ├── Encoder 3: 128 → 256   ─────── Skip ──┐       │     │
    │   MaxPool                                │       │     │
    ├── Encoder 4: 256 → 512   ── Skip ─┐     │       │     │
    │   MaxPool                          │     │       │     │
    ├── Bottleneck: 512 → 1024           │     │       │     │
    │                                    │     │       │     │
    ├── Decoder 4: 1024 → 512  ──── + ──┘     │       │     │
    │   UpConv                                 │       │     │
    ├── Decoder 3: 512 → 256   ──────── + ────┘       │     │
    │   UpConv                                         │     │
    ├── Decoder 2: 256 → 128   ────────────── + ──────┘     │
    │   UpConv                                               │
    ├── Decoder 1: 128 → 64    ──────────────────── + ──────┘
    │
    └── 1×1 Conv → Output (1×128×128)
```

Each encoder/decoder block uses `(Conv3×3 → BatchNorm → ReLU) × 2`.

### Attention U-Net

Based on [Oktay et al. (2018)](https://arxiv.org/abs/1804.03999), adds **attention gates** at each skip connection. The gates learn to highlight cell regions and suppress irrelevant background, particularly useful for noisy microscopy images.

### Cell Counting Pipeline

```
Microscopy Image → U-Net Segmentation → Threshold → Morphological Cleanup
    → Connected Components → Cell Count
         └── (Optional) Watershed Separation for Touching Cells
```

## Project Structure

```
unet-cell-segmentation/
├── configs/
│   ├── default.yaml          # Standard U-Net config
│   ├── attention.yaml        # Attention U-Net config
│   └── lightweight.yaml      # Lightweight config for quick experiments
├── src/
│   ├── model/
│   │   ├── unet.py           # Standard U-Net architecture
│   │   ├── attention_unet.py # Attention U-Net with attention gates
│   │   ├── losses.py         # BCE, Dice, Focal, Tversky losses
│   │   └── postprocess.py    # Morphological ops + watershed
│   ├── data/
│   │   ├── dataset.py        # CellDataset + DataLoader utilities
│   │   └── augmentation.py   # Augmentation transforms
│   ├── utils/
│   │   ├── metrics.py        # Dice, IoU, accuracy, cell counting
│   │   └── visualization.py  # Plotting utilities
│   ├── train.py              # Training pipeline
│   ├── evaluate.py           # Evaluation with visualizations
│   ├── predict.py            # Kaggle submission generation
│   └── compare.py            # Model comparison experiments
├── tests/
│   ├── test_model.py         # Architecture + loss function tests
│   └── test_dataset.py       # Dataset, augmentation, metric tests
├── scripts/
│   └── run_experiment.sh     # Full pipeline automation
├── results/                  # Generated plots and metrics
├── configs/                  # YAML experiment configs
├── requirements.txt
└── README.md
```

## Quick Start

### Installation

```bash
git clone https://github.com/allureking/unet-cell-segmentation.git
cd unet-cell-segmentation
pip install -r requirements.txt
```

### Download Data

Download the dataset from the [Kaggle competition page](https://www.kaggle.com/competitions/counting-cells-in-microscopy-images-2023) and place files in `data/`:

```
data/
├── train_data.npz    # 2000 images + masks (128×128, grayscale)
└── test_images.npz   # 2000 test images
```

### Training

```bash
# Standard U-Net
python -m src.train --config configs/default.yaml

# Attention U-Net
python -m src.train --config configs/attention.yaml

# Lightweight (quick experiment)
python -m src.train --config configs/lightweight.yaml
```

### Evaluation

```bash
python -m src.evaluate --checkpoint checkpoints/best_model.pth
```

### Generate Kaggle Submission

```bash
python -m src.predict \
    --checkpoint checkpoints/best_model.pth \
    --test data/test_images.npz \
    --output submission.csv
```

### Compare Models

```bash
python -m src.compare --config configs/default.yaml
```

### Run Tests

```bash
pytest tests/ -v
```

## Results

### Kaggle Competition

| Model | Augmentation | Loss | MAE (Private) | MAE (Public) |
|-------|-------------|------|---------------|--------------|
| U-Net (32-base) | None | BCE | 9.57 | 10.01 |
| U-Net (64-base) | Flips + Rotation | BCE+Dice | **2.186** | **2.297** |

### Segmentation Metrics (Validation Set)

| Metric | U-Net | Attention U-Net |
|--------|-------|-----------------|
| Dice Coefficient | 0.891 | 0.903 |
| IoU (Jaccard) | 0.824 | 0.839 |
| Pixel Accuracy | 0.967 | 0.971 |

## Loss Functions

| Loss | Formula | Best For |
|------|---------|----------|
| **BCE** | Standard binary cross-entropy | Baseline |
| **Dice** | 1 − 2\|A∩B\| / (\|A\|+\|B\|) | Class imbalance |
| **BCE + Dice** | α·BCE + (1−α)·Dice | General use (default) |
| **Focal** | −α(1−p)^γ · log(p) | Hard examples |
| **Tversky** | 1 − TP/(TP+α·FP+β·FN) | Penalize missed cells |

## Data Augmentation

Following the original U-Net paper's emphasis on augmentation for biomedical images:

- **Geometric**: Random flips (H/V), 90° rotations
- **Intensity**: Gaussian noise, brightness/contrast adjustment
- **Elastic**: Random elastic deformation with smoothed displacement fields
- All transforms applied jointly to image-mask pairs

## Technical Details

- **Input**: Single-channel grayscale microscopy images (128×128 pixels), normalized to [0, 1]
- **Output**: Binary segmentation mask (cell vs. background)
- **Optimizer**: Adam with weight decay (1e-5)
- **LR Schedule**: ReduceLROnPlateau (factor=0.5, patience=5)
- **Checkpointing**: Best model saved based on validation loss
- **Device**: Auto-selects CUDA → MPS → CPU

## References

1. Ronneberger, O., Fischer, P., & Brox, T. (2015). *U-Net: Convolutional Networks for Biomedical Image Segmentation*. MICCAI. [arXiv:1505.04597](https://arxiv.org/abs/1505.04597)

2. Oktay, O. et al. (2018). *Attention U-Net: Learning Where to Look for the Pancreas*. MIDL. [arXiv:1804.03999](https://arxiv.org/abs/1804.03999)

3. Abraham, N. & Khan, N.M. (2019). *A Novel Focal Tversky Loss Function with Improved Attention U-Net for Lesion Segmentation*. ISBI.

## License

MIT License — see [LICENSE](LICENSE) for details.
