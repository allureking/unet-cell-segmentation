# Changelog

## [1.0.0] - 2026-02-26

### Added
- Three model variants: standard U-Net, Attention U-Net (Oktay et al. 2018), and Residual U-Net
- Five loss functions: BCE, Dice, BCE+Dice, Focal Loss, Tversky Loss
- U-Net paper augmentation: elastic deformation, random flips, rotations, Gaussian noise, brightness/contrast
- Morphological post-processing with watershed separation for touching cells
- Connected-component cell counting with area filtering
- YAML-based experiment configs (`default.yaml`, `attention.yaml`, `lightweight.yaml`)
- Comprehensive evaluation with Dice, IoU, pixel accuracy, and count MAE metrics
- Side-by-side model comparison script (`compare.py`)
- Kaggle submission generation (`predict.py`)
- pytest test suites for models and datasets
- Training visualization utilities: sample grids, prediction overlays, training curves, count distributions
