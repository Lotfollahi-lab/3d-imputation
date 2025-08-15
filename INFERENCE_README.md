# 3D Imputation Inference

This script runs inference on a trained 3D imputation model checkpoint.

## Features

- **Configurable Training Slide Assistance**: Add random training slides to the test set to help with imputation
- **Reproducible Results**: Uses the same train/test split as during training
- **Detailed Output**: Saves predictions, metrics, and per-spot analysis
- **Flexible Device Support**: Automatic CPU/GPU detection

## Usage

### Basic Usage
```bash
python inference_3d.py --checkpoint /path/to/model_checkpoint.pth
```

### With Custom Training Slide Assistance
```bash
# Add 3 random training slides to test set for better imputation
python inference_3d.py --checkpoint /path/to/model_checkpoint.pth --k_train_slides 3

# Use only test slides (no training assistance)
python inference_3d.py --checkpoint /path/to/model_checkpoint.pth --k_train_slides 0

# Default behavior (1 training slide)
python inference_3d.py --checkpoint /path/to/model_checkpoint.pth --k_train_slides 1
```

### With Custom Output Path
```bash
python inference_3d.py --checkpoint /path/to/model_checkpoint.pth --output /path/to/results.h5ad --k_train_slides 2
```

## Parameters

- `--checkpoint`: Path to the trained model checkpoint (.pth file) **(required)**
- `--output`: Output path for results (.h5ad file). If not specified, auto-generated based on checkpoint name
- `--k_train_slides`: Number of random training slides to add to test set for imputation assistance (default: 1)
  - Set to 0 to use only test slides
  - The script will automatically handle cases where k > number of available training slides

## Training Slide Assistance

The `k_train_slides` parameter controls how many random training slides are added to the test set during inference. This mimics the behavior used during training and can help improve imputation quality by providing additional reference samples.

- **k=0**: Pure test set inference (most challenging)
- **k=1**: Default behavior, adds 1 random training slide (balanced)
- **k>1**: More training assistance (may improve imputation but less realistic)

The training slides are selected randomly with a fixed seed (42) for reproducibility.

## Output

The script generates:

1. **Results AnnData file** (`.h5ad`): Contains predictions, true labels, and metadata
2. **Metrics file** (`.json`): Overall performance metrics (MSE, MAE, PCC)

### AnnData Structure
- `X`: Predicted gene expression
- `obsm['true_expression']`: Ground truth expression
- `obsm['predicted_expression']`: Predicted expression (same as X)
- `obs`: Metadata including per-spot metrics and spatial coordinates

## Example with Real Checkpoint

```bash
# Run inference on fold1 best model with 2 training slides
python inference_3d.py \
  --checkpoint embryo_results_1/model_fold1_best.pth \
  --k_train_slides 2 \
  --output results_fold1_k2.h5ad
```

## Requirements

Make sure you have the same environment as used for training, including:
- PyTorch
- AnnData
- scanpy
- All custom modules (`models/`, `dataloader_3d_refactor.py`)
- Required data files in the expected locations 