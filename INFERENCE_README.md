# 3D Imputation Inference Scripts

This directory contains scripts for running inference on trained 3D imputation models and analyzing the results.

## Scripts Overview

1. **`inference_3d.py`** - Main inference script for single checkpoints
2. **`batch_inference.py`** - Batch inference for multiple checkpoints  
3. **`analyze_results.py`** - Detailed analysis and visualization of results

## Usage Examples

### Single Model Inference

Run inference on a single trained model checkpoint:

```bash
python inference_3d.py --checkpoint ./weight/ours/model_fold1_best.pth
```

With custom output path:
```bash
python inference_3d.py \
    --checkpoint ./weight/ours/model_fold1_best.pth \
    --output ./results/fold1_inference.h5ad
```

### Batch Inference on Multiple Models

Run inference on all best models from cross-validation:

```bash
python batch_inference.py --checkpoint_dir ./weight/ours
```

Compare performance across all folds:
```bash
python batch_inference.py \
    --checkpoint_dir ./weight/ours \
    --compare_folds \
    --pattern "model_fold*_best.pth"
```

Run on final epoch models instead of best:
```bash
python batch_inference.py \
    --checkpoint_dir ./weight/ours \
    --pattern "model_fold*_final.pth" \
    --output_dir ./results/final_models
```

### Results Analysis

Analyze inference results with detailed metrics and visualizations:

```bash
python analyze_results.py --results ./results/model_fold1_best_inference_results.h5ad
```

With custom output directory:
```bash
python analyze_results.py \
    --results ./results/model_fold1_best_inference_results.h5ad \
    --output_dir ./analysis/fold1
```

## Output Structure

### Inference Outputs

Each inference run produces:

- **`*_inference_results.h5ad`** - AnnData object with predictions and metadata
- **`*_metrics.json`** - Performance metrics (MSE, MAE, PCC)

The AnnData object contains:
- `X`: Predicted expression values
- `obs`: Spot metadata including `spot_id`, prediction statistics
- `obsm['predicted_expression']`: Predicted values (same as X)
- `obsm['true_expression']`: True expression values
- `obs['mse_per_spot']`: Per-spot MSE
- `obs['pcc_per_spot']`: Per-spot correlation

### Batch Inference Outputs

Batch inference creates:
- `batch_inference_results/` directory
- Individual result files for each checkpoint
- `batch_inference_metrics.xlsx` - Comparative metrics table
- `fold_comparison.xlsx` - Cross-validation performance summary

### Analysis Outputs

Analysis generates:
- `analysis_report.md` - Comprehensive performance report
- `visualizations/` directory with plots:
  - `global_analysis.png` - Overall performance plots
  - `sample_analysis.png` - Per-sample performance (if applicable)
  - `gene_analysis.png` - Gene-level performance analysis
- `per_spot_metrics.xlsx` - Detailed per-spot metrics
- `per_gene_metrics.xlsx` - Detailed per-gene metrics

## Key Features

### Inference Features
- ✅ Load trained model weights and configuration
- ✅ Recreate exact train/test splits from checkpoint
- ✅ Handle mixed precision (FP16) inference
- ✅ Proper spot ID tracking and mapping
- ✅ Comprehensive error handling

### Analysis Features
- ✅ Global and per-spot/per-gene metrics
- ✅ Correlation analysis (Pearson, Spearman)
- ✅ Performance distribution visualizations
- ✅ Sample-level performance breakdown
- ✅ Gene expression level vs performance analysis
- ✅ Automated report generation

### Output Features
- ✅ Results saved as AnnData objects (compatible with scanpy)
- ✅ Proper metadata preservation (spot_id, spatial coordinates)
- ✅ Excel exports for further analysis
- ✅ High-quality visualizations (300 DPI)

## Requirements

```bash
pip install torch torchvision torchaudio
pip install anndata scanpy pandas numpy
pip install matplotlib seaborn scipy scikit-learn
pip install tifffile pillow tqdm tensorboardX
```

## Expected Data Structure

The inference scripts expect the following files to be present:

```
3d-imputation/
├── models/                     # Model definitions
├── graph_outputs/
│   └── embryo.pt              # Pre-computed graph structure
├── embryo_data/
│   └── spot_iou/
│       └── embryo_iou_results_new_1.csv  # IoU data
└── weight/ours/               # Trained checkpoints
    ├── model_fold1_best.pth
    ├── model_fold2_best.pth
    └── fold1_split_info.pkl   # Cross-validation splits
```

## Checkpoint Structure

Checkpoints contain:
- `model_state_dict`: Trained model weights
- `split_info`: Train/test split information
- `args`: Training arguments (data paths, hyperparameters)
- `metrics`: Training performance metrics
- `fold`, `epoch`: Training metadata

## Troubleshooting

### Common Issues

1. **Missing dependencies**: Ensure all required packages are installed
2. **Path issues**: Use absolute paths when possible
3. **Memory errors**: Reduce batch size in inference
4. **CUDA errors**: Check GPU availability and memory

### Performance Tips

1. Use `--fp16` flag for mixed precision inference (faster, less memory)
2. Adjust `num_workers` based on available CPU cores
3. For large datasets, consider processing in smaller batches

## Example Workflow

Complete workflow from training to analysis:

```bash
# 1. Train model (using main_3d_refactor.py)
python main_3d_refactor.py --adata_224 data.h5ad --adata_counts counts.h5ad --adata_feats features.h5ad --tiff images.tiff

# 2. Run batch inference on all folds
python batch_inference.py --checkpoint_dir ./weight/ours --compare_folds

# 3. Analyze best performing model
python analyze_results.py --results ./weight/ours/batch_inference_results/model_fold1_best_results.h5ad

# 4. Compare results across folds
python analyze_results.py --results ./weight/ours/batch_inference_results/model_fold2_best_results.h5ad
``` 