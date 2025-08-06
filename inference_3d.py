#!/usr/bin/env python3
"""
Inference script for 3D imputation model.
Loads a trained checkpoint and runs inference on the test set.

Features:
- Configurable number of training slides to add to test set for imputation assistance
- Reproduces the same train/test split as used during training
- Saves detailed results including per-spot metrics
- Supports both CPU and GPU inference
"""

import os
import sys
import argparse
import logging
import pickle
from typing import Dict, Tuple, List

import torch
import torch.nn as nn
from torch.utils.data import Subset
import torchvision.transforms as T
import numpy as np
import pandas as pd
import anndata as ad
import scanpy as sc
from tqdm import tqdm

from models.graph_construction import edge_index_to_adj_matrix
from models.model_3d import ST_GTC_3d
from models.losses import calculate_pcc

from dataloader_3d_refactor import (
    AnnDataImageGraphDataset,
    collate_fn_result
)

def load_checkpoint_and_args(checkpoint_path: str) -> Tuple[Dict, argparse.Namespace, Dict]:
    """Load checkpoint and extract model state, args, and split info."""
    print(f"Loading checkpoint from {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
    
    # Extract necessary components
    model_state_dict = checkpoint['model_state_dict']
    args = checkpoint['args']
    split_info = checkpoint['split_info']
    
    print(f"Loaded checkpoint from fold {checkpoint['fold']}, epoch {checkpoint['epoch']}")
    if 'pcc' in checkpoint:
        print(f"Checkpoint PCC: {checkpoint['pcc']:.4f}")
    
    return checkpoint, args, split_info

def create_test_dataset_and_loader(args: argparse.Namespace, split_info: Dict, device: torch.device, k_train_slides: int = 1):
    """Recreate the test dataset and dataloader using split info.
    
    Args:
        args: Training arguments
        split_info: Split information from checkpoint
        device: PyTorch device
        k_train_slides: Number of random training slides to add to test set for imputation assistance
    """
    
    # Create dataset with same parameters as training
    ds = AnnDataImageGraphDataset(
        args.adata_224, args.adata_counts, args.adata_feats,
        args.tiff,
        'graph_outputs/embryo.pt',
        'embryo_data/spot_iou/embryo_iou_results_new_1.csv',
        spatial_scale=args.spatial_scale,
        patch_size=args.patch_size,
        transform=T.Compose([
            T.ToTensor(),
            T.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])
        ])  # No augmentation for inference
    )
    
    # Use the saved test indices
    test_slides = np.array(split_info['test_slides'])
    train_slides = np.array(split_info['train_slides'])
    
    print(f"Original test slides: {test_slides}")
    print(f"Available train slides: {train_slides}")
    
    # Randomly select k training slides to add to test set for imputation assistance
    if k_train_slides > 0 and len(train_slides) > 0:
        if k_train_slides > len(train_slides):
            print(f"Warning: Requested {k_train_slides} training slides but only {len(train_slides)} available. Using all {len(train_slides)} slides.")
            k_actual = len(train_slides)
        else:
            k_actual = k_train_slides
            
        np.random.seed(42)  # For reproducibility
        selected_train_slides = np.random.choice(train_slides, size=k_actual, replace=False)
        print(f"Adding {k_actual} random training slides to test set: {selected_train_slides}")
        
        # Combine test slides with selected training slides
        temp = np.concatenate((test_slides, selected_train_slides))
    elif k_train_slides == 0:
        temp = test_slides
        print("k_train_slides=0: No training slides added to test set")
    else:
        temp = test_slides
        print("No training slides available or k_train_slides < 0: No training slides added to test set")
    
    # Recreate test indices based on the combined slide list
    temp = np.concatenate((test_slides, train_slides[0:1]))
    test_idx = [
        i for i, s in enumerate(ds.sample_idxs)
        if s in temp
    ]

    print(f"Test set size: {len(test_idx)} samples")
    print(f"Final test slides: {temp}")

    test_ds = Subset(ds, test_idx)
    
    # Create test dataloader
    test_loader = torch.utils.data.DataLoader(
        test_ds, 
        batch_size=args.batch_size, 
        shuffle=True,  # Don't shuffle for inference
        collate_fn=lambda b: collate_fn_result(b, ds.spot_iou, False, train_slides),
        #num_workers=getattr(args, 'num_workers', 4), 
        num_workers=0,
        pin_memory=True
    )
    
    return ds, test_ds, test_loader, test_idx

def run_inference(model: nn.Module, test_loader, device: torch.device, args: argparse.Namespace, test_ds) -> Tuple[np.ndarray, np.ndarray, List[str], List[int]]:
    """Run inference on test set and collect predictions."""
    model.eval()
    
    all_predictions = []
    all_true_labels = []
    all_original_indices = []  # Changed to track original dataset indices directly
    
    # Load graph data (same as in training)
    graph_torch = torch.load('graph_outputs/embryo.pt', weights_only=False)
    
    print("Running inference...")
    with torch.no_grad():
        for batch_idx, (
            images, labels, adj_sub, adj_prop,
            positions, idx_1024, idx_512,
            features_1024, features_512,
            known_label, indices, levels, test_orig_indices  # Now gets original indices as extra field
        ) in enumerate(tqdm(test_loader, desc="Inference")):
            
            # Prepare batch data (same as training)
            g512 = graph_torch['layer_512']
            g1024 = graph_torch['layer_1024']
            bd512 = edge_index_to_adj_matrix(g512)
            bd1024 = edge_index_to_adj_matrix(g1024)

            batch_data = [
                images, adj_sub, adj_prop,
                features_512, features_1024,
                g512.x, g1024.x,
                bd512, bd1024,
                known_label
            ]
            batch_data = [d.to(device) for d in batch_data]

            # Run inference
            if getattr(args, 'fp16', False):
                with torch.amp.autocast('cuda', dtype=torch.float16):
                    pred, _, _ = model(*batch_data)
            else:
                pred, _, _ = model(*batch_data)
            
            # Extract predictions for test indices only
            if len(test_orig_indices) > 0 and len(indices) > 0:
                # indices: batch positions of test spots
                # test_orig_indices: original dataset indices for test spots
                batch_test_positions = torch.tensor(indices)
                
                batch_predictions = pred.cpu()[batch_test_positions].numpy()
                batch_true_labels = labels[batch_test_positions].numpy()
                
                all_predictions.append(batch_predictions)
                all_true_labels.append(batch_true_labels)
                all_original_indices.extend(test_orig_indices)
    
    # Concatenate all results
    if all_predictions:
        all_predictions = np.concatenate(all_predictions, axis=0)
        all_true_labels = np.concatenate(all_true_labels, axis=0)
    else:
        all_predictions = np.array([]).reshape(0, -1)
        all_true_labels = np.array([]).reshape(0, -1)
    
    # Get spot_ids using the original indices directly
    spot_ids = []
    
    # Access the original dataset (not the Subset)
    if hasattr(test_ds, 'dataset'):  # It's a Subset
        original_dataset = test_ds.dataset
    else:
        original_dataset = test_ds
    
    for original_idx in all_original_indices:
        if hasattr(original_dataset, 'adata_224'):
            spot_id = original_dataset.adata_224.obs.iloc[original_idx]['spot_id']
            spot_ids.append(spot_id)
        else:
            spot_ids.append(f"unknown_{original_idx}")
    
    return all_predictions, all_true_labels, spot_ids, all_original_indices

def compute_metrics(predictions: np.ndarray, true_labels: np.ndarray) -> Dict[str, float]:
    """Compute evaluation metrics."""
    if len(predictions) == 0 or len(true_labels) == 0:
        return {'mse': float('inf'), 'mae': float('inf'), 'pcc': 0.0}
    
    # Convert to tensors for PCC calculation
    pred_tensor = torch.tensor(predictions, dtype=torch.float32)
    true_tensor = torch.tensor(true_labels, dtype=torch.float32)
    
    # Compute metrics
    mse = nn.MSELoss()(pred_tensor, true_tensor).item()
    mae = (pred_tensor - true_tensor).abs().mean().item()
    pcc = calculate_pcc(pred_tensor, true_tensor).item()
    
    return {
        'mse': mse,
        'mae': mae,
        'pcc': pcc
    }

def save_results_to_anndata(
    args: argparse.Namespace,
    test_idx: List[int],
    predictions: np.ndarray,
    true_labels: np.ndarray,
    spot_ids: List[str],
    original_indices: List[int],
    output_path: str
):
    """Save inference results to a new AnnData object."""
    
    # Load original AnnData to get structure
    print("Loading original AnnData for result structure...")
    adata_224 = ad.read_h5ad(args.adata_224)
    
    if len(predictions) > 0 and len(spot_ids) > 0:
        # Create AnnData object from predictions
        # Use the original indices to maintain proper mapping
        predicted_spots = [adata_224.obs.iloc[idx]['spot_id'] for idx in original_indices]
        
        # Create new AnnData with predictions
        results_adata = ad.AnnData(
            X=predictions,
            obs=pd.DataFrame({
                'spot_id': spot_ids,
                'original_index': original_indices,
                'prediction_mean': np.nanmean(predictions, axis=1),
                'true_mean': np.nanmean(true_labels, axis=1),
                'prediction_std': np.nanstd(predictions, axis=1),
                'is_predicted': True,
                'inference_timestamp': pd.Timestamp.now()
            }),
            var=adata_224.var.copy()
        )
        
        # Add true expression data
        results_adata.obsm['true_expression'] = true_labels
        results_adata.obsm['predicted_expression'] = predictions
        
        # Copy over spatial coordinates and other metadata if available
        for idx, orig_idx in enumerate(original_indices):
            obs_row = adata_224.obs.iloc[orig_idx]
            for col in ['x', 'y', 'sample']:
                if col in obs_row:
                    results_adata.obs.loc[results_adata.obs.index[idx], col] = obs_row[col]
        
        # Copy spatial coordinates from obsm if available
        if 'spatial' in adata_224.obsm:
            spatial_coords = []
            for orig_idx in original_indices:
                spatial_coords.append(adata_224.obsm['spatial'][orig_idx])
            results_adata.obsm['spatial'] = np.array(spatial_coords)
        
        # Add metrics per spot
        if len(predictions) == len(true_labels):
            # Compute per-spot metrics
            mse_per_spot = np.mean((predictions - true_labels)**2, axis=1)
            mae_per_spot = np.mean(np.abs(predictions - true_labels), axis=1)
            
            results_adata.obs['mse_per_spot'] = mse_per_spot
            results_adata.obs['mae_per_spot'] = mae_per_spot
            
            # Compute PCC per spot (correlation across genes)
            pcc_per_spot = []
            for i in range(len(predictions)):
                if np.std(predictions[i]) > 0 and np.std(true_labels[i]) > 0:
                    corr = np.corrcoef(predictions[i], true_labels[i])[0, 1]
                    pcc_per_spot.append(corr if not np.isnan(corr) else 0.0)
                else:
                    pcc_per_spot.append(0.0)
            results_adata.obs['pcc_per_spot'] = pcc_per_spot
        
    else:
        # Create empty AnnData if no predictions
        print("Warning: No predictions to save")
        results_adata = ad.AnnData(
            X=np.empty((0, adata_224.n_vars)),
            var=adata_224.var.copy(),
            obs=pd.DataFrame(columns=['spot_id', 'is_predicted', 'inference_timestamp'])
        )
    
    # Save results
    print(f"Saving results to {output_path}")
    results_adata.write_h5ad(output_path)
    print(f"Saved {results_adata.n_obs} predictions for {results_adata.n_vars} genes")
    
    return results_adata

def inference_main(checkpoint_path: str, output_path: str = None, k_train_slides: int = 1):
    """Main inference function.
    
    Args:
        checkpoint_path: Path to the trained model checkpoint
        output_path: Output path for results (optional)
        k_train_slides: Number of random training slides to add to test set for imputation assistance
    """
    
    # Load checkpoint and extract information
    checkpoint, args, split_info = load_checkpoint_and_args(checkpoint_path)
    
    # Setup device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Create test dataset and loader
    ds, test_ds, test_loader, test_idx = create_test_dataset_and_loader(args, split_info, device, k_train_slides)
    
    # Initialize and load model
    print("Loading model...")
    model = ST_GTC_3d().to(device)
    try:
        model.load_state_dict(checkpoint['model_state_dict'])
    except Exception as e:
        print(f"\n[ERROR] Failed to load model weights from checkpoint: {e}\n")
        print("Please check that the checkpoint file is correct and matches the model architecture.")
        raise
    
    # Run inference
    predictions, true_labels, spot_ids, original_indices = run_inference(model, test_loader, device, args, test_ds)
    
    # Compute metrics
    metrics = compute_metrics(predictions, true_labels)
    print("\n=== Inference Results ===")
    print(f"MSE: {metrics['mse']:.6f}")
    print(f"MAE: {metrics['mae']:.6f}")
    print(f"PCC: {metrics['pcc']:.6f}")
    print(f"Predictions shape: {predictions.shape if len(predictions) > 0 else 'No predictions'}")
    print(f"True labels shape: {true_labels.shape if len(true_labels) > 0 else 'No true labels'}")
    print(f"Number of spot IDs: {len(spot_ids)}")
    
    # Save results
    if output_path is None:
        # Generate output path based on checkpoint
        checkpoint_dir = os.path.dirname(checkpoint_path)
        checkpoint_name = os.path.splitext(os.path.basename(checkpoint_path))[0]
        output_path = os.path.join(checkpoint_dir, f"{checkpoint_name}_inference_results.h5ad")
    
    results_adata = save_results_to_anndata(args, test_idx, predictions, true_labels, spot_ids, original_indices, output_path)
    
    # Save metrics to file
    metrics_path = output_path.replace('.h5ad', '_metrics.json')
    import json
    with open(metrics_path, 'w') as f:
        json.dump(metrics, f, indent=2)
    print(f"Metrics saved to: {metrics_path}")
    
    return {
        'metrics': metrics,
        'predictions': predictions,
        'true_labels': true_labels,
        'results_adata': results_adata,
        'output_path': output_path
    }

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run inference on trained 3D imputation model")
    parser.add_argument('--checkpoint', type=str, required=True, 
                       help='Path to the trained model checkpoint')
    parser.add_argument('--output', type=str, default=None,
                       help='Output path for results (default: auto-generated)')
    parser.add_argument('--k_train_slides', type=int, default=1,
                       help='Number of random training slides to add to test set for imputation assistance (default: 1, set to 0 to use only test slides)')
    
    args = parser.parse_args()
    
    # Validate arguments
    if args.k_train_slides < 0:
        print("Error: k_train_slides must be non-negative")
        sys.exit(1)
    
    print(f"Running inference with k_train_slides={args.k_train_slides}")
    
    # Run inference
    results = inference_main(args.checkpoint, args.output, args.k_train_slides)
    
    print("\n=== Inference Complete ===")
    print(f"Results saved to: {results['output_path']}") 