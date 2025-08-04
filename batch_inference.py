#!/usr/bin/env python3
"""
Batch inference script for multiple checkpoints.
Compares results across different folds and epochs.
"""

import os
import sys
import argparse
import glob
import json
from typing import List, Dict
import pandas as pd
import numpy as np

from inference_3d import inference_main

def find_checkpoints(checkpoint_dir: str, pattern: str = "*.pth") -> List[str]:
    """Find all checkpoint files in a directory."""
    checkpoint_pattern = os.path.join(checkpoint_dir, pattern)
    checkpoints = glob.glob(checkpoint_pattern)
    return sorted(checkpoints)

def run_batch_inference(
    checkpoint_dir: str, 
    output_dir: str = None,
    checkpoint_pattern: str = "model_fold*_best.pth"
) -> Dict:
    """Run inference on multiple checkpoints."""
    
    if output_dir is None:
        output_dir = os.path.join(checkpoint_dir, "batch_inference_results")
    
    os.makedirs(output_dir, exist_ok=True)
    
    # Find checkpoints
    checkpoints = find_checkpoints(checkpoint_dir, checkpoint_pattern)
    print(f"Found {len(checkpoints)} checkpoints to process")
    
    results = {}
    all_metrics = []
    
    for i, checkpoint_path in enumerate(checkpoints):
        print(f"\n=== Processing checkpoint {i+1}/{len(checkpoints)} ===")
        print(f"Checkpoint: {os.path.basename(checkpoint_path)}")
        
        try:
            # Generate output path for this checkpoint
            checkpoint_name = os.path.splitext(os.path.basename(checkpoint_path))[0]
            output_path = os.path.join(output_dir, f"{checkpoint_name}_results.h5ad")
            
            # Run inference
            result = inference_main(checkpoint_path, output_path)
            
            # Store results
            results[checkpoint_name] = result
            
            # Collect metrics for comparison
            metrics = result['metrics'].copy()
            metrics['checkpoint'] = checkpoint_name
            metrics['checkpoint_path'] = checkpoint_path
            metrics['n_predictions'] = len(result['predictions']) if len(result['predictions']) > 0 else 0
            all_metrics.append(metrics)
            
            print(f"✓ Completed: {checkpoint_name}")
            
        except Exception as e:
            print(f"✗ Failed to process {checkpoint_path}: {str(e)}")
            continue
    
    # Save comparative metrics
    if all_metrics:
        metrics_df = pd.DataFrame(all_metrics)
        metrics_path = os.path.join(output_dir, "batch_inference_metrics.xlsx")
        metrics_df.to_excel(metrics_path, index=False)
        print(f"\n=== Comparative Metrics ===")
        print(metrics_df[['checkpoint', 'mse', 'mae', 'pcc', 'n_predictions']])
        print(f"Metrics saved to: {metrics_path}")
        
        # Find best performing models
        if len(metrics_df) > 0:
            best_pcc_idx = metrics_df['pcc'].idxmax()
            best_mse_idx = metrics_df['mse'].idxmin()
            
            print(f"\n=== Best Performing Models ===")
            print(f"Best PCC: {metrics_df.loc[best_pcc_idx, 'checkpoint']} (PCC: {metrics_df.loc[best_pcc_idx, 'pcc']:.4f})")
            print(f"Best MSE: {metrics_df.loc[best_mse_idx, 'checkpoint']} (MSE: {metrics_df.loc[best_mse_idx, 'mse']:.6f})")
    
    return {
        'results': results,
        'metrics_df': metrics_df if all_metrics else None,
        'output_dir': output_dir
    }

def compare_fold_performance(checkpoint_dir: str) -> pd.DataFrame:
    """Compare performance across different folds."""
    
    # Look for fold-specific checkpoints
    fold_patterns = [
        "model_fold*_best.pth",
        "model_fold*_final.pth", 
        "model_fold*.pth"
    ]
    
    all_fold_metrics = []
    
    for pattern in fold_patterns:
        checkpoints = find_checkpoints(checkpoint_dir, pattern)
        if not checkpoints:
            continue
            
        print(f"\nAnalyzing {pattern} pattern...")
        
        for checkpoint_path in checkpoints:
            try:
                # Extract fold number from filename
                filename = os.path.basename(checkpoint_path)
                if 'fold' in filename:
                    fold_num = filename.split('fold')[1].split('_')[0]
                    
                    # Load checkpoint to get metrics
                    import torch
                    checkpoint = torch.load(checkpoint_path, map_location='cpu')
                    
                    fold_metrics = {
                        'fold': int(fold_num),
                        'checkpoint_type': pattern.replace('*', ''),
                        'checkpoint_path': checkpoint_path,
                        'epoch': checkpoint.get('epoch', 'unknown'),
                        'mse': checkpoint.get('mse', float('nan')),
                        'mae': checkpoint.get('mae', float('nan')),
                        'pcc': checkpoint.get('pcc', float('nan'))
                    }
                    
                    all_fold_metrics.append(fold_metrics)
                    
            except Exception as e:
                print(f"Warning: Could not load {checkpoint_path}: {e}")
                continue
    
    if all_fold_metrics:
        fold_df = pd.DataFrame(all_fold_metrics)
        print("\n=== Fold Performance Summary ===")
        summary = fold_df.groupby(['fold', 'checkpoint_type']).agg({
            'mse': 'mean',
            'mae': 'mean', 
            'pcc': 'mean'
        }).round(4)
        print(summary)
        
        return fold_df
    else:
        print("No fold-specific checkpoints found")
        return pd.DataFrame()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run batch inference on multiple checkpoints")
    parser.add_argument('--checkpoint_dir', type=str, required=True,
                       help='Directory containing checkpoint files')
    parser.add_argument('--output_dir', type=str, default=None,
                       help='Output directory for results (default: checkpoint_dir/batch_inference_results)')
    parser.add_argument('--pattern', type=str, default="model_fold*_best.pth",
                       help='Pattern to match checkpoint files')
    parser.add_argument('--compare_folds', action='store_true',
                       help='Compare performance across folds')
    
    args = parser.parse_args()
    
    if args.compare_folds:
        print("=== Comparing Fold Performance ===")
        fold_df = compare_fold_performance(args.checkpoint_dir)
        if not fold_df.empty:
            output_path = os.path.join(args.checkpoint_dir, "fold_comparison.xlsx")
            fold_df.to_excel(output_path, index=False)
            print(f"Fold comparison saved to: {output_path}")
    
    print("\n=== Running Batch Inference ===")
    batch_results = run_batch_inference(args.checkpoint_dir, args.output_dir, args.pattern)
    
    print(f"\n=== Batch Inference Complete ===")
    print(f"Results saved to: {batch_results['output_dir']}") 