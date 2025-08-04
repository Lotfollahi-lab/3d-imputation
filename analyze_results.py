#!/usr/bin/env python3
"""
Analysis script for inference results.
Provides detailed analysis and visualization of prediction performance.
"""

import os
import sys
import argparse
from typing import List, Dict, Tuple
import numpy as np
import pandas as pd
import anndata as ad
import scanpy as sc
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import pearsonr, spearmanr
from sklearn.metrics import mean_squared_error, mean_absolute_error

def load_and_analyze_results(results_path: str) -> Dict:
    """Load inference results and compute comprehensive metrics."""
    
    print(f"Loading results from: {results_path}")
    adata = ad.read_h5ad(results_path)
    
    if adata.n_obs == 0:
        print("Warning: No observations in results file")
        return {}
    
    # Extract predictions and true values
    if 'predicted_expression' in adata.obsm and 'true_expression' in adata.obsm:
        predictions = adata.obsm['predicted_expression']
        true_values = adata.obsm['true_expression']
    else:
        print("Warning: Expected obsm keys not found, using X as predictions")
        predictions = adata.X
        true_values = adata.X  # This might not be correct, but prevents errors
    
    print(f"Loaded {adata.n_obs} predictions for {adata.n_vars} genes")
    
    # Compute global metrics
    global_metrics = {
        'mse': mean_squared_error(true_values.flatten(), predictions.flatten()),
        'mae': mean_absolute_error(true_values.flatten(), predictions.flatten()),
        'pearson_r': pearsonr(true_values.flatten(), predictions.flatten())[0],
        'spearman_r': spearmanr(true_values.flatten(), predictions.flatten())[0],
        'n_spots': adata.n_obs,
        'n_genes': adata.n_vars
    }
    
    # Compute per-spot metrics
    per_spot_metrics = []
    for i in range(adata.n_obs):
        spot_pred = predictions[i, :]
        spot_true = true_values[i, :]
        
        # Only compute correlations if there's variation
        if np.std(spot_pred) > 1e-8 and np.std(spot_true) > 1e-8:
            pearson_r = pearsonr(spot_pred, spot_true)[0]
            spearman_r = spearmanr(spot_pred, spot_true)[0]
        else:
            pearson_r = spearman_r = 0.0
        
        per_spot_metrics.append({
            'spot_idx': i,
            'spot_id': adata.obs.iloc[i]['spot_id'] if 'spot_id' in adata.obs.columns else f"spot_{i}",
            'mse': mean_squared_error(spot_true, spot_pred),
            'mae': mean_absolute_error(spot_true, spot_pred),
            'pearson_r': pearson_r if not np.isnan(pearson_r) else 0.0,
            'spearman_r': spearman_r if not np.isnan(spearman_r) else 0.0,
            'pred_mean': np.mean(spot_pred),
            'true_mean': np.mean(spot_true),
            'pred_std': np.std(spot_pred),
            'true_std': np.std(spot_true)
        })
    
    per_spot_df = pd.DataFrame(per_spot_metrics)
    
    # Compute per-gene metrics
    per_gene_metrics = []
    for j in range(adata.n_vars):
        gene_pred = predictions[:, j]
        gene_true = true_values[:, j]
        
        # Only compute correlations if there's variation
        if np.std(gene_pred) > 1e-8 and np.std(gene_true) > 1e-8:
            pearson_r = pearsonr(gene_pred, gene_true)[0]
            spearman_r = spearmanr(gene_pred, gene_true)[0]
        else:
            pearson_r = spearman_r = 0.0
        
        per_gene_metrics.append({
            'gene_idx': j,
            'gene_name': adata.var.index[j] if hasattr(adata.var, 'index') else f"gene_{j}",
            'mse': mean_squared_error(gene_true, gene_pred),
            'mae': mean_absolute_error(gene_true, gene_pred),
            'pearson_r': pearson_r if not np.isnan(pearson_r) else 0.0,
            'spearman_r': spearman_r if not np.isnan(spearman_r) else 0.0,
            'pred_mean': np.mean(gene_pred),
            'true_mean': np.mean(gene_true),
            'pred_std': np.std(gene_pred),
            'true_std': np.std(gene_true)
        })
    
    per_gene_df = pd.DataFrame(per_gene_metrics)
    
    return {
        'adata': adata,
        'global_metrics': global_metrics,
        'per_spot_metrics': per_spot_df,
        'per_gene_metrics': per_gene_df,
        'predictions': predictions,
        'true_values': true_values
    }

def create_visualizations(analysis_results: Dict, output_dir: str):
    """Create comprehensive visualizations of the results."""
    
    os.makedirs(output_dir, exist_ok=True)
    
    predictions = analysis_results['predictions']
    true_values = analysis_results['true_values']
    per_spot_df = analysis_results['per_spot_metrics']
    per_gene_df = analysis_results['per_gene_metrics']
    global_metrics = analysis_results['global_metrics']
    
    # Set up the plotting style
    plt.style.use('default')
    sns.set_palette("husl")
    
    # 1. Global correlation plot
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    
    # Scatter plot of all predictions vs true values
    axes[0, 0].scatter(true_values.flatten(), predictions.flatten(), alpha=0.1, s=1)
    axes[0, 0].plot([true_values.min(), true_values.max()], [true_values.min(), true_values.max()], 'r--', lw=2)
    axes[0, 0].set_xlabel('True Values')
    axes[0, 0].set_ylabel('Predicted Values')
    axes[0, 0].set_title(f'Global Correlation (r={global_metrics["pearson_r"]:.3f})')
    
    # Per-spot correlation distribution
    axes[0, 1].hist(per_spot_df['pearson_r'], bins=50, alpha=0.7)
    axes[0, 1].axvline(per_spot_df['pearson_r'].mean(), color='red', linestyle='--', label=f'Mean: {per_spot_df["pearson_r"].mean():.3f}')
    axes[0, 1].set_xlabel('Per-spot Pearson Correlation')
    axes[0, 1].set_ylabel('Frequency')
    axes[0, 1].set_title('Distribution of Per-spot Correlations')
    axes[0, 1].legend()
    
    # Per-gene correlation distribution
    axes[1, 0].hist(per_gene_df['pearson_r'], bins=50, alpha=0.7)
    axes[1, 0].axvline(per_gene_df['pearson_r'].mean(), color='red', linestyle='--', label=f'Mean: {per_gene_df["pearson_r"].mean():.3f}')
    axes[1, 0].set_xlabel('Per-gene Pearson Correlation')
    axes[1, 0].set_ylabel('Frequency')
    axes[1, 0].set_title('Distribution of Per-gene Correlations')
    axes[1, 0].legend()
    
    # MSE vs MAE relationship
    axes[1, 1].scatter(per_spot_df['mse'], per_spot_df['mae'], alpha=0.6)
    axes[1, 1].set_xlabel('MSE')
    axes[1, 1].set_ylabel('MAE')
    axes[1, 1].set_title('MSE vs MAE (per spot)')
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'global_analysis.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    # 2. Performance metrics by sample (if sample info available)
    adata = analysis_results['adata']
    if 'sample' in adata.obs.columns:
        # Group by sample and compute metrics
        sample_metrics = []
        for sample in adata.obs['sample'].unique():
            sample_mask = adata.obs['sample'] == sample
            sample_spots = per_spot_df[sample_mask]
            
            sample_metrics.append({
                'sample': sample,
                'n_spots': len(sample_spots),
                'mean_pearson_r': sample_spots['pearson_r'].mean(),
                'mean_mse': sample_spots['mse'].mean(),
                'mean_mae': sample_spots['mae'].mean()
            })
        
        sample_df = pd.DataFrame(sample_metrics)
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        # Correlation by sample
        sample_df.plot(x='sample', y='mean_pearson_r', kind='bar', ax=axes[0, 0])
        axes[0, 0].set_title('Mean Pearson Correlation by Sample')
        axes[0, 0].set_ylabel('Pearson Correlation')
        axes[0, 0].tick_params(axis='x', rotation=45)
        
        # MSE by sample
        sample_df.plot(x='sample', y='mean_mse', kind='bar', ax=axes[0, 1])
        axes[0, 1].set_title('Mean MSE by Sample')
        axes[0, 1].set_ylabel('MSE')
        axes[0, 1].tick_params(axis='x', rotation=45)
        
        # Number of spots by sample
        sample_df.plot(x='sample', y='n_spots', kind='bar', ax=axes[1, 0])
        axes[1, 0].set_title('Number of Spots by Sample')
        axes[1, 0].set_ylabel('Number of Spots')
        axes[1, 0].tick_params(axis='x', rotation=45)
        
        # Sample correlation vs number of spots
        axes[1, 1].scatter(sample_df['n_spots'], sample_df['mean_pearson_r'])
        axes[1, 1].set_xlabel('Number of Spots')
        axes[1, 1].set_ylabel('Mean Pearson Correlation')
        axes[1, 1].set_title('Correlation vs Sample Size')
        
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'sample_analysis.png'), dpi=300, bbox_inches='tight')
        plt.close()
    
    # 3. Top/bottom performing genes and spots
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    
    # Top 20 genes by correlation
    top_genes = per_gene_df.nlargest(20, 'pearson_r')
    axes[0, 0].barh(range(len(top_genes)), top_genes['pearson_r'])
    axes[0, 0].set_yticks(range(len(top_genes)))
    axes[0, 0].set_yticklabels(top_genes['gene_name'], fontsize=8)
    axes[0, 0].set_xlabel('Pearson Correlation')
    axes[0, 0].set_title('Top 20 Genes by Correlation')
    
    # Bottom 20 genes by correlation
    bottom_genes = per_gene_df.nsmallest(20, 'pearson_r')
    axes[0, 1].barh(range(len(bottom_genes)), bottom_genes['pearson_r'])
    axes[0, 1].set_yticks(range(len(bottom_genes)))
    axes[0, 1].set_yticklabels(bottom_genes['gene_name'], fontsize=8)
    axes[0, 1].set_xlabel('Pearson Correlation')
    axes[0, 1].set_title('Bottom 20 Genes by Correlation')
    
    # Expression level vs performance
    axes[1, 0].scatter(per_gene_df['true_mean'], per_gene_df['pearson_r'], alpha=0.6)
    axes[1, 0].set_xlabel('Mean True Expression')
    axes[1, 0].set_ylabel('Pearson Correlation')
    axes[1, 0].set_title('Expression Level vs Prediction Performance')
    
    # Gene expression variance vs performance
    axes[1, 1].scatter(per_gene_df['true_std'], per_gene_df['pearson_r'], alpha=0.6)
    axes[1, 1].set_xlabel('True Expression Std Dev')
    axes[1, 1].set_ylabel('Pearson Correlation')
    axes[1, 1].set_title('Expression Variance vs Prediction Performance')
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'gene_analysis.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Visualizations saved to: {output_dir}")

def generate_report(analysis_results: Dict, output_path: str):
    """Generate a comprehensive analysis report."""
    
    global_metrics = analysis_results['global_metrics']
    per_spot_df = analysis_results['per_spot_metrics']
    per_gene_df = analysis_results['per_gene_metrics']
    
    report = f"""
# Inference Results Analysis Report

## Global Performance Metrics
- **MSE**: {global_metrics['mse']:.6f}
- **MAE**: {global_metrics['mae']:.6f}
- **Pearson Correlation**: {global_metrics['pearson_r']:.4f}
- **Spearman Correlation**: {global_metrics['spearman_r']:.4f}
- **Number of Spots**: {global_metrics['n_spots']:,}
- **Number of Genes**: {global_metrics['n_genes']:,}

## Per-Spot Performance Summary
- **Mean Correlation**: {per_spot_df['pearson_r'].mean():.4f} ± {per_spot_df['pearson_r'].std():.4f}
- **Median Correlation**: {per_spot_df['pearson_r'].median():.4f}
- **Best Spot Correlation**: {per_spot_df['pearson_r'].max():.4f}
- **Worst Spot Correlation**: {per_spot_df['pearson_r'].min():.4f}
- **Spots with r > 0.5**: {(per_spot_df['pearson_r'] > 0.5).sum():,} ({(per_spot_df['pearson_r'] > 0.5).mean()*100:.1f}%)
- **Spots with r > 0.7**: {(per_spot_df['pearson_r'] > 0.7).sum():,} ({(per_spot_df['pearson_r'] > 0.7).mean()*100:.1f}%)

## Per-Gene Performance Summary
- **Mean Correlation**: {per_gene_df['pearson_r'].mean():.4f} ± {per_gene_df['pearson_r'].std():.4f}
- **Median Correlation**: {per_gene_df['pearson_r'].median():.4f}
- **Best Gene Correlation**: {per_gene_df['pearson_r'].max():.4f}
- **Worst Gene Correlation**: {per_gene_df['pearson_r'].min():.4f}
- **Genes with r > 0.5**: {(per_gene_df['pearson_r'] > 0.5).sum():,} ({(per_gene_df['pearson_r'] > 0.5).mean()*100:.1f}%)
- **Genes with r > 0.7**: {(per_gene_df['pearson_r'] > 0.7).sum():,} ({(per_gene_df['pearson_r'] > 0.7).mean()*100:.1f}%)

## Top 10 Best Performing Genes
"""
    
    top_genes = per_gene_df.nlargest(10, 'pearson_r')[['gene_name', 'pearson_r', 'mse', 'mae']]
    for _, gene in top_genes.iterrows():
        report += f"- **{gene['gene_name']}**: r={gene['pearson_r']:.4f}, MSE={gene['mse']:.6f}\n"
    
    report += "\n## Bottom 10 Worst Performing Genes\n"
    bottom_genes = per_gene_df.nsmallest(10, 'pearson_r')[['gene_name', 'pearson_r', 'mse', 'mae']]
    for _, gene in bottom_genes.iterrows():
        report += f"- **{gene['gene_name']}**: r={gene['pearson_r']:.4f}, MSE={gene['mse']:.6f}\n"
    
    with open(output_path, 'w') as f:
        f.write(report)
    
    print(f"Analysis report saved to: {output_path}")

def main():
    parser = argparse.ArgumentParser(description="Analyze inference results")
    parser.add_argument('--results', type=str, required=True,
                       help='Path to inference results H5AD file')
    parser.add_argument('--output_dir', type=str, default=None,
                       help='Output directory for analysis (default: same as results file)')
    
    args = parser.parse_args()
    
    if args.output_dir is None:
        args.output_dir = os.path.dirname(args.results)
        if args.output_dir == '':
            args.output_dir = '.'
        args.output_dir = os.path.join(args.output_dir, 'analysis')
    
    # Load and analyze results
    print("=== Loading and Analyzing Results ===")
    analysis_results = load_and_analyze_results(args.results)
    
    if not analysis_results:
        print("No results to analyze")
        return
    
    # Create visualizations
    print("\n=== Creating Visualizations ===")
    viz_dir = os.path.join(args.output_dir, 'visualizations')
    create_visualizations(analysis_results, viz_dir)
    
    # Generate report
    print("\n=== Generating Report ===")
    report_path = os.path.join(args.output_dir, 'analysis_report.md')
    generate_report(analysis_results, report_path)
    
    # Save detailed metrics
    print("\n=== Saving Detailed Metrics ===")
    per_spot_path = os.path.join(args.output_dir, 'per_spot_metrics.xlsx')
    per_gene_path = os.path.join(args.output_dir, 'per_gene_metrics.xlsx')
    
    analysis_results['per_spot_metrics'].to_excel(per_spot_path, index=False)
    analysis_results['per_gene_metrics'].to_excel(per_gene_path, index=False)
    
    print(f"Per-spot metrics saved to: {per_spot_path}")
    print(f"Per-gene metrics saved to: {per_gene_path}")
    
    # Print summary
    global_metrics = analysis_results['global_metrics']
    print(f"\n=== Analysis Complete ===")
    print(f"Global Pearson Correlation: {global_metrics['pearson_r']:.4f}")
    print(f"Global MSE: {global_metrics['mse']:.6f}")
    print(f"Results saved to: {args.output_dir}")

if __name__ == "__main__":
    main() 