import os
import time
from typing import List

import numpy as np
import pandas as pd
import scanpy as sc  # using scanpy for AnnData convenience


def compute_sample_gene_means(adata: sc.AnnData) -> pd.DataFrame:
    """
    Compute mean expression of each gene per sample.

    Returns:
        DataFrame of shape (n_genes, n_samples) with mean counts.
    """
    # adata.obs['sample'] must exist
    samples = adata.obs['sample'].unique()
    # Initialize DataFrame
    gene_means = pd.DataFrame(
        index=adata.var_names,
        columns=samples,
        dtype=float
    )
    # For each sample, compute mean across spots
    for sample in samples:
        mask = adata.obs['sample'] == sample
        # slice and take mean across axis=0 (spots)
        mean_vals = np.asarray(adata[mask].X.mean(axis=0)).ravel()
        gene_means[sample] = mean_vals
    return gene_means


def select_top_k_genes(gene_means: pd.DataFrame, k: int) -> List[str]:
    """
    Given per-sample gene means, compute overall gene mean and select top k genes.
    """
    # overall mean across samples
    gene_means['overall'] = gene_means.mean(axis=1)
    # select top k by overall mean
    top_genes = gene_means['overall'].nlargest(k).index.tolist()
    return top_genes


def filter_adata_by_genes(
    adata: sc.AnnData,
    top_genes: List[str]
) -> sc.AnnData:
    """
    Filter AnnData to only top_genes. Preserve obs spot_id and var gene metadata.
    """
    # subset columns (genes)
    ad_filt = adata[:, top_genes].copy()
    # ensure obs has spot_id column
    ad_filt.obs['spot_id'] = ad_filt.obs.index.astype(str)
    return ad_filt


def save_filtered_adata(
    ad_filt: sc.AnnData,
    output_path: str
) -> None:
    """
    Save filtered AnnData object to disk.
    """
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    ad_filt.write_h5ad(output_path)


def filter_top_genes_pipeline(
    adata_path: str,
    output_path: str,
    k: int = 1000
) -> None:
    """
    Load AnnData, compute top k genes by average expression,
    filter, and save new AnnData.
    """
    t0 = time.time()
    print("🔬 Loading AnnData...")
    adata = sc.read_h5ad(adata_path)

    print("📊 Computing gene means per sample...")
    gene_means = compute_sample_gene_means(adata)

    print(f"🎯 Selecting top {k} genes by overall mean...")
    top_genes = select_top_k_genes(gene_means, k)

    print("✂️  Filtering AnnData to top genes...")
    ad_filt = filter_adata_by_genes(adata, top_genes)

    print(f"💾 Saving filtered AnnData to {output_path}...")
    save_filtered_adata(ad_filt, output_path)

    elapsed = time.time() - t0
    print(f"✅ Done in {elapsed:.1f}s. New object has {ad_filt.n_vars} genes and {ad_filt.n_obs} spots.")


# Example usage
if __name__ == "__main__":
    filter_top_genes_pipeline(
        adata_path='/nfs/team361/data/sanger_whole_embryo/adata_CS17.h5ad',
        output_path="./filtered_adata.h5ad",
        k=250
    )
