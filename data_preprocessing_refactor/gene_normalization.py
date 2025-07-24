import os
import time
from typing import List

import numpy as np
import anndata as ad
from scipy import sparse


def normalize_adata_expression(
    adata: ad.AnnData,
    layer: str = None,
    add_one: bool = True,
    copy: bool = True
) -> ad.AnnData:
    """
    Normalize gene expression in an AnnData by:
      1) Adding a pseudo-count of 1 (if add_one)
      2) Normalizing each spot (obs) by its total counts
      3) Taking the natural logarithm

    Args:
        adata: Input AnnData with raw counts in .X or .layers[layer]
        layer: Optional layer name to normalize instead of .X
        add_one: Whether to add 1 to counts before normalization
        copy: Whether to return a copy (default True)

    Returns:
        Normalized AnnData with normalized counts in .X (or specified layer)
    """
    # Choose data matrix
    if layer and layer in adata.layers:
        X = adata.layers[layer]
    else:
        X = adata.X

    # Convert sparse to dense
    if sparse.issparse(X):
        X = X.toarray()

    # Optionally add pseudo-count
    if add_one:
        X = X + 1.0

    # Normalize each spot by total counts
    # sum over genes for each spot (rows)
    total = X.sum(axis=1, keepdims=True)
    # avoid divide by zero
    total[total == 0] = 1.0
    X_norm = X / total

    # Log-transform
    X_norm = np.log(X_norm)

    # Create output AnnData
    out = adata.copy() if copy else adata
    # Assign normalized values back to .X
    out.X = X_norm
    # Remove layer if existed
    if layer and layer in out.layers:
        del out.layers[layer]
    return out


def normalize_multiple_adatas(
    input_paths: List[str],
    output_paths: List[str],
    layer: str = None
) -> None:
    """
    Normalize expression for multiple AnnData files and save results.

    Args:
        input_paths: List of paths to input .h5ad files
        output_paths: List of destinations for normalized .h5ad files
        layer: Optional layer name to read raw counts from
    """
    start = time.time()
    n = len(input_paths)
    if len(output_paths) != n:
        raise ValueError("input_paths and output_paths must have same length")
    for i, (inp, outp) in enumerate(zip(input_paths, output_paths), 1):
        print(f"🔬 [{i}/{n}] Loading AnnData from '{inp}'...")
        adata = ad.read_h5ad(inp)
        print(f"    🧩 Normalizing expression (layer={layer})...")
        adata_norm = normalize_adata_expression(adata, layer=layer)
        print(f"    💾 Saving normalized AnnData to '{outp}'...")
        os.makedirs(os.path.dirname(outp), exist_ok=True)
        adata_norm.write_h5ad(outp)

    elapsed = time.time() - start
    print(f"✅ Done normalizing {n} files in {elapsed:.1f}s")


def example_usage():
    """
    Example: normalize two AnnData files separately.
    """
    inputs = [
        'filtered_adata.h5ad',
        'adata_counts_hierarchy.h5ad'
    ]
    outputs = [
        './filtered_adata_norm.h5ad',
        './adata_counts_hierarchy_norm.h5ad'
    ]
    normalize_multiple_adatas(inputs, outputs)


if __name__ == '__main__':
    example_usage()
