#!/usr/bin/env python3
"""
Cluster 52 slides into user-defined bins, compute per-slide highly variable genes,
and derive a gene panel by intersecting/aggregating within each bin.

Workflow:
 1. Split slides (obs['sample']) into N bins (balanced, consecutive order).
 2. For every slide, compute HVGs using the same routine as preprocess_and_embed_custom.py.
 3. For each bin collect the intersection of per-slide HVGs; if fewer than the requested
    genes_per_bin remain, back-fill with genes that occur most frequently across slides.
 4. Union the per-bin selections to obtain the final gene set.

Outputs:
  * bin_gene_sets.json : mapping bin index -> selected genes
  * union_genes.json   : sorted union of all genes
  * summary.txt        : counts per bin and final panel size
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import anndata as ad
import numpy as np
import pandas as pd
from tqdm import tqdm


def compute_hvgs(adata: ad.AnnData, n_top_genes: int = 2000, flavor: str = "seurat_v3", counts_layer: str = "counts") -> List[str]:
    """
    Return a ranked list of HVGs without mutating `adata`.
    Copied from preprocess_and_embed_custom.py for consistency.
    """
    adata_copy = adata.copy()
    hvg_kwargs: Dict[str, object] = {}

    if flavor == "seurat_v3":
        if counts_layer in adata_copy.layers:
            hvg_kwargs["layer"] = counts_layer
    else:
        from scipy.sparse import issparse  # type: ignore

        xmax = adata_copy.X.max() if not issparse(adata_copy.X) else adata_copy.X.max()
        if xmax > 20:
            import scanpy as sc

            sc.pp.normalize_total(adata_copy, target_sum=1e4)
            sc.pp.log1p(adata_copy)

    import scanpy as sc

    sc.pp.highly_variable_genes(
        adata_copy,
        n_top_genes=n_top_genes,
        flavor=flavor,
        subset=False,
        **hvg_kwargs,
    )

    mask = adata_copy.var["highly_variable"]
    hvgs = adata_copy.var.index[mask]

    if "highly_variable_rank" in adata_copy.var:
        hvgs = adata_copy.var.loc[hvgs].sort_values("highly_variable_rank").index.tolist()
    elif flavor == "seurat_v3":
        hvgs = adata_copy.var.loc[hvgs].sort_values("variances_norm", ascending=False).index.tolist()
    else:
        hvgs = adata_copy.var.loc[hvgs].sort_values("dispersions_norm", ascending=False).index.tolist()

    return hvgs[:n_top_genes]


def balanced_bins(items: Sequence[str], n_bins: int) -> List[List[str]]:
    """Split items into n_bins preserving order with sizes differing by at most 1."""
    bins: List[List[str]] = [[] for _ in range(n_bins)]
    for idx, item in enumerate(items):
        bins[idx % n_bins].append(item)
    return bins


def select_genes_for_bin(hvg_lists: List[List[str]], genes_per_bin: int) -> List[str]:
    """Select genes appearing most frequently; require full intersection first."""
    all_counts = Counter()
    for hvgs in hvg_lists:
        all_counts.update(hvgs)

    full_intersection = [gene for gene, count in all_counts.items() if count == len(hvg_lists)]
    full_intersection_sorted = sorted(full_intersection, key=lambda g: -all_counts[g])

    if len(full_intersection_sorted) >= genes_per_bin:
        return full_intersection_sorted[:genes_per_bin]

    # Backfill: sort by frequency (desc) then name for determinism.
    sorted_candidates = sorted(all_counts.items(), key=lambda kv: (-kv[1], kv[0]))
    chosen: List[str] = []
    used = set()
    for gene in full_intersection_sorted:
        chosen.append(gene)
        used.add(gene)

    for gene, _ in sorted_candidates:
        if gene in used:
            continue
        chosen.append(gene)
        used.add(gene)
        if len(chosen) >= genes_per_bin:
            break
    return chosen


def main() -> None:
    parser = argparse.ArgumentParser(description="Create slide bins and derive HVG panels.")
    parser.add_argument("--adata", required=True, help="Input AnnData H5AD file.")
    parser.add_argument("--output-dir", required=True, help="Directory to write outputs.")
    parser.add_argument("--n-bins", type=int, default=11, help="Number of slide bins (default: 11).")
    parser.add_argument("--genes-per-bin", type=int, default=50, help="Target genes per bin (default: 50).")
    parser.add_argument("--per-slide-hvg", type=int, default=2000, help="Number of HVGs computed per slide.")
    parser.add_argument("--sample-col", type=str, default="sample", help="Observation column storing slide IDs.")
    parser.add_argument("--counts-layer", type=str, default="counts", help="Layer containing raw counts if available.")
    parser.add_argument("--flavor", type=str, default="seurat_v3", help="scanpy HVG flavor (default: seurat_v3).")
    args = parser.parse_args()

    out_dir = Path(args.output_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    print("Loading AnnData...")
    data = ad.read_h5ad(args.adata)
    if args.sample_col not in data.obs.columns:
        raise KeyError(f"Column '{args.sample_col}' not found in adata.obs.")

    slide_ids = data.obs[args.sample_col].astype(str).unique().tolist()
    slide_ids_sorted = sorted(slide_ids)
    if len(slide_ids_sorted) < args.n_bins:
        raise ValueError(f"Number of slides ({len(slide_ids_sorted)}) smaller than requested bins ({args.n_bins}).")

    bins = balanced_bins(slide_ids_sorted, args.n_bins)
    bin_gene_sets: Dict[str, List[str]] = {}
    per_slide_summary: Dict[str, int] = {}

    print("Computing HVGs per slide...")
    slide_hvg_map: Dict[str, List[str]] = {}
    for slide in tqdm(slide_ids_sorted, desc="Slides"):
        subset = data[data.obs[args.sample_col] == slide].copy()
        hvgs = compute_hvgs(subset, n_top_genes=args.per_slide_hvg, flavor=args.flavor, counts_layer=args.counts_layer)
        slide_hvg_map[slide] = hvgs
        per_slide_summary[slide] = len(hvgs)

    print("Aggregating within bins...")
    union_genes: Dict[str, None] = {}
    for bin_idx, slide_list in enumerate(bins):
        if not slide_list:
            continue
        hvg_lists = [slide_hvg_map[slide] for slide in slide_list]
        selected = select_genes_for_bin(hvg_lists, args.genes_per_bin)
        bin_name = f"bin_{bin_idx:02d}"
        bin_gene_sets[bin_name] = selected
        for gene in selected:
            union_genes[gene] = None

    union_gene_list = sorted(union_genes.keys())

    print(f"Total unique genes: {len(union_gene_list)} (expected <= {args.n_bins * args.genes_per_bin})")

    (out_dir / "bin_gene_sets.json").write_text(json.dumps(bin_gene_sets, indent=2))
    (out_dir / "union_genes.json").write_text(json.dumps(union_gene_list, indent=2))

    summary = {
        "n_bins": args.n_bins,
        "genes_per_bin": args.genes_per_bin,
        "per_slide_hvg": args.per_slide_hvg,
        "slides_per_bin": [len(slides) for slides in bins],
        "total_unique_genes": len(union_gene_list),
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2))

    per_slide_df = pd.DataFrame(
        {
            "slide": slide_ids_sorted,
            "hvg_count": [per_slide_summary[s] for s in slide_ids_sorted],
            "bin": [next(i for i, sl in enumerate(bins) if s in sl) for s in slide_ids_sorted],
        }
    )
    per_slide_df.to_csv(out_dir / "per_slide_hvg_counts.csv", index=False)

    print(f"Gene sets saved under {out_dir}")


if __name__ == "__main__":  # pragma: no cover
    main()
