#!/usr/bin/env python3
"""
Streamlined preprocessing + embedding pipeline for custom OME-TIFF + AnnData datasets.

This script combines data preparation and image embedding into a single pass,
directly producing training-ready outputs without intermediate files.

Output format matches STFlow expectations:
- Embeddings in <embed_dir>/<dataset>/<encoder>/fp32/<sample>.h5
- Per-sample AnnData in <output_dir>/adata/<sample>.h5ad
- Gene lists and splits in <output_dir>/

Usage:
    python preprocess_and_embed_custom.py \\
        --adata /path/to/adata.h5ad \\
        --ome_tiff /path/to/images.ome.tif \\  # or --rgb_image /path/to/slide.jpg for single-sample runs
        --output_dir ./dataset/EMBRYO \\
        --embed_dir ./embeddings \\
        --encoder uni_v1_official \\
        --n_genes 50 \\
        --test_samples 0,5,10  # Indices, not IDs
"""

import os
import json
import h5py
import argparse
import numpy as np
import pandas as pd
from tqdm import tqdm
from pathlib import Path

import torch
import scanpy as sc
import anndata as ad
from PIL import Image
import squidpy as sq
import scipy.sparse as sp
from scipy.stats import rankdata

try:
    import tifffile
except ImportError:
    raise ImportError("Please install tifffile: pip install tifffile")

from hest_utils.encoder import load_encoder
from hest_utils.file_utils import save_hdf5
from data.normalize_utils import get_normalize_method


def compute_hvgs(adata, n_top_genes=2000, flavor="seurat_v3", counts_layer="counts"):
    """Return a ranked list of HVGs without mutating `adata`."""
    ad = adata.copy()
    hvg_kwargs = {}

    # Use raw counts for seurat_v3; log-normalized X for others
    if flavor == "seurat_v3":
        if counts_layer in ad.layers:
            hvg_kwargs["layer"] = counts_layer  # explicit counts
        # else assume ad.X already holds counts; do NOT normalize/log1p here
    else:
        # Ensure X is log-normalized for dispersion-based flavors
        from scipy.sparse import issparse
        xmax = ad.X.max() if not issparse(ad.X) else ad.X.max()
        if xmax > 20:  # crude guard if X looks like counts
            import scanpy as sc
            sc.pp.normalize_total(ad, target_sum=1e4)
            sc.pp.log1p(ad)

    import scanpy as sc
    sc.pp.highly_variable_genes(ad, n_top_genes=n_top_genes, flavor=flavor, subset=False, **hvg_kwargs)

    mask = ad.var["highly_variable"]
    hvgs = ad.var.index[mask]

    # Prefer the provided rank; fall back to flavor-specific scores
    if "highly_variable_rank" in ad.var:
        hvgs = ad.var.loc[hvgs].sort_values("highly_variable_rank").index.tolist()
    elif flavor == "seurat_v3":
        hvgs = ad.var.loc[hvgs].sort_values("variances_norm", ascending=False).index.tolist()
    else:
        hvgs = ad.var.loc[hvgs].sort_values("dispersions_norm", ascending=False).index.tolist()

    return hvgs[:n_top_genes]


def _canonicalize_spatial_mode(mode: str) -> str:
    mode_lower = mode.lower()
    if mode_lower in {"morani", "moran"}:
        return "moran"
    if mode_lower in {"geary", "gearyc"}:
        return "geary"
    return mode


def compute_spatially_variable_genes(
    adata,
    n_top_genes=2000,
    mode="moranI",
    spatial_neighbors_kwargs=None,
    candidate_gene_count=None,
    balanced_counts=None,
    sample_col: str = "sample",
):
    """
    Select spatially variable genes using Squidpy spatial autocorrelation.

    Parameters
    ----------
    adata : AnnData
        Full AnnData object containing `.obsm['spatial']`.
    n_top_genes : int
        Number of spatially variable genes to return.
    mode : str
        Autocorrelation metric to use (e.g., 'moranI', 'gearyC').
    spatial_neighbors_kwargs : dict or None
        Optional keyword arguments passed to `sq.gr.spatial_neighbors`.

    Returns
    -------
    List[str]
        Gene names sorted by decreasing spatial variability.
    """
    import warnings

    mode = _canonicalize_spatial_mode(mode)

    if "spatial" not in adata.obsm:
        raise ValueError("AnnData is missing obsm['spatial'], required for spatial gene selection.")

    ad = adata.copy()
    kwargs = spatial_neighbors_kwargs or {}

    if candidate_gene_count is not None:
        candidate_gene_count = min(candidate_gene_count, ad.n_vars)
        candidates = compute_hvgs(
            ad,
            n_top_genes=candidate_gene_count,
            flavor="seurat_v3",
        )
        print(f"  Using {len(candidates)} candidate genes for spatial selection")
    else:
        candidates = ad.var_names.tolist()
        print(f"  Using all {len(candidates)} genes for spatial selection (this may be slow)")

    # Build spatial graph per slide and assemble block-diagonal matrices
    sample_series = ad.obs[sample_col].astype(str)
    unique_samples = list(pd.unique(sample_series))
    index_blocks = [np.where(sample_series == s)[0] for s in unique_samples]
    index_blocks = [idx for idx in index_blocks if idx.size > 0]

    if index_blocks:
        perm_index = np.concatenate(index_blocks)
        ad = ad[perm_index].copy()
    else:
        perm_index = np.arange(ad.n_obs)

    conn_blocks = []
    dist_blocks = []
    offset = 0
    for idx in index_blocks:
        n_rows = len(idx)
        sub = ad[offset: offset + n_rows].copy()
        sq.gr.spatial_neighbors(sub, coord_type="generic", **kwargs)
        conn_blocks.append(sub.obsp["spatial_connectivities"])
        dist_blocks.append(sub.obsp["spatial_distances"])
        offset += n_rows

    if conn_blocks:
        conn_full = sp.block_diag(conn_blocks, format="csr")
        dist_full = sp.block_diag(dist_blocks, format="csr")
        ad.obsp["spatial_connectivities"] = conn_full
        ad.obsp["spatial_distances"] = dist_full
        ad.uns["spatial_neighbors"] = {
            "connectivities_key": "spatial_connectivities",
            "distances_key": "spatial_distances",
            "params": kwargs,
        }
    else:
        # Fallback: single global graph (no block structure)
        sq.gr.spatial_neighbors(ad, coord_type="generic", **kwargs)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        sq.gr.spatial_autocorr(ad, mode=mode, genes=candidates)

    key = mode
    if key not in ad.uns:
        alt = None
        if mode == "moran":
            alt = "moranI"
        elif mode == "geary":
            alt = "gearyC"
        if alt and alt in ad.uns:
            key = alt
        else:
            raise KeyError(f"Spatial autocorr results not found in adata.uns['{mode}'].")

    spatial_df = ad.uns[key]
    # Some modes may return a list; normalize to DataFrame
    if hasattr(spatial_df, "copy"):
        spatial_df = spatial_df.copy()
    else:
        raise TypeError(f"Unexpected data type for adata.uns['{mode}']: {type(spatial_df)}")

    score_col = "I" if "I" in spatial_df.columns else spatial_df.columns[0]
    spatial_df = spatial_df.replace([np.inf, -np.inf], np.nan).dropna(subset=[score_col])
    spatial_df = spatial_df.sort_values(by=score_col, ascending=False)

    sorted_genes = spatial_df.index.tolist()

    def select_balanced_genes(genes, target_count, n_bins=5):
        if target_count >= len(genes):
            return genes
        bins = np.array_split(genes, min(n_bins, len(genes)))
        selected = []
        bin_indices = [list(b) for b in bins if len(b) > 0]
        while len(selected) < target_count and bin_indices:
            for bucket in list(bin_indices):
                if not bucket:
                    bin_indices.remove(bucket)
                    continue
                selected.append(bucket.pop(0))
                if len(selected) == target_count:
                    break
        return selected

    counts = set(balanced_counts or [])
    counts.add(min(n_top_genes, len(sorted_genes)))
    selected = {}
    for count in sorted(counts):
        selected[count] = select_balanced_genes(sorted_genes, count)

    main_list = selected[min(n_top_genes, len(sorted_genes))]
    extra_lists = {k: v for k, v in selected.items() if k != min(n_top_genes, len(sorted_genes))}
    return main_list, extra_lists


def _bh_fdr(pvals: np.ndarray) -> np.ndarray:
    """Benjamini-Hochberg FDR correction; returns q-values in [0,1]."""
    n = pvals.size
    order = np.argsort(pvals)
    ranked = pvals[order]
    q = ranked * n / (np.arange(n) + 1)
    q = np.minimum.accumulate(q[::-1])[::-1]
    out = np.empty_like(q)
    out[order.argsort()] = q
    return np.clip(out, 0.0, 1.0)


def compute_z_variation_genes(
    adata,
    n_top_genes: int,
    sample_col: str = "sample",
    min_sections_expressed: int = 5,
    min_global_mean: float = 0.01,
    moran_min: float = 0.25,
    fdr_max: float = 0.05,
    rho_min: float | None = None,
    cv2_min: float | None = None,
    permutations: int = 200,
    random_seed: int = 0,
    libsize_target: float = 1e4,
):
    """Rank genes by z-variation across sections using mean profiles and Moran's I."""

    import numpy as _np
    from scipy.sparse import issparse as _issparse

    if sample_col not in adata.obs:
        raise ValueError(f"Column '{sample_col}' not found in adata.obs")

    section_ids = _np.array(sorted(_np.unique(adata.obs[sample_col].astype(str))))
    S = section_ids.size
    G = adata.n_vars

    means_per_section = _np.zeros((S, G), dtype=_np.float32)
    for si, sid in enumerate(section_ids):
        mask = (adata.obs[sample_col].astype(str).values == sid)
        if _np.count_nonzero(mask) == 0:
            continue
        X_sub = adata.X[mask, :]
        if _issparse(X_sub):
            row_sums = _np.asarray(X_sub.sum(axis=1)).ravel()
            scale = (libsize_target / _np.maximum(row_sums, 1e-12)).astype(_np.float32)
            X_norm = X_sub.multiply(scale[:, None])
            X_norm.data = _np.log1p(X_norm.data)
            means_per_section[si, :] = _np.asarray(X_norm.mean(axis=0)).ravel()
        else:
            row_sums = X_sub.sum(axis=1, keepdims=True)
            X_norm = X_sub * (libsize_target / _np.maximum(row_sums, 1e-12))
            X_norm = _np.log1p(X_norm)
            means_per_section[si, :] = X_norm.mean(axis=0)

    M = means_per_section

    mean_g = M.mean(axis=0)
    var_g = M.var(axis=0, ddof=1) if S > 1 else _np.zeros(G, dtype=_np.float32)
    sd_g = _np.sqrt(_np.maximum(var_g, 0.0))
    nonzero_sections = (M > 0).sum(axis=0)

    with _np.errstate(divide='ignore', invalid='ignore'):
        ID = _np.where(mean_g > 0, var_g / _np.maximum(mean_g, 1e-12), _np.nan)
        CV2 = _np.where(mean_g > 0, (sd_g / _np.maximum(mean_g, 1e-12)) ** 2, _np.nan)

    ranks = _np.apply_along_axis(rankdata, 0, M).astype(_np.float32)
    z = _np.arange(1, S + 1, dtype=_np.float32)[:, None]
    zc = z - z.mean()
    rc = ranks - ranks.mean(axis=0, keepdims=True)
    num = _np.sum(zc * rc, axis=0)
    den = _np.sqrt(_np.sum(zc ** 2) * _np.sum(rc ** 2, axis=0))
    rho = _np.where(den > 0, num / den, _np.nan)

    x_centered = M - mean_g[None, :]
    prod_sum = _np.sum(x_centered[:-1, :] * x_centered[1:, :], axis=0)
    denom = _np.sum(x_centered ** 2, axis=0)
    W_sum = 2 * (S - 1)
    I = _np.where(denom > 0, (S / W_sum) * (2 * prod_sum) / denom, _np.nan)

    rng = _np.random.default_rng(random_seed)
    more_extreme = _np.zeros(G, dtype=_np.int32)
    abs_I = _np.abs(I)
    for _ in range(permutations):
        perm_idx = rng.permutation(S)
        xp = x_centered[perm_idx, :]
        prod_sum_p = _np.sum(xp[:-1, :] * xp[1:, :], axis=0)
        I_p = _np.where(denom > 0, (S / W_sum) * (2 * prod_sum_p) / denom, _np.nan)
        more_extreme += (_np.abs(I_p) >= abs_I).astype(_np.int32)

    pvals = (more_extreme + 1.0) / (permutations + 1.0)
    qvals = _bh_fdr(pvals)

    keep = (
        (nonzero_sections >= min_sections_expressed)
        & (mean_g > min_global_mean)
        & (I >= moran_min)
        & (qvals <= fdr_max)
    )
    if rho_min is not None:
        keep &= (_np.abs(rho) >= rho_min)
    if cv2_min is not None:
        keep &= (CV2 >= cv2_min)

    order_all = _np.argsort(-_np.nan_to_num(I, nan=-_np.inf))
    keep_indices = _np.where(keep)[0]
    order_keep = order_all[_np.isin(order_all, keep_indices)]
    if order_keep.size >= n_top_genes:
        top_idx = order_keep[:n_top_genes]
    else:
        fill_needed = n_top_genes - order_keep.size
        fallback = order_all[~_np.isin(order_all, order_keep)][:fill_needed]
        top_idx = _np.concatenate([order_keep, fallback])

    return adata.var_names[top_idx].tolist()

def extract_and_embed_sample(
    tiff_path,
    sample_idx,
    adata_sample,
    encoder,
    img_transform,
    device,
    tile_size=224,
    batch_size=128,
    precision=torch.float32,
    coord_scale=15.0,
    rgb_image_path=None,
):
    """
    Extract tiles from OME-TIFF and immediately embed them.
    Returns embeddings, coords, barcodes without saving intermediate tiles.
    """
    print(f"  Processing sample {sample_idx}...")
    
    # Load image data either from an OME-TIFF stack or a standalone RGB image
    if rgb_image_path:
        pil_img = Image.open(rgb_image_path).convert("RGB")
        img = np.array(pil_img)
        print(f"    Sample {sample_idx} loaded from RGB image {rgb_image_path}: shape={img.shape}, dtype={img.dtype}")
    elif tiff_path:
        with tifffile.TiffFile(tiff_path) as tif:
            # For TIFF with shape (52, 3, H, W) stored as 156 pages (52 samples × 3 channels)
            # Pages are organized as: [sample0_ch0, sample0_ch1, sample0_ch2, sample1_ch0, ...]
            page_start = 3 * sample_idx
            page_end = page_start + 3

            if page_end > len(tif.pages):
                raise ValueError(
                    f"Sample index {sample_idx} out of range (needs pages {page_start}-{page_end-1}, but only {len(tif.pages)} pages)"
                )

            # Load the 3 channel pages for this sample
            planes = [tif.pages[i].asarray() for i in range(page_start, page_end)]
            img = np.stack(planes, axis=-1)  # Stack to (H, W, 3)

            print(
                f"    Sample {sample_idx} loaded from pages {page_start}-{page_end-1}: shape={img.shape}, dtype={img.dtype}"
            )
    else:
        raise ValueError("Either tiff_path or rgb_image_path must be provided to extract tiles.")
    
    spatial_coords = adata_sample.obsm['spatial'] * float(coord_scale)
    barcodes = adata_sample.obs_names.values
    n_spots = len(barcodes)
    
    # Add z-index as third coordinate (same for all spots on this slide)
    z_coord = np.full((n_spots, 1), sample_idx, dtype=np.float32)
    spatial_coords_with_z = np.concatenate([spatial_coords, z_coord], axis=1)  # [N, 3]
    
    print(f"    Extracting and embedding {n_spots} tiles (z={sample_idx})...")
    
    half_tile = tile_size // 2
    all_embeddings = []
    
    # Process in batches to manage memory
    for start_idx in tqdm(range(0, n_spots, batch_size), desc="    Batches", leave=False):
        end_idx = min(start_idx + batch_size, n_spots)
        batch_coords = spatial_coords[start_idx:end_idx]
        
        # Extract tiles for this batch
        batch_tiles = []
        for x, y in batch_coords:
            cx, cy = int(round(x)), int(round(y))
            
            x_start = max(0, cx - half_tile)
            y_start = max(0, cy - half_tile)
            x_end = min(img.shape[1], cx + half_tile)
            y_end = min(img.shape[0], cy + half_tile)
            
            tile = img[y_start:y_end, x_start:x_end]
            
            # Pad if needed
            if tile.shape[0] < tile_size or tile.shape[1] < tile_size:
                padded = np.zeros((tile_size, tile_size, 3), dtype=np.uint8)
                pad_y = (tile_size - tile.shape[0]) // 2
                pad_x = (tile_size - tile.shape[1]) // 2
                padded[pad_y:pad_y+tile.shape[0], pad_x:pad_x+tile.shape[1]] = tile
                tile = padded
            
            batch_tiles.append(tile)
        
        # Convert to tensors and apply transforms
        batch_tensors = []
        for tile in batch_tiles:
            tile_pil = Image.fromarray(tile)
            tile_tensor = img_transform(tile_pil)
            batch_tensors.append(tile_tensor)
        
        batch_input = torch.stack(batch_tensors).to(device)
        
        # Embed
        with torch.inference_mode(), torch.cuda.amp.autocast(dtype=precision):
            embeddings = encoder(batch_input)
        
        all_embeddings.append(embeddings.cpu().numpy())
    
    # Concatenate all batches
    embeddings_all = np.concatenate(all_embeddings, axis=0)
    
    return {
        'embeddings': embeddings_all.astype(np.float32),
        'coords': spatial_coords_with_z.astype(np.float32),  # Now includes z as 3rd coordinate
        'barcodes': barcodes
    }


def create_sample_adata(full_adata, sample_id, gene_list, output_path):
    """Create per-sample AnnData filtered by genes."""
    sample_mask = full_adata.obs['sample'] == sample_id
    adata_sample = full_adata[sample_mask, :].copy()
    
    available_genes = [g for g in gene_list if g in adata_sample.var_names]
    adata_sample = adata_sample[:, available_genes].copy()
    
    adata_sample.write_h5ad(output_path)
    return adata_sample

def create_split_csvs(sample_ids, test_sample_ids, output_dir, dataset_name='sample'):
    """Create train/test split CSVs."""
    splits_dir = os.path.join(output_dir, 'splits')
    os.makedirs(splits_dir, exist_ok=True)
    
    test_set = set(test_sample_ids)
    train_samples = [s for s in sample_ids if s not in test_set]
    test_samples = [s for s in sample_ids if s in test_set]
    
    print(f"\nCreating splits: Train={len(train_samples)}, Test={len(test_samples)}")
    
    def make_df(samples):
        return pd.DataFrame({
            'sample_id': [f'{dataset_name}_{s}' for s in samples],
            'patches_path': [f'patches/{dataset_name}_{s}.h5' for s in samples],
            'expr_path': [f'adata/{dataset_name}_{s}.h5ad' for s in samples]
        })
    
    train_df = make_df(train_samples)
    test_df = make_df(test_samples)
    
    train_path = os.path.join(splits_dir, 'train_0.csv')
    test_path = os.path.join(splits_dir, 'test_0.csv')
    
    train_df.to_csv(train_path, index=False)
    test_df.to_csv(test_path, index=False)
    
    print(f"  Saved {train_path}")
    print(f"  Saved {test_path}")


def main():
    parser = argparse.ArgumentParser(
        description='Streamlined preprocessing + embedding for custom datasets',
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    
    # Input/Output
    parser.add_argument('--adata', type=str, required=True,
                        help='Path to input AnnData H5AD file')
    parser.add_argument('--ome_tiff', type=str, default=None,
                        help='Path to multi-sample OME-TIFF file (expects 3 pages per sample)')
    parser.add_argument('--rgb_image', type=str, default=None,
                        help='Path to a standalone RGB image (JPG/PNG/TIFF) for single-sample runs')
    parser.add_argument('--output_dir', type=str, required=True,
                        help='Output directory for dataset files (adata/, splits/, gene lists)')
    parser.add_argument('--embed_dir', type=str, required=True,
                        help='Output directory for embeddings')
    
    # Encoder
    parser.add_argument('--encoder', type=str, default='uni_v1_official',
                        help='Image encoder name')
    parser.add_argument('--weights_root', type=str, default=None,
                        help='Path to model weights (optional)')
    parser.add_argument('--device', type=int, default=0,
                        help='GPU device ID')
    
    # Processing
    parser.add_argument('--n_genes', type=int, default=50,
                        help='Number of genes to export when auto-selecting')
    parser.add_argument('--gene_list_override', type=str, default=None,
                        help='Path to JSON file containing {"genes": [...]} to force a custom gene panel/order')
    parser.add_argument('--test_samples', type=str, default=None,
                        help='Comma-separated test sample indices (0-based positions in sorted sample list, e.g., "0,5,10")')
    parser.add_argument('--dataset_name', type=str, default='sample',
                        help='Dataset/sample name prefix')
    parser.add_argument('--sample_col', type=str, default='sample',
                        help='Column in adata.obs with sample IDs')
    parser.add_argument('--tile_size', type=int, default=224,
                        help='Tile size for extraction')
    parser.add_argument('--batch_size', type=int, default=128,
                        help='Batch size for embedding')
    parser.add_argument('--precision', type=str, default='fp32',
                        choices=['fp16', 'fp32'],
                        help='Precision for embedding')
    parser.add_argument('--reuse_embeddings', action='store_true',
                        help='Reuse existing embeddings if the H5 files already exist')
    parser.add_argument('--coord_scale', type=float, default=15.0,
                        help='Scale applied to spatial coordinates when extracting tiles')
    parser.add_argument('--hvg_flavor', type=str, default='seurat_v3',
                        help='HVG selection method')
    parser.add_argument('--gene_selection', type=str,
                        choices=['hvg', 'spatial', 'intersection', 'z_variation', 'csv'], default='hvg',
                        help='Strategy for gene selection: hvg | spatial | intersection (HVG ∩ spatial) | z_variation | csv')
    parser.add_argument('--gene_selection_csv', type=str, default=None,
                        help='CSV file with columns including "names" that list Ensembl gene IDs when using --gene_selection csv')
    parser.add_argument('--spatial_mode', type=str, default='moran',
                        help='Spatial autocorrelation metric (moran or geary) used when --gene_selection spatial')
    parser.add_argument('--spatial_candidate_genes', type=int, default=5000,
                        help='Number of candidate genes to consider for spatial variability (default: 5000)')
    parser.add_argument('--z_min_sections_expressed', type=int, default=5,
                        help='Min number of sections where gene must be expressed (>0)')
    parser.add_argument('--z_min_global_mean', type=float, default=0.01,
                        help='Min global mean expression across sections')
    parser.add_argument('--z_moran_min', type=float, default=0.25,
                        help="Min Moran's I threshold for z-variation selection")
    parser.add_argument('--z_fdr_max', type=float, default=0.05,
                        help="Max FDR threshold for Moran's I permutations")
    parser.add_argument('--z_rho_min', type=float, default=None,
                        help='Optional |Spearman rho| threshold with section index')
    parser.add_argument('--z_cv2_min', type=float, default=None,
                        help='Optional CV^2 threshold')
    parser.add_argument('--z_permutations', type=int, default=200,
                        help='Number of permutations for Moran I p-values (z-variation)')
    parser.add_argument('--z_seed', type=int, default=0,
                        help='Random seed for z-variation permutations')
    parser.add_argument('--balanced_gene_counts', type=str, default=None,
                        help='Comma-separated extra gene counts to generate balanced lists for (e.g., "50,100,150")')
    parser.add_argument('--normalize_method', type=str, default='log1p',
                        help='Normalization method for expression')
    
    args = parser.parse_args()

    if bool(args.ome_tiff) == bool(args.rgb_image):
        parser.error("Please provide exactly one of --ome_tiff or --rgb_image.")

    if args.ome_tiff:
        args.ome_tiff = str(Path(args.ome_tiff).expanduser())
    if args.rgb_image:
        args.rgb_image = str(Path(args.rgb_image).expanduser())
    
    # Setup
    device = torch.device(f"cuda:{args.device}" if torch.cuda.is_available() else "cpu")
    precision = torch.float16 if args.precision == 'fp16' else torch.float32
    
    output_dir = Path(args.output_dir)
    adata_dir = output_dir / 'adata'
    adata_dir.mkdir(parents=True, exist_ok=True)

    embed_output_dir = Path(args.embed_dir) / args.dataset_name.upper() / args.encoder / args.precision
    embed_output_dir.mkdir(parents=True, exist_ok=True)
    
    print("="*80)
    print("STREAMLINED PREPROCESSING + EMBEDDING PIPELINE")
    print("="*80)
    print(f"Input AnnData:    {args.adata}")
    if args.ome_tiff:
        print(f"Input OME-TIFF:   {args.ome_tiff}")
    else:
        print(f"Input RGB image:  {args.rgb_image}")
    print(f"Dataset output:   {output_dir}")
    print(f"Embedding output: {embed_output_dir}")
    print(f"Encoder:          {args.encoder}")
    print(f"Device:           {device}")
    print()
    
    # ========== Load AnnData ==========
    print("[1/5] Loading AnnData...")
    adata_full = ad.read_h5ad(args.adata)
    print(f"  Shape: {adata_full.shape[0]} spots × {adata_full.shape[1]} genes")
    
    if args.sample_col not in adata_full.obs.columns:
        raise ValueError(f"Column '{args.sample_col}' not in adata.obs")
    
    if 'spatial' not in adata_full.obsm:
        raise ValueError("Missing obsm['spatial']. Run prepare_adata_spatial.py first.")
    
    sample_ids = sorted(adata_full.obs[args.sample_col].unique())
    print(f"  Found {len(sample_ids)} samples")
    
    if args.rgb_image and len(sample_ids) != 1:
        raise ValueError("--rgb_image mode expects the AnnData file to contain exactly one sample.")
    
    # ========== Gene Selection ==========
    print("\n[2/5] Selecting genes...")
    override_genes = None
    if args.gene_list_override:
        override_path = Path(args.gene_list_override)
        if not override_path.exists():
            raise FileNotFoundError(f"--gene_list_override file not found: {override_path}")
        with override_path.open("r") as fh:
            payload = json.load(fh)
        override_genes = payload.get("genes")
        if not isinstance(override_genes, list) or not override_genes:
            raise ValueError(f"Invalid gene list in {override_path}; expected non-empty 'genes' array.")
        if args.n_genes and args.n_genes != len(override_genes):
            print(f"  [warning] --n_genes ({args.n_genes}) differs from override list length ({len(override_genes)}); using override length.")
        args.n_genes = len(override_genes)
        print(f"  Loaded {len(override_genes)} genes from override file (order preserved).")
    balanced_counts = []
    if args.balanced_gene_counts:
        try:
            balanced_counts = sorted({int(x) for x in args.balanced_gene_counts.split(',') if x})
        except ValueError:
            raise ValueError("balanced_gene_counts must be a comma-separated list of integers.")

    if override_genes is not None:
        top_genes = override_genes
        extra_gene_lists = {}
        selection_suffix = "_custom"
    elif args.gene_selection == 'hvg':
        print(f"  Strategy: HVG (flavor={args.hvg_flavor})")
        top_genes = compute_hvgs(adata_full, n_top_genes=args.n_genes, flavor=args.hvg_flavor)
        extra_gene_lists = {}
    elif args.gene_selection == 'csv':
        if not args.gene_selection_csv:
            raise ValueError("When using --gene_selection csv, you must supply --gene_selection_csv with a CSV file path.")
        csv_path = Path(args.gene_selection_csv)
        if not csv_path.exists():
            raise FileNotFoundError(f"--gene_selection_csv file not found: {csv_path}")
        try:
            csv_df = pd.read_csv(csv_path)
        except Exception as exc:
            raise ValueError(f"Failed to read gene CSV '{csv_path}': {exc}") from exc

        names_column = None
        for candidate in ("names", "name"):
            if candidate in csv_df.columns:
                names_column = candidate
                break
        if names_column is None:
            raise ValueError(f"Gene CSV '{csv_path}' must contain a 'names' column with Ensembl gene IDs.")

        raw_series = csv_df[names_column].dropna().astype(str).str.strip()
        raw_series = raw_series[raw_series != ""]
        dedup_series = raw_series.drop_duplicates()
        csv_genes = dedup_series.tolist()

        if not csv_genes:
            raise ValueError(f"No genes found in column '{names_column}' of {csv_path}.")

        if args.n_genes and args.n_genes != len(csv_genes):
            print(f"  [warning] --n_genes ({args.n_genes}) differs from CSV gene count ({len(csv_genes)}); using CSV length.")

        args.n_genes = len(csv_genes)
        top_genes = csv_genes
        extra_gene_lists = {}
        print(f"  Loaded {len(top_genes)} genes from CSV file {csv_path} (column '{names_column}').")
    elif args.gene_selection == 'spatial':
        canonical_mode = _canonicalize_spatial_mode(args.spatial_mode)
        print(f"  Strategy: spatial variability (mode={canonical_mode})")
        top_genes, extra_gene_lists = compute_spatially_variable_genes(
            adata_full,
            n_top_genes=args.n_genes,
            mode=canonical_mode,
            candidate_gene_count=args.spatial_candidate_genes,
            balanced_counts=balanced_counts,
            sample_col=args.sample_col,
        )
    elif args.gene_selection == 'z_variation':
        print("  Strategy: z-variation across ordered sections")
        print(
            f"  Thresholds: min_sections={args.z_min_sections_expressed}, min_mean={args.z_min_global_mean}, "
            f"moran_min={args.z_moran_min}, fdr_max={args.z_fdr_max}, "
            f"rho_min={args.z_rho_min}, cv2_min={args.z_cv2_min}, permutations={args.z_permutations}"
        )
        top_genes = compute_z_variation_genes(
            adata_full,
            n_top_genes=args.n_genes,
            sample_col=args.sample_col,
            min_sections_expressed=args.z_min_sections_expressed,
            min_global_mean=args.z_min_global_mean,
            moran_min=args.z_moran_min,
            fdr_max=args.z_fdr_max,
            rho_min=args.z_rho_min,
            cv2_min=args.z_cv2_min,
            permutations=args.z_permutations,
            random_seed=args.z_seed,
        )
        extra_gene_lists = {}
    else:
        canonical_mode = _canonicalize_spatial_mode(args.spatial_mode)
        print(f"  Strategy: intersection (HVG ∩ spatial), hvg_flavor={args.hvg_flavor}, spatial_mode={canonical_mode}")
        hvgs = compute_hvgs(adata_full, n_top_genes=args.n_genes, flavor=args.hvg_flavor)
        spatial_top, _ = compute_spatially_variable_genes(
            adata_full,
            n_top_genes=args.n_genes,
            mode=canonical_mode,
            candidate_gene_count=args.spatial_candidate_genes,
            balanced_counts=None,
            sample_col=args.sample_col,
        )
        spatial_set = set(spatial_top)
        inter = [g for g in hvgs if g in spatial_set]
        if len(inter) < args.n_genes:
            fill = [g for g in hvgs if g not in spatial_set]
            inter = (inter + fill)[:args.n_genes]
        top_genes = inter
        extra_gene_lists = {}

    if override_genes is None and args.gene_selection != 'spatial' and balanced_counts:
        for count in balanced_counts:
            if count == args.n_genes:
                continue
            extra_gene_lists[count] = top_genes[:min(count, len(top_genes))]

    if override_genes is None:
        selection_suffix = "" if args.gene_selection == "hvg" else f"_{args.gene_selection}"

    gene_sets = {args.n_genes: top_genes}
    for count, genes in extra_gene_lists.items():
        gene_sets[count] = genes

    for count, genes in sorted(gene_sets.items()):
        gene_list_path = output_dir / f'var_{count}genes{selection_suffix}.json'
        with open(gene_list_path, 'w') as f:
            json.dump({'genes': genes}, f, indent=2)
        print(f"  Saved {gene_list_path}")

        adata_subset = adata_full[:, genes].copy()
        mean_expr = np.array(adata_subset.X.mean(axis=0)).flatten().tolist()
        mean_dict = {gene: float(mean_expr[i]) for i, gene in enumerate(genes)}
        mean_path = output_dir / f'mean_{count}genes{selection_suffix}.json'
        with open(mean_path, 'w') as f:
            json.dump({'genes': mean_dict}, f, indent=2)
        print(f"  Saved {mean_path}")
    
    # ========== Load Encoder ==========
    print("\n[3/5] Loading encoder...")
    if args.weights_root:
        encoder, img_transforms, _ = load_encoder(args.encoder, device, args.weights_root, None)
    else:
        # Try to find weights automatically
        script_dir = Path(__file__).parent
        weights_root = script_dir.parent.parent.parent / 'model_weights'
        if weights_root.exists():
            encoder, img_transforms, _ = load_encoder(args.encoder, device, str(weights_root), None)
        else:
            raise ValueError(f"Weights not found. Please specify --weights_root")
    
    encoder.eval()
    print(f"  Loaded {args.encoder}")
    
    # ========== Process Each Sample ==========
    print("\n[4/5] Processing samples (extract + embed + save)...")
    
    normalize_method = get_normalize_method(args.normalize_method)
    
    for i, sample_id in enumerate(tqdm(sample_ids, desc="Samples")):
        sample_name = f'{args.dataset_name}_{sample_id}'
        
        # Filter AnnData for this sample
        sample_mask = adata_full.obs[args.sample_col] == sample_id
        adata_sample = adata_full[sample_mask, :].copy()
        
        embed_h5_path = embed_output_dir / f'{sample_name}.h5'
        adata_path = adata_dir / f'{sample_name}.h5ad'

        if args.reuse_embeddings and embed_h5_path.exists():
            print(f"  Reusing existing embeddings for {sample_name}")
        else:
            try:
                result = extract_and_embed_sample(
                    tiff_path=args.ome_tiff,
                    sample_idx=i,
                    adata_sample=adata_sample,
                    encoder=encoder,
                    img_transform=img_transforms,
                    device=device,
                    tile_size=args.tile_size,
                    batch_size=args.batch_size,
                    precision=precision,
                    coord_scale=args.coord_scale,
                    rgb_image_path=args.rgb_image,
                )

                with h5py.File(embed_h5_path, 'w') as f:
                    f.create_dataset('embeddings', data=result['embeddings'], compression='gzip')
                    f.create_dataset('coords', data=result['coords'], compression='gzip')
                    f.create_dataset('barcodes', data=result['barcodes'].astype('S'))
            except Exception as e:
                print(f"  ERROR embedding sample {sample_id}: {e}")
                continue

        try:
            create_sample_adata(adata_full, sample_id, gene_sets[args.n_genes], str(adata_path))
        except Exception as e:
            print(f"  ERROR writing AnnData for sample {sample_id}: {e}")
            continue
    
    # ========== Create Splits ==========
    print("\n[5/5] Creating train/test splits...")
    
    if args.test_samples:
        # Parse as indices (0-based positions in sample_ids list)
        test_indices = [int(s.strip()) for s in args.test_samples.split(',')]
        
        # Validate indices
        invalid_indices = [idx for idx in test_indices if idx < 0 or idx >= len(sample_ids)]
        if invalid_indices:
            raise ValueError(f"Invalid test sample indices: {invalid_indices}. Valid range: 0-{len(sample_ids)-1}")
        
        # Convert indices to actual sample IDs
        test_ids = [sample_ids[idx] for idx in test_indices]
        print(f"  Test sample indices: {test_indices}")
        print(f"  Test sample IDs: {test_ids}")
    else:
        # Default: use last 20% as test
        n_test = max(1, len(sample_ids) // 5)
        test_ids = sample_ids[-n_test:]
        print(f"  Auto-selected last {n_test} samples for test")
    
    create_split_csvs(
        sample_ids=sample_ids,
        test_sample_ids=test_ids,
        output_dir=str(output_dir),
        dataset_name=args.dataset_name
    )
    
    # ========== Summary ==========
    print("\n" + "="*80)
    print("✓ PIPELINE COMPLETE")
    print("="*80)
    print(f"Processed {len(sample_ids)} samples")
    print()
    print("Output structure:")
    print(f"  {output_dir}/")
    print(f"    ├── adata/              ({len(list(adata_dir.glob('*.h5ad')))} files)")
    print(f"    ├── var_{args.n_genes}genes{selection_suffix}.json")
    print(f"    ├── mean_{args.n_genes}genes{selection_suffix}.json")
    print(f"    └── splits/")
    print(f"        ├── train_0.csv")
    print(f"        └── test_0.csv")
    print()
    print(f"  {embed_output_dir}/")
    print(f"    └── {args.dataset_name}_*.h5  ({len(list(embed_output_dir.glob('*.h5')))} embeddings)")
    print()
    print("Ready for training! Run:")
    print(f"  cd app/flow")
    print(f"  python train.py \\")
    print(f"    --dataset {args.dataset_name.upper()} \\")
    print(f"    --source_dataroot {output_dir.parent}/ \\")
    print(f"    --embed_dataroot {Path(args.embed_dir)}/ \\")
    print(f"    --feature_encoder {args.encoder} \\")
    print(f"    --gene_list var_{args.n_genes}genes{selection_suffix}.json \\")
    print(f"    --n_genes {args.n_genes}")


if __name__ == '__main__':
    main()
