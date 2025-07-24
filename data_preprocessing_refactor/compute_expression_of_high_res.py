import os
import time
from typing import List, Dict, Any

import numpy as np
import pandas as pd
import anndata as ad


def aggregate_patch_counts(
    adata_path: str,
    windows_csv: str,
    output_adata_path: str,
    count_layer: str = 'X'
) -> None:
    """
    Load an AnnData (adata_path) and a CSV of patch definitions (windows_csv).  
    For 512-sized patches:
      - parse 'spot_ids' as semicolon-separated list of adata.obs_names
      - sum counts (adata.layers[count_layer] or .X) over those spots
    For 1024-sized patches:
      - parse 'spot_ids' as list of 512-patch_ids
      - sum the previously computed 512-patch counts
    Build a new AnnData where each obs is one patch (512 or 1024),
      .X holds the aggregated counts (shape n_patches × n_genes),
      .var is copied from the input adata,
      .obs contains all columns from the CSV plus 'count' and 'patch_id', 'size'.
    Save to output_adata_path.
    """
    t0 = time.time()
    print("🔬 Loading AnnData and CSV...")
    adata = ad.read_h5ad(adata_path)
    df = pd.read_csv(windows_csv)

    # Fetch count matrix
    if count_layer in adata.layers:
        Xmat = adata.layers[count_layer]
    else:
        Xmat = adata.X
    # densify if sparse
    Xmat = Xmat.toarray() if hasattr(Xmat, 'toarray') else Xmat

    # Prepare 512 patches
    print("📦 Processing 512-sized patches...")
    df512 = df[df['size'] == 512].copy()
    df512['spot_list'] = df512['spot_ids'].str.split(';')
    # predefine 'count' as object dtype so we can store lists

    patch512_counts: Dict[str, np.ndarray] = {}
    for idx, row in df512.iterrows():
        spots = row['spot_list']  # list of spot_ids
        # map to adata row indices
        indices = [adata.obs_names.get_loc(s) for s in spots if s in adata.obs_names]
        if not indices:
            vec = np.zeros(adata.n_vars)
        else:
            vec = Xmat[indices, :].sum(axis=0)
        patch512_counts[row['spot_id']] = vec
        # store the raw list of counts per gene

    # Prepare 1024 patches
    print("📦 Processing 1024-sized patches...")
    df1024 = df[df['size'] == 1024].copy()
    df1024['patch512_list'] = df1024['spot_ids'].str.split(';')
    # predefine 'count' as object dtype

    patch1024_counts: Dict[str, np.ndarray] = {}
    for idx, row in df1024.iterrows():
        pids = row['patch512_list']  # list of 512-patch_ids
        mats = [patch512_counts[pid] for pid in pids if pid in patch512_counts]
        if not mats:
            vec = np.zeros(adata.n_vars)
        else:
            vec = np.vstack(mats).sum(axis=0)
        patch1024_counts[row['spot_id']] = vec

    # Concatenate df512 + df1024
    df_all = pd.concat([df512, df1024], axis=0, ignore_index=True)

    # Build data matrix for new AnnData
    print("🔗 Building new AnnData...")
    counts_list: List[np.ndarray] = []
    for pid in df_all['spot_id']:
        if pid in patch512_counts:
            counts_list.append(patch512_counts[pid])
        else:
            counts_list.append(patch1024_counts[pid])
    X_new = np.vstack(counts_list)  # shape n_patches × n_genes

    # Create new AnnData
    ad_new = ad.AnnData(X=X_new)
    # copy var (gene info)
    ad_new.var = adata.var.copy()
    # copy obs from df_all (all CSV columns, including 'count','patch_id','size','spot_ids', etc.)
    ad_new.obs = df_all.drop(columns=['spot_list', 'patch512_list'], errors='ignore')
    #save the spatial coordinates
    ad_new.obsm['spatial'] = ad_new.obs[['x', 'y']].to_numpy()
    # Save
    print(f"💾 Saving aggregated AnnData to {output_adata_path}...")
    os.makedirs(os.path.dirname(output_adata_path), exist_ok=True)
    ad_new.write_h5ad(output_adata_path)
    print(f"✅ Done in {time.time() - t0:.1f}s: {ad_new.n_obs} patches × {ad_new.n_vars} genes.")


def example_usage():
    ADATA1 = "filtered_adata.h5ad"
    WINDOWS_CSV = "./embryo_data/spot_windows_hits_hierarchy.csv"
    OUTPUT = "./adata_counts_hierarchy.h5ad"
    aggregate_patch_counts(
        adata_path=ADATA1,
        windows_csv=WINDOWS_CSV,
        output_adata_path=OUTPUT
    )


if __name__ == "__main__":
    example_usage()
