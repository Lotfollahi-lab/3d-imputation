import os
import time
from typing import Dict, List

import numpy as np
import pandas as pd
import anndata as ad


def aggregate_counts_to_adata(
    adata1_path: str,
    windows_csv: str,
    adata2_path: str,
    output_adata_path: str,
    layer: str = "X"
) -> None:
    """
    1) Load adata1 (spots × genes) and windows CSV.
    2) Compute per-spot counts_224, counts_512, counts_1024.
    3) Load adata2, copy adata1.var → adata2.var.
    4) Attach counts_* to adata2.obsm, merge extra CSV columns into adata2.obs.
    5) Save updated adata2.
    """
    t0 = time.time()
    print("🔬 Loading data...")
    adata1 = ad.read_h5ad(adata1_path)
    df = pd.read_csv(windows_csv)
    adata2 = ad.read_h5ad(adata2_path)

    # Sanity: adata2.obs must include a 'spot_id' column matching adata1.obs_names
    if 'spot_id' not in adata2.obs.columns:
        raise ValueError("adata2.obs must have a 'spot_id' column to match spots")
    spot_index = list(adata2.obs['spot_id'].astype(str))
    n_spots = len(spot_index)

    print("📊 Computing counts_224 (raw per-spot counts)...")
    # Convert X to dense array if sparse
    X1 = adata1.X.toarray() if hasattr(adata1.X, "toarray") else adata1.X
    # Build counts_224 in the order of spot_index
    idx_map = {sid: i for i, sid in enumerate(adata1.obs_names)}
    order = [idx_map[sid] for sid in spot_index]
    counts_224 = X1[order, :]  # shape (n_spots, n_genes)

    print("📦 Aggregating 512×512 patch sums...")
    df512 = df[df['window_size'] == 512].copy()
    # parse spot_ids list
    df512['spot_list'] = df512['spot_ids'].str.split(';')
    # compute patch512 → summed vector
    patch512_counts: Dict[str, np.ndarray] = {}
    for _, row in df512.iterrows():
        spots = row['spot_list']
        indices = [idx_map[s] for s in spots]
        patch512_counts[row['patch_id']] = X1[indices, :].sum(axis=0)

    # map each spot to its 512 patch
    spot_to_512patch: Dict[str, str] = {}
    for _, row in df512.iterrows():
        pid = row['patch_id']
        for s in row['spot_list']:
            spot_to_512patch[s] = pid

    # build counts_512 per spot in adata2 order
    counts_512 = np.vstack([
        patch512_counts[spot_to_512patch[sid]]
        for sid in spot_index
    ])

    print("📦 Aggregating 1024×1024 patch sums (by grouping 512 patches)...")
    df1024 = df[df['window_size'] == 1024].copy()
    # here spot_ids column holds semicolon list of 512-patch_ids
    df1024['patch512_list'] = df1024['spot_ids'].str.split(';')
    # compute patch1024 → summed vector
    patch1024_counts: Dict[str, np.ndarray] = {}
    for _, row in df1024.iterrows():
        p512 = row['patch512_list']
        # sum the precomputed 512 counts
        mats = [patch512_counts[pid] for pid in p512]
        patch1024_counts[row['patch_id']] = np.vstack(mats).sum(axis=0)

    # map each spot to its 1024 patch
    # we need the reverse mapping: each 1024 patch contains many 512 patches, each 512 has spots
    # build a 512patch→spot_list from df512
    patch512_to_spots = {row['patch_id']: row['spot_list'] for _, row in df512.iterrows()}
    # now for each 1024 patch, collect all spots
    spot_to_1024patch: Dict[str, str] = {}
    for _, row in df1024.iterrows():
        pid1024 = row['patch_id']
        sub512 = row['patch512_list']
        for pid512 in sub512:
            for s in patch512_to_spots.get(pid512, []):
                spot_to_1024patch[s] = pid1024

    # build counts_1024 per spot in adata2 order
    counts_1024 = np.vstack([
        patch1024_counts[spot_to_1024patch[sid]]
        for sid in spot_index
    ])

    # Copy var from adata1 → adata2
    print("🔧 Copying var (gene metadata)...")
    adata2.var = adata1.var.copy()

    # Attach counts to adata2.obsm
    print("📥 Attaching counts to adata2.obsm...")
    adata2.obsm['counts_224'] = counts_224
    adata2.obsm['counts_512'] = counts_512
    adata2.obsm['counts_1024'] = counts_1024

    # Finally, merge any extra CSV columns into adata2.obs
    print("🧩 Merging extra CSV columns into adata2.obs...")
    # columns in df not in obs
    extras = [c for c in df.columns if c not in adata2.obs.columns and c not in
              ('spot_ids','patch_id','window_size','x','y')]
    if extras:
        # need a mapping from spot_id → row(s) in df512 & df1024
        # we’ll merge per-spot, using spot_to_512patch and spot_to_1024patch
        merge_df = pd.DataFrame({'spot_id': spot_index})
        for col in extras:
            # choose value from df512 if present, else df1024
            def lookup(sid):
                pid512 = spot_to_512patch.get(sid)
                row512 = df512[df512['patch_id']==pid512]
                if not row512.empty and col in row512:
                    return row512.iloc[0][col]
                pid1024 = spot_to_1024patch.get(sid)
                row1024 = df1024[df1024['patch_id']==pid1024]
                if not row1024.empty and col in row1024:
                    return row1024.iloc[0][col]
                return None
            merge_df[col] = merge_df['spot_id'].map(lookup)
        # join into adata2.obs (on spot_id)
        adata2.obs = adata2.obs.merge(merge_df.drop(columns=[]),
                                      on='spot_id', how='left', right_index=False)

    # Save
    print(f"💾 Saving updated AnnData to {output_adata_path}…")
    os.makedirs(os.path.dirname(output_adata_path), exist_ok=True)
    adata2.write_h5ad(output_adata_path)

    print(f"✅ Done in {time.time()-t0:.1f}s")
