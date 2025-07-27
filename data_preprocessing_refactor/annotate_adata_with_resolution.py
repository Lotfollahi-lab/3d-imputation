import os
import time
from typing import Dict, List, Optional

import pandas as pd
import anndata as ad


def annotate_adata_with_containment(
    adata_path: str,
    windows_csv: str,
    output_adata_path: str
) -> None:
    """
    Load an AnnData and a CSV of patch windows. For each obs in the AnnData:
      - If size==224, find the 512×512 patch that contains it (by checking df size=512 spot_ids),
        and the 1024×1024 patch that contains that 512 patch (df size=1024 spot_ids).
      - Otherwise, set containing_512 and containing_1024 to None.
    Adds two obs columns:
      'containing_512' and 'containing_1024'
    Saves the updated AnnData.

    Args:
        adata_path: Path to input .h5ad (must have obs['size'] and obs_names as 224 spot_ids)
        windows_csv: CSV with columns ['sample','size','x','y','spot_id','spot_ids']
        output_adata_path: Path to write updated .h5ad
    """
    t0 = time.time()
    print("🔬 Loading AnnData and CSV…")
    adata = ad.read_h5ad(adata_path)
    df = pd.read_csv(windows_csv)

    # Split into 512 and 1024 windows
    df512 = df[df['size'] == 512].copy()
    df512['spot_list'] = df512['spot_ids'].str.split(';')
    df1024 = df[df['size'] == 1024].copy()
    df1024['patch512_list'] = df1024['spot_ids'].str.split(';')

    # Build containment maps
    # 224 spot_id -> single 512 patch_id
    spot224_to_512: Dict[str, str] = {}
    for _, row in df512.iterrows():
        pid512 = row['spot_id']
        for sid224 in row['spot_list']:
            if sid224 not in spot224_to_512:
                spot224_to_512[sid224] = pid512
            else:
                raise ValueError(f"Spot {sid224} already in spot224_to_512")
    # 512 patch_id -> single 1024 patch_id
    pid512_to_1024: Dict[str, str] = {}
    for _, row in df1024.iterrows():
        pid1024 = row['spot_id']
        for pid512 in row['patch512_list']:
            if pid512 not in pid512_to_1024:
                pid512_to_1024[pid512] = pid1024
            else:
                raise ValueError(f"Patch {pid512} already in pid512_to_1024")
    # Prepare new obs columns
    containing_512: List[Optional[str]] = []
    containing_1024: List[Optional[str]] = []

    print("🔎 Annotating each spot…")
    for row_idx, row in adata.obs.iterrows():
        size = row.get('size')
        sid = row.get('spot_id')
        if size == 224:
            c512 = spot224_to_512.get(sid)
            c1024 = pid512_to_1024.get(c512) if c512 is not None else None
        else:
            c512 = None
            c1024 = None
        containing_512.append(c512)
        containing_1024.append(c1024)

    # Assign to obs
    adata.obs['containing_512'] = containing_512
    adata.obs['containing_1024'] = containing_1024

    # Save
    print(f"💾 Saving annotated AnnData to {output_adata_path}…")
    os.makedirs(os.path.dirname(output_adata_path), exist_ok=True)
    adata.write_h5ad(output_adata_path)
    print(f"✅ Done in {time.time() - t0:.1f}s")


def example_usage():
    annotate_adata_with_containment(
        adata_path='../adata_with_features.h5ad',
        windows_csv='../embryo_data/spot_windows_hits_hierarchy.csv',
        output_adata_path='../adata_with_features.h5ad'
    )

if __name__ == '__main__':
    example_usage()
