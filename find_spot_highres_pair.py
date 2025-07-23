import os
import time
from typing import List, Dict, Any

import numpy as np
import pandas as pd
import anndata as ad
from joblib import Parallel, delayed
from tqdm import tqdm


def process_sample_windows(
    sample_id: str,
    coords: np.ndarray,
    spot_ids: List[str],
    image_width: int,
    image_height: int,
    window_size: int
) -> List[Dict[str, Any]]:
    """
    Slide non-overlapping windows of `window_size` over one sample’s slice,
    returning windows that contain ≥1 item, along with those spot_ids and a patch_id.

    `spot_ids` can be spot IDs (for 512) or 512-patch IDs (for 1024).
    """
    hits: List[Dict[str, Any]] = []
    x_starts = np.arange(0, image_width - window_size + 1, window_size)
    y_starts = np.arange(0, image_height - window_size + 1, window_size)

    for y0 in y_starts:
        y_mask = (coords[:,1] >= y0) & (coords[:,1] < y0 + window_size)
        if not y_mask.any():
            continue
        for x0 in x_starts:
            mask = (
                (coords[:,0] >= x0) &
                (coords[:,0] < x0 + window_size) &
                y_mask
            )
            if mask.any():
                contained = [spot_ids[i] for i in np.where(mask)[0]]
                patch_id = f"{sample_id}_{window_size}_{x0}_{y0}"
                hits.append({
                    'sample':      sample_id,
                    'window_size': window_size,
                    'x':           int(x0),
                    'y':           int(y0),
                    'patch_id':    patch_id,
                    'spot_ids':    contained
                })
    return hits


def find_spot_windows(
    adata_path: str,
    image_width: int,
    image_height: int,
    spatial_scale: float,
    n_jobs: int = -1,
    output_csv_path: str = './spot_windows.csv'
) -> None:
    """
    1) Compute 512×512 windows over raw spots → get spot-based patch_ids.
    2) Compute 1024×1024 windows over 512-patch grid → get lists of 512-patch patch_ids.
    3) Save both to CSV with columns: sample, window_size, x, y, patch_id, spot_ids.
    """
    t0 = time.time()
    print("🔬 Loading AnnData...")
    adata = ad.read_h5ad(adata_path)
    samples = adata.obs['sample'].unique()
    print(f"🧬 {adata.n_obs} spots across {len(samples)} samples")

    # build per-sample spot coords & ids
    sample_spots = {}
    sample_spot_ids = {}
    for sample in tqdm(samples, desc="Extracting spot coords"):
        mask = adata.obs['sample'] == sample
        coords = adata.obsm['spatial'][mask] * spatial_scale
        sample_spots[sample] = np.asarray(coords)
        sample_spot_ids[sample] = adata.obs.index[mask].tolist()

    # Step 1: 512-window over spots
    tasks512 = [
        (s, sample_spots[s], sample_spot_ids[s], image_width, image_height, 512)
        for s in samples
    ]
    print(f"🚀 Running {len(tasks512)} parallel tasks for 512×512...")
    results512 = Parallel(n_jobs=n_jobs, backend='loky', verbose=5)(
        delayed(process_sample_windows)(*t) for t in tasks512
    )
    hits512 = [h for sub in results512 for h in sub]

    # prepare 512 patch grid coords & ids for step 2
    sample_512_coords = {}
    sample_512_ids = {}
    for sample in samples:
        sub = [h for h in hits512 if h['sample'] == sample]
        coords512 = np.array([[h['x'], h['y']] for h in sub])
        ids512 = [h['patch_id'] for h in sub]
        sample_512_coords[sample] = coords512
        sample_512_ids[sample] = ids512

    # Step 2: 1024-window over 512 patches
    tasks1024 = [
        (s, sample_512_coords[s], sample_512_ids[s], image_width, image_height, 1024)
        for s in samples
    ]
    print(f"🚀 Running {len(tasks1024)} parallel tasks for 1024×1024...")
    results1024 = Parallel(n_jobs=n_jobs, backend='loky', verbose=5)(
        delayed(process_sample_windows)(*t) for t in tasks1024
    )
    hits1024 = [h for sub in results1024 for h in sub]

    # Combine and output
    all_hits = hits512 + hits1024
    df = pd.DataFrame(all_hits)
    # list -> semicolon string
    df['spot_ids'] = df['spot_ids'].apply(lambda lst: ";".join(lst))

    os.makedirs(os.path.dirname(output_csv_path), exist_ok=True)
    df.to_csv(output_csv_path, index=False)

    elapsed = time.time() - t0
    print("✅ Done!")
    print(f"   512 windows: {len(hits512)}")
    print(f"  1024 windows: {len(hits1024)}")
    print(f"   Saved to {output_csv_path}")
    print(f"   Elapsed {elapsed:.1f}s")


def example_usage():
    IMAGE_WIDTH = 30000
    IMAGE_HEIGHT = 30000
    ADATA_PATH = '/nfs/team361/data/sanger_whole_embryo/adata_CS17.h5ad'
    OUTPUT_CSV = './embryo_data/spot_windows_hits_hierarchy.csv'
    find_spot_windows(
        adata_path=ADATA_PATH,
        image_width=IMAGE_WIDTH,
        image_height=IMAGE_HEIGHT,
        spatial_scale=15.0,
        n_jobs=-1,
        output_csv_path=OUTPUT_CSV
    )


if __name__ == "__main__":
    example_usage()
