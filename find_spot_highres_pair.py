import os
import time
from typing import List, Dict, Tuple

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
) -> List[Dict]:
    """
    Slide non‐overlapping windows over one sample’s image slice,
    record windows containing ≥1 spot, along with those spot_ids and a patch_id.

    Args:
        sample_id    (str): Sample name
        coords        (np.ndarray): (n_spots,2) scaled (x,y) coords
        spot_ids      (List[str]): length‐n_spots list of spot obs indices
        image_width  (int): Image width in px
        image_height (int): Image height in px
        window_size  (int): Square window size in px

    Returns:
        List[Dict]: Each dict has:
          - sample
          - window_size
          - x, y (top‐left of window)
          - patch_id (e.g. "SampleA_512_0_0")
          - spot_ids (list of obs indices in that window)
    """
    hits: List[Dict] = []
    # grid of top‐left corners
    x_starts = np.arange(0, image_width - window_size + 1, window_size)
    y_starts = np.arange(0, image_height - window_size + 1, window_size)

    # for each window, test full mask
    for y0 in y_starts:
        # quick y‐range filter
        y_mask = (coords[:,1] >= y0) & (coords[:,1] < y0 + window_size)
        if not y_mask.any():
            continue
        for x0 in x_starts:
            # full 2D mask for this window
            mask = (
                (coords[:,0] >= x0) &
                (coords[:,0] < x0 + window_size) &
                y_mask
            )
            if mask.any():
                ids_in_window = [spot_ids[i] for i in np.where(mask)[0]]
                patch_id = f"{sample_id}_{window_size}_{x0}_{y0}"
                hits.append({
                    'sample':       sample_id,
                    'window_size':  window_size,
                    'x':            int(x0),
                    'y':            int(y0),
                    'patch_id':     patch_id,
                    'spot_ids':     ids_in_window
                })
    return hits


def find_spot_windows(
    adata_path: str,
    image_width: int,
    image_height: int,
    spatial_scale: float,
    window_sizes: List[int] = [512, 1024],
    n_jobs: int = -1,
    output_csv_path: str = './spot_windows.csv'
) -> None:
    """
    Load AnnData, slide windows of given sizes over each sample’s spatial coords,
    record windows with spots, including their spot_ids and a patch_id, then save CSV.

    The CSV will have columns:
      sample, window_size, x, y, patch_id, spot_ids
    where spot_ids is a semicolon‐separated list.
    """
    t0 = time.time()
    print("🔬 Loading AnnData...")
    adata = ad.read_h5ad(adata_path)
    samples = adata.obs['sample'].unique()
    print(f"🧬 {adata.n_obs} spots across {len(samples)} samples")

    # build per‐sample coords and spot_ids
    sample_coords: Dict[str, np.ndarray] = {}
    sample_spot_ids: Dict[str, List[str]] = {}
    for sample in tqdm(samples, desc="Extracting sample coords"):
        mask = (adata.obs['sample'] == sample).values
        coords = adata.obsm['spatial'][mask] * spatial_scale
        sample_coords[sample] = np.asarray(coords)
        sample_spot_ids[sample] = adata.obs.index[mask].tolist()

    # prepare parallel tasks
    tasks = []
    for sample in samples:
        coords = sample_coords[sample]
        ids = sample_spot_ids[sample]
        for ws in window_sizes:
            tasks.append((sample, coords, ids, image_width, image_height, ws))

    print(f"🚀 Launching {len(tasks)} tasks "
          f"({len(samples)} samples × {len(window_sizes)} window sizes)")

    # run in parallel
    results = Parallel(n_jobs=n_jobs, backend='loky', verbose=5)(
        delayed(process_sample_windows)(*task) for task in tasks
    )

    # flatten and build DataFrame
    all_hits = [hit for sub in results for hit in sub]
    df = pd.DataFrame(all_hits)
    # convert list to semicolon‐separated string for CSV
    df['spot_ids'] = df['spot_ids'].apply(lambda lst: ";".join(lst))

    os.makedirs(os.path.dirname(output_csv_path), exist_ok=True)
    df.to_csv(output_csv_path, index=False)

    elapsed = time.time() - t0
    total_wins = sum(
        ((image_width - ws)//ws + 1) * ((image_height - ws)//ws + 1)
        for ws in window_sizes
    ) * len(samples)

    print("✅ Done!")
    print(f"   Total windows scanned: {total_wins}")
    print(f"   Windows with ≥1 spot: {len(df)}")
    print(f"   Coverage: {100*len(df)/total_wins:.2f}%")
    print(f"   Results written to: {output_csv_path}")
    print(f"   Elapsed time: {elapsed:.1f}s")


def example_usage():
    IMAGE_WIDTH = 30000
    IMAGE_HEIGHT = 30000
    ADATA_PATH = '/nfs/team361/data/sanger_whole_embryo/adata_CS17.h5ad'
    OUTPUT_CSV = './embryo_data/spot_windows_hits_new.csv'
    find_spot_windows(
        adata_path=ADATA_PATH,
        image_width=IMAGE_WIDTH,
        image_height=IMAGE_HEIGHT,
        spatial_scale=15.0,
        window_sizes=[512, 1024],
        n_jobs=-1,
        output_csv_path=OUTPUT_CSV
    )


if __name__ == "__main__":
    example_usage()
