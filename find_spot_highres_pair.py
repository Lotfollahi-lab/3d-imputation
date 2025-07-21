import os
import time
from typing import List, Tuple, Dict

import numpy as np
import pandas as pd
import anndata as ad
from joblib import Parallel, delayed
from tqdm import tqdm


def process_sample_windows(
    sample_id: str,
    coords: np.ndarray,
    image_width: int,
    image_height: int,
    window_size: int
) -> List[Dict]:
    """
    Slide non-overlapping windows over the image for one sample and window size,
    and record windows containing at least one spot.

    Args:
        sample_id (str): Sample name
        coords (np.ndarray): Array of shape (n_spots, 2) with scaled (x, y) coords
        image_width (int): Image width in pixels
        image_height (int): Image height in pixels
        window_size (int): Size of square window (e.g. 512 or 1024)

    Returns:
        List[Dict]: Each dict has keys: 'sample', 'window_size', 'x', 'y'
    """
    hits: List[Dict] = []
    # compute grid of top-left corners
    x_starts = np.arange(0, image_width - window_size + 1, window_size)
    y_starts = np.arange(0, image_height - window_size + 1, window_size)

    for y0 in y_starts:
        # Vectorized mask pre-filter for this row of windows: spots with y in [y0, y0+window_size)
        y_mask = (coords[:, 1] >= y0) & (coords[:, 1] < y0 + window_size)
        if not y_mask.any():
            continue  # no spots in any window on this row
        row_coords = coords[y_mask]
        for x0 in x_starts:
            # check x-range for this window
            mask = (row_coords[:, 0] >= x0) & (row_coords[:, 0] < x0 + window_size)
            if mask.any():
                hits.append({
                    'sample': sample_id,
                    'window_size': window_size,
                    'x': int(x0),
                    'y': int(y0),
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
    Load an AnnData object and, for each sample and each window size,
    find all 512×512 and 1024×1024 windows that contain at least one spot.

    Args:
        adata_path (str): Path to the .h5ad file
        image_width (int): Width of the full image
        image_height (int): Height of the full image
        spatial_scale (float): Scaling factor to apply to raw spatial coords
        window_sizes (List[int]): List of window sizes to scan
        n_jobs (int): Number of parallel jobs for joblib (default: -1 uses all cores)
        output_csv_path (str): Where to save the resulting CSV
    """
    start_time = time.time()
    print("🔬 Loading AnnData...")
    adata = ad.read_h5ad(adata_path)
    print(f"🧬 Loaded {adata.n_obs} spots across {adata.obs['sample'].nunique()} samples")

    # Precompute scaled coords per sample
    sample_coords: Dict[str, np.ndarray] = {}
    for sample in tqdm(adata.obs['sample'].unique(), desc="Extracting sample coords"):
        mask = adata.obs['sample'] == sample
        coords = adata[mask].obsm['spatial'] * spatial_scale
        sample_coords[sample] = np.asarray(coords)

    # Build tasks: one task per (sample, window_size)
    tasks = [
        (sample, coords, image_width, image_height, ws)
        for sample, coords in sample_coords.items()
        for ws in window_sizes
    ]
    print(f"🚀 Launching {len(tasks)} parallel tasks "
          f"({len(sample_coords)} samples × {len(window_sizes)} window sizes)")

    # Parallel execution
    results_nested = Parallel(n_jobs=n_jobs, backend='loky', verbose=5)(
        delayed(process_sample_windows)(*task) for task in tasks
    )

    # Flatten and save
    all_hits = [hit for sublist in results_nested for hit in sublist]
    df = pd.DataFrame(all_hits, columns=['sample', 'window_size', 'x', 'y'])
    os.makedirs(os.path.dirname(output_csv_path), exist_ok=True)
    df.to_csv(output_csv_path, index=False)

    elapsed = time.time() - start_time
    total_windows = sum(
        ((image_width - ws) // ws + 1) * ((image_height - ws) // ws + 1)
        for ws in window_sizes
        for _ in sample_coords
    )
    print("✅ Done!")
    print(f"   Total windows scanned: {total_windows}")
    print(f"   Windows with ≥1 spot: {len(df)}")
    print(f"   Coverage: {100 * len(df) / total_windows:.2f}%")
    print(f"   Results written to: {output_csv_path}")
    print(f"   Elapsed time: {elapsed:.1f}s")


def example_usage():
    IMAGE_WIDTH = 30000
    IMAGE_HEIGHT = 30000
    ADATA_PATH = '/nfs/team361/data/sanger_whole_embryo/adata_CS17.h5ad'
    OUTPUT_CSV = './embryo_data/spot_windows_hits.csv'
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
