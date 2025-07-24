import numpy as np
import pandas as pd
import anndata as ad
from shapely.geometry import Polygon, Point
from tqdm import tqdm
import math
import os
import time
from typing import Tuple, List, Dict
from joblib import Parallel, delayed


def create_circle_polygon(center_x: float, center_y: float, radius: float, num_points: int = 100) -> Polygon:
    """
    Create a circular polygon around a given center point.
    
    Args:
        center_x (float): X coordinate of the circle center
        center_y (float): Y coordinate of the circle center
        radius (float): Radius of the circle
        num_points (int): Number of points to approximate the circle (default: 100)
    
    Returns:
        Polygon: Shapely polygon representing the circle
    """
    angles = np.linspace(0, 2 * np.pi, num_points, endpoint=False)
    points = [(center_x + radius * np.cos(angle), center_y + radius * np.sin(angle)) for angle in angles]
    return Polygon(points)


def is_iou_zero(center1: Tuple[float, float], r1: float, center2: Tuple[float, float], r2: float) -> bool:
    """
    Fast check if IoU between two circles would be zero (no overlap).
    
    Args:
        center1 (Tuple[float, float]): Center coordinates of first circle
        r1 (float): Radius of first circle
        center2 (Tuple[float, float]): Center coordinates of second circle
        r2 (float): Radius of second circle
    
    Returns:
        bool: True if IoU would be zero (no overlap), False otherwise
    """
    dx = center1[0] - center2[0]
    dy = center1[1] - center2[1]
    # Using squared distance to avoid sqrt calculation
    distance_squared = dx*dx + dy*dy
    threshold_squared = (r1 + r2) * (r1 + r2)
    return distance_squared >= threshold_squared


def compute_iou_pair(poly1: Polygon, poly2: Polygon) -> float:
    """
    Compute the IoU (Intersection over Union) of two polygons.
    
    Args:
        poly1 (Polygon): First polygon
        poly2 (Polygon): Second polygon
    
    Returns:
        float: IoU score (0 to 1)
    """
    try:
        inter_area = poly1.intersection(poly2).area
        union_area = poly1.union(poly2).area
        return inter_area / union_area if union_area > 0 else 0.0
    except Exception as e:
        print(f"Warning: IoU computation failed: {e}")
        return 0.0


def compute_iou_pair_optimized(coord1: Tuple[float, float], coord2: Tuple[float, float], 
                              radius: float) -> float:
    """
    Optimized IoU computation with fast zero-check.
    
    Args:
        coord1 (Tuple[float, float]): Center coordinates of first circle
        coord2 (Tuple[float, float]): Center coordinates of second circle
        radius (float): Radius for both circles
    
    Returns:
        float: IoU score (0 to 1)
    """
    # Fast check: if circles don't overlap, IoU = 0
    if is_iou_zero(coord1, radius, coord2, radius):
        return 0.0
    
    # If circles might overlap, compute actual IoU using polygons
    poly1 = create_circle_polygon(coord1[0], coord1[1], radius)
    poly2 = create_circle_polygon(coord2[0], coord2[1], radius)
    
    return compute_iou_pair(poly1, poly2)


def extract_spots_from_sample(adata: ad.AnnData, sample_id: str, spatial_scale: float = 15.0) -> Dict:
    """
    Extract spot information for a specific sample from AnnData object.
    
    Args:
        adata (AnnData): AnnData object containing spatial transcriptomics data
        sample_id (str): Sample identifier to filter spots
        spatial_scale (float): Scaling factor for spatial coordinates (default: 15.0)
    
    Returns:
        Dict: Dictionary containing spot IDs, coordinates, and polygons
    """
    # Filter spots for the specific sample
    sample_mask = adata.obs['sample'] == sample_id
    sample_adata = adata[sample_mask].copy()
    
    spot_info = {
        'spot_ids': [],
        'coordinates': [],
        'polygons': []
    }
    
    for i, spot_id in enumerate(sample_adata.obs.index):
        # Get spatial coordinates and scale them
        x_coord = sample_adata.obsm['spatial'][i, 0] * spatial_scale
        y_coord = sample_adata.obsm['spatial'][i, 1] * spatial_scale
        
        spot_info['spot_ids'].append(spot_id)
        spot_info['coordinates'].append((x_coord, y_coord))
    
    return spot_info

def compute_iou_chunk_vectorized(spots1_chunk: Dict, spots2_data: Dict, radius: float) -> List[Tuple]:
    """
    Vectorized IoU computation for a chunk of spots using numpy operations.
    
    Args:
        spots1_chunk (Dict): Chunk of spots from first sample
        spots2_data (Dict): All spots from second sample  
        radius (float): Circle radius
    
    Returns:
        List[Tuple]: List of (spot1_id, spot2_id, iou_score) tuples
    """
    results = []
    
    # Convert to numpy arrays for vectorized operations
    coords1 = np.array(spots1_chunk['coordinates'])
    coords2 = np.array(spots2_data['coordinates'])
    
    # Vectorized distance computation for all pairs in this chunk
    for i, (spot1_id, coord1) in enumerate(zip(spots1_chunk['spot_ids'], coords1)):
        # Vectorized distance calculation to all spots in sample 2
        dx = coords2[:, 0] - coord1[0]
        dy = coords2[:, 1] - coord1[1]
        distances = np.sqrt(dx*dx + dy*dy)
        
        # Find potentially overlapping spots (distance < 2*radius)
        overlap_mask = distances < (2 * radius)
        overlapping_indices = np.where(overlap_mask)[0]
        
        # Compute IoU only for potentially overlapping spots
        for j in overlapping_indices:
            spot2_id = spots2_data['spot_ids'][j]
            coord2 = tuple(coords2[j])
            
            # Fast distance check
            if not is_iou_zero(coord1, radius, coord2, radius):
                iou_score = compute_iou_pair_optimized(coord1, coord2, radius)
                if iou_score > 0.0:
                    results.append((spot1_id, spot2_id, iou_score))
    
    return results


def compute_iou_between_samples(spots1: Dict, spots2: Dict, radius: float, n_jobs: int = -1, chunk_size: int = 100) -> List[Tuple]:
    """
    Compute IoU between all spot pairs from two samples using chunked vectorized parallel processing.
    
    Args:
        spots1 (Dict): Spot information from first sample
        spots2 (Dict): Spot information from second sample
        radius (float): Radius for circle creation around each spot
        n_jobs (int): Number of parallel jobs
        chunk_size (int): Number of spots to process in each chunk
    
    Returns:
        List[Tuple]: List of (spot1_id, spot2_id, iou_score) tuples
    """
    n_spots1 = len(spots1['spot_ids'])
    
    # Create chunks of spots1 for parallel processing
    chunks = []
    for i in range(0, n_spots1, chunk_size):
        end_idx = min(i + chunk_size, n_spots1)
        chunk = {
            'spot_ids': spots1['spot_ids'][i:end_idx],
            'coordinates': spots1['coordinates'][i:end_idx]
        }
        chunks.append(chunk)
    
    print(f"    🔢 Processing {len(chunks)} chunks of ~{chunk_size} spots each")
    
    # Parallel computation using chunked vectorized approach
    results_nested = Parallel(n_jobs=n_jobs, backend='threading', verbose=0)(
        delayed(compute_iou_chunk_vectorized)(chunk, spots2, radius) 
        for chunk in tqdm(chunks, desc="Processing chunks", leave=False)
    )
    
    # Flatten results
    all_results = []
    for chunk_results in results_nested:
        all_results.extend(chunk_results)
    
    total_possible = n_spots1 * len(spots2['spot_ids'])
    print(f"    📊 Non-zero IoU pairs: {len(all_results)}/{total_possible} ({100*len(all_results)/total_possible:.1f}%)")
    
    return all_results


def compute_sample_pair_iou(args: Tuple) -> List[Dict]:
    """
    Compute IoU for a single sample pair (for parallel processing of sample pairs).
    
    Args:
        args (Tuple): (sample1, sample2, sample_spots, radius, n_jobs_inner, chunk_size)
    
    Returns:
        List[Dict]: List of IoU result dictionaries
    """
    sample1, sample2, sample_spots, radius, n_jobs_inner, chunk_size = args
    
    start_time = time.time()
    print(f"🔍 Processing pair: {sample1} vs {sample2}")
    
    # Compute IoU between the two samples
    pair_results = compute_iou_between_samples(
        sample_spots[sample1], 
        sample_spots[sample2], 
        radius,
        n_jobs_inner,
        chunk_size
    )
    
    elapsed_time = time.time() - start_time
    print(f"    ⏱️  Completed in {elapsed_time:.2f} seconds")
    
    # Add sample information to results
    results_with_samples = []
    for spot1_id, spot2_id, iou_score in pair_results:
        results_with_samples.append({
            'spot1_id': spot1_id,
            'spot2_id': spot2_id,
            'sample1': sample1,
            'sample2': sample2,
            'iou_score': iou_score
        })
    
    return results_with_samples


def compute_iou(adata_path: str, output_csv_path: str, radius: float = 112.0, 
                spatial_scale: float = 15.0, n_jobs: int = -1, chunk_size: int = 100,
                parallel_samples: bool = True) -> None:
    """
    Compute IoU scores between spots across different samples/layers from h5ad data.
    
    This function loads an AnnData object and computes IoU scores between circular 
    regions around each spot across all sample pairs using optimized parallel processing.
    
    Args:
        adata_path (str): Path to the h5ad AnnData file
        output_csv_path (str): Path to save the output CSV file
        radius (float): Radius for creating circles around each spot (default: 112.0)
        spatial_scale (float): Scaling factor for spatial coordinates (default: 15.0)
        n_jobs (int): Number of parallel jobs for computation (default: -1, use all cores)
        chunk_size (int): Number of spots to process per chunk (default: 100)
        parallel_samples (bool): Whether to parallelize sample pair processing (default: True)
    
    Output:
        CSV file with columns: ['spot1_id', 'spot2_id', 'sample1', 'sample2', 'iou_score']
    """
    print("🔬 Loading data...")
    
    # Load AnnData
    adata = ad.read_h5ad(adata_path)
    print(f"🧬 AnnData shape: {adata.shape}")
    print(f"📍 Unique samples: {adata.obs['sample'].unique()}")
    
    # Get unique sample IDs in order
    unique_samples = adata.obs['sample'].unique()
    n_samples = len(unique_samples)
    
    print(f"🔄 Processing {n_samples} samples...")
    
    # Extract spot information for each sample
    sample_spots = {}
    for sample_id in tqdm(unique_samples, desc="Extracting spots"):
        sample_spots[sample_id] = extract_spots_from_sample(adata, sample_id, spatial_scale)
        print(f"Sample {sample_id}: {len(sample_spots[sample_id]['spot_ids'])} spots")
    
    # Compute IoU between all sample pairs (avoiding duplicates and same-layer pairs)
    total_pairs = (n_samples * (n_samples - 1)) // 2  # Exclude self-pairs
    
    print(f"🧮 Computing IoU for {total_pairs} sample pairs...")
    
    # Prepare sample pairs for processing
    sample_pairs = []
    for i, sample1 in enumerate(unique_samples):
        for j, sample2 in enumerate(unique_samples):
            # Only compute upper triangle to avoid duplicates, exclude diagonal (same layer)
            if j > i:
                sample_pairs.append((sample1, sample2))
    
    if parallel_samples and len(sample_pairs) > 1:
        # Parallel processing of sample pairs
        print(f"🚀 Using parallel sample pair processing with {len(sample_pairs)} pairs")
        
        # Determine optimal job distribution
        n_jobs_outer = min(len(sample_pairs), max(1, n_jobs // 2)) if n_jobs > 0 else min(len(sample_pairs), 4)
        n_jobs_inner = max(1, n_jobs // n_jobs_outer) if n_jobs > 0 else -1
        
        print(f"    📊 Jobs: {n_jobs_outer} sample pairs × {n_jobs_inner} inner jobs each")
        
        # Prepare arguments for parallel processing
        pair_args = [
            (sample1, sample2, sample_spots, radius, n_jobs_inner, chunk_size)
            for sample1, sample2 in sample_pairs
        ]
        
        # Parallel computation of all sample pairs
        results_nested = Parallel(n_jobs=n_jobs_outer, backend='loky', verbose=1)(
            delayed(compute_sample_pair_iou)(args) for args in pair_args
        )
        
        # Flatten results
        all_results = []
        for pair_results in results_nested:
            all_results.extend(pair_results)
            
    else:
        # Sequential processing (fallback or single pair)
        print(f"🔄 Using sequential sample pair processing")
        all_results = []
        
        for sample1, sample2 in tqdm(sample_pairs, desc="Processing sample pairs"):
            pair_results = compute_sample_pair_iou(
                (sample1, sample2, sample_spots, radius, n_jobs, chunk_size)
            )
            all_results.extend(pair_results)
    
    # Create DataFrame and save to CSV
    print("💾 Saving results...")
    results_df = pd.DataFrame(all_results)
    
    # Create output directory if it doesn't exist
    os.makedirs(os.path.dirname(output_csv_path), exist_ok=True)
    
    # Save to CSV
    results_df.to_csv(output_csv_path, index=False)
    
    # Print summary statistics
    print("✅ IoU computation completed!")
    print(f"📈 Total spot pairs: {len(results_df)}")
    print(f"📊 IoU statistics:")
    print(f"   Mean: {results_df['iou_score'].mean():.4f}")
    print(f"   Std:  {results_df['iou_score'].std():.4f}")
    print(f"   Max:  {results_df['iou_score'].max():.4f}")
    print(f"   Min:  {results_df['iou_score'].min():.4f}")
    print(f"💾 Results saved to: {output_csv_path}")



# Example usage function
def example_usage():
    """
    Example of how to use the compute_iou function.
    """
    ADATA_PATH = '/nfs/team361/data/sanger_whole_embryo/adata_CS17.h5ad'
    #ADATA_PATH = '/lustre/scratch126/cellgen/lotfollahi/mv10/3d-imputation-data/adata_CS17_2.h5ad'

    OUTPUT_CSV = './embryo_data/spot_iou/embryo_iou_results_new_1.csv'
    
    # Compute IoU
    compute_iou(
        adata_path=ADATA_PATH,
        output_csv_path=OUTPUT_CSV,
        radius=112.0,           # Adjust radius as needed
        spatial_scale=15.0,     # Adjust scaling as needed
        n_jobs=-1,              # Use all CPU cores
        chunk_size=100,         # Process 100 spots per chunk
        parallel_samples=True   # Parallelize sample pairs
    )


if __name__ == "__main__":
    example_usage() 