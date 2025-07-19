import numpy as np
import pandas as pd
import anndata as ad
from shapely.geometry import Polygon, Point
from tqdm import tqdm
import math
import os
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


def compute_iou_between_samples(spots1: Dict, spots2: Dict, radius: float) -> List[Tuple]:
    """
    Compute IoU between all spot pairs from two samples.
    
    Args:
        spots1 (Dict): Spot information from first sample
        spots2 (Dict): Spot information from second sample
        radius (float): Radius for circle creation around each spot
    
    Returns:
        List[Tuple]: List of (spot1_id, spot2_id, iou_score) tuples
    """
    results = []
    
    # Create polygons for sample 1
    polygons1 = [create_circle_polygon(coord[0], coord[1], radius) for coord in spots1['coordinates']]
    
    # Create polygons for sample 2
    polygons2 = [create_circle_polygon(coord[0], coord[1], radius) for coord in spots2['coordinates']]
    
    # Compute IoU for all pairs
    for i, (spot1_id, poly1) in tqdm(enumerate(zip(spots1['spot_ids'], polygons1))):
        if (i +1 ) % 100 == 0:
            break
        for j, (spot2_id, poly2) in tqdm(enumerate(zip(spots2['spot_ids'], polygons2))):
            iou_score = compute_iou_pair(poly1, poly2)
            results.append((spot1_id, spot2_id, iou_score))
    
    return results


def compute_iou(adata_path: str, output_csv_path: str, radius: float = 112.0, 
                spatial_scale: float = 15.0, n_jobs: int = -1) -> None:
    """
    Compute IoU scores between spots across different samples/layers from h5ad data.
    
    This function loads an AnnData object and computes IoU scores between circular 
    regions around each spot across all sample pairs.
    
    Args:
        adata_path (str): Path to the h5ad AnnData file
        output_csv_path (str): Path to save the output CSV file
        radius (float): Radius for creating circles around each spot (default: 112.0)
        spatial_scale (float): Scaling factor for spatial coordinates (default: 15.0)
        n_jobs (int): Number of parallel jobs for computation (default: -1, use all cores)
    
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
    all_results = []
    total_pairs = (n_samples * (n_samples - 1)) // 2  # Exclude self-pairs
    
    print(f"🧮 Computing IoU for {total_pairs} sample pairs...")
    
    # Progress bar for sample pairs
    with tqdm(total=total_pairs, desc="Computing IoU") as pbar:
        for i, sample1 in enumerate(unique_samples):
            for j, sample2 in enumerate(unique_samples):
                # Only compute upper triangle to avoid duplicates, exclude diagonal (same layer)
                if j > i:
                    print(f"🔍 Processing pair: {sample1} vs {sample2}")
                    
                    # Compute IoU between the two samples
                    pair_results = compute_iou_between_samples(
                        sample_spots[sample1], 
                        sample_spots[sample2], 
                        radius
                    )
                    
                    # Add sample information to results
                    for spot1_id, spot2_id, iou_score in pair_results:
                        all_results.append({
                            'spot1_id': spot1_id,
                            'spot2_id': spot2_id,
                            'sample1': sample1,
                            'sample2': sample2,
                            'iou_score': iou_score
                        })
                    
                    pbar.update(1)
    
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
    ADATA_PATH = '/lustre/scratch126/cellgen/lotfollahi/mv10/3d-imputation-data/adata_CS17_2.h5ad'

    OUTPUT_CSV = './embryo_data/spot_iou/embryo_iou_results.csv'
    
    # Compute IoU
    compute_iou(
        adata_path=ADATA_PATH,
        output_csv_path=OUTPUT_CSV,
        radius=112.0,  # Adjust radius as needed
        spatial_scale=15.0,  # Adjust scaling as needed
        n_jobs=-1
    )


if __name__ == "__main__":
    example_usage() 