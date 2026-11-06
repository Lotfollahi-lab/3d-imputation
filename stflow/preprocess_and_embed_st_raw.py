#!/usr/bin/env python3
"""
Preprocessing + embedding pipeline for ST-raw dataset format.

Input format:
- Images: HE_*.jpg (e.g., HE_BC23209_C1.jpg)
- Expression: *_stdata.tsv.gz (e.g., BC23209_C1_stdata.tsv.gz)
- Spot coordinates: spots_*.csv.gz (e.g., spots_BC23209_C1.csv.gz)

Samples are grouped by patient ID (e.g., BC23209 has slides C1, C2, D1).
Creates 4-fold cross-validation splits where:
- All slides from same patient go to same split
- Each fold: some patients in train, some in test

Output format matches STFlow expectations:
- Embeddings in <embed_dir>/<dataset>/<encoder>/fp32/<sample>.h5
- Per-sample AnnData in <output_dir>/adata/<sample>.h5ad
- Gene lists and splits in <output_dir>/

Usage:
    python preprocess_and_embed_st_raw.py \\
        --data_dir /path/to/ST-raw \\
        --output_dir ./dataset/ST_RAW \\
        --embed_dir ./embeddings \\
        --encoder uni_v1_official \\
        --gene_list_path ./selected_genes.json \\
        --n_genes 50
"""

import os
import json
import h5py
import argparse
import numpy as np
import pandas as pd
from tqdm import tqdm
from pathlib import Path
from collections import defaultdict
import gzip

import torch
import scanpy as sc
import anndata as ad
from PIL import Image
import scipy.sparse as sp

from hest_utils.encoder import load_encoder
from hest_utils.file_utils import save_hdf5
from data.normalize_utils import get_normalize_method
from tqdm import tqdm
# Curated ST-RAW subset with explicit prefixes per modality.
TARGET_PATIENTS = {
    '23272': {'image_prefix': 'BT', 'expr_prefix': 'BC', 'slides': ['D2', 'E1', 'E2']},
    '23277': {'image_prefix': 'BT', 'expr_prefix': 'BC', 'slides': ['D2', 'E1', 'E2']},
    '23377': {'image_prefix': 'BT', 'expr_prefix': 'BC', 'slides': ['C1', 'C2', 'D1']},
    '23450': {'image_prefix': 'BT', 'expr_prefix': 'BC', 'slides': ['D2', 'E1', 'E2']},
    '23506': {'image_prefix': 'BT', 'expr_prefix': 'BC', 'slides': ['C1', 'C2', 'D1']},
    '23508': {'image_prefix': 'BT', 'expr_prefix': 'BC', 'slides': ['D2', 'E1', 'E2']},
    '23567': {'image_prefix': 'BT', 'expr_prefix': 'BC', 'slides': ['D2', 'E1', 'E2']},
    '23803': {'image_prefix': 'BC', 'expr_prefix': 'BC', 'slides': ['D2', 'E1', 'E2']},
    '23810': {'image_prefix': 'BT', 'expr_prefix': 'BC', 'slides': ['D2', 'E1', 'E2']},
    '23895': {'image_prefix': 'BT', 'expr_prefix': 'BC', 'slides': ['C1', 'C2', 'D1']},
    '23903': {'image_prefix': 'BT', 'expr_prefix': 'BC', 'slides': ['C1', 'C2', 'D1']},
    '23944': {'image_prefix': 'BT', 'expr_prefix': 'BC', 'slides': ['D2', 'E1', 'E2']},
    '24044': {'image_prefix': 'BT', 'expr_prefix': 'BC', 'slides': ['D2', 'E1', 'E2']},
    '24105': {'image_prefix': 'BC', 'expr_prefix': 'BC', 'slides': ['C1', 'C2', 'D1']},
    '24220': {'image_prefix': 'BC', 'expr_prefix': 'BC', 'slides': ['D2', 'E1', 'E2']},
    '24223': {'image_prefix': 'BT', 'expr_prefix': 'BC', 'slides': ['D2', 'E1', 'E2']},
}

EXPR_EXCLUDE_COLUMNS = {'Unnamed: 0', 'barcode', 'Barcode', 'BARCODE'}

def normalize_gene_name(gene_name: str) -> str:
    """Convert composite genes (containing '+') to their __ambiguous[...] keys."""
    gene_name = (gene_name or '').strip()
    if not gene_name:
        return gene_name
    if '+' in gene_name and not gene_name.startswith('__ambiguous['):
        return f'__ambiguous[{gene_name}]'
    return gene_name


def normalize_gene_list(gene_list):
    normalized = []
    for gene in gene_list:
        normalized_name = normalize_gene_name(gene)
        if normalized_name:
            normalized.append(normalized_name)
    return normalized


def compute_shared_genes(samples):
    """Return sorted list of genes present in every sample's expression table."""
    shared = None
    for info in tqdm(samples.values(), desc="Computing shared genes"):
        header = pd.read_csv(info['expr'], sep='\t', compression='gzip', nrows=0)
        columns = {
            normalize_gene_name(col)
            for col in header.columns
            if col not in EXPR_EXCLUDE_COLUMNS
        }
        if shared is None:
            shared = columns
        else:
            shared &= columns
        if not shared:
            break
    return sorted(shared) if shared else []


def parse_sample_name(filename):
    """Extract patient ID and slide ID from filename.
    
    Examples:
        HE_BC23209_C1.jpg -> (BC23209, C1)
        BC23209_C1_stdata.tsv.gz -> (BC23209, C1)
        spots_BC23209_C1.csv.gz -> (BC23209, C1)
    """
    # Remove prefixes and suffixes
    name = filename.replace('HE_', '').replace('spots_', '')
    name = name.replace('_stdata.tsv.gz', '').replace('.jpg', '').replace('.csv.gz', '')
    
    # Split into patient and slide
    parts = name.rsplit('_', 1)
    if len(parts) == 2:
        return parts[0], parts[1]
    else:
        return name, 'unknown'


def discover_samples(data_dir, target_patients=TARGET_PATIENTS):
    """Build the ST-RAW sample list from the curated patient/slide mapping.
    
    Returns:
        dict: {sample_id: {'image': path, 'expr': path, 'spots': path,
                          'patient': numeric str, 'slide': str}}
    """
    data_path = Path(data_dir)
    samples = {}
    patients = defaultdict(list)
    
    for patient_id in sorted(target_patients.keys(), key=int):
        cfg = target_patients[patient_id]
        image_prefix = cfg['image_prefix']
        expr_prefix = cfg.get('expr_prefix', image_prefix)
        spots_prefix = cfg.get('spots_prefix', image_prefix)
        slides = cfg['slides']
        
        for slide in slides:
            sample_id = f'{image_prefix}{patient_id}_{slide}'
            expr_sample = f'{expr_prefix}{patient_id}_{slide}'
            spots_sample = f'{spots_prefix}{patient_id}_{slide}'
            paths = {
                'image': data_path / f'HE_{sample_id}.jpg',
                'expr': data_path / f'{expr_sample}_stdata.tsv.gz',
                'spots': data_path / f'spots_{spots_sample}.csv.gz',
            }
            missing = [kind for kind, path in paths.items() if not path.exists()]
            if missing:
                raise FileNotFoundError(
                    f"Missing {', '.join(missing)} for patient {patient_id} slide {slide} "
                    f"(expected files like {paths['image'].name})"
                )
            
            samples[sample_id] = {
                **paths,
                'patient': patient_id,
                'slide': slide,
            }
            patients[patient_id].append(sample_id)
    
    print(f"Discovered {len(samples)} samples from curated ID list ({len(patients)} patients)")
    for patient_id in sorted(patients.keys(), key=int):
        slide_order = target_patients[patient_id]['slides']
        sample_slides = ', '.join(slide_order)
        print(f"  Patient {patient_id}: {len(patients[patient_id])} slides ({sample_slides})")
    
    return samples, patients


def load_expression_data(expr_path, spots_path, selected_genes=None):
    """Load expression and spot coordinates, merge on barcode.
    
    Args:
        expr_path: Path to *_stdata.tsv.gz
        spots_path: Path to spots_*.csv.gz
        selected_genes: List of gene names/IDs to keep (None = keep all)
    
    Returns:
        tuple[pd.DataFrame, list[str]]: dataframe with barcode/x/y/genes and
        the ordered list of gene columns actually present.
    """
    # Load expression (barcodes in 'Unnamed: 0' column)
    expr_df = pd.read_csv(expr_path, sep='\t', compression='gzip')
    
    # Rename barcode column
    if 'Unnamed: 0' in expr_df.columns:
        expr_df = expr_df.rename(columns={'Unnamed: 0': 'barcode'})
    
    # Load spots (barcodes in 'Unnamed: 0', coords in X, Y)
    spots_df = pd.read_csv(spots_path, compression='gzip')
    if 'Unnamed: 0' in spots_df.columns:
        spots_df = spots_df.rename(columns={'Unnamed: 0': 'barcode'})
    
    # Merge on barcode
    merged = pd.merge(spots_df, expr_df, on='barcode', how='inner')
    
    # Filter genes if provided
    if selected_genes is not None:
        available_genes = []
        missing = []
        for gene in selected_genes:
            if gene in merged.columns:
                available_genes.append(gene)
            else:
                missing.append(gene)
        if missing:
            print(f"    Warning: {len(missing)} genes missing from expression data: {missing[:5]}{'...' if len(missing) > 5 else ''}")
        merged = merged[['barcode', 'X', 'Y'] + available_genes]
    else:
        available_genes = [c for c in merged.columns if c not in ('barcode', 'X', 'Y')]
    
    return merged, available_genes


def create_anndata_from_sample(sample_data, sample_id, image_path=None, max_image_dim=20000):
    """Convert sample dataframe to AnnData.
    
    Args:
        sample_data: DataFrame with barcode, X, Y, and gene columns
        sample_id: Sample identifier
        image_path: Optional path to histology image for this sample
        max_image_dim: Maximum size (pixels) for longest image edge when storing in AnnData
    
    Returns:
        AnnData object
    """
    # Extract gene expression matrix
    gene_cols = [c for c in sample_data.columns if c not in ['barcode', 'X', 'Y']]
    X = sample_data[gene_cols].values.astype(np.float32)
    
    # Create obs (cell metadata)
    obs = pd.DataFrame({
        'barcode': sample_data['barcode'].values,
        'sample_id': sample_id,
    })
    obs.index = obs['barcode']
    
    # Create var (gene metadata)
    var = pd.DataFrame(index=gene_cols)
    var['gene_name'] = gene_cols
    
    # Create spatial coordinates
    spatial_coords = sample_data[['X', 'Y']].values.astype(np.float32)
    
    # Create AnnData
    adata = ad.AnnData(
        X=X,
        obs=obs,
        var=var,
        dtype=np.float32,
    )
    
    # Add spatial info
    adata.obsm['spatial'] = spatial_coords

    images_dict = {}
    scalefactors = {
        'tissue_hires_scalef': 1.0,
        'spot_diameter_fullres': 1.0,
    }

    if image_path is not None:
        image_path = Path(image_path)
        if image_path.exists():
            try:
                img_full = Image.open(image_path).convert('RGB')
                images_dict['fullres'] = np.array(img_full)

                orig_width, orig_height = img_full.size
                longest_edge = max(orig_width, orig_height)
                scale = 1.0
                if longest_edge > max_image_dim and max_image_dim > 0:
                    scale = max_image_dim / float(longest_edge)
                    new_size = (
                        max(1, int(round(orig_width * scale))),
                        max(1, int(round(orig_height * scale))),
                    )
                    img_hires = img_full.resize(new_size, Image.BILINEAR)
                else:
                    img_hires = img_full
                images_dict['hires'] = np.array(img_hires)
                scalefactors['tissue_hires_scalef'] = float(scale)
            except Exception as exc:
                print(f"  [warn] Failed to attach image for {sample_id}: {exc}")

    adata.uns['spatial'] = {
        sample_id: {
            'images': images_dict,
            'scalefactors': scalefactors,
        }
    }
    
    return adata


def extract_and_embed_sample(
    image_path,
    adata_sample,
    encoder,
    img_transform,
    device,
    tile_size=224,
    batch_size=32,
    precision='fp32',
):
    """Extract tiles from image and embed them.
    
    Args:
        image_path: Path to HE image
        adata_sample: AnnData for this sample
        encoder: Pre-loaded encoder model
        img_transform: Image transformation pipeline
        device: torch device
        tile_size: Size of tiles to extract
        batch_size: Batch size for embedding
        precision: 'fp32' or 'fp16'
    
    Returns:
        dict with keys: embeddings, coords, barcodes
    """
    # Load image
    img = Image.open(image_path).convert('RGB')
    img_array = np.array(img)
    
    # Get spot coordinates
    coords = adata_sample.obsm['spatial']  # [N, 2]
    barcodes = adata_sample.obs['barcode'].values
    
    # Extract tiles
    tiles = []
    valid_indices = []
    
    for i, (x, y) in enumerate(coords):
        x_int, y_int = int(x), int(y)
        
        # Extract tile centered on spot
        x1 = max(0, x_int - tile_size // 2)
        y1 = max(0, y_int - tile_size // 2)
        x2 = min(img_array.shape[1], x1 + tile_size)
        y2 = min(img_array.shape[0], y1 + tile_size)
        
        tile = img_array[y1:y2, x1:x2]
        
        # Pad if needed
        if tile.shape[0] < tile_size or tile.shape[1] < tile_size:
            padded = np.zeros((tile_size, tile_size, 3), dtype=np.uint8)
            padded[:tile.shape[0], :tile.shape[1]] = tile
            tile = padded
        
        tiles.append(Image.fromarray(tile))
        valid_indices.append(i)
    
    # Transform tiles
    tile_tensors = torch.stack([img_transform(t) for t in tiles])
    
    # Embed in batches
    embeddings_list = []
    encoder.eval()
    
    with torch.no_grad():
        for i in range(0, len(tile_tensors), batch_size):
            batch = tile_tensors[i:i+batch_size].to(device)
            
            if precision == 'fp16':
                with torch.cuda.amp.autocast():
                    emb = encoder(batch)
            else:
                emb = encoder(batch)
            
            embeddings_list.append(emb.cpu().numpy())
    
    embeddings = np.vstack(embeddings_list).astype(np.float32)
    
    return {
        'embeddings': embeddings,
        'coords': coords[valid_indices],
        'barcodes': barcodes[valid_indices],
    }


def create_4fold_splits(patients, samples, output_dir, dataset_name, patient_config):
    """Create 4-fold cross-validation splits.
    
    Each fold: some patients in train, some in test. Additionally, the first
    slide (as listed in TARGET_PATIENTS) of every test patient is also exposed
    to the training split as requested.
    
    Args:
        patients: dict {patient_id: [sample_ids]}
        samples: dict {sample_id: info}
        output_dir: Output directory
        dataset_name: Dataset name prefix
        patient_config: mapping used to determine slide ordering/prefix
    """
    splits_dir = Path(output_dir) / 'splits'
    splits_dir.mkdir(parents=True, exist_ok=True)
    
    patient_ids = sorted(patients.keys(), key=int)
    n_patients = len(patient_ids)
    fold_size = n_patients // 4
    
    print(f"\nCreating 4-fold splits ({n_patients} patients, ~{fold_size} per fold)")
    
    for fold in range(4):
        start = fold * fold_size
        end = start + fold_size if fold < 3 else n_patients  # Last fold gets remainder
        
        test_patients = patient_ids[start:end]
        train_patients = [p for p in patient_ids if p not in test_patients]
    
        # Get all samples for train/test patients
        train_samples = []
        for p in train_patients:
            train_samples.extend(patients[p])
        
        test_samples = []
        for p in test_patients:
            test_samples.extend(patients[p])

        # Add first slide from each test patient into training split as well
        carryover_samples = []
        for p in test_patients:
            cfg = patient_config.get(p)
            if not cfg:
                continue
            if not cfg['slides']:
                continue
            first_slide = cfg['slides'][0]
            image_prefix = cfg['image_prefix']
            sample_name = f"{image_prefix}{p}_{first_slide}"
            if sample_name in patients[p]:
                carryover_samples.append(sample_name)
            else:
                print(f"  Warning: {sample_name} expected for shared split but not found among samples")
        if carryover_samples:
            # Preserve ordering while deduplicating
            ordered = []
            seen = set()
            for name in train_samples + carryover_samples:
                if name not in seen:
                    seen.add(name)
                    ordered.append(name)
            train_samples = ordered
    
        def build_split_df(sample_list):
            full_ids = [f'{dataset_name}_{s}' for s in sample_list]
            return pd.DataFrame({
                'sample_id': full_ids,
                'patches_path': [f'patches/{sid}.h5' for sid in full_ids],
                'expr_path': [f'adata/{sid}.h5ad' for sid in full_ids],
            })
        
        train_df = build_split_df(train_samples)
        test_df = build_split_df(test_samples)
        
        train_path = splits_dir / f'train_{fold}.csv'
        test_path = splits_dir / f'test_{fold}.csv'
        
        train_df.to_csv(train_path, index=False)
        test_df.to_csv(test_path, index=False)
        
        print(
            f"  Fold {fold}: {len(train_patients)} train patients ({len(train_samples)} slides incl. "
            f"{len(carryover_samples)} shared), {len(test_patients)} test patients ({len(test_samples)} slides)"
        )


def main():
    parser = argparse.ArgumentParser(description='Preprocess ST-raw dataset')
    parser.add_argument('--data_dir', type=str, required=True,
                        help='Directory containing HE_*.jpg, *_stdata.tsv.gz, spots_*.csv.gz')
    parser.add_argument('--output_dir', type=str, required=True,
                        help='Output directory for processed data')
    parser.add_argument('--embed_dir', type=str, required=True,
                        help='Output directory for embeddings')
    parser.add_argument('--encoder', type=str, default='uni_v1_official',
                        help='Encoder name (uni_v1_official, gigapath, etc.)')
    parser.add_argument('--weights_root', type=str, default=None,
                        help='Path to encoder weights directory')
    parser.add_argument('--gene_list_path', type=str, required=True,
                        help='Path to JSON file with selected gene list (Ensembl IDs)')
    parser.add_argument('--n_genes', type=int, default=50,
                        help='Number of genes to use')
    parser.add_argument('--dataset_name', type=str, default='ST_RAW',
                        help='Dataset name prefix')
    parser.add_argument('--tile_size', type=int, default=224,
                        help='Tile size for extraction')
    parser.add_argument('--batch_size', type=int, default=32,
                        help='Batch size for embedding')
    parser.add_argument('--precision', type=str, default='fp32', choices=['fp32', 'fp16'],
                        help='Precision for embeddings')
    parser.add_argument('--normalize_method', type=str, default='log1p',
                        help='Normalization method')
    parser.add_argument('--device', type=int, default=0,
                        help='GPU device ID')
    parser.add_argument('--reuse_embeddings', action='store_true',
                        help='Skip embedding if output file exists')
    
    args = parser.parse_args()
    
    # Setup
    device = torch.device(f'cuda:{args.device}' if torch.cuda.is_available() else 'cpu')
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    adata_dir = output_dir / 'adata'
    adata_dir.mkdir(exist_ok=True)
    
    embed_output_dir = Path(args.embed_dir) / args.dataset_name / args.encoder / args.precision
    embed_output_dir.mkdir(parents=True, exist_ok=True)
    
    print("="*80)
    print("ST-RAW PREPROCESSING PIPELINE")
    print("="*80)
    print(f"Data directory: {args.data_dir}")
    print(f"Output directory: {output_dir}")
    print(f"Embedding directory: {embed_output_dir}")
    print(f"Encoder: {args.encoder}")
    print(f"Device: {device}")
    print()
    
    # ========== Discover Samples ==========
    print("[1/5] Discovering samples...")
    samples, patients = discover_samples(args.data_dir, TARGET_PATIENTS)
    
    if len(samples) == 0:
        raise ValueError("No complete samples found! Check file naming.")
    
    # ========== Load Gene List ==========
    print("\n[2/5] Loading gene list...")
    gene_list_path = Path(args.gene_list_path)
    
    if gene_list_path.suffix == '.txt':
        # Load from text file (one gene per line)
        with open(gene_list_path, 'r') as f:
            all_genes = [line.strip() for line in f if line.strip()]
        selected_genes = all_genes[:args.n_genes]
    elif gene_list_path.suffix == '.json':
        # Load from JSON file
        with open(gene_list_path, 'r') as f:
            gene_data = json.load(f)
        
        if isinstance(gene_data, dict) and 'genes' in gene_data:
            selected_genes = gene_data['genes'][:args.n_genes]
        elif isinstance(gene_data, list):
            selected_genes = gene_data[:args.n_genes]
        else:
            raise ValueError(f"Unexpected gene list format in {args.gene_list_path}")
    else:
        raise ValueError(f"Unsupported gene list format: {gene_list_path.suffix}. Use .txt or .json")
    
    selected_genes = normalize_gene_list(selected_genes)
    print(f"  Loaded {len(selected_genes)} genes from {gene_list_path}")

    shared_genes = compute_shared_genes(samples)
    if not shared_genes:
        raise ValueError("No shared genes found across all expression files.")

    shared_set = set(shared_genes)
    filtered_genes = [gene for gene in selected_genes if gene in shared_set]
    if len(filtered_genes) < len(selected_genes):
        print(f"  Filtered gene list down to {len(filtered_genes)} shared genes.")
    selected_genes = filtered_genes
    args.n_genes = len(selected_genes)
    
    # Save gene list to output dir
    gene_list_path = output_dir / f'var_{args.n_genes}genes.json'
    with open(gene_list_path, 'w') as f:
        json.dump({'genes': selected_genes}, f, indent=2)
    print(f"  Saved {gene_list_path}")
    gene_list_synced = False
    
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
    print("\n[4/5] Processing samples (load + embed + save)...")
    
    for sample_id, info in tqdm(sorted(samples.items()), desc="Samples"):
        sample_name = f'{args.dataset_name}_{sample_id}'
        
        embed_h5_path = embed_output_dir / f'{sample_name}.h5'
        adata_path = adata_dir / f'{sample_name}.h5ad'
        
        # Load expression and spots
        try:
            sample_data, sample_genes = load_expression_data(
                info['expr'],
                info['spots'],
                selected_genes=selected_genes
            )

            if not gene_list_synced:
                if sample_genes != selected_genes:
                    with open(gene_list_path, 'w') as f:
                        json.dump({'genes': sample_genes}, f, indent=2)
                    print(f"  Synced gene list with AnnData columns ({len(sample_genes)} genes).")
                    selected_genes = sample_genes
                gene_list_synced = True
            # Create AnnData
            adata_sample = create_anndata_from_sample(sample_data, sample_id, image_path=info['image'])
            
            # Save AnnData
            adata_sample.write_h5ad(adata_path)
            
        except Exception as e:
            print(f"  ERROR loading data for {sample_id}: {e}")
            continue
        
        # Embed
        if args.reuse_embeddings and embed_h5_path.exists():
            print(f"  Reusing existing embeddings for {sample_name}")
        else:
            try:
                result = extract_and_embed_sample(
                    info['image'],
                    adata_sample,
                    encoder,
                    img_transforms,
                    device,
                    tile_size=args.tile_size,
                    batch_size=args.batch_size,
                    precision=args.precision,
                )
                
                with h5py.File(embed_h5_path, 'w') as f:
                    f.create_dataset('embeddings', data=result['embeddings'], compression='gzip')
                    f.create_dataset('coords', data=result['coords'], compression='gzip')
                    f.create_dataset('barcodes', data=result['barcodes'].astype('S'))
                    
            except Exception as e:
                print(f"  ERROR embedding sample {sample_id}: {e}")
                continue
    
    # ========== Create Splits ==========
    print("\n[5/5] Creating 4-fold cross-validation splits...")
    create_4fold_splits(patients, samples, output_dir, args.dataset_name, TARGET_PATIENTS)
    
    # ========== Summary ==========
    print("\n" + "="*80)
    print("✓ PIPELINE COMPLETE")
    print("="*80)
    print(f"Processed {len(samples)} samples from {len(patients)} patients")
    print()
    print("Output structure:")
    print(f"  {output_dir}/")
    print(f"    ├── adata/              ({len(list(adata_dir.glob('*.h5ad')))} files)")
    print(f"    ├── var_{args.n_genes}genes.json")
    print(f"    └── splits/")
    print(f"        ├── train_0.csv, test_0.csv")
    print(f"        ├── train_1.csv, test_1.csv")
    print(f"        ├── train_2.csv, test_2.csv")
    print(f"        └── train_3.csv, test_3.csv")
    print()
    print(f"  {embed_output_dir}/")
    print(f"    └── {args.dataset_name}_*.h5  ({len(list(embed_output_dir.glob('*.h5')))} embeddings)")
    print()
    print("Ready for training! Run:")
    print(f"  python app/flow/train.py \\")
    print(f"    --dataset {args.dataset_name} \\")
    print(f"    --source_dataroot {output_dir.parent}/ \\")
    print(f"    --embed_dataroot {Path(args.embed_dir)}/ \\")
    print(f"    --feature_encoder {args.encoder} \\")
    print(f"    --gene_list var_{args.n_genes}genes.json \\")
    print(f"    --n_genes {args.n_genes} \\")
    print(f"    --split_indices 0 1 2 3")


if __name__ == '__main__':
    main()
