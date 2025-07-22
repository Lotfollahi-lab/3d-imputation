import os
import time
from typing import List, Dict, Any, Tuple

import numpy as np
import pandas as pd
import torch
from torch import nn
from torchvision import models, transforms
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import tifffile
import anndata as ad


class PatchDataset(Dataset):
    """
    Dataset of image patches across multiple samples & sizes.
    Each item returns (sample_idx, size, patch_tensor, patch_id).
    """

    def __init__(
        self,
        img_array: np.ndarray,
        adata: ad.AnnData,
        windows_csv: str,
        spatial_scale: float,
        preprocess: transforms.Compose,
    ):
        """
        Args:
          img_array    : np array [n_samples, H, W, C] or [n_samples, C, H, W]
          adata        : loaded AnnData with obs['sample'] and obsm['spatial']
          windows_csv  : CSV path with columns [sample,window_size,x,y]
          spatial_scale: float
          preprocess   : torchvision transforms to apply to each patch
        """
        # ensure shape [n_samples, H, W, C]
        if img_array.ndim == 4 and img_array.shape[1] in (1,3,4):
            # move channel from axis=1 to last
            img_array = np.moveaxis(img_array, 1, -1)
        self.img_array = img_array
        self.H, self.W = img_array.shape[1], img_array.shape[2]

        # map sample names to integer indices
        samples = list(adata.obs['sample'].unique())
        self.sample_to_idx = {s: i for i, s in enumerate(samples)}

        # build metadata list
        self.meta: List[Dict[str, Any]] = []
        half224 = 224 // 2
        coords = (adata.obsm['spatial'] * spatial_scale).astype(int)
        for sample, (xc, yc) in zip(adata.obs['sample'], coords):
            idx = self.sample_to_idx[sample]
            x0, y0 = xc - half224, yc - half224
            if 0 <= x0 <= self.W - 224 and 0 <= y0 <= self.H - 224:
                self.meta.append({
                    'sample_idx': idx,
                    'size': 224,
                    'x0': x0,
                    'y0': y0,
                    'patch_id': f"{sample}_224_{x0}_{y0}"
                })

        df = pd.read_csv(windows_csv)
        for _, row in df.iterrows():
            sname = str(row['sample'])
            if sname not in self.sample_to_idx:
                continue
            idx = self.sample_to_idx[sname]
            sz = int(row['window_size'])
            x0, y0 = int(row['x']), int(row['y'])
            if 0 <= x0 <= self.W - sz and 0 <= y0 <= self.H - sz:
                self.meta.append({
                    'sample_idx': idx,
                    'size': sz,
                    'x0': x0,
                    'y0': y0,
                    'patch_id': f"{sname}_{sz}_{x0}_{y0}"
                })

        # group indices by size for homogeneous batching
        self.by_size: Dict[int, List[int]] = {}
        for idx, m in enumerate(self.meta):
            self.by_size.setdefault(m['size'], []).append(idx)

        self.preprocess = preprocess

    def __len__(self) -> int:
        return len(self.meta)

    def __getitem__(self, idx: int) -> Tuple[int, torch.Tensor, str]:
        """
        Returns:
          sample_idx (int), tensor [3, size, size], patch_id (str)
        """
        m = self.meta[idx]
        img_patch = self.img_array[
            m['sample_idx'],
            m['y0']:m['y0']+m['size'],
            m['x0']:m['x0']+m['size'],
            :
        ]
        # convert to PIL and preprocess
        patch = Image.fromarray(img_patch)
        tensor = self.preprocess(patch)
        return m['sample_idx'], tensor, m['patch_id']


def collate_homogeneous(batch: List[Tuple[int, torch.Tensor, str]]) -> Tuple[torch.Tensor, List[str]]:
    """
    Collate function that stacks only if all sizes match in the batch.
    Returns:
       batch_tensor [B,3,sz,sz], list of patch_ids
    """
    # batch is guaranteed homogeneous by how we'll feed indices to DataLoader
    _, tensors, ids = zip(*batch)
    stacked = torch.stack(tensors, dim=0)
    return stacked, list(ids)


def extract_all_features(
    image_path: str,
    adata_path: str,
    windows_csv: str,
    spatial_scale: float,
    target_dim: int = 1024,
    batch_size: int = 32,
    num_workers: int = 4,
    output_adata_path: str = './adata_with_features.h5ad'
) -> None:
    """
    Driver using PatchDataset + DataLoader to batch preprocess CPU-side
    and GPU-forward each homogeneous batch.
    """
    t0 = time.time()
    print("🔬 Loading data...")
    img_arr = tifffile.imread(image_path)
    adata = ad.read_h5ad(adata_path)

    # preprocessing & model
    preprocess = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485,0.456,0.406],
                             std=[0.229,0.224,0.225]),
    ])
    resnet = models.resnet50(pretrained=True)
    backbone = nn.Sequential(*list(resnet.children())[:-1]).eval()
    reducer = nn.Linear(2048, target_dim).eval()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    backbone.to(device); reducer.to(device)

    # build dataset
    ds = PatchDataset(img_arr, adata, windows_csv, spatial_scale, preprocess)

    # prepare storage
    feat_store: Dict[int, Dict[str, np.ndarray]] = {}

    # for each size, create a DataLoader over that subset
    for size, indices in ds.by_size.items():
        print(f"▶ Processing patches of size {size} ({len(indices)} total)")
        loader = DataLoader(
            ds,
            batch_size=batch_size,
            sampler=torch.utils.data.SubsetRandomSampler(indices),
            collate_fn=collate_homogeneous,
            num_workers=num_workers,
            pin_memory=True,
        )

        for batch_idx, (batch_tensor, patch_ids) in enumerate(loader):
            batch_tensor = batch_tensor.to(device)
            with torch.no_grad():
                feats2048 = backbone(batch_tensor).view(batch_tensor.size(0), -1)
                feats = reducer(feats2048).cpu().numpy()

            # store per-patch
            for pid, fv in zip(patch_ids, feats):
                feat_store.setdefault(size, {})[pid] = fv

            if (batch_idx + 1) % 10 == 0:
                print(f"    processed {batch_idx+1}/{len(loader)} batches")

    # attach to AnnData
    adata.uns['patch_features'] = {str(sz): feat_store[sz] for sz in sorted(feat_store)}

    # save
    print("💾 Saving AnnData...")
    os.makedirs(os.path.dirname(output_adata_path), exist_ok=True)
    adata.write_h5ad(output_adata_path)

    print(f"✅ Finished in {(time.time()-t0):.1f}s")
    for sz, d in adata.uns['patch_features'].items():
        print(f"   size {sz}: {len(d)} patches")


if __name__ == "__main__":
    extract_all_features(
        #image_path='/nfs/team361/data/sanger_whole_embryo/aligned_downsampled_2_reg.ome.tif',
        #adata_path='/nfs/team361/data/sanger_whole_embryo/adata_CS17.h5ad',
        image_path='//lustre/scratch126/cellgen/lotfollahi/mv10/3d-imputation-data/aligned_downsampled_2_reg.ome.tif',
        adata_path='//lustre/scratch126/cellgen/lotfollahi/mv10/3d-imputation-data/adata_CS17_2.h5ad',
        windows_csv='./embryo_data/spot_windows_hits.csv',
        spatial_scale=15.0,
        target_dim=1024,
        batch_size=32,
        num_workers=8,
        output_adata_path="./adata_with_features.h5ad"
    )

