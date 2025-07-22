import os
import time
from typing import List, Dict, Any, Tuple, Optional

import numpy as np
import pandas as pd
import torch
from torch import nn
from torchvision import models, transforms
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import tifffile
import anndata as ad
from tqdm import tqdm


class PatchDataset(Dataset):
    def __init__(
        self,
        img_array: np.ndarray,
        adata: ad.AnnData,
        windows_csv: str,
        spatial_scale: float,
        preprocess: transforms.Compose,
    ):
        # ensure [n_samples,H,W,C]
        if img_array.ndim == 4 and img_array.shape[1] in (1,3,4):
            img_array = np.moveaxis(img_array, 1, -1)
        self.img_array = img_array
        self.H, self.W = img_array.shape[1], img_array.shape[2]

        # sample name → slice index
        samples = list(adata.obs['sample'].unique())
        self.sample_to_idx = {s: i for i, s in enumerate(samples)}

        # build meta rows
        self.meta: List[Dict[str, Any]] = []
        half224 = 224 // 2
        coords = (adata.obsm['spatial'] * spatial_scale).astype(int)
        # track spot index mapping
        spot_ids = list(adata.obs.index)
        for spot_idx, (sample, (xc, yc)) in enumerate(zip(adata.obs['sample'], coords)):
            idx = self.sample_to_idx[sample]
            x0, y0 = xc - half224, yc - half224
            if 0 <= x0 <= self.W - 224 and 0 <= y0 <= self.H - 224:
                self.meta.append({
                    'sample_idx': idx,
                    'sample_id': sample,
                    'size': 224,
                    'x': x0,
                    'y': y0,
                    'spot_id': spot_ids[spot_idx],  # store AnnData obs index
                })

        df = pd.read_csv(windows_csv)
        for _, row in df.iterrows():
            s = str(row['sample'])
            if s not in self.sample_to_idx:
                continue
            idx = self.sample_to_idx[s]
            sz = int(row['window_size'])
            x0, y0 = int(row['x']), int(row['y'])
            if 0 <= x0 <= self.W - sz and 0 <= y0 <= self.H - sz:
                self.meta.append({
                    'sample_idx': idx,
                    'sample_id': s,
                    'size': sz,
                    'x': x0,
                    'y': y0,
                    'spot_id': '',  # empty for non-224 patches
                })

        # group for homogeneous batches
        self.by_size: Dict[int, List[int]] = {}
        for i, m in enumerate(self.meta):
            self.by_size.setdefault(m['size'], []).append(i)

        self.preprocess = preprocess

    def __len__(self) -> int:
        return len(self.meta)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, str, int, int, int, Optional[str]]:
        m = self.meta[idx]
        patch = self.img_array[
            m['sample_idx'], m['y']:m['y']+m['size'],
            m['x']:m['x']+m['size'], :
        ]
        img = Image.fromarray(patch)
        tensor = self.preprocess(img)
        # return tensor plus metadata including spot_id
        return tensor, m['sample_id'], m['size'], m['x'], m['y'], m.get('spot_id', '')


def collate_homogeneous(
    batch: List[Tuple[torch.Tensor, str, int, int, int, Optional[str]]]
) -> Tuple[torch.Tensor, List[str], List[int], List[int], List[int], List[Optional[str]]]:
    imgs, samples, sizes, xs, ys, spot_ids = zip(*batch)
    return (
        torch.stack(imgs, 0),
        list(samples),
        list(sizes),
        list(xs),
        list(ys),
        list(spot_ids),
    )


def build_feature_adata(
    sample_ids: List[str],
    sizes: List[int],
    xs: List[int],
    ys: List[int],
    spot_ids: List[Optional[str]],
    features: np.ndarray,
    target_dim: int
) -> ad.AnnData:
    """
    Create a new AnnData where obs has sample_id, size, x, y, spot_id;
    obsm['features'] holds matrix.
    """
    obs = pd.DataFrame({
        'sample_id': sample_ids,
        'size': sizes,
        'x': xs,
        'y': ys,
        'spot_id': spot_ids,
    })
    ad_new = ad.AnnData(
        obs = obs,
        dtype=np.float32
    )
    ad_new.obsm['features'] = features   # shape [n_patches, target_dim]
    return ad_new


def extract_all_features(
    image_path: str,
    adata_path: str,
    windows_csv: str,
    spatial_scale: float,
    target_dim: int = 1024,
    batch_size: int = 32,
    num_workers: int = 4,
    output_adata_path: str = './patch_features.h5ad'
) -> None:
    t0 = time.time()
    print("🧬 Loading image & AnnData…")
    #img_arr = tifffile.imread(image_path)
    adata = ad.read_h5ad(adata_path)

    # preprocessing + model
    preprocess = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225]),
    ])
    resnet = models.resnet50(pretrained=True)
    backbone = nn.Sequential(*list(resnet.children())[:-1]).eval()
    reducer  = nn.Linear(2048, target_dim).eval()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    backbone.to(device); reducer.to(device)

    # build dataset
    ds = PatchDataset(img_arr, adata, windows_csv, spatial_scale, preprocess)

    # collect all metadata + features
    all_samples: List[str] = []
    all_sizes:   List[int] = []
    all_xs:      List[int] = []
    all_ys:      List[int] = []
    all_spot_ids:List[Optional[str]] = []
    all_feats:   List[np.ndarray] = []

    # process each size separately
    for size, idxs in ds.by_size.items():
        print(f"Processing size {size}, {len(idxs)} patches…")
        loader = DataLoader(
            ds, batch_size=batch_size,
            sampler=torch.utils.data.SubsetRandomSampler(idxs),
            collate_fn=collate_homogeneous,
            num_workers=num_workers,
            pin_memory=True
        )
        for batch_imgs, samples, sizes, xs, ys, spot_ids in tqdm(loader):
            batch_imgs = batch_imgs.to(device)
            with torch.no_grad():
                f2048 = backbone(batch_imgs).view(len(batch_imgs), -1)
                f1024 = reducer(f2048).cpu().numpy()
            all_feats.append(f1024)
            all_samples.extend(samples)
            all_sizes.extend(sizes)
            all_xs.extend(xs)
            all_ys.extend(ys)
            all_spot_ids.extend(spot_ids)

    # stack features
    features = np.vstack(all_feats)  # [n_patches, target_dim]

    # build and save new AnnData
    ad_out = build_feature_adata(
        all_samples,
        all_sizes,
        all_xs,
        all_ys,
        all_spot_ids,
        features,
        target_dim
    )

    print("Saving output AnnData…")
    os.makedirs(os.path.dirname(output_adata_path), exist_ok=True)
    ad_out.write_h5ad(output_adata_path)

    print(f"Done in {time.time()-t0:.1f}s. Written {ad_out.n_obs} patches.")


if __name__ == "__main__":
    extract_all_features(
        image_path='/nfs/team361/data/sanger_whole_embryo/aligned_downsampled_2_reg.ome.tif',
        adata_path='/nfs/team361/data/sanger_whole_embryo/adata_CS17.h5ad',
        windows_csv='./embryo_data/spot_windows_hits.csv',
        spatial_scale=15.0,
        target_dim=1024,
        batch_size=32,
        num_workers=8,
        output_adata_path="./adata_with_features.h5ad"
    )
