# dataloader_3d.py

import numpy as np
import pandas as pd
import torch
import tifffile
from torch.utils.data import Dataset
from PIL import Image
from tqdm import tqdm
import scanpy as sc

from models.graph_construction import (
    extract_submatrix_3D,
    calcADJ_from_distance_matrix_3D,
    inter_layer_from_distance_matrix,
    build_adj_matrix_with_similarity_and_known_labels,
    combined_cosine_iou
)


def find_closest_column(target_sequence, tensor):
    target = torch.tensor(target_sequence)
    dists  = torch.norm(tensor - target, dim=1)
    return torch.argmin(dists).item()

def retain_random_rows(labels, retain_ratio=0.5):
    mask = torch.rand(labels.size(0)) < retain_ratio
    return labels * mask.unsqueeze(1).to(labels.dtype)

class AnnDataImageGraphDataset(Dataset):
    """
    Three‐AnnData + multi‐page TIFF loader.
     - adata_224      → .X, .obs['x','y','sample']
     - adata_counts   → .obs['label_512','label_1024']
     - adata_features → .obs['containing_512','containing_1024'], .obsm['feature_512','feature_1024']
     - TIFF            → shape (T,3,H,W)
    """
    def __init__(
        self,
        adata_224_path,
        adata_counts_path,
        adata_features_path,
        tiff_path: str,
        graph_path: str,
        spot_iou_path: str,
        spatial_scale: float = 15.0,
        patch_size: int = 224,
        transform=None
    ):
        super().__init__()
        self.transform   = transform
        self.patch_size  = patch_size

        # 1) memory-map the TIFF volume (T,3,H,W) → (T,H,W,C)
        self.vol = tifffile.imread(tiff_path)
        if self.vol.ndim == 4 and self.vol.shape[1] in (1,3,4):
            self.vol = np.moveaxis(self.vol, 1, -1)

        # 2) open AnnData in backed mode
        self.adata_224      = sc.read_h5ad(adata_224_path, backed='r')
        self.adata_counts   = sc.read_h5ad(adata_counts_path, backed='r')
        self.adata_features = sc.read_h5ad(adata_features_path, backed='r')

        obs224 = self.adata_224.obs
        # map sample names → TIFF slice index
        samples = list(obs224['sample'].astype(str).unique())
        self.sample_to_idx = {s:i for i,s in enumerate(samples)}

        # compute integer coords (scaled)
        coords        = (self.adata_224.obsm['spatial'] * spatial_scale).astype(int)
        self.x_coords = coords[:,0]
        self.y_coords = coords[:,1]
        self.sample_idxs = obs224['sample'].astype(str).map(self.sample_to_idx).to_numpy().astype(int)

        # 5) graph & adjacency CSV
        self.graph      = torch.load(graph_path, weights_only=False)
        self.spot_iou = pd.read_csv(spot_iou_path)

    def __len__(self):
        return self.adata_224.shape[0]

    def __getitem__(self, idx):
        # crop patch
        si = self.sample_idxs[idx]
        x, y = self.x_coords[idx], self.y_coords[idx]
        h = w = self.patch_size
        y0, x0 = y - h//2, x - w//2

        img = self.vol[si, y0:y0+h, x0:x0+w, :]
        img   = Image.fromarray(img)
        if self.transform:
            img = self.transform(img)

        spot_id = self.adata_224.obs.iloc[idx]['spot_id']
        # features for 224
        f224 = self.adata_features[self.adata_features.obs['spot_id']==spot_id].obsm['features']
        # gather labels & features
        c512 = self.adata_features[self.adata_features.obs['spot_id']==spot_id].obs['containing_512'].item()
        c1024= self.adata_features[self.adata_features.obs['spot_id']==spot_id].obs['containing_1024'].item()
        # pull counts from adata_counts.obs
        l224 = self.adata_224.X[idx]
        l512 = self.adata_counts[self.adata_counts.obs['spot_id']==c512].X
        l10  = self.adata_counts[self.adata_counts.obs['spot_id']==c1024].X
        f512 = self.adata_features[self.adata_features.obs['spot_id']==c512].obsm['features']
        f10  = self.adata_features[self.adata_features.obs['spot_id']==c1024].obsm['features']

        # graph indices
        idx_1024 = find_closest_column(l10,  self.graph['layer_1024'].y)
        idx_512  = find_closest_column(l512, self.graph['layer_512'].y)

        # slide index
        level = si

        return (
            img, l224, (x,y),
            idx_1024, idx_512,
            f10, f512, f224,
            spot_id, level
        )
#section_029_AACTTGCTAACGGTTC-1'
def collate_fn(batch, spot_iou, graph, is_weighted=False):
    imgs, l224s, poss, idx10s, idx5s, f10s, f5s, f224s, spot_id, levels = zip(*batch)
    images    = torch.stack(imgs)
    labels    = torch.tensor(np.array(l224s))
    positions = torch.tensor(np.array(poss))
    f224 = torch.tensor(np.array(f224s).squeeze(1))

    idx_arr   = np.array(spot_id)
    k         = min(idx_arr.shape[0], 12)
    subM      = combined_cosine_iou(f224, spot_id, spot_iou)
    inter     = inter_layer_from_distance_matrix(subM, positions, k=4)
    adj_sub   = calcADJ_from_distance_matrix_3D(subM, k=k) + inter
    adj_sub[adj_sub>0] = 1

    feat10 = torch.tensor(np.array(f10s).squeeze(1))
    feat5  = torch.tensor(np.array(f5s).squeeze(1))
    i10    = torch.tensor(idx10s)
    i5     = torch.tensor(idx5s)
    known  = retain_random_rows(labels, retain_ratio=0.2)

    prop = torch.tensor(0)
    if is_weighted:
        prop = adj_sub * (subM + 1).detach().clone()       # good


    return (images, labels, adj_sub, prop,
            positions, i10, i5, feat10, feat5,
            graph, known)

def collate_fn_result(batch, spot_iou, graph, is_weighted=False, train_slides=None):
    imgs, l224s, poss, idx10s, idx5s, f10s, f5s, f224s, spot_id, levels = zip(*batch)
    images    = torch.stack(imgs)
    labels    = torch.tensor(np.array(l224s))
    positions = torch.tensor(np.array(poss))
    f224 = torch.tensor(np.array(f224s).squeeze(1))

    feat10 = torch.tensor(f10s)
    feat5  = torch.tensor(f5s)
    i10    = torch.tensor(idx10s)
    i5     = torch.tensor(idx5s)

    # Set default train_slides if not provided (fallback to old behavior)
    if train_slides is None:
        indices      = [i for i, lvl in enumerate(levels) if lvl >  0]
        indices_eval = [i for i, lvl in enumerate(levels) if lvl == 0]
    else:
        # New logic: indices_eval = training slides, indices = validation slides
        indices_eval = [i for i, lvl in enumerate(levels) if lvl in train_slides]
        indices      = [i for i, lvl in enumerate(levels) if lvl not in train_slides]

    idx_arr = np.array(spot_id)
    k       = min(idx_arr.shape[0]-1, 12)
    subM    = combined_cosine_iou(f224, spot_id, spot_iou)
    inter   = inter_layer_from_distance_matrix(subM, positions, k=4)
    adj_sub = (
        build_adj_matrix_with_similarity_and_known_labels(
            subM, indices_eval, k=k, num_random_edges=min(idx_arr.shape[0]-1,10)
        ) + inter
    )
    adj_sub[adj_sub>0] = 1

    known = labels.clone()
    eval_ = labels.clone()
    known[indices]      = 0
    eval_[indices_eval] = 0

    prop = torch.tensor(0)
    if is_weighted:
        prop = adj_sub * (subM + 1).detach().clone()       # good

    return (images, eval_, adj_sub, prop,
            positions, i10, i5, feat10, feat5,
            graph, known, indices, torch.tensor(levels))

def create_dataloaders_for_train(
    adata_224, adata_counts, adata_features,
    tiff_path, graph_path, spot_iou_path,
    spatial_scale=15.0,
    batch_size=32, patch_size=224, transform=None
):
    ds = AnnDataImageGraphDataset(
        adata_224, adata_counts, adata_features,
        tiff_path, graph_path, spot_iou_path,
        spatial_scale=spatial_scale,
        patch_size=patch_size,
        transform=transform
    )
    return {
        'train': torch.utils.data.DataLoader(
            ds,
            batch_size=batch_size,
            shuffle=True,
            collate_fn=lambda b: collate_fn(b, ds.spot_iou, ds.graph, True),
            num_workers=4,
            pin_memory=True
        )
    }

def create_dataloaders_for_result(
    adata_224, adata_counts, adata_features,
    tiff_path, graph_path, spot_iou_path,
    spatial_scale=15.0,
    batch_size=32, patch_size=224, transform=None, train_slides=None
):
    ds = AnnDataImageGraphDataset(
        adata_224, adata_counts, adata_features,
        tiff_path, graph_path, spot_iou_path,
        spatial_scale=spatial_scale,
        patch_size=patch_size,
        transform=transform
    )
    return {
        'train': torch.utils.data.DataLoader(
            ds,
            batch_size=batch_size,
            shuffle=True,
            collate_fn=lambda b: collate_fn(b, ds.spot_iou, ds.graph, False),
            num_workers=4,
            pin_memory=True
        ),
        'test': torch.utils.data.DataLoader(
            ds,
            batch_size=batch_size,
            shuffle=True,
            collate_fn=lambda b: collate_fn_result(b, ds.spot_iou, ds.graph, False, train_slides),
            num_workers=4,
            pin_memory=True
        )
    }
