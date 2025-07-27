import os
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader, Subset
from PIL import Image

from models.graph_construction import (
    calcADJ_from_distance_matrix_3D,
    extract_submatrix_3D,
    inter_layer_from_distance_matrix,
    build_adj_matrix_with_similarity_and_known_labels
)


def find_closest_column(target_sequence, tensor):
    target_sequence = torch.tensor(target_sequence)
    distances = torch.norm(tensor - target_sequence, dim=1)
    return torch.argmin(distances).item()


def retain_random_rows(labels, retain_ratio=0.5):
    mask = torch.rand(labels.size(0)) < retain_ratio
    return labels * mask.unsqueeze(1).to(labels.dtype)


class AnnDataImageGraphDataset(Dataset):
    def __init__(
        self,
        adata_224,            # AnnData with .X = 224‑counts, .obs[['x','y','sample']]
        adata_counts,         # AnnData with .obs['label_512'], .obs['label_1024']
        adata_features,       # AnnData with .obsm['feature_512'], .obsm['feature_1024']
        graph_path: str,
        adj_matrix_path: str,
        patch_size: int = 224,
        transform=None,
        level_split: int = 1,
        root_path: str = './'
    ):
        super().__init__()
        # --- store inputs ---
        self.adata_224      = adata_224
        self.adata_counts   = adata_counts
        self.adata_features = adata_features
        self.transform      = transform
        self.patch_size     = patch_size
        self.root_path      = root_path
        self.level_split    = level_split

        # --- prepare index list of all spot_ids ---
        self.spot_ids = list(self.adata_224.obs_names)

        # --- load graph & adjacency ---
        self.graph      = torch.load(graph_path, weights_only=False)
        self.adj_matrix = pd.read_csv(adj_matrix_path)

        # --- precompute train/test splits by ‘section_###’ in obs['sample'] ---
        levels = (
            self.adata_224.obs['sample']
            .str.extract(r'section_(\d+)', expand=False)
            .astype(int)
        )
        self.train_idx = [i for i, lvl in enumerate(levels) if lvl == self.level_split]
        self.test_idx  = [i for i, lvl in enumerate(levels) if lvl >  self.level_split]

    def __len__(self):
        return len(self.spot_ids)

    def __getitem__(self, idx):
        spot_id = self.spot_ids[idx]
        obs     = self.adata_224.obs.loc[spot_id]

        # 1) load & crop image patch around (x,y)
        slide_name = obs['sample']               # e.g. 'section_001'
        img_file   = os.path.join(self.root_path, f"{slide_name}.tiff")
        img        = Image.open(img_file).convert('RGB')

        x, y = obs[['x','y']].values.astype(int)
        half = self.patch_size // 2
        patch = img.crop((x-half, y-half, x-half+self.patch_size, y-half+self.patch_size))
        if self.transform:
            patch = self.transform(patch)

        # 2) gather labels & features
        label_224   = self.adata_224.X[idx].astype(np.float32)
        label_512   = self.adata_counts.obs.loc[spot_id, 'label_512'].astype(np.float32)
        label_1024  = self.adata_counts.obs.loc[spot_id, 'label_1024'].astype(np.float32)
        feat_512    = self.adata_features.obsm['feature_512'][idx].astype(np.float32)
        feat_1024   = self.adata_features.obsm['feature_1024'][idx].astype(np.float32)

        # 3) graph indices (closest in embedding space)
        idx_1024 = find_closest_column(label_1024, self.graph['layer_1024'].y)
        idx_512  = find_closest_column(label_512, self.graph['layer_512'].y)

        # 4) adjacency‑matrix lookup (column name pattern must match your CSV)
        col_name    = f"{slide_name[:2]}_{spot_id}"
        idx_matrix  = self.adj_matrix.columns.get_loc(col_name)

        # 5) the slide‑level integer
        level = int(obs['sample'].split('_')[-1])

        return (
            patch,                                # image tensor
            label_224,                            # counts @224
            (x, y),                               # position
            idx_1024,                             # graph idx 1024
            idx_512,                              # graph idx 512
            feat_1024,                            # feature vec 1024
            feat_512,                             # feature vec 512
            idx_matrix,                           # adjacency col
            level                                 # sample level
        )

    def get_train_dataset(self):
        return Subset(self, self.train_idx)


# --- reuse your original collate fns without change ---  
def collate_fn(batch, adj_matrix, graph, is_weighted=False):
    images, labels, positions, idx_1024_list, idx_512_list, features_1024, features_512, idx_matrix, level = zip(*batch)
    images     = torch.stack(images)
    labels     = torch.tensor(labels)
    positions  = torch.tensor(positions)
    # build sub‑matrices exactly as before…
    k = min(np.array(idx_matrix).shape[0], 12)
    sub_matrix      = extract_submatrix_3D(adj_matrix, np.array(idx_matrix))
    sub_inter_matrix= inter_layer_from_distance_matrix(sub_matrix, positions, k=4)
    adj_sub         = calcADJ_from_distance_matrix_3D(sub_matrix, k=k) + sub_inter_matrix
    adj_sub[adj_sub>0] = 1

    features_1024 = torch.tensor(features_1024)
    features_512  = torch.tensor(features_512)
    idx_1024      = torch.tensor(idx_1024_list)
    idx_512       = torch.tensor(idx_512_list)
    known_label   = retain_random_rows(labels, retain_ratio=0.2)

    adj_propagate = torch.tensor(0)
    if is_weighted:
        adj_propagate = adj_sub * torch.tensor((sub_matrix+1))

    return images, labels, adj_sub, adj_propagate, positions, idx_1024, idx_512, features_1024, features_512, graph, known_label


def collate_fn_result(batch, adj_matrix, graph, is_weighted=False):
    images, labels, positions, idx_1024_list, idx_512_list, features_1024, features_512, idx_matrix, level = zip(*batch)
    images    = torch.stack(images)
    labels    = torch.tensor(labels)
    positions = torch.tensor(positions)
    features_1024 = torch.tensor(features_1024)
    features_512  = torch.tensor(features_512)
    idx_1024      = torch.tensor(idx_1024_list)
    idx_512       = torch.tensor(idx_512_list)

    indices      = [i for i, lvl in enumerate(level) if lvl > 1]
    indices_eval = [i for i, lvl in enumerate(level) if lvl == 1]

    k = min(np.array(idx_matrix).shape[0]-1, 12)
    num_random_edges = min(np.array(idx_matrix).shape[0]-1, 10)
    sub_matrix      = extract_submatrix_3D(adj_matrix, np.array(idx_matrix))
    sub_inter_matrix= inter_layer_from_distance_matrix(sub_matrix, positions, k=4)

    adj_sub = (
        build_adj_matrix_with_similarity_and_known_labels(
            sub_matrix, indices_eval, k=k, num_random_edges=num_random_edges
        )
        + sub_inter_matrix
    )
    adj_sub[adj_sub>0] = 1

    known_label = labels.clone()
    eval_label  = labels.clone()
    known_label[indices]      = 0
    eval_label[indices_eval]  = 0

    adj_propagate = torch.tensor(0)
    if is_weighted:
        adj_propagate = adj_sub * torch.tensor((sub_matrix+1))
    level = torch.tensor(level)

    return images, eval_label, adj_sub, adj_propagate, positions, idx_1024, idx_512, features_1024, features_512, graph, known_label, indices, level


# --- factories for train/result loaders ---
def create_dataloaders_for_train(
    adata_224, adata_counts, adata_features,
    graph_path, adj_matrix_path,
    batch_size=32, transform=None,
    patch_size=224, level_split=1, root_path='./'
):
    dataset   = AnnDataImageGraphDataset(
        adata_224, adata_counts, adata_features,
        graph_path, adj_matrix_path,
        patch_size=patch_size, transform=transform,
        level_split=level_split, root_path=root_path
    )
    adj_df    = dataset.adj_matrix
    graph_obj = dataset.graph

    def collate(batch):
        return collate_fn(batch, adj_df, graph_obj, is_weighted=True)

    return {
        'train': DataLoader(
            dataset.get_train_dataset(),
            batch_size=batch_size, shuffle=True,
            collate_fn=collate
        )
    }


def create_dataloaders_for_result(
    adata_224, adata_counts, adata_features,
    graph_path, adj_matrix_path,
    batch_size=32, transform=None,
    patch_size=224, level_split=1, root_path='./'
):
    dataset = AnnDataImageGraphDataset(
        adata_224, adata_counts, adata_features,
        graph_path, adj_matrix_path,
        patch_size=patch_size, transform=transform,
        level_split=level_split, root_path=root_path
    )
    adj_df    = dataset.adj_matrix
    graph_obj = dataset.graph

    def collate_train(batch):
        return collate_fn(batch, adj_df, graph_obj, is_weighted=False)

    def collate_test(batch):
        return collate_fn_result(batch, adj_df, graph_obj, is_weighted=False)

    return {
        'train': DataLoader(
            dataset.get_train_dataset(),
            batch_size=batch_size, shuffle=True,
            collate_fn=collate_train
        ),
        'test': DataLoader(
            dataset,
            batch_size=batch_size, shuffle=False,
            collate_fn=collate_test
        )
    }
