import os
import json
import numpy as np
from typing import List
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.data import Dataset

from stflow.hest_utils.st_dataset import load_adata
from stflow.hest_utils.file_utils import read_assets_from_h5
from .sampling_utils import PatchSampler
from tqdm import tqdm
from scipy.spatial import KDTree


def _ensure_coords_have_z(coords):
    """
    Guarantee coords have three columns (x, y, z). If z is missing, append zeros.
    Works for both numpy arrays and torch tensors.
    """
    if coords.ndim != 2:
        raise ValueError(f"Expected 2D coordinates array, got shape {coords.shape}")
    if coords.shape[1] == 3:
        return coords
    if coords.shape[1] != 2:
        raise ValueError(f"Coordinates must have 2 or 3 columns, got {coords.shape[1]}")

    if isinstance(coords, np.ndarray):
        zeros = np.zeros((coords.shape[0], 1), dtype=coords.dtype)
        return np.concatenate([coords, zeros], axis=1)
    elif isinstance(coords, torch.Tensor):
        zeros = torch.zeros(coords.shape[0], 1, dtype=coords.dtype, device=coords.device)
        return torch.cat([coords, zeros], dim=1)
    else:
        raise TypeError(f"Unsupported coordinate type: {type(coords)}")

class HESTDatasetPath:
    name: str | None = None
    h5_path: str | None = None
    h5ad_path: str | None = None
    gene_list_path: str | None = None

    def __init__(self, name, h5_path, h5ad_path, gene_list_path, **kwargs):
        self.name = name
        self.h5_path = h5_path
        self.h5ad_path = h5ad_path
        self.gene_list_path = gene_list_path

        for k, v in kwargs.items():
            setattr(self, k, v)


class SPData:
    features: torch.Tensor | None = None
    labels: torch.Tensor | None = None
    coords: torch.Tensor | None = None
    coords_original: torch.Tensor | None = None
    detected_counts: torch.Tensor | None = None
    sample_name: str | None = None
    barcodes: list[str] | None = None

    def __init__(
        self,
        features,
        labels,
        coords,
        barcodes: list[str] | None = None,
        use_global_counts: bool = False,
    ):
        self.features = features
        self.labels = labels
        coords = _ensure_coords_have_z(coords)
        if isinstance(coords, torch.Tensor):
            self.coords_original = coords.clone()  # Store original coords (x, y, z) before decentering
            self.coords = coords.clone()
        else:
            raise TypeError("coords must be a torch.Tensor when constructing SPData")

        self.detected_counts = (self.labels > 0).sum(dim=1).float()
        self.original_indices = torch.arange(self.features.size(0), dtype=torch.long)
        self.slide_index = -1
        self.global_detected_counts: torch.Tensor | None = None
        self.global_detected_mask: torch.Tensor | None = None
        self.global_max_spots: int | None = None
        self._use_global_counts = use_global_counts
        if barcodes is not None:
            if len(barcodes) != self.features.size(0):
                raise ValueError(
                    f"Expected {self.features.size(0)} barcodes, received {len(barcodes)}."
                )
            self.barcodes = list(barcodes)
        else:
            self.barcodes = None

        # Decenter only x and y, keep z unchanged
        self.coords[:, 0] = self.coords[:, 0] - self.coords[:, 0].mean()
        self.coords[:, 1] = self.coords[:, 1] - self.coords[:, 1].mean()
        # coords[:, 2] (z) remains the same for all spots on a slide

    def __len__(self):
        return len(self.features)
    
    @property
    def z_index(self):
        """Get the z-index for this slide (all spots have same z)."""
        return self.coords_original[0, 2].item()

    def chunk(self, index):
        chunked = SPData.__new__(SPData)
        chunked.features = self.features[index]
        chunked.labels = self.labels[index]
        chunked.coords = self.coords[index]
        chunked.coords_original = self.coords_original[index]
        chunked.detected_counts = self.detected_counts[index]
        chunked.original_indices = self.original_indices[index]
        chunked.slide_index = self.slide_index
        if self.barcodes is not None:
            if isinstance(index, torch.Tensor):
                idx_list = index.detach().cpu().long().tolist()
            else:
                idx_list = [int(i) for i in np.atleast_1d(index)]
            chunked.barcodes = [self.barcodes[i] for i in idx_list]
        else:
            chunked.barcodes = None
        if self._use_global_counts:
            chunked.global_detected_counts = self.global_detected_counts
            chunked.global_detected_mask = self.global_detected_mask
            chunked.global_max_spots = self.global_max_spots
        else:
            chunked.global_detected_counts = None
            chunked.global_detected_mask = None
            chunked.global_max_spots = None
        chunked._use_global_counts = self._use_global_counts
        chunked.sample_name = self.sample_name
        return chunked


class HESTDataset(Dataset):
    def __init__(
        self,
        dataset: HESTDatasetPath,
        normalize_method,
        distribution="beta_3_1",
        sample_times=5,
        use_global_counts: bool = False,
    ):
        super().__init__()

        self.name = dataset.name
        data_dict, _ = read_assets_from_h5(dataset.h5_path)
        barcodes = data_dict["barcodes"].flatten().astype(str).tolist()
        coords = _ensure_coords_have_z(data_dict["coords"])
        embeddings = data_dict["embeddings"]

        with open(os.path.join(dataset.gene_list_path), 'r') as f:
            genes = json.load(f)['genes']
        
        self.gene_list = genes
        self.h5ad_path = dataset.h5ad_path
        self.barcode_list = barcodes
        labels = load_adata(dataset.h5ad_path, genes=genes, barcodes=barcodes, normalize_method=normalize_method)
        labels = labels.values

        self.n_chunks = sample_times
        self.patch_sampler = PatchSampler(distribution)

        self.sp_dataset = SPData(
            features=torch.from_numpy(embeddings).float(),
            labels=torch.from_numpy(labels).float(),
            coords=torch.from_numpy(coords).float(),
            barcodes=barcodes,
            use_global_counts=use_global_counts,
        )
        self.sp_dataset.sample_name = dataset.name
        if use_global_counts:
            self.sp_dataset.slide_index = 0
            self.sp_dataset.global_detected_counts = self.sp_dataset.detected_counts.clone()
            self.sp_dataset.global_detected_mask = torch.ones_like(self.sp_dataset.detected_counts, dtype=torch.bool)
            self.sp_dataset.global_max_spots = self.sp_dataset.detected_counts.size(0)
        
    def __len__(self):
        return self.n_chunks

    def __getitem__(self, idx):
        # For sampling, use only x,y coordinates (first 2 dims)
        return self.sp_dataset.chunk(self.patch_sampler(self.sp_dataset.coords[:, :2]))


class MultiHESTDataset(Dataset):
    def __init__(
        self,
        dataset_list: List[HESTDatasetPath],
        normalize_method,
        distribution="beta_3_1",
        sample_times=5,
        neighbor_mode: str = "spatial",
        cross_slide_neighbors: int = 2,
        neighbor_candidate_count: int = 64,
        neighbor_alpha: float = 0.5,
        neighbor_vector: str = "expression",
        use_global_counts: bool = False,
    ):
        super().__init__()

        self.dataset_list = dataset_list
        self.sp_datasets = []
        self.n_chunks, self.sample_times = [], sample_times
        self.patch_sampler = PatchSampler(distribution)
        self.use_global_counts = use_global_counts

        # Neighbour selection options
        valid_modes = {"spatial", "cosine"}
        if neighbor_mode not in valid_modes:
            raise ValueError(f"neighbor_mode must be one of {valid_modes}, got {neighbor_mode}")
        self.neighbor_mode = neighbor_mode
        self.cross_slide_neighbors = max(0, cross_slide_neighbors)
        self.neighbor_candidate_count = max(1, neighbor_candidate_count)
        # Blend weight between cosine similarity and geometric proximity
        self.neighbor_alpha = float(np.clip(neighbor_alpha, 0.0, 1.0))
        valid_vectors = {"expression", "embedding"}
        if neighbor_vector not in valid_vectors:
            raise ValueError(f"neighbor_vector must be one of {valid_vectors}, got {neighbor_vector}")
        self.neighbor_vector = neighbor_vector

        slide_detected_counts: list[torch.Tensor] = []

        for i, dataset in enumerate(tqdm(self.dataset_list)):
            data_dict, _ = read_assets_from_h5(dataset.h5_path)
            barcodes = data_dict["barcodes"].flatten().astype(str).tolist()
            coords = _ensure_coords_have_z(data_dict["coords"])
            embeddings = data_dict["embeddings"]

            with open(os.path.join(dataset.gene_list_path), 'r') as f:
                genes = json.load(f)['genes']

            labels = load_adata(dataset.h5ad_path, genes=genes, barcodes=barcodes, normalize_method=normalize_method)
            labels = labels.values

            self.n_chunks.append(sample_times)

            self.sp_datasets.append(
                SPData(
                    features=torch.from_numpy(embeddings).float(),
                    labels=torch.from_numpy(labels).float(),
                    coords=torch.from_numpy(coords).float(),  # [N, 3]
                    barcodes=barcodes,
                    use_global_counts=use_global_counts,
                )
            )
            if use_global_counts:
                sp_data = self.sp_datasets[-1]
                sp_data.slide_index = i
                slide_detected_counts.append(sp_data.detected_counts.clone())
            self.sp_datasets[-1].sample_name = dataset.name
        
        # Build z-index to dataset mapping for fast nearest-slide lookup
        self.z_to_dataset_idx = {sp.z_index: i for i, sp in enumerate(self.sp_datasets)}

        if self.use_global_counts:
            if not slide_detected_counts:
                raise ValueError("No slides loaded; cannot build detected-count matrix.")

            self.max_spots = max(counts.size(0) for counts in slide_detected_counts)
            num_slides = len(slide_detected_counts)
            detected_matrix = torch.zeros((num_slides, self.max_spots), dtype=torch.float32)
            detected_mask = torch.zeros((num_slides, self.max_spots), dtype=torch.bool)

            for idx, counts in enumerate(slide_detected_counts):
                length = counts.size(0)
                detected_matrix[idx, :length] = counts
                detected_mask[idx, :length] = True
                self.sp_datasets[idx].global_detected_counts = detected_matrix[idx].clone()
                self.sp_datasets[idx].global_detected_mask = detected_mask[idx].clone()
                self.sp_datasets[idx].global_max_spots = self.max_spots

            self.detected_count_matrix = detected_matrix
            self.detected_count_mask = detected_mask
        else:
            self.max_spots = None
            self.detected_count_matrix = None
            self.detected_count_mask = None

    def _select_candidate_slides(self, target_z: float):
        """Return a list of dataset indices sorted by |Δz|."""

        print('test')
        available = list(self.z_to_dataset_idx.items())
        if not available:
            raise ValueError("No slides available in training dataset")
        if self.cross_slide_neighbors == 0:
            return []
        # Exclude target z (if present) when possible to focus on adjacent slides
        filtered = [(z, idx) for z, idx in available if z != target_z]
        if not filtered:
            filtered = available
        # Sort by absolute z distance
        filtered.sort(key=lambda item: abs(item[0] - target_z))
        return filtered[: self.cross_slide_neighbors]

    @staticmethod
    def _ensure_2d_numpy(coords):
        if isinstance(coords, torch.Tensor):
            coords_np = coords.detach().cpu().numpy()
        else:
            coords_np = coords
        return coords_np[:, :2]

    @staticmethod
    def _normalize_tensor(values: torch.Tensor, eps: float = 1e-6):
        if values.numel() == 0:
            return values
        min_v = values.min()
        max_v = values.max()
        if (max_v - min_v) < eps:
            return torch.ones_like(values)
        return (values - min_v) / (max_v - min_v + eps)

    def _lookup_target_vectors(
        self,
        target_z: float,
        coords_target: torch.Tensor,
        target_expression: torch.Tensor | None,
        target_features: torch.Tensor | None,
    ):
        """
        Retrieve the vector representation for each target spot used in cosine similarity.
        Preference order:
            1. Provided tensors (expression or embedding).
            2. Lookup from stored slide with identical z.
        """
        if self.neighbor_vector == "expression":
            if target_expression is not None:
                return target_expression.detach().cpu()
            source_attr = "labels"
        else:
            if target_features is not None:
                return target_features.detach().cpu()
            source_attr = "features"

        if target_z not in self.z_to_dataset_idx:
            raise ValueError(
                "Target z not found in training slides and no target vector supplied; "
                "cannot compute cosine similarity."
            )

        dataset_idx = self.z_to_dataset_idx[target_z]
        slide = self.sp_datasets[dataset_idx]
        source_matrix = getattr(slide, source_attr)  # torch.Tensor

        # Match coordinates via KDTree to handle patch subsets
        slide_coords_np = slide.coords_original[:, :2].cpu().numpy()
        tree = KDTree(slide_coords_np)
        coords_target_np = self._ensure_2d_numpy(coords_target)
        _, matched_indices = tree.query(coords_target_np, k=1)
        return source_matrix[matched_indices]

    def _get_neighbor_data_spatial(self, target_z, coords_target, k):
        """
        Original spatial-only neighbour lookup used by downstream code.
        """
        available_z_values = list(self.z_to_dataset_idx.keys())
        if not available_z_values:
            raise ValueError("No slides available in training dataset")

        if len(available_z_values) > 1:
            candidate_z_values = [z for z in available_z_values if z != target_z]
            if not candidate_z_values:
                candidate_z_values = available_z_values
        else:
            candidate_z_values = available_z_values

        z_distances = [abs(target_z - z) for z in candidate_z_values]
        nearest_z = candidate_z_values[z_distances.index(min(z_distances))]
        nearest_dataset_idx = self.z_to_dataset_idx[nearest_z]

        neighbor_data = self.sp_datasets[nearest_dataset_idx]
        neighbor_coords = neighbor_data.coords_original[:, :2].cpu().numpy()
        neighbor_labels = neighbor_data.labels

        tree = KDTree(neighbor_coords)
        target_coords_np = self._ensure_2d_numpy(coords_target)

        k_actual = min(k, len(neighbor_coords))
        distances, indices = tree.query(target_coords_np, k=k_actual)

        if k_actual == 1:
            indices = np.expand_dims(indices, axis=1)

        neighbor_expr = neighbor_labels[indices]  # [N_spots, k_actual, N_genes]
        return neighbor_expr

    def _get_neighbor_data_cosine(
        self,
        target_z,
        coords_target,
        k,
        target_expression: torch.Tensor | None = None,
        target_features: torch.Tensor | None = None,
    ):
        if self.neighbor_vector == "embedding":
            raise NotImplementedError(
                "neighbor_vector='embedding' is not yet supported for cosine ranking."
            )

        # Resolve target vectors for cosine similarity
        target_vectors = self._lookup_target_vectors(
            target_z, coords_target, target_expression, target_features
        )

        candidate_slides = self._select_candidate_slides(target_z)
        if not candidate_slides:
            # Fallback to spatial if nothing else available
            return self._get_neighbor_data_spatial(target_z, coords_target, k)

        target_coords_np = self._ensure_2d_numpy(coords_target)
        N_spots = target_coords_np.shape[0]

        # Accumulate candidate information across slides
        candidate_vectors_per_spot = [[] for _ in range(N_spots)]
        candidate_dist_per_spot = [[] for _ in range(N_spots)]

        for z_value, dataset_idx in candidate_slides:
            neighbor_data = self.sp_datasets[dataset_idx]
            neighbor_coords = neighbor_data.coords_original[:, :2].cpu().numpy()
            if len(neighbor_coords) == 0:
                continue

            tree = KDTree(neighbor_coords)
            k_candidates = min(self.neighbor_candidate_count, len(neighbor_coords))
            distances, indices = tree.query(target_coords_np, k=k_candidates)

            if k_candidates == 1:
                distances = np.expand_dims(distances, axis=1)
                indices = np.expand_dims(indices, axis=1)

            candidate_matrix = neighbor_data.labels

            for spot_idx in range(N_spots):
                idx_row = torch.as_tensor(indices[spot_idx], dtype=torch.long)
                dist_row = torch.from_numpy(distances[spot_idx]).float()
                candidate_vectors_per_spot[spot_idx].append(candidate_matrix[idx_row])
                candidate_dist_per_spot[spot_idx].append(dist_row)

        neighbor_expr_outputs = []
        for spot_idx in range(N_spots):
            vectors_list = candidate_vectors_per_spot[spot_idx]
            dist_list = candidate_dist_per_spot[spot_idx]

            if not vectors_list:
                # No candidates gathered (e.g., single slide). Fallback to spatial.
                spatial_neighbors = self._get_neighbor_data_spatial(target_z, coords_target[spot_idx : spot_idx + 1], k)
                spatial_neighbors = spatial_neighbors.squeeze(0)
                neighbor_expr_outputs.append(spatial_neighbors)
                continue

            candidate_vectors = torch.cat(vectors_list, dim=0)  # [C, dim]
            candidate_dists = torch.cat(dist_list, dim=0)  # [C]

            # Cosine similarity between target vector and candidates
            target_vec = target_vectors[spot_idx].unsqueeze(0)
            cos_scores = F.cosine_similarity(candidate_vectors, target_vec, dim=1)
            cos_scores = (cos_scores + 1.0) * 0.5  # map [-1,1] -> [0,1]

            # Convert distances to similarity (closer -> higher weight)
            distance_scores = torch.reciprocal(candidate_dists + 1e-3)
            distance_scores = self._normalize_tensor(distance_scores)

            blended_scores = self.neighbor_alpha * cos_scores + (1.0 - self.neighbor_alpha) * distance_scores

            k_actual = min(k, candidate_vectors.shape[0])
            top_scores, top_indices = torch.topk(blended_scores, k=k_actual, largest=True)

            neighbor_expr_outputs.append(candidate_vectors[top_indices])

        neighbor_expr = torch.stack(
            [
                expr if expr.dim() > 1 else expr.unsqueeze(0)
                for expr in neighbor_expr_outputs
            ],
            dim=0,
        )
        return neighbor_expr

    def get_neighbor_data(
        self,
        target_z,
        coords_target,
        k,
        target_expression: torch.Tensor | None = None,
        target_features: torch.Tensor | None = None,
    ):
        """
        Get expression data from nearest slides with optional cosine re-ranking.
        """
        print('I am here')
        if self.neighbor_mode == "cosine":
            try:
                return self._get_neighbor_data_cosine(
                    target_z,
                    coords_target,
                    k,
                    target_expression=target_expression,
                    target_features=target_features,
                )
            except ValueError:
                # Fallback to spatial if cosine ranking cannot be computed
                return self._get_neighbor_data_spatial(target_z, coords_target, k)
        else:
            return self._get_neighbor_data_spatial(target_z, coords_target, k)
        
    def __len__(self):
        return sum(self.n_chunks)

    def __getitem__(self, idx):
        for i, n_chunk in enumerate(self.n_chunks):
            if idx < n_chunk:
                # Use only x,y for patch sampling
                return self.sp_datasets[i].chunk(
                        self.patch_sampler(self.sp_datasets[i].coords[:, :2])
                    )
            idx -= n_chunk


def padding_batcher(return_global_counts: bool = False):
    def batcher_dev(batch):
        features = [d.features for d in batch]
        labels = [d.labels for d in batch]
        coords = [d.coords for d in batch]  # [N, 3] with z
        coords_original = [d.coords_original for d in batch]  # [N, 3] with z
        detected_counts = [d.detected_counts for d in batch]
        sample_names = [getattr(d, "sample_name", "") for d in batch]
        barcode_lists = [getattr(d, "barcodes", None) for d in batch]
        if return_global_counts:
            global_detected_counts = [d.global_detected_counts for d in batch]
            global_detected_masks = [d.global_detected_mask for d in batch]
            spot_index_lists = [d.original_indices for d in batch]
        
        # Extract z-indices (all spots on same slide have same z)
        z_indices = [d.coords_original[0, 2] for d in batch]  # Get z from first spot

        lengths = [x.size(0) for x in features]
        max_len = max(lengths)
        features = torch.stack([F.pad(x, (0, 0, 0, max_len - x.size(0))) for x in features])
        labels = torch.stack([F.pad(x, (0, 0, 0, max_len - x.size(0))) for x in labels])
        coords = torch.stack([F.pad(x, (0, 0, 0, max_len - x.size(0))) for x in coords])
        coords_original = torch.stack([F.pad(x, (0, 0, 0, max_len - x.size(0))) for x in coords_original])
        detected_counts = torch.stack([F.pad(x, (0, max_len - x.size(0))) for x in detected_counts])

        spot_valid_mask = []
        for length in lengths:
            mask = torch.zeros(max_len, dtype=torch.bool)
            mask[:length] = True
            spot_valid_mask.append(mask)
        spot_valid_mask = torch.stack(spot_valid_mask)

        padded_barcodes: list[list[str]] = []
        for bc_list, length in zip(barcode_lists, lengths):
            if bc_list is None:
                base_list = [""] * length
            else:
                base_list = list(bc_list)
                if len(base_list) < length:
                    base_list = base_list + [""] * (length - len(base_list))
                elif len(base_list) > length:
                    base_list = base_list[:length]
            padded = base_list + [""] * (max_len - length)
            padded_barcodes.append(padded)

        if return_global_counts:
            padded_spot_indices = []
            for idx in spot_index_lists:
                pad_tensor = torch.full((max_len,), -1, dtype=torch.long)
                pad_tensor[: idx.size(0)] = idx.long()
                padded_spot_indices.append(pad_tensor)
            spot_indices = torch.stack(padded_spot_indices)

            global_detected_counts = torch.stack([g.clone() for g in global_detected_counts])
            global_detected_masks = torch.stack([m.clone() for m in global_detected_masks])
        z_indices = torch.tensor(z_indices, dtype=torch.float32)

        # Return only 2D coords to the model; keep full coords_original (x,y,z) for prior
        coords_2d = coords[:, :, :2]

        if return_global_counts:
            return (
                features,
                coords_2d,
                labels,
                coords_original,
                z_indices,
                detected_counts,
                spot_valid_mask,
                sample_names,
                padded_barcodes,
                global_detected_counts,
                global_detected_masks,
                spot_indices,
            )
        else:
            return (
                features,
                coords_2d,
                labels,
                coords_original,
                z_indices,
                detected_counts,
                spot_valid_mask,
                sample_names,
                padded_barcodes,
            )
    return batcher_dev