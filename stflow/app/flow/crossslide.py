import bisect
import json
from typing import Dict, List, Optional

import torch
from scipy.spatial import KDTree

from stflow.hest_utils.file_utils import read_assets_from_h5
from stflow.hest_utils.st_dataset import load_adata
from stflow.data.dataset import _ensure_coords_have_z, HESTDatasetPath


class SlideRepository:
    """
    Cache of the profiled (training) sections: per-slide coordinates, expression and a
    KD-tree on (a, b). For a query section it returns, for every spot, the expression of
    the nearest spot by (a, b) on the closest profiled section above (z+1) and below (z-1).
    """

    def __init__(self, dataset_paths: List[HESTDatasetPath], normalize_method) -> None:
        self.normalize_method = normalize_method
        self.slides: Dict[str, Dict[str, object]] = {}
        for path in dataset_paths:
            self.slides[path.name] = self._load_slide(path)

        sorted_pairs = sorted(((data["z"], name) for name, data in self.slides.items()), key=lambda item: item[0])
        self.sorted_z = [z for z, _ in sorted_pairs]
        self.sorted_names = [name for _, name in sorted_pairs]
        self._name_to_index = {name: idx for idx, name in enumerate(self.sorted_names)}

    def _load_slide(self, dataset_path: HESTDatasetPath) -> Dict[str, object]:
        data_dict, _ = read_assets_from_h5(dataset_path.h5_path)
        barcodes = data_dict["barcodes"].flatten().astype(str).tolist()
        coords = _ensure_coords_have_z(torch.from_numpy(data_dict["coords"]).float())

        with open(dataset_path.gene_list_path, "r") as f:
            genes = json.load(f)["genes"]
        labels_df = load_adata(
            dataset_path.h5ad_path,
            genes=genes,
            barcodes=barcodes,
            normalize_method=self.normalize_method,
        )
        coords_np = coords[:, :2].cpu().numpy()
        return {
            "tree": KDTree(coords_np),
            "labels": torch.from_numpy(labels_df.values).float(),
            "z": coords[0, 2].item(),
        }

    def _neighbor_name(self, sample_name: str, direction: int, target_z: Optional[float]) -> Optional[str]:
        """Closest profiled section above (direction=+1) or below (direction=-1)."""
        if sample_name in self._name_to_index:
            idx = self._name_to_index[sample_name] + (1 if direction > 0 else -1)
        elif target_z is not None:
            # Section that is not profiled (held out / to be imputed): locate it in the z-stack.
            if direction > 0:
                idx = bisect.bisect_right(self.sorted_z, target_z)
            else:
                idx = bisect.bisect_left(self.sorted_z, target_z) - 1
        else:
            return None
        if 0 <= idx < len(self.sorted_names):
            return self.sorted_names[idx]
        return None

    def get_neighbor_expression(
        self,
        sample_name: str,
        coords_target: torch.Tensor,
        direction: int,
    ) -> Optional[torch.Tensor]:
        """
        Args:
            sample_name: query section
            coords_target: [N_valid, 3] original coordinates (a, b, z) of the query spots
            direction: +1 for the section above, -1 for the section below
        Returns:
            [N_valid, n_genes] expression of the matched spots, or None at the ends of the stack.
        """
        if coords_target.numel() == 0:
            return None
        target_z = float(coords_target[0, 2].item()) if coords_target.shape[-1] >= 3 else None
        neighbor_name = self._neighbor_name(sample_name, direction, target_z)
        if neighbor_name is None:
            return None
        neighbor = self.slides[neighbor_name]
        _, indices = neighbor["tree"].query(coords_target[:, :2].detach().cpu().numpy(), k=1)
        return neighbor["labels"][indices]


def prepare_crossslide_condition(
    sample_names: List[str],
    coords_original: torch.Tensor,
    spot_valid_mask: torch.Tensor,
    slide_repo: Optional[SlideRepository],
    n_genes: int,
    dtype: torch.dtype,
    device: torch.device,
) -> Optional[Dict[str, torch.Tensor]]:
    """
    Builds padded [B, N, n_genes] tensors with the matched expression of the sections
    above ('upper') and below ('lower') for each batch element.
    """
    if slide_repo is None:
        return None

    B, N, _ = coords_original.shape
    result: Dict[str, torch.Tensor] = {}
    for direction, key in ((1, "upper"), (-1, "lower")):
        tensors = []
        has_data = False
        for b, name in enumerate(sample_names):
            valid_mask = spot_valid_mask[b]
            expr = slide_repo.get_neighbor_expression(name, coords_original[b, valid_mask], direction)
            buffer = torch.zeros(N, n_genes, dtype=dtype)
            if expr is not None:
                buffer[valid_mask] = expr.to(dtype=dtype)
                has_data = True
            tensors.append(buffer)
        if has_data:
            result[key] = torch.stack(tensors).to(device=device, dtype=dtype)
    return result if result else None
