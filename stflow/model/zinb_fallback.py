import re
from typing import Dict, Iterable, List, Optional

import numpy as np
import torch
from scipy.spatial import KDTree


def _extract_slide_id(sample_name: str, fallback: int) -> int:
    """Parse trailing integer identifier from a slide name."""
    match = re.search(r"(\d+)$", sample_name)
    if match:
        return int(match.group(1))
    return fallback


class GeneCountFallback:
    """
    Build nearest-neighbour lookups over training slides so that gene-count
    statistics can be borrowed for slides that lack them.
    """

    def __init__(self, train_sp_datasets: Iterable):
        self._entries: Dict[str, Dict[str, object]] = {}
        self._neighbor_cache: Dict[str, str] = {}

        for idx, sp_data in enumerate(train_sp_datasets):
            sample_name = getattr(sp_data, "sample_name", None)
            if not sample_name:
                continue

            coords_tensor = getattr(sp_data, "coords_original", None)
            labels_tensor = getattr(sp_data, "labels", None)
            if coords_tensor is None or labels_tensor is None:
                continue

            coords_np = coords_tensor[:, :2].detach().cpu().numpy()
            if coords_np.size == 0:
                continue

            counts = labels_tensor.detach().cpu().clone()
            slide_id = _extract_slide_id(sample_name, fallback=idx)

            self._entries[sample_name] = {
                "id": slide_id,
                "coords": coords_np,
                "counts": counts,
                "tree": KDTree(coords_np) if coords_np.shape[0] > 0 else None,
            }

    def has_slides(self) -> bool:
        return bool(self._entries)

    def _lookup_entry(self, sample_name: str) -> Dict[str, object]:
        if not self._entries:
            raise RuntimeError("No training slides registered for GeneCountFallback.")

        if sample_name in self._entries:
            return self._entries[sample_name]

        if sample_name in self._neighbor_cache:
            neighbor_name = self._neighbor_cache[sample_name]
            return self._entries[neighbor_name]

        target_id = _extract_slide_id(sample_name, fallback=0)
        best_name: Optional[str] = None
        best_gap: Optional[int] = None

        for name, entry in self._entries.items():
            gap = abs(entry["id"] - target_id)
            if best_gap is None or gap < best_gap:
                best_gap = gap
                best_name = name

        if best_name is None:
            best_name = next(iter(self._entries))

        self._neighbor_cache[sample_name] = best_name
        return self._entries[best_name]

    def borrow_counts(
        self,
        sample_name: str,
        coords_original: torch.Tensor,
        valid_mask: torch.Tensor,
    ) -> torch.Tensor:
        """
        Borrow per-spot gene expression counts from the nearest training slide.
        """
        entry = self._lookup_entry(sample_name)
        mask_cpu = valid_mask.detach().cpu().bool()
        coords_cpu = coords_original.detach().cpu()
        coords_np = coords_cpu[mask_cpu, :2].numpy()

        counts_template = entry["counts"]
        num_genes = counts_template.shape[1]
        out = torch.zeros((coords_original.shape[0], num_genes), dtype=counts_template.dtype)

        if coords_np.size == 0:
            return out

        tree: KDTree = entry["tree"]
        if tree is None:
            return out

        _, indices = tree.query(coords_np, k=1)
        indices = np.atleast_1d(indices)
        borrowed = counts_template[indices]
        out[mask_cpu] = borrowed
        return out


def prepare_count_tensor(
    sample_names: List[str],
    coords_original: torch.Tensor,
    raw_counts: torch.Tensor,
    valid_mask: torch.Tensor,
    fallback_helper: Optional[GeneCountFallback],
    force_fallback: bool = False,
) -> torch.Tensor:
    """
    Produce counts suitable for phi_e feature construction, optionally borrowing
    statistics from neighbouring training slides.
    """
    counts = raw_counts.clone().float()
    B = counts.shape[0]

    for b in range(B):
        spot_mask = valid_mask[b].to(device=counts.device, dtype=torch.bool)
        counts[b, ~spot_mask] = 0

        if fallback_helper is None:
            continue

        need_fallback = force_fallback or torch.count_nonzero(counts[b, spot_mask]) == 0
        if not need_fallback:
            continue

        borrowed = fallback_helper.borrow_counts(
            sample_names[b],
            coords_original[b],
            spot_mask,
        )
        counts[b] = borrowed.to(counts.device)

    return counts
