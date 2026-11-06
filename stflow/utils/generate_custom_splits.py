#!/usr/bin/env python3
"""
Generate multiple train/test split CSVs for STFlow datasets without touching the
existing `train_0.csv` and `test_0.csv`.

Usage:
    python generate_custom_splits.py \
        --source_dataroot /path/to/dataset/root \
        --dataset EMBRYO
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path
from typing import Iterable, List, Tuple

import pandas as pd


def load_sample_ids(dataset_root: Path) -> List[str]:
    """Return sorted sample IDs based on the per-sample AnnData files."""
    adata_dir = dataset_root / "adata"
    if not adata_dir.is_dir():
        raise FileNotFoundError(f"Cannot find adata directory at {adata_dir}")
    sample_ids = sorted(p.stem for p in adata_dir.glob("*.h5ad"))
    if not sample_ids:
        raise ValueError(f"No .h5ad files found under {adata_dir}")
    return sample_ids


def make_split_frame(sample_ids: Iterable[str]) -> pd.DataFrame:
    """Build the standard split DataFrame structure."""
    sample_ids = list(sample_ids)
    return pd.DataFrame(
        {
            "sample_id": sample_ids,
            "patches_path": [f"patches/{sid}.h5" for sid in sample_ids],
            "expr_path": [f"adata/{sid}.h5ad" for sid in sample_ids],
        }
    )


def _next_split_indices(split_dir: Path, n: int) -> Iterable[Tuple[int, Path, Path]]:
    """
    Yield successive indices (train_i/test_i) based on existing files.
    Ensures new files don't clobber existing train/test CSVs.
    """
    existing = []
    pattern = re.compile(r"train_(\d+)\.csv")
    for train_file in split_dir.glob("train_*.csv"):
        match = pattern.fullmatch(train_file.name)
        if match:
            existing.append(int(match.group(1)))
    next_idx = max(existing) + 1 if existing else 0
    for _ in range(n):
        yield_idx = next_idx
        train_path = split_dir / f"train_{yield_idx}.csv"
        test_path = split_dir / f"test_{yield_idx}.csv"
        next_idx += 1
        yield yield_idx, train_path, test_path


def save_split(dataset_root: Path, train_samples: Iterable[str], test_samples: Iterable[str], description: str) -> None:
    """Persist train/test CSVs under dataset_root/splits using sequential indices."""
    split_dir = dataset_root / "splits"
    split_dir.mkdir(exist_ok=True)
    iterator = _next_split_indices(split_dir, 1)
    idx, train_path, test_path = next(iterator)
    train_df = make_split_frame(train_samples)
    test_df = make_split_frame(test_samples)
    train_df.to_csv(train_path, index=False)
    test_df.to_csv(test_path, index=False)
    print(
        f"  - Saved scenario '{description}' as train_{idx}.csv ({len(train_df)} samples) "
        f"and test_{idx}.csv ({len(test_df)} samples)"
    )


def first_half_vs_last_half(samples: List[str], dataset_root: Path) -> None:
    mid = len(samples) // 2
    train, test = samples[:mid], samples[mid:]
    print(f"First half train ({len(train)}) vs last half test ({len(test)})")
    save_split(dataset_root, train, test, "first_half_vs_last_half")


def middle_vs_ends(samples: List[str], dataset_root: Path, start: int = 10, end: int = 40) -> None:
    start = max(0, start)
    end = min(len(samples) - 1, end)
    train = [sid for idx, sid in enumerate(samples) if start <= idx <= end]
    test = [sid for idx, sid in enumerate(samples) if idx < start or idx > end]
    print(f"Middle indices {start}-{end} train ({len(train)}) vs ends test ({len(test)})")
    save_split(dataset_root, train, test, f"middle_{start}_{end}_vs_ends")


def even_vs_odd(samples: List[str], dataset_root: Path) -> None:
    train = [sid for idx, sid in enumerate(samples) if idx % 2 == 0]
    test = [sid for idx, sid in enumerate(samples) if idx % 2 == 1]
    print(f"Even-index train ({len(train)}) vs odd-index test ({len(test)})")
    save_split(dataset_root, train, test, "even_idx_vs_odd_idx")


def first_vs_rest(samples: List[str], dataset_root: Path) -> None:
    if not samples:
        raise ValueError("No samples found for first-vs-rest scenario.")
    first = [samples[0]]
    rest = samples[1:]
    print(f"First slide train ({len(first)}) vs rest test ({len(rest)})")
    save_split(dataset_root, first, rest, "first_vs_rest")


def last_vs_rest(samples: List[str], dataset_root: Path) -> None:
    if not samples:
        raise ValueError("No samples found for last-vs-rest scenario.")
    last = [samples[-1]]
    rest = samples[:-1]
    print(f"Last slide train ({len(last)}) vs rest test ({len(rest)})")
    save_split(dataset_root, last, rest, "last_vs_rest")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate alternative split CSVs.")
    parser.add_argument(
        "--source_dataroot",
        required=True,
        help="Root directory containing <dataset>/adata, patches, splits, etc.",
    )
    parser.add_argument(
        "--dataset",
        required=True,
        help="Dataset folder name under source_dataroot (e.g., EMBRYO).",
    )
    parser.add_argument(
        "--mid_start",
        type=int,
        default=10,
        help="Starting index (inclusive) for the middle-slice training set.",
    )
    parser.add_argument(
        "--mid_end",
        type=int,
        default=40,
        help="Ending index (inclusive) for the middle-slice training set.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    dataset_root = Path(args.source_dataroot) / args.dataset
    print(f"Loading sample IDs from {dataset_root / 'adata'}")
    samples = load_sample_ids(dataset_root)
    print(f"Found {len(samples)} samples.")

    first_half_vs_last_half(samples, dataset_root)
    middle_vs_ends(samples, dataset_root, start=args.mid_start, end=args.mid_end)
    even_vs_odd(samples, dataset_root)
    first_vs_rest(samples, dataset_root)
    last_vs_rest(samples, dataset_root)

    print("Done. Existing train_0/test_0.csv were left untouched.")


if __name__ == "__main__":
    main()
