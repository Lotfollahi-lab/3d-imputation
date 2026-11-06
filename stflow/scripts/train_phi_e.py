import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

import anndata as ad
from scipy import sparse

from scvi.distributions import ZeroInflatedNegativeBinomial

from stflow.model.phi_zinb import PhiEZINB, build_spot_features


class SlideDataset(Dataset):
    def __init__(self, counts: np.ndarray, slide_ids: np.ndarray):
        self.slides = []
        unique_ids = np.unique(slide_ids)
        for sid in unique_ids:
            idx = np.where(slide_ids == sid)[0]
            slide_counts = counts[idx]
            self.slides.append(torch.from_numpy(slide_counts).float())

    def __len__(self):
        return len(self.slides)

    def __getitem__(self, idx):
        return self.slides[idx]


def collate_slides(batch):
    max_len = max(x.size(0) for x in batch)
    gene_dim = batch[0].size(1)
    counts = torch.zeros(len(batch), max_len, gene_dim)
    mask = torch.zeros(len(batch), max_len, 1, dtype=torch.bool)
    for i, slide in enumerate(batch):
        counts[i, : slide.size(0)] = slide
        mask[i, : slide.size(0), 0] = True
    return counts, mask


def load_counts_from_dir(adata_dir: Path, sample_ids: list[str], gene_list: list[str] | None):
    """Concatenate per-section .h5ad files (<adata_dir>/<sample_id>.h5ad, raw counts in X)."""
    parts = []
    for sid in sample_ids:
        a = ad.read_h5ad(adata_dir / f"{sid}.h5ad")
        if gene_list is not None:
            a = a[:, gene_list]
        a = a.copy()
        a.obs = a.obs.iloc[:, :0]
        a.obs["_section"] = sid
        a.obs_names = [f"{sid}:{b}" for b in a.obs_names]
        parts.append(a)
    adata = ad.concat(parts)
    X = adata.X.toarray() if sparse.issparse(adata.X) else adata.X
    return X.astype(np.float32), adata.obs["_section"].astype(str).values, list(adata.var_names)


def load_counts(adata_path: Path, gene_list: list[str] | None, sample_col: str):
    adata = ad.read_h5ad(adata_path)
    if gene_list is not None:
        adata = adata[:, gene_list]
    X = adata.X
    if sparse.issparse(X):
        X = X.toarray()
    counts = X.astype(np.float32)
    slide_ids = adata.obs[sample_col].astype(str).values
    genes = list(adata.var_names)
    return counts, slide_ids, genes


def zinb_nll(dist, counts, mask):
    log_prob = dist.log_prob(counts)
    if mask is not None:
        if mask.dim() == 3:
            mask_f = mask.float()
        else:
            mask_f = mask.unsqueeze(-1).float()
        log_prob = log_prob * mask_f
        total = mask_f.sum()
    else:
        total = log_prob.numel()
    return -log_prob.sum() / torch.clamp(total, min=1.0)


def build_distribution(mu, theta, pi):
    eps = 1e-6
    theta_exp = theta.view(1, 1, -1).expand_as(mu)
    nb_logits = torch.log(theta_exp + eps) - torch.log(mu + eps)
    zi_logits = torch.log(pi + eps) - torch.log1p(-pi + eps)
    return theta_exp, nb_logits, zi_logits


def _normalize_id(s: str) -> str:
    s = s.strip()
    lowered = s.lower()
    prefixes = ["embryo_", "her2_", "her2-"]
    for pref in prefixes:
        if lowered.startswith(pref):
            return s[len(pref):]
    return s


def _load_split_ids(csv_path: Path, column: str = "sample_id") -> list[str]:
    if not csv_path.exists():
        raise FileNotFoundError(f"Split CSV not found: {csv_path}")
    df = pd.read_csv(csv_path)
    if column not in df.columns:
        raise ValueError(f"Column '{column}' missing from {csv_path}")
    ids = []
    for x in df[column].tolist():
        if isinstance(x, str):
            ids.append(_normalize_id(x))
        elif pd.notna(x):
            ids.append(_normalize_id(str(x)))
    if not ids:
        raise ValueError(f"No entries found in column '{column}' of {csv_path}")
    return ids


def train(args):
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    if args.adata_dir is not None:
        if not args.train_ids_csv:
            raise ValueError("--adata_dir requires --train_ids_csv")
        ids = pd.read_csv(args.train_ids_csv)["sample_id"].astype(str).str.strip().tolist()
        if args.val_ids_csv:
            ids += [i for i in pd.read_csv(args.val_ids_csv)["sample_id"].astype(str).str.strip() if i not in ids]
        counts, slide_ids_raw, genes = load_counts_from_dir(Path(args.adata_dir), ids, args.gene_list)
    else:
        counts, slide_ids_raw, genes = load_counts(Path(args.adata), args.gene_list, args.sample_col)
    slide_ids = np.array([_normalize_id(s) for s in slide_ids_raw])

    all_slides = np.unique(slide_ids)
    slide_set = set(all_slides)

    train_ids = None
    if args.train_ids_csv:
        train_ids = _load_split_ids(Path(args.train_ids_csv))
        missing = sorted(set(train_ids) - slide_set)
        if missing:
            print(f"[warning] {len(missing)} IDs from train split not found in adata (first: {missing[:3]})")
        train_ids = [sid for sid in train_ids if sid in slide_set]
        if not train_ids:
            raise ValueError("After filtering, no train IDs remain from --train_ids_csv")
    else:
        train_ids = list(all_slides)

    val_ids = None
    if args.val_ids_csv:
        val_ids = _load_split_ids(Path(args.val_ids_csv))
        missing = sorted(set(val_ids) - set(train_ids))
        if missing:
            print(f"[warning] {len(missing)} val IDs not present in training set; they will be ignored.")
        val_ids = [sid for sid in val_ids if sid in train_ids]
        if not val_ids:
            raise ValueError("After filtering, no validation IDs remain from --val_ids_csv")
        val_slides = set(val_ids)
    else:
        rng = np.random.default_rng(args.seed)
        shuffled = np.array(train_ids)
        rng.shuffle(shuffled)
        val_count = max(1, int(len(train_ids) * args.val_ratio))
        val_slides = set(shuffled[:val_count])

    train_slides = set(train_ids) - val_slides
    if not train_slides:
        raise ValueError("No training slides remain after reserving validation set.")

    mask_keep = np.isin(slide_ids, list(train_slides | val_slides))
    counts = counts[mask_keep]
    slide_ids = slide_ids[mask_keep]

    train_mask = np.array([sid in train_slides for sid in slide_ids])
    val_mask = np.array([sid in val_slides for sid in slide_ids])
    if not train_mask.any():
        raise ValueError("Filtered dataset contains no training spots.")
    if not val_mask.any():
        raise ValueError("Filtered dataset contains no validation spots.")

    train_dataset = SlideDataset(counts[train_mask], slide_ids[train_mask])
    val_dataset = SlideDataset(counts[val_mask], slide_ids[val_mask])

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, collate_fn=collate_slides)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, collate_fn=collate_slides)

    device = torch.device(f"cuda:{args.device}" if args.device >= 0 and torch.cuda.is_available() else "cpu")

    model = PhiEZINB(
        num_genes=len(genes),
        spot_feat_dim=len(genes),
        embed_dim=args.embed_dim,
        hidden_dim=args.hidden_dim,
        dropout=args.dropout,
        spot_encoder_layers=args.spot_encoder_layers,
    ).to(device)
    param_count = sum(p.numel() for p in model.parameters())
    print(f"phi_e parameter count: {param_count:,}")

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    metrics_dir = Path(args.save_dir)
    metrics_dir.mkdir(parents=True, exist_ok=True)
    log_file = metrics_dir / "training_log.jsonl"
    if log_file.exists():
        log_file.unlink()

    best_val = float("inf")
    patience = args.patience
    wait = 0

    print("Starting phi_e training...")
    for epoch in range(1, args.epochs + 1):
        model.train()
        train_loss = 0.0
        for counts_batch, mask_batch in train_loader:
            counts_batch = counts_batch.to(device)
            mask_batch = mask_batch.to(device)
            features, auto_mask = build_spot_features(counts_batch)
            mask = mask_batch & auto_mask

            mu, theta, pi = model(features, mask=mask)
            theta_exp, nb_logits, zi_logits = build_distribution(mu, theta, pi)
            dist = ZeroInflatedNegativeBinomial(
                total_count=theta_exp,
                logits=nb_logits,
                zi_logits=zi_logits,
            )
            loss = zinb_nll(dist, counts_batch, mask)
            if args.lambda_mu > 0:
                reg = (mu * mask.float()).pow(2).sum() / torch.clamp(mask.sum(), min=1.0)
                loss = loss + args.lambda_mu * reg

            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), args.clip_norm)
            optimizer.step()
            train_loss += loss.item()

        train_loss /= max(len(train_loader), 1)

        model.eval()
        with torch.no_grad():
            val_loss = 0.0
            for counts_batch, mask_batch in val_loader:
                counts_batch = counts_batch.to(device)
                mask_batch = mask_batch.to(device)
                features, auto_mask = build_spot_features(counts_batch)
                mask = mask_batch & auto_mask

                mu, theta, pi = model(features, mask=mask)
                theta_exp, nb_logits, zi_logits = build_distribution(mu, theta, pi)
                dist = ZeroInflatedNegativeBinomial(
                    total_count=theta_exp,
                    logits=nb_logits,
                    zi_logits=zi_logits,
                )
                loss = zinb_nll(dist, counts_batch, mask)
                val_loss += loss.item()

            val_loss /= max(len(val_loader), 1)

        gap = val_loss - train_loss
        print(f"Epoch {epoch:03d} | train {train_loss:.4f} | val {val_loss:.4f} | gap {gap:+.4f}")
        with open(log_file, "a", encoding="utf-8") as f:
            json.dump(
                {
                    "epoch": epoch,
                    "train_loss": train_loss,
                    "val_loss": val_loss,
                    "gap": gap,
                    "best_val": best_val,
                },
                f,
            )
            f.write("\n")

        if val_loss < best_val:
            best_val = val_loss
            wait = 0
            ckpt = {
                "model": model.state_dict(),
                "config": vars(args),
                "genes": genes,
            }
            torch.save(ckpt, metrics_dir / "phi_e.pt")
        else:
            wait += 1
            if wait >= patience and patience > 0:
                print("Early stopping")
                break


def parse_args():
    parser = argparse.ArgumentParser(description="Train phi_e ZINB prior")
    parser.add_argument("--adata", type=str, default=None,
                        help="One .h5ad with raw counts of all sections (section id in obs[--sample_col])")
    parser.add_argument("--adata_dir", type=str, default=None,
                        help="Alternatively: folder with one .h5ad per section (<sample_id>.h5ad), e.g. <dataset>/adata")
    parser.add_argument("--gene_list", type=str, default=None)
    parser.add_argument("--sample_col", type=str, default="sample")
    parser.add_argument("--train_ids_csv", type=str, default=None,
                        help="CSV (with column 'sample_id') listing slides to use for training/validation")
    parser.add_argument("--val_ids_csv", type=str, default=None,
                        help="CSV (same format) listing slides reserved for validation; overrides --val_ratio")
    parser.add_argument("--save_dir", type=str, required=True)
    parser.add_argument("--device", type=int, default=-1)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight_decay", type=float, default=1e-5)
    parser.add_argument("--embed_dim", type=int, default=64)
    parser.add_argument("--hidden_dim", type=int, default=128)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--spot_encoder_layers", type=int, default=3,
                        help="Number of residual layers in the spot encoder")
    parser.add_argument("--lambda_mu", type=float, default=0.0)
    parser.add_argument("--clip_norm", type=float, default=1.0)
    parser.add_argument("--val_ratio", type=float, default=0.1)
    parser.add_argument("--patience", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    if (args.adata is None) == (args.adata_dir is None):
        raise SystemExit("Provide exactly one of --adata or --adata_dir")
    if args.gene_list is not None:
        with open(args.gene_list) as f:
            gene_names = json.load(f)["genes"]
        args.gene_list = gene_names
    train(args)
