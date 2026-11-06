import torch
import torch.nn as nn
import torch.nn.functional as F


def build_spot_features(counts: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Build spot-level feature vectors directly from gene counts.

    Args:
        counts: Tensor [B, N, G] of gene expression counts (floats).

    Returns:
        features: [B, N, G] log-transformed feature tensor.
        mask: [B, N, 1] boolean mask indicating valid spots.
    """
    counts = counts.float()
    spot_sum = counts.sum(dim=-1, keepdim=True)
    mask = spot_sum > 0

    features = torch.log1p(counts)
    features = features * mask
    return features, mask


class ResidualBlock(nn.Module):
    def __init__(self, dim: int, hidden_dim: int, dropout: float):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.net(self.norm(x))


class SpotEncoder(nn.Module):
    def __init__(
        self,
        in_dim: int,
        hidden_dim: int,
        out_dim: int,
        dropout: float = 0.1,
        n_layers: int = 3,
    ):
        super().__init__()
        self.input_proj = nn.Linear(in_dim, hidden_dim)
        self.blocks = nn.ModuleList(
            ResidualBlock(hidden_dim, hidden_dim * 2, dropout) for _ in range(max(1, n_layers))
        )
        self.output_norm = nn.LayerNorm(hidden_dim)
        self.output_proj = nn.Linear(hidden_dim, out_dim)
        self.gate = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, out_dim),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.input_proj(x)
        for block in self.blocks:
            x = block(x)
        x = self.output_norm(x)
        gated = self.output_proj(x)
        gate = self.gate(x)
        return gated * gate


class PhiEZINB(nn.Module):
    """Modern phi_e module using per-gene log-count features."""

    def __init__(
        self,
        num_genes: int,
        spot_feat_dim: int,
        embed_dim: int = 64,
        hidden_dim: int = 128,
        dropout: float = 0.1,
        use_dropout: bool = True,
        spot_encoder_layers: int = 3,
    ):
        super().__init__()
        self.num_genes = num_genes
        self.embed_dim = embed_dim

        self.gene_embed = nn.Parameter(torch.randn(num_genes, embed_dim))
        self.spot_encoder = SpotEncoder(
            spot_feat_dim,
            hidden_dim,
            embed_dim,
            dropout=dropout if use_dropout else 0.0,
            n_layers=spot_encoder_layers,
        )

        def head():
            layers: list[nn.Module] = [
                nn.Linear(embed_dim, embed_dim),
                nn.SiLU(),
            ]
            if use_dropout:
                layers.append(nn.Dropout(dropout))
            layers.append(nn.Linear(embed_dim, 1))
            return nn.Sequential(*layers)

        self.mu_head = head()
        self.pi_head = head()
        self.theta_head = head()

        nn.init.normal_(self.gene_embed, mean=0.0, std=0.02)

    def forward(self, spot_features: torch.Tensor, mask: torch.Tensor | None = None):
        B, N, _ = spot_features.shape
        device = spot_features.device

        if mask is None:
            mask = torch.ones(B, N, 1, dtype=torch.bool, device=device)

        spot_flat = spot_features.view(B * N, -1)
        spot_emb = self.spot_encoder(spot_flat).view(B, N, self.embed_dim)

        gene_emb = self.gene_embed.to(device)
        combined = spot_emb.unsqueeze(2) + gene_emb.unsqueeze(0).unsqueeze(0)

        mu_logit = self.mu_head(combined).squeeze(-1)
        pi_logit = self.pi_head(combined).squeeze(-1)
        theta_logit = self.theta_head(gene_emb).squeeze(-1)

        mu = F.softplus(mu_logit + 1.0)
        theta = F.softplus(theta_logit + 1.0)
        pi = torch.sigmoid(pi_logit).clamp(min=1e-4, max=1 - 1e-4)

        mask_expanded = mask.to(mu.dtype)
        mu = mu * mask_expanded
        pi = pi * mask_expanded
        return mu, theta, pi
