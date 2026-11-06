import math
import torch
import torch.nn as nn

from .transformer import SpatialTransformer
from .config import ModelConfig
from .controlnet import GeneMapBuilder, ControlNet, sample_control_tokens


class TimestepEmbedder(nn.Module):
    """
    Embeds scalar timesteps into vector representations.
    time emb to frequency_embedding_size dim, then to hidden_size
    """
    def __init__(self, hidden_size, frequency_embedding_size=256):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(frequency_embedding_size, hidden_size, bias=True),
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size, bias=True),
        )
        
        self.frequency_embedding_size = frequency_embedding_size

    @staticmethod
    def timestep_embedding(t, dim, max_period=10000):
        """
        Create sinusoidal timestep embeddings.
        :param t: a 1-D Tensor of N indices, one per batch element.
                          These may be fractional.
        :param dim: the dimension of the output.
        :param max_period: controls the minimum frequency of the embeddings.
        :return: an (N, D) Tensor of positional embeddings.
        """
        # https://github.com/openai/glide-text2im/blob/main/glide_text2im/nn.py
        half = dim // 2
        freqs = torch.exp(
            -math.log(max_period) * torch.arange(start=0, end=half, dtype=torch.float32) / half
        ).to(device=t.device)
        args = t[..., None].float() * freqs[None]
        embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
        if dim % 2:
            embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1)
        return embedding

    def forward(self, t):
        t_freq = self.timestep_embedding(t, self.frequency_embedding_size)
        t_emb = self.mlp(t_freq)
        return t_emb



class ControlBranch(nn.Module):
    def __init__(self, n_genes, control_channels, grid_size, hidden_dim, intermediate_channels, variant):
        super().__init__()
        self.builder = GeneMapBuilder(
            n_genes=n_genes,
            control_channels=control_channels,
            grid_size=grid_size,
        )
        self.net = ControlNet(
            in_channels=control_channels,
            hidden_dim=hidden_dim,
            intermediate_channels=intermediate_channels,
            variant=variant,
        )
        self.layer_norm = nn.LayerNorm(hidden_dim)

    def forward(self, gene_exp, coords, pad_mask):
        control_map, sampling_grid = self.builder(gene_exp, coords, pad_mask)
        control_map = self.net(control_map)
        control_tokens = sample_control_tokens(control_map, sampling_grid)
        control_tokens = self.layer_norm(control_tokens)
        return control_tokens


class Denoiser(nn.Module):
    def __init__(self, config) -> None:
        super(Denoiser, self).__init__()

        self.backbone = SpatialTransformer(
            ModelConfig(
                n_genes=config.n_genes,
                d_input=config.feature_dim,
                d_model=config.hidden_dim,
                d_edge_model=config.pairwise_hidden_dim,
                n_layers=config.n_layers,
                n_heads=config.n_heads,
                dropout=config.dropout,
                attn_dropout=config.attn_dropout,
                n_neighbors=config.n_neighbors,
                act=config.activation,
                use_inducing=getattr(config, "use_inducing", False),
                n_inducing=getattr(config, "n_inducing", 32),
                use_hf_branch=getattr(config, "use_hf_branch", False),
            )
        )
        self.loss_func = nn.MSELoss()

        self.fourier_proj = TimestepEmbedder(config.hidden_dim)
        self.image_transform = nn.Linear(config.feature_dim, config.hidden_dim)

        self.control_scale = getattr(config, "controlnet_scale", 1.0)
        self.controlnet_enabled = getattr(config, "enable_controlnet", False)
        self.control_warmup = 1.0
        if self.controlnet_enabled:
            control_channels = getattr(config, "controlnet_channels", 32)
            control_grid = getattr(config, "controlnet_grid_size", 64)
            control_mid = getattr(config, "controlnet_intermediate_channels", 64)
            control_variant = getattr(config, "controlnet_variant", "standard")
            self.control_branch = ControlBranch(
                n_genes=config.n_genes,
                control_channels=control_channels,
                grid_size=control_grid,
                hidden_dim=config.hidden_dim,
                intermediate_channels=control_mid,
                variant=control_variant,
            )
        else:
            self.control_branch = None

        self.crossslide_enabled = getattr(config, "enable_crossslide_condition", False)
        self.crossslide_scale = getattr(config, "crossslide_scale", 1.0)
        self.crossslide_warmup = 1.0
        # Cross-slide branches: one ControlNet-style branch per direction (section above / below).
        self.crossslide_branches = None
        if self.crossslide_enabled:
            control_channels = getattr(config, "controlnet_channels", 32)
            control_grid = getattr(config, "controlnet_grid_size", 64)
            control_mid = getattr(config, "controlnet_intermediate_channels", 64)
            control_variant = getattr(config, "controlnet_variant", "standard")
            self.crossslide_branches = nn.ModuleDict(
                {
                    "upper": ControlBranch(
                        n_genes=config.n_genes,
                        control_channels=control_channels,
                        grid_size=control_grid,
                        hidden_dim=config.hidden_dim,
                        intermediate_channels=control_mid,
                        variant=control_variant,
                    ),
                    "lower": ControlBranch(
                        n_genes=config.n_genes,
                        control_channels=control_channels,
                        grid_size=control_grid,
                        hidden_dim=config.hidden_dim,
                        intermediate_channels=control_mid,
                        variant=control_variant,
                    ),
                }
            )

    def inference(self, noisy_exp, img_features, coords, t_steps, predict=False, crossslide_expr=None):
        # noisy_exp: [B, n_cells, n_genes]
        # img_features: [B, n_cells, n_features]
        # coords: [B, n_cells, 2]
        # t_steps: [B]

        pad_mask = (img_features.abs().sum(dim=-1) == 0)

        img_features = self.image_transform(img_features)
        time_emb = self.fourier_proj(t_steps)[:, None].expand(noisy_exp.shape[0], noisy_exp.shape[1], -1)
        features = img_features + time_emb
        features[pad_mask] = 0.0

        if self.control_branch is not None:
            control_tokens = self.control_branch(noisy_exp, coords, pad_mask)
            control_tokens = control_tokens * t_steps[:, None, None]
            warmup = self.control_warmup
            if isinstance(warmup, torch.Tensor):
                warmup = warmup.item()
            control_tokens = control_tokens * float(warmup)
            control_tokens = control_tokens * self.control_scale
            control_tokens[pad_mask] = 0.0
            features = features + control_tokens
            features[pad_mask] = 0.0

        if self.crossslide_branches is not None and crossslide_expr:
            warmup = self.crossslide_warmup
            if isinstance(warmup, torch.Tensor):
                warmup = warmup.item()
            for key, branch in self.crossslide_branches.items():
                expr = crossslide_expr.get(key) if isinstance(crossslide_expr, dict) else None
                if expr is None:
                    continue
                branch_tokens = branch(expr, coords, pad_mask)
                branch_tokens = branch_tokens * t_steps[:, None, None]
                branch_tokens = branch_tokens * float(warmup)
                branch_tokens = branch_tokens * self.crossslide_scale
                branch_tokens[pad_mask] = 0.0
                features = features + branch_tokens
                features[pad_mask] = 0.0

        prediction = self.backbone(
            gene_exp=noisy_exp,
            features=features,
            coords=coords
        )
        return prediction

    def forward(self, exp, img_features, coords, labels, t_steps, crossslide_expr=None):
        prediction = self.inference(exp, img_features, coords, t_steps, crossslide_expr=crossslide_expr)
        pad_mask = img_features.abs().sum(-1) == 0
        loss = self.loss_func(prediction[~pad_mask], labels[~pad_mask])
        return prediction, loss

    def set_control_warmup(self, factor: float):
        if not self.controlnet_enabled:
            return
        factor_clamped = max(0.0, min(1.0, factor))
        self.control_warmup = float(factor_clamped)

    def set_crossslide_warmup(self, factor: float):
        if not self.crossslide_enabled or self.crossslide_branches is None:
            return
        factor_clamped = max(0.0, min(1.0, factor))
        self.crossslide_warmup = float(factor_clamped)
