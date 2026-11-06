import torch
import torch.nn as nn
import torch.nn.functional as F


class GeneMapBuilder(nn.Module):
    """
    Projects gene expression vectors to a low-dimensional channel space
    and scatters them onto a fixed-resolution 2D grid using spatial
    coordinates. Returns both the aggregated map as well as the normalized
    grid coordinates (in [-1, 1]) for sampling back per-spot control features.
    """

    def __init__(self, n_genes: int, control_channels: int, grid_size: int):
        super().__init__()
        self.grid_size = grid_size
        self.control_channels = control_channels

        self.proj = nn.Linear(n_genes, control_channels)

    def forward(self, gene_exp: torch.Tensor, coords: torch.Tensor, pad_mask: torch.Tensor):
        """
        Args:
            gene_exp: [B, N, n_genes]
            coords: [B, N, 2]
            pad_mask: [B, N] boolean mask (True where padded)
        Returns:
            control_map: [B, control_channels, grid, grid]
            sampling_grid: [B, N, 2] coordinates normalized to [-1, 1]
        """
        B, N, _ = gene_exp.shape
        device = gene_exp.device

        gene_proj = self.proj(gene_exp)  # [B, N, control_channels]
        control_map = gene_proj.new_zeros(B, self.control_channels, self.grid_size, self.grid_size)
        counts = gene_proj.new_zeros(B, 1, self.grid_size, self.grid_size)
        sampling_grid = coords.new_zeros((B, N, 2))

        for b in range(B):
            valid = ~pad_mask[b]
            if valid.sum() == 0:
                continue

            coords_b = coords[b, valid]
            proj_b = gene_proj[b, valid]

            # Normalize coordinates to [0, 1]
            min_xy = coords_b.min(dim=0).values
            max_xy = coords_b.max(dim=0).values
            span_xy = torch.clamp(max_xy - min_xy, min=1e-6)
            norm_xy = (coords_b - min_xy) / span_xy
            norm_xy = torch.clamp(norm_xy, 0.0, 1.0)

            # Convert to discrete indices
            idx_x = torch.clamp((norm_xy[:, 0] * (self.grid_size - 1)).long(), 0, self.grid_size - 1)
            idx_y = torch.clamp((norm_xy[:, 1] * (self.grid_size - 1)).long(), 0, self.grid_size - 1)
            linear_idx = idx_y * self.grid_size + idx_x  # [num_valid]

            # Scatter gene channels into map
            map_flat = control_map[b].view(self.control_channels, -1)
            proj_t = proj_b.transpose(0, 1)  # [control_channels, num_valid]
            map_flat.scatter_add_(1, linear_idx.unsqueeze(0).expand(self.control_channels, -1), proj_t)

            # Track counts for averaging
            count_flat = counts[b].view(1, -1)
            ones = torch.ones(1, linear_idx.size(0), device=device, dtype=count_flat.dtype)
            count_flat.scatter_add_(1, linear_idx.unsqueeze(0), ones)

            # Save normalized grid coordinates for grid sampling later
            sampling_grid[b, valid] = norm_xy * 2.0 - 1.0  # scale to [-1, 1]

        control_map = control_map / torch.clamp(counts, min=1.0)
        return control_map, sampling_grid


class ControlNet(nn.Module):
    """
    Lightweight CNN that processes the gene control map into hidden-dimensional
    features that are later sampled back onto spot coordinates.

    variant="standard" reproduces the original smoothing-heavy network,
    while variant="sharp" keeps a residual skip to preserve high-frequency
    structure.
    """

    def __init__(
        self,
        in_channels: int,
        hidden_dim: int,
        intermediate_channels: int = 64,
        variant: str = "standard",
    ):
        super().__init__()

        variant = variant.lower()
        if variant not in {"standard", "sharp"}:
            raise ValueError(f"Unknown ControlNet variant '{variant}'")
        self.variant = variant

        num_groups = min(8, intermediate_channels)
        if intermediate_channels % num_groups != 0:
            num_groups = 1

        if self.variant == "standard":
            layers = [
                nn.Conv2d(in_channels, intermediate_channels, kernel_size=3, padding=1),
                nn.GroupNorm(num_groups, intermediate_channels),
                nn.GELU(),
                nn.Conv2d(intermediate_channels, intermediate_channels, kernel_size=3, padding=1),
                nn.GroupNorm(num_groups, intermediate_channels),
                nn.GELU(),
                nn.Conv2d(intermediate_channels, hidden_dim, kernel_size=1),
            ]
            self.net = nn.Sequential(*layers)
            nn.init.zeros_(self.net[-1].weight)
            if self.net[-1].bias is not None:
                nn.init.zeros_(self.net[-1].bias)
            self.skip = None
        else:  # sharp variant
            layers = [
                nn.Conv2d(in_channels, intermediate_channels, kernel_size=3, padding=1),
                nn.GroupNorm(num_groups, intermediate_channels),
                nn.GELU(),
                nn.Conv2d(intermediate_channels, hidden_dim, kernel_size=1),
            ]
            self.net = nn.Sequential(*layers)
            self.skip = nn.Conv2d(in_channels, hidden_dim, kernel_size=1)

            nn.init.zeros_(self.net[-1].weight)
            if self.net[-1].bias is not None:
                nn.init.zeros_(self.net[-1].bias)
            nn.init.zeros_(self.skip.weight)
            if self.skip.bias is not None:
                nn.init.zeros_(self.skip.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.net(x)
        if getattr(self, "skip", None) is not None:
            out = out + self.skip(x)
        return out


def sample_control_tokens(control_map: torch.Tensor, sampling_grid: torch.Tensor) -> torch.Tensor:
    """
    Bilinearly samples the hidden control map at per-spot coordinates.

    Args:
        control_map: [B, hidden_dim, H, W]
        sampling_grid: [B, N, 2] coordinates in [-1, 1]
    Returns:
        control_tokens: [B, N, hidden_dim]
    """
    B, _, H, W = control_map.shape
    _, N, _ = sampling_grid.shape

    # grid_sample expects [B, H_out, W_out, 2]; use a degenerate axis for per-spot queries
    grid = sampling_grid.view(B, 1, N, 2)
    sampled = F.grid_sample(control_map, grid, mode='bilinear', align_corners=True)
    # sampled shape: [B, hidden_dim, 1, N]
    control_tokens = sampled.squeeze(2).transpose(1, 2).contiguous()
    return control_tokens
