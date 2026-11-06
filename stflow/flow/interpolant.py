import torch
from .noise import PriorSampler


class Interpolant:
    def __init__(self, prior_sample_type, normalize=True, **kwargs):
        if (torch.cuda.is_available()):
            self.device = "cuda"
        else:
            self.device = "cpu"
        
        self.prior_sampler = PriorSampler(prior_sample_type, device=self.device, **kwargs)
        self.normalize = normalize
    
    def sample_from_prior(
        self,
        shape,
        z_indices=None,
        coords_original=None,
        k_neighbors=None,
        reference_exp=None,
        reference_features=None,
        detected_counts=None,
        spot_features=None,
        spot_mask=None,
    ):
        """
        Sample from prior distribution.
        
        Args:
            shape: tuple [B, N_spots, N_genes]
            z_indices: torch.Tensor [B] or None - z-coordinates for spatial_empirical
            coords_original: torch.Tensor [B, N_spots, 3] or None - original coordinates (x, y, z)
            k_neighbors: int or None - number of neighbors for spatial_empirical
            
        Returns:
            torch.Tensor [B, N_spots, N_genes] - sampled expression values
        """
        if self.prior_sampler.requires_context:
            exp = self.prior_sampler.sample(
                shape,
                z_indices=z_indices,
                coords_original=coords_original,
                k_neighbors=k_neighbors,
                reference_exp=reference_exp,
                reference_features=reference_features,
                detected_counts=detected_counts,
                spot_features=spot_features,
                spot_mask=spot_mask,
            ).to(self.device)
        else:
            exp = self.prior_sampler.sample(shape).to(self.device)
        
        if self.normalize:
            exp = torch.log(exp + 1)
        return exp

    def sample_t(self, shape):
        return torch.rand(shape)

    def corrupt_exp(
        self,
        exp,
        z_indices=None,
        coords_original=None,
        k_neighbors=None,
        reference_features=None,
        detected_counts=None,
        spot_features=None,
        spot_mask=None,
    ):
        """
        Corrupt expression data by interpolating between prior and data.
        
        Args:
            exp: torch.Tensor [B, n_cells, n_genes] - true expression
            z_indices: torch.Tensor [B] or None - z-coordinates for spatial_empirical
            coords_original: torch.Tensor [B, N_spots, 3] or None - original coordinates (x, y, z)
            k_neighbors: int or None - number of neighbors for spatial_empirical
            
        Returns:
            tuple: (corrupted expression, timesteps)
        """
        # exp: [B, n_cells, n_genes]
        t = self.sample_t((exp.shape[0],)).to(self.device)
        if exp.shape[0] > 1:
            t = t.squeeze(-1)
        exp_0 = self.sample_from_prior(
            exp.shape,
            z_indices=z_indices,
            coords_original=coords_original,
            k_neighbors=k_neighbors,
            reference_exp=exp,
            reference_features=reference_features,
            detected_counts=detected_counts,
            spot_features=spot_features,
            spot_mask=spot_mask,
        ).to(self.device)
        return exp_0 * (1 - t[:, None, None]) + exp * t[:, None, None], t

    def denoise(self, exp_1, exp_t, t, d_t):
        # exp_1: [B, n_cells, n_genes]
        # exp_t: [B, n_cells, n_genes]
        # t: [B]
        # d_t: [B]
        exp_vf = (exp_1 - exp_t) / (1 - t[:, None, None])
        return exp_t + d_t[:, None, None] * exp_vf

    def prior_guidance(self, x, params=None):
        if hasattr(self.prior_sampler, "guidance"):
            return self.prior_sampler.guidance(x, params=params)
        return None
