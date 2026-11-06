import torch
import numpy as np


from scvi.distributions import ZeroInflatedNegativeBinomial


class SpatialEmpiricalPrior:
    """
    Prior sampler that draws from empirical distributions of neighboring slides.
    For each spot and gene, samples from K nearest neighbors from the nearest training slide in z.
    """
    def __init__(self, dataset, **kwargs):
        self.dataset = dataset  # MultiHESTDataset reference
        self.device = kwargs.get('device', 'cpu')
        
    def sample(
        self,
        shape,
        z_indices,
        coords_original,
        k_neighbors,
        reference_exp=None,
        reference_features=None,
        **kwargs,
    ):
        """
        Sample from empirical distributions of nearest neighbor slide in z.
        
        Args:
            shape: tuple [B, N_spots, N_genes]
            z_indices: torch.Tensor [B] - z-coordinate of each slide in batch
            coords_original: torch.Tensor [B, N_spots, 3] - original (uncentered) coordinates (x, y, z)
            k_neighbors: int - number of nearest spatial neighbors to sample from
            
        Returns:
            torch.Tensor [B, N_spots, N_genes] - sampled expression values
        """
        B, N, G = shape
        samples = torch.zeros(shape, device=self.device)
        
        for b in range(B):
            target_z = z_indices[b].item()
            coords_b = coords_original[b]  # [N_spots, 3]
            
            # Get nearest training slide's expression data for K spatial neighbors
            ref_exp = None
            ref_feat = None
            if reference_exp is not None:
                ref_exp = reference_exp[b]
            if reference_features is not None:
                ref_feat = reference_features[b]

            neighbor_expr = self.dataset.get_neighbor_data(
                target_z,
                coords_b,
                k_neighbors,
                target_expression=ref_exp,
                target_features=ref_feat,
            )
            # neighbor_expr: [N_spots, K, G]
            
            # Vectorized sampling: sample one index per (spot, gene) pair
            K = neighbor_expr.shape[1]
            random_indices = torch.randint(0, K, (N, G), device=neighbor_expr.device)
            
            # Gather values using advanced indexing
            spot_indices = torch.arange(N, device=neighbor_expr.device).unsqueeze(1).expand(N, G)
            gene_indices = torch.arange(G, device=neighbor_expr.device).unsqueeze(0).expand(N, G)
            samples[b] = neighbor_expr[spot_indices, random_indices, gene_indices]
        
        return samples.to(self.device)


class PriorSampler:
    def __init__(self, prior_sample_type, **kwargs):
        self.prior_sample_type = prior_sample_type
        self.requires_context = False  # Flag for context-dependent samplers

        if prior_sample_type == "gaussian":
            self.prior_sampler = gaussian_prior
        elif prior_sample_type == "zero":
            self.prior_sampler = all_zeros
        elif prior_sample_type == "zinb":
            # https://github.com/scverse/scvi-tools/blob/main/src/scvi/distributions/_negative_binomial.py#L433
            prior_sampler = ZeroInflatedNegativeBinomial(
                                        total_count=kwargs.get("total_count", None),
                                        logits=kwargs.get("logits", None),
                                        zi_logits=kwargs.get("zi_logits", None),  # real number
                                    )
            self.prior_sampler = lambda shape: prior_sampler.sample(shape).squeeze(-1)
        elif prior_sample_type == "spatial_empirical":
            self.prior_sampler = SpatialEmpiricalPrior(
                dataset=kwargs.get("dataset", None),
                device=kwargs.get("device", "cpu")
            )
            self.requires_context = True
        elif prior_sample_type == "learned_zinb":
            phi_e = kwargs.get("phi_e", None)
            if phi_e is None:
                raise ValueError("phi_e must be provided for learned_zinb prior")
            self.prior_sampler = LearnedZINBPrior(
                phi_e=phi_e,
                device=kwargs.get("device", "cpu"),
            )
            self.requires_context = True
        else:
            raise ValueError("Invalid prior sample type")

    def sample(
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
        if self.requires_context:
            return self.prior_sampler.sample(
                shape,
                z_indices,
                coords_original,
                k_neighbors,
                reference_exp=reference_exp,
                reference_features=reference_features,
                detected_counts=detected_counts,
                spot_features=spot_features,
                spot_mask=spot_mask,
            )
        else:
            return self.prior_sampler(shape)

    def guidance(self, x, params=None):
        if hasattr(self.prior_sampler, "guidance"):
            return self.prior_sampler.guidance(x, params=params)
        return None


def gaussian_prior(shape):
    return torch.randn(shape)


def all_zeros(shape):
    return torch.zeros(shape)


class LearnedZINBPrior:
    def __init__(self, phi_e, device="cpu"):
        self.phi_e = phi_e
        self.device = device
        self._latest_params = None
        self._latest_valid_mask = None

    def sample(
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
        if spot_features is None or spot_mask is None:
            raise ValueError("spot_features and spot_mask must be provided for learned_zinb prior")

        spot_features = spot_features.to(self.device)
        spot_mask = spot_mask.to(self.device)

        mu, theta, pi = self.phi_e(spot_features, mask=spot_mask)

        eps = 1e-6
        theta_exp = theta.view(1, 1, -1).expand_as(mu)
        nb_logits = torch.log(theta_exp + eps) - torch.log(mu + eps)
        zi_logits = torch.log(pi + eps) - torch.log1p(-pi + eps)

        params = (mu, theta, pi)
        self._latest_params = params
        self._latest_valid_mask = spot_mask

        dist = ZeroInflatedNegativeBinomial(
            total_count=theta_exp,
            logits=nb_logits,
            zi_logits=zi_logits,
        )
        samples = dist.sample().float()
        samples = samples * spot_mask.float()
        return samples.to(self.device)

    def log_prob(self, x, params=None):
        if params is None:
            if self._latest_params is None:
                raise RuntimeError("No parameters available for log_prob computation")
            params = self._latest_params
        mu, theta, pi = params
        eps = 1e-6
        theta_exp = theta.view(1, 1, -1).expand_as(mu)
        nb_logits = torch.log(theta_exp + eps) - torch.log(mu + eps)
        zi_logits = torch.log(pi + eps) - torch.log1p(-pi + eps)
        dist = ZeroInflatedNegativeBinomial(
            total_count=theta_exp,
            logits=nb_logits,
            zi_logits=zi_logits,
        )
        counts = torch.expm1(x)
        log_p = dist.log_prob(counts)
        if self._latest_valid_mask is not None:
            log_p = log_p * self._latest_valid_mask.float()
        return log_p

    def guidance(self, x, params=None):
        if params is None:
            params = self._latest_params
        if params is None:
            raise RuntimeError("No parameters available for guidance computation")

        mu, theta, pi = params
        eps = 1e-6
        theta_exp = theta.view(1, 1, -1).expand_as(mu)
        nb_logits = torch.log(theta_exp + eps) - torch.log(mu + eps)
        zi_logits = torch.log(pi + eps) - torch.log1p(-pi + eps)
        dist = ZeroInflatedNegativeBinomial(
            total_count=theta_exp,
            logits=nb_logits,
            zi_logits=zi_logits,
        )

        x = x.requires_grad_(True)
        counts = torch.expm1(x)
        log_prob = dist.log_prob(counts)
        if self._latest_valid_mask is not None:
            mask = self._latest_valid_mask.float()
            log_prob = log_prob * mask
        grad_counts = torch.autograd.grad(log_prob.sum(), counts, retain_graph=True)[0]
        grad_x = grad_counts * torch.exp(x)
        x = x.detach()
        if self._latest_valid_mask is not None:
            grad_x = grad_x * self._latest_valid_mask.float()
        return grad_x
