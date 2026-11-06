import torch
import torch.nn as nn
import torch.nn.functional as F
from timm.models.vision_transformer import Mlp, SwiGLUPacked
from torch_geometric.utils import to_dense_batch
from einops import rearrange, einsum

from .fa import FrameAveraging


def get_activation(activation="gelu"):
    return {
        "gelu": nn.GELU,
        "silu": nn.SiLU,
        "relu": nn.ReLU,
    }[activation]

class InducingSetAttention(nn.Module):
    """Low-rank global attention via inducing tokens (Set Transformer ISAB-style).
    
    Provides O(N*m) global context with m inducing tokens instead of O(N^2).
    Each point attends to m inducing tokens, then attends back from inducing tokens.
    """
    def __init__(self, d_model, n_inducing=32, n_heads=1, attn_drop=0., proj_drop=0.):
        super().__init__()
        self.n_inducing = n_inducing
        self.n_heads = n_heads
        self.d_head = d_model // n_heads
        
        # Learnable inducing tokens
        self.inducing_tokens = nn.Parameter(torch.randn(n_inducing, d_model))
        
        # Two-stage attention: X -> I, then I -> X
        self.qkv_I = nn.Linear(d_model, d_model * 3)  # for inducing tokens to attend to X
        self.qkv_X = nn.Linear(d_model, d_model * 3)  # for X to attend to inducing tokens
        
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(d_model, d_model)
        self.proj_drop = nn.Dropout(proj_drop)
        self.norm = nn.LayerNorm(d_model)
        
    def forward(self, x, batch_idx):
        """
        Args:
            x: [N, d_model] - flattened tokens
            batch_idx: [N] - batch assignment for each token
        Returns:
            global_context: [N, d_model] - global context for each token
        """
        N, d_model = x.shape
        device = x.device
        
        # Expand inducing tokens for each batch
        unique_batches = torch.unique(batch_idx)
        B = unique_batches.size(0)
        I = self.inducing_tokens.unsqueeze(0).expand(B, -1, -1)  # [B, m, d]
        
        # Stage 1: Inducing tokens attend to X (aggregate X into I)
        # Build per-batch grouped attention
        H_list = []
        for b_idx in unique_batches:
            mask = (batch_idx == b_idx)
            x_b = x[mask]  # [N_b, d]
            I_b = I[b_idx.item() if B > 1 else 0]  # [m, d]
            
            # Q from inducing, K/V from x_b
            q_i = self.qkv_I(I_b).chunk(3, dim=-1)[0]  # [m, d]
            k_x, v_x = self.qkv_I(x_b).chunk(3, dim=-1)[1:]  # [N_b, d]
            
            q_i = rearrange(q_i, 'm (h d) -> h m d', h=self.n_heads)
            k_x = rearrange(k_x, 'n (h d) -> h n d', h=self.n_heads)
            v_x = rearrange(v_x, 'n (h d) -> h n d', h=self.n_heads)
            
            # Attention: [h, m, d] @ [h, d, N_b] -> [h, m, N_b]
            attn = torch.einsum('hmd,hnd->hmn', q_i, k_x) / (self.d_head ** 0.5)
            attn = F.softmax(attn, dim=-1)
            attn = self.attn_drop(attn)
            
            # [h, m, N_b] @ [h, N_b, d] -> [h, m, d]
            H_b = torch.einsum('hmn,hnd->hmd', attn, v_x)
            H_b = rearrange(H_b, 'h m d -> m (h d)')
            H_list.append(H_b)
        
        # Stage 2: X attends to updated inducing tokens H
        global_out = []
        h_idx = 0
        for b_idx in unique_batches:
            mask = (batch_idx == b_idx)
            x_b = x[mask]  # [N_b, d]
            H_b = H_list[h_idx]  # [m, d]
            h_idx += 1
            
            # Q from x_b, K/V from H_b
            q_x = self.qkv_X(x_b).chunk(3, dim=-1)[0]  # [N_b, d]
            k_h, v_h = self.qkv_X(H_b).chunk(3, dim=-1)[1:]  # [m, d]
            
            q_x = rearrange(q_x, 'n (h d) -> h n d', h=self.n_heads)
            k_h = rearrange(k_h, 'm (h d) -> h m d', h=self.n_heads)
            v_h = rearrange(v_h, 'm (h d) -> h m d', h=self.n_heads)
            
            # Attention: [h, N_b, d] @ [h, d, m] -> [h, N_b, m]
            attn = torch.einsum('hnd,hmd->hnm', q_x, k_h) / (self.d_head ** 0.5)
            attn = F.softmax(attn, dim=-1)
            attn = self.attn_drop(attn)
            
            # [h, N_b, m] @ [h, m, d] -> [h, N_b, d]
            out_b = torch.einsum('hnm,hmd->hnd', attn, v_h)
            out_b = rearrange(out_b, 'h n d -> n (h d)')
            global_out.append(out_b)
        
        # Concatenate all batches back
        global_context = torch.cat(global_out, dim=0)  # [N, d]
        global_context = self.proj_drop(self.proj(global_context))
        global_context = self.norm(global_context)
        
        return global_context

class HighFrequencyBranch(nn.Module):
    """High-frequency (detail) branch: x - mean_neighbors(x).
    
    Computes graph Laplacian-like residual to capture local contrast/edges.
    """
    def __init__(self, d_model, proj_drop=0.):
        super().__init__()
        self.proj = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.Dropout(proj_drop),
        )
        self.norm = nn.LayerNorm(d_model)
        
    def forward(self, features, neighbor_indices, neighbor_masks=None):
        """
        Args:
            features: [N, d_model]
            neighbor_indices: [N, k]
            neighbor_masks: [N, k] optional mask
        Returns:
            hf_features: [N, d_model] - high-frequency residual
        """
        N, k = neighbor_indices.shape
        
        # Gather neighbor features
        neighbor_feats = features[neighbor_indices]  # [N, k, d]
        
        # Compute mean (handle masks)
        if neighbor_masks is not None:
            # Set masked neighbors to 0 and adjust mean
            neighbor_feats = neighbor_feats.masked_fill(neighbor_masks.unsqueeze(-1), 0)
            counts = (~neighbor_masks).sum(dim=1, keepdim=True).clamp(min=1)  # [N, 1]
            neighbor_mean = neighbor_feats.sum(dim=1) / counts  # [N, d]
        else:
            neighbor_mean = neighbor_feats.mean(dim=1)  # [N, d]
        
        # High-frequency residual: x - mean(neighbors)
        hf_residual = features - neighbor_mean
        
        # Project and normalize
        hf_features = self.proj(hf_residual)
        hf_features = self.norm(hf_features)
        
        return hf_features


class GeneUpdate(nn.Module):
    def __init__(
            self, 
            d_model, 
            n_genes,
            proj_drop=0.,
            non_negative: bool = False,
        ):
        super(GeneUpdate, self).__init__()    

        self.non_negative = non_negative
        self.output = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.LayerNorm(d_model),
            nn.Dropout(proj_drop),
            nn.Linear(d_model, n_genes),
            nn.Dropout(proj_drop),
        )
    
    def forward(self, features):
        update = self.output(features) 
        if self.non_negative:
            update = F.relu(update)
        return update


class MLPAttnEdgeAggregation(FrameAveraging):
    def __init__(
            self, 
            d_model, 
            d_edge_model,
            n_genes,
            n_heads=1,
            proj_drop=0.,
            attn_drop=0.,
            activation='gelu',
        ):
        super(MLPAttnEdgeAggregation, self).__init__(dim=2)
        
        self.d_head, self.d_edge_head, self.n_heads = d_model // n_heads, d_edge_model // n_heads, n_heads
        self.n_genes = n_genes

        self.layernorm_qkv = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, d_model * 3),
        )

        if activation == "swiglu":
            self.mlp_attn = SwiGLUPacked(
                in_features=self.d_head*2+self.d_edge_head+n_genes, hidden_features=d_model, 
                out_features=1, drop=proj_drop, norm_layer=nn.LayerNorm
            )
            self.edge_trans = SwiGLUPacked(
                in_features=self.dim+1, hidden_features=d_edge_model, 
                out_features=d_edge_model, drop=proj_drop, norm_layer=nn.LayerNorm
            )
            self.W_output = SwiGLUPacked(
                in_features=d_model+d_edge_model, hidden_features=d_model, 
                out_features=d_model, drop=proj_drop, norm_layer=nn.LayerNorm
            )
        else:
            self.mlp_attn = Mlp(
                in_features=self.d_head*2+self.d_edge_head+n_genes, hidden_features=d_model, 
                out_features=1, drop=proj_drop, norm_layer=nn.LayerNorm
            )
            self.edge_trans = Mlp(
                in_features=self.dim+1, hidden_features=d_edge_model, out_features=d_edge_model, 
                act_layer=get_activation(activation), drop=proj_drop, norm_layer=nn.LayerNorm
            )
            self.W_output = Mlp(
                in_features=d_model+d_edge_model, hidden_features=d_model, out_features=d_model, 
                act_layer=get_activation(activation), drop=proj_drop, norm_layer=nn.LayerNorm
            )

        self.attn_dropout = nn.Dropout(attn_drop)

    def forward(self, gene_exp, token_embs, coords, neighbor_indices, neighbor_masks=None):
        # gene_exp: [N, N_genes], token_embs: [N, -1], geo_token_embs: [N, 3]
        # neighbor_indices: [N, N_neighbor], neighbor_masks: [N, N_neighbor]
        n_tokens, n_neighbors = token_embs.size(0), neighbor_indices.size(1)
        n_heads, d_head, d_edge_head = self.n_heads, self.d_head, self.d_edge_head

        q_s, k_s, v_s = self.layernorm_qkv(token_embs).chunk(3, dim=-1)
        q_s, k_s, v_s = map(lambda x: rearrange(x, 'n (h d) -> n h d', h=n_heads), (q_s, k_s, v_s))

        """build pairwise representation with FA"""
        radial_coords = coords[neighbor_indices] - coords.unsqueeze(dim=1)  # [N, N_neighbor, 2]
        radial_coord_norm = radial_coords.norm(dim=-1).unsqueeze(-1)  # [N, N_neighbor, 1]

        frame_feats, _, _ = self.create_frame(radial_coords, neighbor_masks)  # [N*8, N_neighbors, 3]
        frame_feats = frame_feats.view(n_tokens, self.n_frames, n_neighbors, -1)  # [N, 8, N_neighbors, d_model]

        radial_coord_norm = radial_coord_norm.unsqueeze(dim=1).expand(n_tokens, self.n_frames, n_neighbors, -1)
        frame_feats = self.edge_trans(torch.cat([frame_feats, radial_coord_norm], dim=-1)).mean(dim=1)  # [N, N_neighbors, d_edge_model]

        """gene expression features"""
        gene_exp_diff = gene_exp[neighbor_indices] - gene_exp.unsqueeze(dim=1)  # [N, N_neighbor, N_genes]
        gene_exp_feats_expand = gene_exp_diff[..., None, :].expand(n_tokens, n_neighbors, n_heads, -1)  # [N, N_neighbor, n_heads, N_genes]

        """attention map"""
        q_s = q_s.unsqueeze(dim=1).expand(n_tokens, n_neighbors, n_heads, d_head)
        frame_feats = frame_feats.view(n_tokens, n_neighbors, n_heads, d_edge_head)
        message = torch.cat([q_s, k_s[neighbor_indices], frame_feats, gene_exp_feats_expand], dim=-1)
        
        attn_map = self.mlp_attn(message).squeeze(-1)
        if neighbor_masks is not None:
            attn_map.masked_fill_(neighbor_masks.unsqueeze(dim=-1), -1e9)
        attn_map = self.attn_dropout(nn.Softmax(dim=-1)(attn_map.transpose(1, 2)))  # [N, n_heads, N_neighbor]

        """context aggregation"""
        v_s_neighs = v_s[neighbor_indices].view(n_tokens, -1, n_heads, d_head)  # [N, n_heads, N_neighbor, D]
        scalar_context = einsum(attn_map, v_s_neighs, 'n h m, n m h d -> n h d').view(n_tokens, -1)  # [N, n_heads*D]
        edge_context = einsum(attn_map, frame_feats, 'n h m, n m h d -> n h d').view(n_tokens, -1)  # [N, n_heads*D]
        return self.W_output(torch.cat([scalar_context, edge_context], dim=-1))


class TransformerBlock(nn.Module):
    def __init__(            
            self,
            d_model,
            d_edge_model,
            n_genes,
            n_heads=1,
            activation="gelu",
            attn_drop=0.,
            proj_drop=0.,
            gene_exp_non_negative=True,
            mlp_ratio=4.0,
            # New hybrid attention parameters
            use_inducing=False,
            n_inducing=32,
            use_hf_branch=False,
            layer_idx=0,
            n_layers=1,
        ):
        super(TransformerBlock, self).__init__()

        self.layer_idx = layer_idx
        self.n_layers = n_layers
        self.use_inducing = use_inducing
        self.use_hf_branch = use_hf_branch

        # Local attention (original)
        self.attn = MLPAttnEdgeAggregation(
            d_model=d_model, d_edge_model=d_edge_model, n_genes=n_genes, n_heads=n_heads, 
            proj_drop=proj_drop, attn_drop=attn_drop, activation=activation
        )

        # Global low-rank attention via inducing tokens
        if use_inducing:
            self.global_attn = InducingSetAttention(
                d_model=d_model, n_inducing=n_inducing, n_heads=n_heads,
                attn_drop=attn_drop, proj_drop=proj_drop
            )
        
        # High-frequency (detail) branch
        if use_hf_branch:
            self.hf_branch = HighFrequencyBranch(d_model=d_model, proj_drop=proj_drop)
        
        # Multi-path fusion with progressive scheduling
        # Compute input dimension based on active branches
        fusion_in_dim = d_model  # local always present
        if use_inducing:
            fusion_in_dim += d_model  # global
        if use_hf_branch:
            fusion_in_dim += d_model  # HF
        
        self.fusion = nn.Sequential(
            nn.Linear(fusion_in_dim, d_model),
            nn.GELU(),
            nn.LayerNorm(d_model),
            nn.Dropout(proj_drop),
        )

        if activation == "swiglu":
            self.mlp = SwiGLUPacked(
                in_features=d_model, hidden_features=int(d_model * mlp_ratio), drop=proj_drop, norm_layer=nn.LayerNorm
            )
        else:
            self.mlp = Mlp(
                in_features=d_model, hidden_features=int(d_model * mlp_ratio), 
                act_layer=get_activation(activation), drop=proj_drop, norm_layer=nn.LayerNorm
            )
        
        self.gene_updater = GeneUpdate(d_model, n_genes, proj_drop=proj_drop, non_negative=gene_exp_non_negative)

    def _get_schedule_weights(self):
        """Compute depth-dependent fusion weights (progressive scheduling).
        
        Early layers (low layer_idx): favor local + HF (high-freq details)
        Later layers (high layer_idx): favor global (long-range context)
        
        Returns:
            dict with keys 'local', 'global', 'hf' (sum to 1.0)
        """
        if self.n_layers == 1:
            # Single layer: balanced
            return {'local': 0.5, 'global': 0.3, 'hf': 0.2}
        
        # Normalize layer position to [0, 1]
        depth_ratio = self.layer_idx / max(1, self.n_layers - 1)
        
        # Early layers: high local/HF, low global
        # Late layers: lower local/HF, higher global
        w_local = 0.4 - 0.3 * depth_ratio  # 0.6 -> 0.3
        w_global = 0.4 + 0.4 * depth_ratio  # 0.2 -> 0.6
        w_hf = 0.2 - 0.1 * depth_ratio      # 0.2 -> 0.1
        
        # Normalize to sum to 1
        total = w_local + (w_global if self.use_inducing else 0) + (w_hf if self.use_hf_branch else 0)
        return {
            'local': w_local / total,
            'global': (w_global / total) if self.use_inducing else 0.0,
            'hf': (w_hf / total) if self.use_hf_branch else 0.0,
        }

    def forward(self, gene_exp, token_embs, coords, neighbor_indices, batch_idx=None):
        """Forward pass with multi-path fusion.
        
        Args:
            gene_exp: [N, N_genes]
            token_embs: [N, d_model]
            coords: [N, 3]
            neighbor_indices: [N, k]
            batch_idx: [N] - batch assignment (required if use_inducing=True)
        """
        # Local attention (original path)
        context_local = self.attn(gene_exp, token_embs, coords, neighbor_indices)
        
        # Collect all paths for fusion
        paths = [context_local]
        
        # Global attention via inducing tokens
        if self.use_inducing:
            if batch_idx is None:
                raise ValueError("batch_idx required when use_inducing=True")
            context_global = self.global_attn(token_embs, batch_idx)
            paths.append(context_global)
        
        # High-frequency branch
        if self.use_hf_branch:
            context_hf = self.hf_branch(context_local, neighbor_indices)
            paths.append(context_hf)
        
        # Fuse paths with progressive scheduling
        if len(paths) > 1:
            # Concatenate and project
            fused = self.fusion(torch.cat(paths, dim=-1))
            
            # Optional: apply scheduled weighting (currently handled by fusion MLP)
            # weights = self._get_schedule_weights()
            # (Could add explicit weighted combination here if needed)
        else:
            fused = context_local
        
        # Add residual
        token_embs = token_embs + fused

        # MLP and gene update
        token_embs = token_embs + self.mlp(token_embs)
        gene_exp = self.gene_updater(token_embs)

        return gene_exp, token_embs


class SpatialTransformer(nn.Module):
    def __init__(self, config):
        super(SpatialTransformer, self).__init__()

        self.n_neighbors = config.n_neighbors
        self.n_layers = config.n_layers
        
        # Inducing tokens and HF branch flags
        use_inducing = getattr(config, 'use_inducing', False)
        n_inducing = getattr(config, 'n_inducing', 32)
        use_hf_branch = getattr(config, 'use_hf_branch', False)

        self.blks = nn.ModuleList([
            TransformerBlock(
                config.d_model, config.d_edge_model, 
                n_genes=config.n_genes, n_heads=config.n_heads, 
                activation=config.act, attn_drop=config.attn_dropout, 
                proj_drop=config.dropout,
                use_inducing=use_inducing,
                n_inducing=n_inducing,
                use_hf_branch=use_hf_branch,
                layer_idx=i,
                n_layers=config.n_layers,
            )
            for i in range(config.n_layers)
        ])

    def _build_graph(self, coords, batch_idx, n_neighbors, exclude_self=True):
        """kNN graph over spots of the same section (computed on |x|, |y|).

        Args:
            coords: [N, 2 or 3] spatial coordinates
            batch_idx: [N] batch assignment
            n_neighbors: number of neighbours
        Returns:
            nearest_indices: [N, k]
        """
        N = coords.shape[0]
        xy = torch.abs(coords[:, :2].clone())
        same_batch = batch_idx.unsqueeze(0) == batch_idx.unsqueeze(1)
        rel_pos = rearrange(xy, 'n d -> n 1 d') - rearrange(xy, 'n d -> 1 n d')
        rel_dist = rel_pos.norm(dim=-1).detach()  # [N, N]
        mask_invalid = ~same_batch
        if exclude_self:
            mask_invalid |= torch.eye(N, dtype=torch.bool, device=coords.device)
        rel_dist = rel_dist.masked_fill(mask_invalid, 1e9)
        _, nearest_indices = rel_dist.topk(n_neighbors, dim=-1, largest=False)
        return nearest_indices

    def forward(self, gene_exp, features, coords):
        # gene_exp: [B, N_cells, N_genes], features: [B, N_cells, -1], coords: [B, N_cells, 2]
        B, N_cells, N_genes = gene_exp.shape[0], gene_exp.shape[1], gene_exp.shape[-1]
        device = features.device
        
        pad_mask = features.sum(dim=-1) == 0  # [B, N_cells], True if the token is padding
        batch_idx = torch.arange(B, device=device).unsqueeze(-1).repeat(1, N_cells)[~pad_mask]

        features = features[~pad_mask]  # [N_total, d_model]
        coords = coords[~pad_mask]  # [N_total, 3]
        gene_exp = gene_exp[~pad_mask]  # [N_total, N_genes]

        all_gene_exp = []
        for layer_idx, blk in enumerate(self.blks):
            k = min(self.n_neighbors, N_cells)
            nearest_indices = self._build_graph(coords, batch_idx, k, exclude_self=True)

            # Forward through block with batch_idx for inducing tokens
            gene_exp, features = blk(
                gene_exp, features, coords, nearest_indices, batch_idx=batch_idx
            )
            all_gene_exp.append(gene_exp)
        gene_exp = torch.stack(all_gene_exp, dim=0).mean(dim=0)  # [B, N_cells, N_genes]
        
        # average the gene expression among the neighbors
        gene_exp, _ =   to_dense_batch(gene_exp, batch=batch_idx, fill_value=0, max_num_nodes=N_cells)  # [B, N_cells, N_genes]
        return gene_exp
