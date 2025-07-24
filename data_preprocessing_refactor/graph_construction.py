import os
import time
from typing import Dict, List, Any

import numpy as np
import torch
from torch_geometric.data import HeteroData
import anndata as ad
from scipy.spatial import distance_matrix
from tqdm import tqdm

def build_graph_from_dict(
    node_dict: Dict[str, Dict[str, Any]],
    distance_threshold: float
):
    node_ids = list(node_dict.keys())
    coords   = np.vstack([node_dict[n]['position'] for n in node_ids])
    feats    = np.vstack([node_dict[n]['feature']  for n in node_ids])
    labs     = np.vstack([node_dict[n]['label']    for n in node_ids])

    dist_mat = distance_matrix(coords, coords)
    edges, weights = [], []
    N = len(node_ids)
    for i in range(N):
        for j in range(i+1, N):
            d = dist_mat[i, j]
            if d < distance_threshold:
                edges.append((i, j))
                weights.append(distance_threshold - d)

    edge_index  = torch.tensor(edges, dtype=torch.long).t().contiguous()
    edge_attr   = torch.tensor(weights, dtype=torch.float)
    x           = torch.tensor(feats, dtype=torch.float)
    y           = torch.tensor(labs, dtype=torch.float)
    return edge_index, edge_attr, x, y, node_ids


def build_cross_layer_edges(
    upper_ids: List[str],
    lower_ids: List[str],
    rel_map: Dict[str, List[str]]
):
    edges = []
    for up in upper_ids:
        if up not in rel_map:
            continue
        ui = upper_ids.index(up)
        for low in rel_map[up]:
            if low in lower_ids:
                li = lower_ids.index(low)
                edges.append((ui, li))
    return edges


def build_multires_graphs(
    adata_224_path: str,
    adata_hier_path: str,
    adata_feat_path: str,
    output_dir: str,
    distance_threshold: float = 700.0
) -> None:
    """
    Build and save one HeteroData per sample, using spot_id-based lookups
    into adata_feat.obsm['spatial'], .obsm['features'], adata_224.X and adhier.X.
    """
    t0 = time.time()
    os.makedirs(output_dir, exist_ok=True)

    # Load the three Anndata objects
    ad224  = ad.read_h5ad(adata_224_path)
    adhier = ad.read_h5ad(adata_hier_path)
    adfeat = ad.read_h5ad(adata_feat_path)

    # Prepare spot_id → row‐index maps for each AnnData
    idx224   = {sid: i for i, sid in enumerate(ad224.obs['spot_id'])}
    idxhier  = {sid: i for i, sid in enumerate(adhier.obs['spot_id'])}
    idxfeat  = {sid: i for i, sid in enumerate(adfeat.obs['spot_id'])}

    # Iterate samples
    for sample in tqdm(ad224.obs['sample'].unique()):
        # --- layer 224 dictionary ---
        layer224 = {}
        for sid in ad224.obs.query("sample == @sample")['spot_id']:
            i224 = idx224[sid]
            if sid not in idxfeat:
                continue  # skip if no feature/position
            if sid not in idx224:
                continue
            if sid not in idxhier and sid not in idxhier:
                pass
            ifeat = idxfeat[sid]
            layer224[sid] = {
                'position': adfeat.obsm['spatial'][ifeat],
                'feature':  adfeat.obsm['features'][ifeat],
                'label':    ad224.X[i224].A1 if hasattr(ad224.X, "A1") else ad224.X[i224]
            }

        # --- layer 512 & 1024 plus relationships ---
        layer512: Dict[str,Any] = {}
        layer1024: Dict[str,Any] = {}
        rel_512_224: Dict[str,List[str]] = {}
        rel_1024_512: Dict[str,List[str]] = {}

        subobs = adhier.obs.query("sample == @sample")
        # 512
        for _, row in subobs.query("size == 512").iterrows():
            pid = row['spot_id']
            if pid not in idxfeat or pid not in idxhier:
                continue
            iher = idxhier[pid]
            ifeat = idxfeat[pid]
            layer512[pid] = {
                'position': adfeat.obsm['spatial'][ifeat],
                'feature':  adfeat.obsm['features'][ifeat],
                'label':    adhier.X[iher].A1 if hasattr(adhier.X, "A1") else adhier.X[iher]
            }
            rel_512_224[pid] = row['spot_ids'].split(';')

        # 1024
        for _, row in subobs.query("size == 1024").iterrows():
            pid = row['spot_id']
            if pid not in idxfeat or pid not in idxhier:
                continue
            iher = idxhier[pid]
            ifeat = idxfeat[pid]
            layer1024[pid] = {
                'position': adfeat.obsm['spatial'][ifeat],
                'feature':  adfeat.obsm['features'][ifeat],
                'label':    adhier.X[iher].A1 if hasattr(adhier.X, "A1") else adhier.X[iher]
            }
            rel_1024_512[pid] = row['spot_ids'].split(';')

        # Build intra‐layer graphs
        e224,w224,x224,y224,ids224   = build_graph_from_dict(layer224, distance_threshold)
        e512,w512,x512,y512,ids512   = build_graph_from_dict(layer512, distance_threshold)
        e1024,w1024,x1024,y1024,ids1024 = build_graph_from_dict(layer1024, distance_threshold)

        # Build cross‐layer edges
        edges512_224  = build_cross_layer_edges(ids512,  ids224,  rel_512_224)
        edges1024_512 = build_cross_layer_edges(ids1024, ids512, rel_1024_512)

        # Assemble HeteroData
        data = HeteroData()
        data['layer_224'].x          = x224
        data['layer_224'].edge_index = e224
        data['layer_224'].edge_attr  = w224
        data['layer_224'].y          = y224

        data['layer_512'].x          = x512
        data['layer_512'].edge_index = e512
        data['layer_512'].edge_attr  = w512
        data['layer_512'].y          = y512

        data['layer_1024'].x          = x1024
        data['layer_1024'].edge_index = e1024
        data['layer_1024'].edge_attr  = w1024
        data['layer_1024'].y          = y1024

        data['layer_512','to','layer_224'].edge_index  = torch.tensor(edges512_224).t().contiguous()
        data['layer_1024','to','layer_512'].edge_index = torch.tensor(edges1024_512).t().contiguous()

        out_path = os.path.join(output_dir, f"{sample}.pt")
        torch.save(data, out_path)
        print(f"[{sample}] hetero‐graph saved to {out_path}")

    print(f"All done in {time.time()-t0:.1f}s")


def example_usage():
    build_multires_graphs(
        adata_224_path="../filtered_adata_norm.h5ad",
        adata_hier_path="../adata_counts_hierarchy_norm.h5ad",
        adata_feat_path="../adata_with_features.h5ad",
        output_dir="../graph_outputs",
        distance_threshold=2300.0
    )


if __name__ == "__main__":
    example_usage()
