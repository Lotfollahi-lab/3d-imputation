import os
import time
from typing import List

import torch
from torch_geometric.data import HeteroData


def merge_section_graphs(
    folder_path: str,
    output_path: str
) -> None:
    """
    Load all .pt hetero‐graphs from folder_path (sorted by filename),
    merge them into one big HeteroData, and write to output_path.
    """
    t0 = time.time()
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    # Collect and sort filenames alphabetically
    files = sorted([f for f in os.listdir(folder_path) if f.endswith('.pt')])
    if not files:
        raise RuntimeError(f"No .pt files found in {folder_path}")

    # Load graphs in order
    graphs = [torch.load(os.path.join(folder_path, f), weights_only=False) for f in files]

    # Start from the first graph
    merged = graphs[0]
    offset_512  = merged['layer_512'].x.size(0)
    offset_1024 = merged['layer_1024'].x.size(0)

    # Merge the rest
    for g in graphs[1:]:
        # 512‐layer
        merged['layer_512'].x = torch.cat([merged['layer_512'].x, g['layer_512'].x], dim=0)
        merged['layer_512'].y = torch.cat([merged['layer_512'].y, g['layer_512'].y], dim=0)
        e512 = g['layer_512'].edge_index + offset_512
        merged['layer_512'].edge_index = torch.cat([merged['layer_512'].edge_index, e512], dim=1)

        # 1024‐layer
        merged['layer_1024'].x = torch.cat([merged['layer_1024'].x, g['layer_1024'].x], dim=0)
        merged['layer_1024'].y = torch.cat([merged['layer_1024'].y, g['layer_1024'].y], dim=0)
        e1024 = g['layer_1024'].edge_index + offset_1024
        merged['layer_1024'].edge_index = torch.cat([merged['layer_1024'].edge_index, e1024], dim=1)

        # advance offsets
        offset_512  += g['layer_512'].x.size(0)
        offset_1024 += g['layer_1024'].x.size(0)

    # Save final merged graph
    torch.save(merged, output_path)
    print(f"Merged {len(files)} sections into {output_path} in {time.time()-t0:.1f}s")


def example_usage():
    SECTION_DIR = '../graph_outputs/'
    OUTPUT_PT   = '../graph_outputs/embryo.pt'
    merge_section_graphs(SECTION_DIR, OUTPUT_PT)


if __name__ == '__main__':
    example_usage()
