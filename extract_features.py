import os
import time
from typing import List, Tuple, Dict, Any

import numpy as np
import pandas as pd
import torch
from torch import nn
from torchvision import models, transforms
from PIL import Image
from joblib import Parallel, delayed
from tqdm import tqdm
import anndata as ad


def load_model_and_preprocess(target_dim: int = 1024) -> Tuple[nn.Module, transforms.Compose, torch.device]:
    """
    Load a pretrained ResNet50 backbone (no head) plus a reducer linear layer,
    and build the preprocessing pipeline.

    Args:
        target_dim (int): Dimension to reduce the 2048‑d output to.

    Returns:
        backbone (nn.Module): ResNet50 up to avgpool, outputs [B×2048×1×1]
        reducer  (nn.Module): Linear layer 2048→target_dim
        preprocess (Compose): torchvision transforms for ResNet50
        device   (torch.device): cuda if available, else cpu
    """
    # preprocessing
    preprocess = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225]),
    ])

    # load ResNet50, strip the final fc
    resnet = models.resnet50(pretrained=True)
    backbone = nn.Sequential(*list(resnet.children())[:-1])
    reducer = nn.Linear(2048, target_dim)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    backbone.to(device).eval()
    reducer.to(device).eval()

    return backbone, reducer, preprocess, device


def extract_patch_feature(
    img: Image.Image,
    backbone: nn.Module,
    reducer: nn.Module,
    preprocess: transforms.Compose,
    device: torch.device,
    sample: str,
    x: int,
    y: int,
    size: int,
    centered: bool,
) -> Dict[str, Any]:
    """
    Crop one image patch, run through model, and return its feature vector + metadata.

    Args:
        img      (PIL.Image): full image
        backbone (nn.Module): feature extractor
        reducer  (nn.Module): 2048→target_dim
        preprocess (Compose): transforms
        device   (torch.device)
        sample   (str): sample_id
        x, y     (int): if centered, patch center; if not, top-left corner
        size     (int): patch size (e.g. 224, 512, 1024)
        centered (bool): True if (x,y) is center, False if top-left

    Returns:
        dict with keys ['patch_id','sample','x','y','size','feature']
    """
    # determine top-left
    if centered:
        x0 = int(x - size // 2)
        y0 = int(y - size // 2)
    else:
        x0, y0 = x, y

    # skip out-of-bounds
    W, H = img.width, img.height
    if x0 < 0 or y0 < 0 or x0 + size > W or y0 + size > H:
        return None

    patch = img.crop((x0, y0, x0 + size, y0 + size))
    tensor = preprocess(patch).unsqueeze(0).to(device)

    with torch.no_grad():
        feat2048 = backbone(tensor).reshape(1, -1)          # [1×2048]
        feat      = reducer(feat2048).cpu().numpy().squeeze()  # [target_dim]

    patch_id = f"{sample}_{size}_{x0}_{y0}"
    return {
        'patch_id':  patch_id,
        'sample':    sample,
        'x':         x0,
        'y':         y0,
        'size':      size,
        'feature':   feat
    }


def extract_all_features(
    image_path: str,
    adata_path: str,
    windows_csv: str,
    spatial_scale: float,
    target_dim: int = 1024,
    n_jobs: int = -1,
    output_adata_path: str = './adata_with_features.h5ad',
) -> None:
    """
    Main driver: load image, AnnData, and CSV windows;
    crop 224×224 patches at each spot center;
    crop 512×512 & 1024×1024 at CSV-specified top-lefts;
    extract features in parallel; store in adata.uns; save.

    Args:
        image_path          (str): path to .tiff
        adata_path          (str): path to .h5ad
        windows_csv         (str): path to CSV with columns [sample,window_size,x,y]
        spatial_scale       (float): scale factor for adata.obsm['spatial']
        target_dim          (int): final feature dim
        n_jobs              (int): joblib parallel jobs
        output_adata_path   (str): where to write updated AnnData
    """
    t0 = time.time()
    print("🔬 Loading image and data...")
    img   = Image.open(image_path).convert('RGB')
    ad    = ad.read_h5ad(adata_path)
    dfwin = pd.read_csv(windows_csv)

    print(f"🧬 AnnData: {ad.n_obs} spots, samples: {ad.obs['sample'].nunique()}")
    print(f"📑 Windows: {len(dfwin)} entries")

    # prepare spot-centered tasks
    spots = ad.obsm['spatial'] * spatial_scale
    samples = ad.obs['sample'].values
    tasks_224 = [
        (img, *model_args, sample, float(xc), float(yc), 224, True)
        for (model_args, sample, (xc, yc)) in zip(
            [()] * len(spots),
            samples,
            spots
        )
    ]
    # Instead of the above zip hack, better:
    tasks_224 = []
    for sample, coord in zip(ad.obs['sample'], spots):
        tasks_224.append((img, ), )  # wrong...

    # Actually build tasks properly below after loading model

    # load model once
    backbone, reducer, preprocess, device = load_model_and_preprocess(target_dim)

    # build all tasks
    tasks: List[Tuple] = []

    # 1) 224×224 centered on each spot
    for sample, coord in zip(ad.obs['sample'], spots):
        x_c, y_c = coord
        tasks.append((img, backbone, reducer, preprocess, device, sample, x_c, y_c, 224, True))

    # 2) windows from CSV
    for _, row in dfwin.iterrows():
        tasks.append((img, backbone, reducer, preprocess, device,
                      str(row['sample']), int(row['x']), int(row['y']),
                      int(row['window_size']), False))

    print(f"🚀 Extracting features for {len(tasks)} patches in parallel...")
    results = Parallel(n_jobs=n_jobs, backend='threading', verbose=5)(
        delayed(extract_patch_feature)(*task) for task in tasks
    )

    # filter out any None
    feats = [r for r in results if r is not None]
    print(f"✅ Extracted {len(feats)} feature vectors")

    # Organize into dicts per size
    feat_dict: Dict[int, Dict[str, np.ndarray]] = {}
    for r in feats:
        feat_dict.setdefault(r['size'], {})[r['patch_id']] = r['feature']

    # Store into adata.uns
    ad.uns['patch_features'] = {}
    for size, d in feat_dict.items():
        ad.uns['patch_features'][f"{size}"] = d

    # Save updated AnnData
    print("💾 Writing updated AnnData...")
    os.makedirs(os.path.dirname(output_adata_path), exist_ok=True)
    ad.write_h5ad(output_adata_path)

    print("🕒 Done in {:.1f}s".format(time.time() - t0))


# Example usage
if __name__ == "__main__":
    extract_all_features(
        image_path='./image.tiff',
        adata_path='./adata.h5ad',
        windows_csv='./spot_windows.csv',
        spatial_scale=15.0,
        target_dim=1024,
        n_jobs=-1,
        output_adata_path='./adata_with_features.h5ad'
    )
