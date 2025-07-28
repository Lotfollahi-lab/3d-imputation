import os
import sys
import random
import time
import logging
import argparse

import torch
import torch.nn as nn
import torch.optim as optim
import torch.backends.cudnn as cudnn
import torchvision
from torchvision import transforms
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm
from tensorboardX import SummaryWriter

import numpy as np
import pandas as pd
import scanpy as sc  # <-- new
from models.graph_construction import edge_index_to_adj_matrix
from models.model_3d import ST_GTC_3d
from models.losses import calculate_pcc, pcc_loss
import warnings

from dataloader_3d_refactor import (
    AnnDataImageGraphDataset,
    collate_fn, collate_fn_result,
    create_dataloaders_for_train,
    create_dataloaders_for_result
)

warnings.filterwarnings("ignore", category=UserWarning,
                        message="Creating a tensor from a list of numpy.ndarrays is extremely slow")
warnings.filterwarnings("ignore", category=FutureWarning,
                        message="You are using `torch.load` with `weights_only=False`")

parser = argparse.ArgumentParser()
parser.add_argument('--root_path',      type=str, default='./ST_Breast_3D')
parser.add_argument('--adata_224',      type=str, required=True,
                    help='.h5ad with 224‑resolution counts & obs[x,y,sample]')
parser.add_argument('--adata_counts',   type=str, required=True,
                    help='.h5ad with 512/1024 labels in obs')
parser.add_argument('--adata_features', type=str, required=True,
                    help='.h5ad with feature_512/feature_1024 in obsm')
parser.add_argument('--max_iterations', type=int, default=25000)
parser.add_argument('--base_lr',        type=float, default=0.001)
parser.add_argument('--lr_decay',       type=float, default=0.9)
parser.add_argument('--seed',           type=int, default=1337)
parser.add_argument('--gpu',            type=str, default='0')
parser.add_argument('--batch_size',     type=int, default=128)
parser.add_argument('--rate',           type=int, default=10)
args = parser.parse_args()

# ——— Setup ——————————————————————————————————————————————————————————
os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
device    = torch.device("cuda" if torch.cuda.is_available() else "cpu")
cudnn.benchmark = True
random.seed(args.seed)
torch.manual_seed(args.seed)

# ——— Load AnnData —————————————————————————————————————————————————————
adata_224      = sc.read_h5ad(args.adata_224)
adata_counts   = sc.read_h5ad(args.adata_counts)
adata_features = sc.read_h5ad(args.adata_features)

# ——— Logging / transforms —————————————————————————————————————————————————
save_path = './weight/ours'
os.makedirs(save_path, exist_ok=True)
logging.basicConfig(
    filename=os.path.join(save_path, "log.txt"),
    level=logging.INFO,
    format='[%(asctime)s.%(msecs)03d] %(message)s',
    datefmt='%H:%M:%S'
)
logging.getLogger().addHandler(logging.StreamHandler(sys.stdout))
logging.info(str(args))

img_transform = transforms.Compose([
    torchvision.transforms.RandomHorizontalFlip(),
    torchvision.transforms.RandomVerticalFlip(),
    torchvision.transforms.RandomApply([torchvision.transforms.RandomRotation((90,90))]),
    torchvision.transforms.ToTensor(),
    transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])
])

mse_loss_fn = nn.MSELoss()
# ————————————————————————————————————————————————————————————————————————

# Cross‑validation splits by first letter of sample
all_samples = adata_224.obs['sample'].astype(str).unique().tolist()
fold_splits = [
    {'A','E'},
    {'B','F'},
    {'C','G'},
    {'D','H'},
]

for fold_index, fold_split in enumerate(fold_splits):
    logging.info(f"\n=== Fold {fold_index+1} (val on {fold_split}) ===")

    # Partition slides by their first letter
    train_slides = [s for s in all_samples if s[0] not in fold_split]
    val_slides   = [s for s in all_samples if s[0]     in fold_split]

    # ——— Build train loaders per‑slide —————————————————————
    train_loaders = {}
    for slide in train_slides:
        gp = os.path.join(args.root_path, f'3D_graph_data/{slide}.pt')
        ap = os.path.join(args.root_path, f'3D_combined_information/{slide}.csv')
        dl = create_dataloaders_for_train(
            adata_224, adata_counts, adata_features,
            graph_path=gp, adj_matrix_path=ap,
            batch_size=args.batch_size,
            transform=img_transform,
            root_path=args.root_path
        )
        train_loaders[slide] = dl['train']

    # ——— Build val/train‑part loaders per‑slide —————————————————
    val_train_loaders = {}
    val_test_loaders  = {}
    for slide in val_slides:
        gp = os.path.join(args.root_path, f'3D_graph_data/{slide}.pt')
        ap = os.path.join(args.root_path, f'3D_combined_information/{slide}.csv')
        dl = create_dataloaders_for_result(
            adata_224, adata_counts, adata_features,
            graph_path=gp, adj_matrix_path=ap,
            batch_size=args.batch_size,
            transform=img_transform,
            root_path=args.root_path
        )
        val_train_loaders[slide] = dl['train']
        val_test_loaders[slide]  = dl['test']

    # Merge val‑part into train for semi‑supervised
    train_loaders.update(val_train_loaders)

    # ——— Model / optimizer ——————————————————————————————————————————  
    model     = ST_GTC_3d().to(device)
    optimizer = optim.SGD(model.parameters(), lr=args.base_lr, momentum=0.9)
    writer    = SummaryWriter(os.path.join(save_path,'log'))

    iter_num = 0
    max_epoch = 51
    for epoch in tqdm(range(max_epoch), ncols=70):
        model.train()
        # shuffle slide order
        slides = list(train_loaders.keys()); random.shuffle(slides)

        for slide in slides:
            loader = train_loaders[slide]
            for param_group in optimizer.param_groups:
                param_group['lr'] = args.base_lr * (1 - iter_num/args.max_iterations)**args.lr_decay

            for (
                images, labels, adj_sub, adj_prop,
                positions, idx_1024, idx_512,
                feat_1024, feat_512,
                graph_torch, known_label
            ) in loader:
                # assemble batch_data exactly as before…
                g512 = graph_torch['layer_512']
                g1024= graph_torch['layer_1024']
                batch_data = [
                    images, adj_sub, adj_prop, feat_512, feat_1024,
                    g512.x, g1024.x,
                    edge_index_to_adj_matrix(g512),
                    edge_index_to_adj_matrix(g1024),
                    known_label
                ]
                batch_data = [d.to(device) for d in batch_data]

                pred, p512, p1024 = model(*batch_data)
                p512_con = p512.cpu()[idx_512.cpu()]
                p1024_con= p1024.cpu()[idx_1024.cpu()]

                loss_other = pcc_loss(g512.y.to(device), p512) \
                           + pcc_loss(g1024.y.to(device), p1024)
                loss_cons  = pcc_loss(p512_con.to(device), labels.to(device)) \
                           + pcc_loss(p1024_con.to(device), labels.to(device))
                loss_spot  = 0.75*mse_loss_fn(pred, labels.to(device)) \
                           + 0.25*pcc_loss(pred, labels.to(device))
                loss = loss_spot + 0.125*loss_other + 0.125*loss_cons

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                iter_num += 1
                writer.add_scalar('lr', optimizer.param_groups[0]['lr'], iter_num)
                writer.add_scalar('loss', loss.item(), iter_num)
                logging.info(f"iter {iter_num} — loss {loss.item():.4f}")

                if iter_num >= args.max_iterations:
                    break
            if iter_num >= args.max_iterations:
                break

        # ——— Validation every 10 epochs —————————————————————————————————
        if epoch % 10 == 0 or epoch+1 == max_epoch:
            model.eval()
            with torch.no_grad():
                for slide, loader in val_test_loaders.items():
                    val_loss = val_mae = val_pcc = n_samples = 0
                    for (
                        images, labels, adj_sub, adj_prop,
                        positions, idx_1024, idx_512,
                        feat_1024, feat_512,
                        graph_torch, known_label, indices, _
                    ) in loader:
                        g512 = graph_torch['layer_512']
                        g1024= graph_torch['layer_1024']
                        batch_data = [
                            images, adj_sub, adj_prop, feat_512, feat_1024,
                            g512.x, g1024.x,
                            edge_index_to_adj_matrix(g512),
                            edge_index_to_adj_matrix(g1024),
                            known_label
                        ]
                        batch_data = [d.to(device) for d in batch_data]
                        pred, _, _ = model(*batch_data)

                        idxs = torch.tensor(indices, device=device)
                        loss = mse_loss_fn(pred[idxs], labels[idxs].to(device))
                        val_loss += loss.item() * len(indices)
                        val_mae  += (pred[idxs].cpu() - labels[idxs].cpu()).abs().mean().item() * len(indices)
                        val_pcc  += calculate_pcc(pred[idxs], labels[idxs].to(device)).item() * len(indices)
                        n_samples+= len(indices)

                    avg = lambda x: x / n_samples if n_samples else 0
                    logging.info(
                        f"VAL {slide} — MSE {avg(val_loss):.4f}, MAE {avg(val_mae):.4f}, PCC {avg(val_pcc):.4f}"
                    )

    # ——— Save metrics —————————————————————————————————————————————————————
    metrics = {
        'mse': [], 'mae': [], 'pcc': []
    }
    df = pd.DataFrame(metrics)
    df.to_excel(os.path.join(save_path, 'ours_new_metrics.xlsx'), index=False)
    logging.info("All done, metrics written.")
