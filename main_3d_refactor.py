# main_3d.py

import os
import sys
import random
import argparse
import logging

import torch
import torch.nn as nn
import torch.optim as optim
import torch.backends.cudnn as cudnn
from torch.utils.data import Subset
import torchvision.transforms as T
from tqdm import tqdm
from tensorboardX import SummaryWriter
import anndata as ad

import numpy as np
import pandas as pd
import scanpy as sc
from sklearn.model_selection import KFold

from models.graph_construction import edge_index_to_adj_matrix
from models.model_3d          import ST_GTC_3d
from models.losses            import calculate_pcc, pcc_loss

from dataloader_3d_refactor import (
    AnnDataImageGraphDataset,
    collate_fn, collate_fn_result
)

from torch.amp import autocast, GradScaler
#import torch.multiprocessing as mp
#mp.set_start_method("spawn", force=True)

parser = argparse.ArgumentParser()
parser.add_argument('--adata_224',    type=str, required=True)
parser.add_argument('--adata_counts', type=str, required=True)
parser.add_argument('--adata_feats',  type=str, required=True)
parser.add_argument('--tiff',         type=str, required=True)
parser.add_argument('--root_path',    type=str, default='.')
parser.add_argument('--n_splits',     type=int, default=4)
parser.add_argument('--max_iters',    type=int, default=25000)
parser.add_argument('--base_lr',      type=float, default=0.001)
parser.add_argument('--lr_decay',     type=float, default=0.9)
parser.add_argument('--seed',         type=int, default=1337)
parser.add_argument('--gpu',          type=str, default='0')
parser.add_argument('--batch_size',   type=int, default=128)
parser.add_argument('--spatial_scale',type=float, default=15.0)
parser.add_argument('--patch_size',   type=int, default=224)
parser.add_argument('--fp16', action='store_true', help='Enable mixed precision (fp16) training')
parser.add_argument('--num_workers', type=int, default=10)
parser.add_argument('--save_path', type=str, default='./weight/ours')
args = parser.parse_args()

# Setup
os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
cudnn.benchmark = True
random.seed(args.seed); torch.manual_seed(args.seed)

# Instead, get unique slides from AnnData file (efficiently)
adata_224_obs = ad.read_h5ad(args.adata_224, backed='r').obs
unique_slides = list(adata_224_obs['sample'].astype(str).unique())

# Dataset instance (shared across folds)
ds = AnnDataImageGraphDataset(
    args.adata_224, args.adata_counts, args.adata_feats,
    args.tiff,
    'graph_outputs/embryo.pt',
    'embryo_data/spot_iou/embryo_iou_results_new_1.csv',
    spatial_scale=args.spatial_scale,
    patch_size=args.patch_size,
    transform=T.Compose([
        T.RandomHorizontalFlip(),
        T.RandomVerticalFlip(),
        T.RandomRotation(90),
        T.ToTensor(),
        T.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])
    ])
)

# Cross‐validation setup
n_slides = len(unique_slides)
kf = KFold(n_splits=args.n_splits, shuffle=True, random_state=args.seed)

# Logging
save_path = args.save_path
os.makedirs(save_path, exist_ok=True)
logging.basicConfig(
    filename=os.path.join(save_path,'log.txt'),
    level=logging.INFO,
    format='[%(asctime)s] %(message)s'
)
logging.getLogger().addHandler(logging.StreamHandler(sys.stdout))
logging.info(str(args))

# AMP scaler (if needed)
scaler = GradScaler() if args.fp16 else None

metrics = {'mse':[], 'mae':[], 'pcc':[]}
fold = 0
best_pcc = -1.0
best_fold = 0

for train_slides, test_slides in kf.split(unique_slides):
    fold += 1
    logging.info(f"\n=== Fold {fold} ===")

    # determine obs‐indices for train/test
    train_idx = [
        i for i, s in enumerate(ds.sample_idxs)
        if s in train_slides
    ]
    test_idx = [
        i for i, s in enumerate(ds.sample_idxs)
        if s in test_slides
    ]

    train_ds = Subset(ds, train_idx)
    test_ds  = Subset(ds, test_idx)

    train_loader = torch.utils.data.DataLoader(
        train_ds, batch_size=args.batch_size, shuffle=True,
        collate_fn=lambda b: collate_fn(b, ds.spot_iou, ds.graph, True),
        num_workers=args.num_workers, pin_memory=True
    )
    test_loader = torch.utils.data.DataLoader(
        test_ds, batch_size=args.batch_size, shuffle=False,
        collate_fn=lambda b: collate_fn_result(b, ds.spot_iou, ds.graph, False, train_slides),
        num_workers=args.num_workers, pin_memory=True
    )

    # Model & optimizer
    model     = ST_GTC_3d().to(device)
    optimizer = optim.SGD(model.parameters(), lr=args.base_lr, momentum=0.9)
    writer    = SummaryWriter(os.path.join(save_path, f'fold{fold}'))

    iter_num = 0
    max_epoch = 51
    best_fold_pcc = -1.0

    for epoch in tqdm(range(1, max_epoch+1), desc=f"Fold {fold}"):
        model.train()
        for (
            images, labels, adj_sub, adj_prop,
            positions, idx_1024, idx_512,
            features_1024, features_512,
            graph_torch, known_label
        ) in tqdm(train_loader, desc=f"Fold {fold} Epoch {epoch}"):
            # adjust LR
            for g in optimizer.param_groups:
                g['lr'] = args.base_lr * (1 - iter_num/args.max_iters)**args.lr_decay

            g512  = graph_torch['layer_512']
            g1024 = graph_torch['layer_1024']
            bd512 = edge_index_to_adj_matrix(g512)
            bd1024= edge_index_to_adj_matrix(g1024)

            batch_data = [
                images, adj_sub, adj_prop,
                features_512, features_1024,
                g512.x, g1024.x,
                bd512, bd1024,
                known_label
            ]
            batch_data = [d.to(device) for d in batch_data]

            optimizer.zero_grad()
            if args.fp16:
                with autocast('cuda',dtype=torch.float16):
                    pred, p512, p1024 = model(*batch_data)
                    p512c = p512.cpu()[idx_512.cpu()]
                    p1024c= p1024.cpu()[idx_1024.cpu()]
                    loss_other = pcc_loss(g512.y.to(device), p512) \
                               + pcc_loss(g1024.y.to(device), p1024)
                    loss_cons  = pcc_loss(p512c.to(device), labels.to(device)) \
                               + pcc_loss(p1024c.to(device), labels.to(device))
                    loss_spot  = 0.75*nn.MSELoss()(pred, labels.to(device)) \
                               + 0.25*pcc_loss(pred, labels.to(device))
                    loss = loss_spot + 0.125*loss_other + 0.125*loss_cons
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                pred, p512, p1024 = model(*batch_data)
                p512c = p512.cpu()[idx_512.cpu()]
                p1024c= p1024.cpu()[idx_1024.cpu()]
                loss_other = pcc_loss(g512.y.to(device), p512) \
                           + pcc_loss(g1024.y.to(device), p1024)
                loss_cons  = pcc_loss(p512c.to(device), labels.to(device)) \
                           + pcc_loss(p1024c.to(device), labels.to(device))
                loss_spot  = 0.75*nn.MSELoss()(pred, labels.to(device)) \
                           + 0.25*pcc_loss(pred, labels.to(device))
                loss = loss_spot + 0.125*loss_other + 0.125*loss_cons
                loss.backward()
                optimizer.step()

            iter_num += 1
            writer.add_scalar('lr', optimizer.param_groups[0]['lr'], iter_num)
            writer.add_scalar('loss', loss.item(), iter_num)
            logging.info(f"iter {iter_num} — loss {loss.item():.4f}")
            break
            if iter_num >= args.max_iters:
                break
        if iter_num >= args.max_iters:
            break
        
        # Save model after each epoch
        torch.save({
            'fold': fold,
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'iter_num': iter_num,
            'loss': loss.item() if 'loss' in locals() else 0.0,
            'args': args
        }, os.path.join(save_path, f'model_fold{fold}_epoch{epoch}.pth'))
        logging.info(f"Saved model after epoch {epoch}")

        # Evaluation every 10 epochs and last epoch
        if epoch % 1 == 0 or epoch == max_epoch:
            model.eval()
            val_loss = val_mae = val_pcc = total = 0
            with torch.no_grad():
                for (
                    images, labels, adj_sub, adj_prop,
                    positions, idx_1024, idx_512,
                    features_1024, features_512,
                    graph_torch, known_label, indices, levels
                ) in test_loader:
                    g512  = graph_torch['layer_512']
                    g1024 = graph_torch['layer_1024']
                    bd512 = edge_index_to_adj_matrix(g512)
                    bd1024= edge_index_to_adj_matrix(g1024)

                    batch_data = [
                        images, adj_sub, adj_prop,
                        features_512, features_1024,
                        g512.x, g1024.x,
                        bd512, bd1024,
                        known_label
                    ]
                    batch_data = [d.to(device) for d in batch_data]

                    if args.fp16:
                        with autocast('cuda',dtype=torch.float16):
                            pred, _, _ = model(*batch_data)
                            idxs = torch.tensor(indices)
                            loss = nn.MSELoss()(pred.cpu()[idxs], labels[idxs])
                    else:
                        pred, _, _ = model(*batch_data)
                        idxs = torch.tensor(indices)
                        loss = nn.MSELoss()(pred.cpu()[idxs], labels[idxs])

                    val_loss += loss.item() * len(indices)
                    val_mae  += (pred.cpu()[idxs] - labels[idxs]).abs().mean().item() * len(indices)
                    val_pcc  += calculate_pcc(pred.cpu()[idxs], labels[idxs]).item() * len(indices)
                    total    += len(indices)

            fold_mse = val_loss/total
            fold_mae = val_mae/total
            fold_pcc = val_pcc/total
            
            metrics['mse'].append(fold_mse)
            metrics['mae'].append(fold_mae)
            metrics['pcc'].append(fold_pcc)
            
            logging.info(
                f"VAL Fold{fold} E{epoch}: MSE={fold_mse:.4f}, " +
                f"MAE={fold_mae:.4f}, PCC={fold_pcc:.4f}"
            )
            
            # Save best model for this fold
            if fold_pcc > best_fold_pcc:
                best_fold_pcc = fold_pcc
                torch.save({
                    'fold': fold,
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'mse': fold_mse,
                    'mae': fold_mae,
                    'pcc': fold_pcc,
                    'args': args
                }, os.path.join(save_path, f'model_fold{fold}_best.pth'))
                logging.info(f"Saved best model for fold {fold} with PCC: {fold_pcc:.4f}")

    # Save final model for this fold
    torch.save({
        'fold': fold,
        'epoch': max_epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'mse': metrics['mse'][-1],
        'mae': metrics['mae'][-1],
        'pcc': metrics['pcc'][-1],
        'args': args
    }, os.path.join(save_path, f'model_fold{fold}_final.pth'))
    
    # Track global best model
    if metrics['pcc'][-1] > best_pcc:
        best_pcc = metrics['pcc'][-1]
        best_fold = fold
        torch.save({
            'fold': fold,
            'epoch': max_epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'mse': metrics['mse'][-1],
            'mae': metrics['mae'][-1],
            'pcc': metrics['pcc'][-1],
            'args': args
        }, os.path.join(save_path, 'model_best_overall.pth'))
        logging.info(f"Saved best overall model from fold {fold} with PCC: {best_pcc:.4f}")

    writer.close()

# Save metrics
pd.DataFrame(metrics).to_excel(
    os.path.join(save_path,'crossval_metrics.xlsx'),
    index=False
)
logging.info(f"All done. Metrics saved. Best model from fold {best_fold} with PCC: {best_pcc:.4f}")
