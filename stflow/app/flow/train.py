import os
import json
import argparse
import shutil
from typing import Optional
import numpy as np
import pandas as pd
from time import time
from tqdm import tqdm
from operator import itemgetter

import torch

from stflow.utils import set_random_seed, merge_fold_results
from stflow.data.dataset import HESTDatasetPath, MultiHESTDataset, padding_batcher, HESTDataset
from stflow.data.normalize_utils import get_normalize_method
from stflow.model.denoiser import Denoiser
from stflow.model.phi_zinb import PhiEZINB, build_spot_features
from stflow.model.zinb_fallback import GeneCountFallback, prepare_count_tensor
from stflow.flow.interpolant import Interpolant
from stflow.app.flow.test import test
from stflow.app.flow.crossslide import SlideRepository, prepare_crossslide_condition
from stflow.hest_utils.utils import save_pkl


def _to_native(obj):
    """Recursively convert numpy/torch types to Python-native types for JSON serialization."""
    # numpy scalar
    if isinstance(obj, (np.generic,)):
        return obj.item()
    # torch scalar tensor
    if isinstance(obj, torch.Tensor):
        if obj.ndim == 0:
            return obj.item()
        return obj.detach().cpu().tolist()
    # numpy array
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    # basic containers
    if isinstance(obj, dict):
        return {k: _to_native(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_to_native(v) for v in obj]
    # leave other JSON-serializable types as is
    return obj


def main(args, split_id, train_sample_ids, test_sample_ids, val_save_dir, checkpoint_save_dir):
    normalize_method = get_normalize_method(args.normalize_method)

    print("Dataset Loading")
    use_global_counts = False
    args.use_global_counts = use_global_counts

    def _make_path(sample_id: str) -> HESTDatasetPath:
        return HESTDatasetPath(
            name=sample_id,
            h5_path=os.path.join(args.embed_dataroot, args.dataset, args.feature_encoder, f"fp32/{sample_id}.h5"),
            h5ad_path=os.path.join(args.source_dataroot, args.dataset, f"adata/{sample_id}.h5ad"),
            gene_list_path=os.path.join(args.source_dataroot, args.dataset, args.gene_list),
        )

    sample_id_paths = [_make_path(sample_id) for sample_id in tqdm(train_sample_ids)]
    train_dataset = MultiHESTDataset(
        sample_id_paths,
        distribution=args.patch_distribution,
        normalize_method=normalize_method,
        sample_times=args.sample_times,
        neighbor_mode=args.neighbor_mode,
        cross_slide_neighbors=args.cross_slide_neighbors,
        neighbor_candidate_count=args.neighbor_candidate_count,
        neighbor_alpha=args.neighbor_alpha,
        use_global_counts=use_global_counts,
    )
    fallback_helper = None
    if args.prior_sampler == "learned_zinb":
        fallback_helper = GeneCountFallback(train_dataset.sp_datasets)
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        collate_fn=padding_batcher(return_global_counts=use_global_counts),
    )

    # using test sample ids for validation
    val_loaders = []
    for i, sample_id in enumerate(tqdm(test_sample_ids)):
        sample_id_path = HESTDatasetPath(
            name=sample_id,
            h5_path=os.path.join(args.embed_dataroot, args.dataset, args.feature_encoder, f"fp32/{sample_id}.h5"),
            h5ad_path=os.path.join(args.source_dataroot, args.dataset, f"adata/{sample_id}.h5ad"),
            gene_list_path=os.path.join(args.source_dataroot, args.dataset, args.gene_list),
        )
        
        val_loader = torch.utils.data.DataLoader(
            HESTDataset(
                sample_id_path, distribution="constant_1.0", 
                normalize_method=normalize_method,
                sample_times=1,
                use_global_counts=use_global_counts,
            ),
            batch_size=1,
            collate_fn=padding_batcher(return_global_counts=use_global_counts),
        )
        val_loaders.append(val_loader)
    slide_repo = None
    if getattr(args, "enable_crossslide_condition", False):
        train_paths = [_make_path(sample_id) for sample_id in train_sample_ids]
        slide_repo = SlideRepository(train_paths, normalize_method)

    device = torch.device(f"cuda:{args.device}" if torch.cuda.is_available() else "cpu")
    model = Denoiser(args).to(device)

    # Pass dataset reference if using spatial_empirical prior
    diffusier_kwargs = {
        'total_count': torch.tensor([args.zinb_total_count]),
        'logits': torch.tensor([args.zinb_logits]),
        'zi_logits': args.zinb_zi_logits,
        'normalize': (args.prior_sampler != "gaussian") and (args.prior_sampler != "spatial_empirical"),
    }
    
    phi_e_module = None
    if args.prior_sampler == "spatial_empirical":
        # Use train_dataset for neighbor lookups (even at test time)
        diffusier_kwargs['dataset'] = train_dataset
    elif args.prior_sampler == "learned_zinb":
        gene_list_path = os.path.join(args.source_dataroot, args.dataset, args.gene_list)
        if not os.path.exists(gene_list_path):
            raise FileNotFoundError(f"Gene list not found: {gene_list_path}")
        with open(gene_list_path, 'r') as f:
            gene_names = json.load(f)['genes']
        if args.phi_e_ckpt is None:
            raise ValueError("--phi_e_ckpt must be provided when using learned_zinb prior")
        state = torch.load(args.phi_e_ckpt, map_location=device)
        model_state = state.get("model", state)
        config = state.get("config", {})

        embed_dim = config.get("embed_dim", args.prior_phi_embed)
        hidden_dim = config.get("hidden_dim", args.prior_phi_hidden)
        dropout = config.get("dropout", args.prior_phi_dropout)
        spot_encoder_layers = config.get("spot_encoder_layers", 3)

        if "spot_encoder.input_proj.weight" in model_state:
            spot_feat_dim = model_state["spot_encoder.input_proj.weight"].shape[1]
        else:
            first_weight = model_state.get("spot_encoder.net.0.weight")
            if first_weight is None:
                raise RuntimeError("phi_e checkpoint missing spot encoder weights; cannot determine architecture.")
            if first_weight.dim() != 1:
                raise RuntimeError(
                    "phi_e checkpoint was trained with legacy 8-D features. "
                    "Please retrain phi_e using full gene-count inputs."
                )
            spot_feat_dim = len(gene_names)

        phi_e_module = PhiEZINB(
            num_genes=len(gene_names),
            spot_feat_dim=spot_feat_dim,
            embed_dim=embed_dim,
            hidden_dim=hidden_dim,
            dropout=dropout,
            spot_encoder_layers=spot_encoder_layers,
        ).to(device)

        phi_e_module.load_state_dict(model_state, strict=True)
        phi_e_module.eval()
        for param in phi_e_module.parameters():
            param.requires_grad_(False)
        diffusier_kwargs['phi_e'] = phi_e_module

    diffusier = Interpolant(args.prior_sampler, **diffusier_kwargs)
    optim_params = list(model.parameters())
    optimizer = torch.optim.Adam(optim_params, lr=args.lr)

    # ------------------------------------------------------------------
    # Eval-only: load a trained checkpoint, impute the sections listed in
    # the test split once, save metrics (and, with --save_per_slide_h5ad,
    # the imputed expression per section).
    # ------------------------------------------------------------------
    if getattr(args, 'eval_only', False):
        if not args.eval_checkpoint:
            raise ValueError("--eval_only requires --eval_checkpoint <path>")
        print(f"[eval_only] loading checkpoint: {args.eval_checkpoint}")
        ckpt = torch.load(args.eval_checkpoint, map_location=device, weights_only=False)
        model.load_state_dict(ckpt['model'])
        if phi_e_module is not None and 'phi_e' in ckpt:
            phi_e_module.load_state_dict(ckpt['phi_e'])
        model.eval()
        if phi_e_module is not None:
            phi_e_module.eval()
        # Re-seed right before inference so the stochastic prior sampler
        # (ZINB draws) produces the same trajectory across sweep configs.
        set_random_seed(args.seed)

        _t0 = time()
        val_perf_dict = test(
            args,
            diffusier,
            model,
            val_loaders,
            slide_repo=slide_repo,
            return_all=False,
            fallback_helper=fallback_helper,
            save_per_slide_h5ad=bool(args.save_per_slide_h5ad),
            output_dir=(args.slide_output_dir or os.path.join(val_save_dir, 'slides')) if args.save_per_slide_h5ad else None,
        )
        eval_wall = time() - _t0
        os.makedirs(val_save_dir, exist_ok=True)
        for patch_name, dataset_res in val_perf_dict.items():
            with open(os.path.join(val_save_dir, f'{patch_name}_results.json'), 'w') as f:
                json.dump(_to_native(dataset_res), f, sort_keys=True, indent=4)

        eval_meta = {
            'eval_checkpoint': args.eval_checkpoint,
            'n_sample_steps': args.n_sample_steps,
            'seed': args.seed,
            'eval_wall_seconds': eval_wall,
            'n_slides_eval': len(val_loaders),
            'sec_per_slide': eval_wall / max(1, len(val_loaders)),
        }
        with open(os.path.join(val_save_dir, 'eval_metadata.json'), 'w') as f:
            json.dump(eval_meta, f, indent=2, default=str)
        print(f"[eval_only] DONE in {eval_wall:.1f}s; results -> {val_save_dir}")
        return val_perf_dict.get('all', {})

    print("Training")
    best_pearson, best_val_dict = float("-inf"), None
    best_epoch = None
    early_stop_step = 0
    # Resource tracking (Exp 2 needs peak mem, sec/epoch, param counts).
    _log_resources = bool(getattr(args, "log_resources", False))
    epoch_times = []
    if _log_resources and torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
    epoch_iter = tqdm(range(1, args.epochs + 1), ncols=100)
    for epoch in epoch_iter:
        _epoch_start = time()
        avg_loss = 0
        model.train()
        if phi_e_module is not None:
            phi_e_module.eval()

        if model.controlnet_enabled:
            if args.controlnet_warmup_epochs > 0:
                warmup_factor = min(1.0, epoch / args.controlnet_warmup_epochs)
            else:
                warmup_factor = 1.0
            model.set_control_warmup(warmup_factor)
        if getattr(model, "crossslide_branches", None):
            if getattr(args, "crossslide_warmup_epochs", 0) > 0:
                cross_factor = min(1.0, epoch / args.crossslide_warmup_epochs)
            else:
                cross_factor = 1.0
            model.set_crossslide_warmup(cross_factor)

        for step, batch in enumerate(train_loader):
            if use_global_counts:
                (
                    img_features,
                    coords,
                    gene_exp,
                    coords_original,
                    z_indices,
                    detected_counts,
                    spot_valid_mask,
                    sample_names,
                    barcodes,
                    global_detected_counts,
                    global_detected_masks,
                    spot_indices,
                ) = batch
            else:
                (
                    img_features,
                    coords,
                    gene_exp,
                    coords_original,
                    z_indices,
                    detected_counts,
                    spot_valid_mask,
                    sample_names,
                    barcodes,
                ) = batch
            if args.prior_sampler == "learned_zinb":
                counts_for_phi = prepare_count_tensor(
                    sample_names=sample_names,
                    coords_original=coords_original,
                    raw_counts=gene_exp,
                    valid_mask=spot_valid_mask,
                    fallback_helper=fallback_helper,
                    force_fallback=False,
                )
                spot_features, spot_mask = build_spot_features(counts_for_phi)
            else:
                spot_features = None
                spot_mask = None

            crossslide_inputs = None
            if slide_repo is not None:
                crossslide_inputs = prepare_crossslide_condition(
                    sample_names=sample_names,
                    coords_original=coords_original,
                    spot_valid_mask=spot_valid_mask,
                    slide_repo=slide_repo,
                    n_genes=gene_exp.size(-1),
                    dtype=gene_exp.dtype,
                    device=device,
                )

            img_features = img_features.to(device)
            coords = coords.to(device)  # [B, N, 3] with decentered x,y and original z
            gene_exp = gene_exp.to(device)
            coords_original = coords_original.to(device)  # [B, N, 3] with original x,y,z
            z_indices = z_indices.to(device)  # [B] z-coordinate per slide
            detected_counts = detected_counts.to(device)

            if spot_mask is not None:
                spot_features = spot_features.to(device)
                spot_mask = spot_mask.to(device)
            
            # For spatial_empirical, pass original coordinates and z info
            if args.prior_sampler == "spatial_empirical" or args.prior_sampler == "learned_zinb":
                # Use user-provided k for empirical distribution neighbors
                k_neighbors = args.empirical_k_neighbors
                corrupt_kwargs = dict(
                    z_indices=z_indices,
                    coords_original=coords_original,
                    k_neighbors=k_neighbors,
                    reference_features=img_features,
                    detected_counts=detected_counts,
                    spot_features=spot_features,
                    spot_mask=spot_mask,
                )
                noisy_exp, t_steps = diffusier.corrupt_exp(
                    gene_exp,
                    **corrupt_kwargs,
                )
            else:
                corrupt_kwargs = dict(
                    detected_counts=detected_counts,
                    spot_features=spot_features,
                    spot_mask=spot_mask,
                )
                noisy_exp, t_steps = diffusier.corrupt_exp(
                    gene_exp,
                    **corrupt_kwargs,
                )
            
            model_kwargs = dict(
                exp=noisy_exp,
                img_features=img_features,
                coords=coords,
                labels=gene_exp,
                t_steps=t_steps,
            )
            if crossslide_inputs is not None and getattr(model, "crossslide_branches", None) is not None:
                model_kwargs["crossslide_expr"] = crossslide_inputs
            pred_exp, loss = model(**model_kwargs)

            optimizer.zero_grad()
            model.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip_norm)
            optimizer.step()

            if args.use_wandb:
                wandb.log({f"{args.dataset}/Train/{split_id}/loss": loss.cpu().item()})

            avg_loss += loss.cpu().item()
        
        avg_loss /= len(train_loader)
        epoch_iter.set_description(f"epoch: {epoch}, avg_loss: {avg_loss:.3f}")

        if args.save_step > 0 and epoch % args.save_step == 0:
            state = {'model': model.state_dict()}
            if phi_e_module is not None:
                state['phi_e'] = phi_e_module.state_dict()
            torch.save(state, os.path.join(checkpoint_save_dir, f"{epoch}.pth"))

        if epoch % args.eval_step == 0 or epoch == args.epochs:
            val_perf_dict = test(
                args,
                diffusier,
                model,
                val_loaders,
                slide_repo=slide_repo,
                return_all=False,
                fallback_helper=fallback_helper,
                save_per_slide_h5ad=False,
                output_dir=None,
            )

            pearson_mean = val_perf_dict["all"]['pearson_mean']
            is_finite = np.isfinite(pearson_mean)

            if is_finite and pearson_mean > best_pearson:
                best_pearson = float(pearson_mean)
                best_val_dict = val_perf_dict
                best_epoch = epoch
                best_state = {'model': model.state_dict(), 'epoch': epoch}
                if phi_e_module is not None:
                    best_state['phi_e'] = phi_e_module.state_dict()
                torch.save(best_state, os.path.join(checkpoint_save_dir, 'best.pth'))
                for patch_name, dataset_res in val_perf_dict.items():
                    with open(os.path.join(val_save_dir, f'{patch_name}_results.json'), 'w') as f:
                        json.dump(_to_native(dataset_res), f, sort_keys=True, indent=4)
                
                if args.save_per_slide_h5ad:
                    base_slide_dir = (
                        args.slide_output_dir
                        if args.slide_output_dir is not None
                        else os.path.join(val_save_dir, 'slides')
                    )
                    if args.slide_output_dir is not None:
                        best_slide_dir = os.path.join(base_slide_dir, 'best')
                    else:
                        best_slide_dir = base_slide_dir
                    if os.path.isdir(best_slide_dir):
                        shutil.rmtree(best_slide_dir)
                    test(
                        args,
                        diffusier,
                        model,
                        val_loaders,
                        return_all=False,
                        fallback_helper=fallback_helper,
                        save_per_slide_h5ad=True,
                        output_dir=best_slide_dir,
                    )
                
                # save_pkl(os.path.join(val_save_dir, 'inference_dump.pkl'), pred_dump)
                early_stop_step = 0

            else:
                early_stop_step += 1
                patience = args.early_stop_patience
                if patience > 0 and early_stop_step >= patience:
                    print("Early stopping")
                    break

            if args.use_wandb:
                for patch_name, dataset_res in val_perf_dict.items():
                    wandb.log({
                        f"{args.dataset}/Val/{split_id}/{patch_name}/pearson_mean": dataset_res['pearson_mean'],
                        f"{args.dataset}/Val/{split_id}/{patch_name}/pearson_std": dataset_res['pearson_std'],
                        f"{args.dataset}/Val/{split_id}/{patch_name}/l2_error_q1": dataset_res['l2_error_q1'],
                        f"{args.dataset}/Val/{split_id}/{patch_name}/l2_error_q2": dataset_res['l2_error_q2'],
                        f"{args.dataset}/Val/{split_id}/{patch_name}/l2_error_q3": dataset_res['l2_error_q3'],
                        f"{args.dataset}/Val/{split_id}/{patch_name}/r2_score_q1": dataset_res['r2_score_q1'],
                        f"{args.dataset}/Val/{split_id}/{patch_name}/r2_score_q2": dataset_res['r2_score_q2'],
                        f"{args.dataset}/Val/{split_id}/{patch_name}/r2_score_q3": dataset_res['r2_score_q3'],
                    })

        epoch_times.append(time() - _epoch_start)

    if _log_resources:
        cond_params = 0
        if getattr(model, "crossslide_branches", None) is not None:
            cond_params = sum(p.numel() for p in model.crossslide_branches.parameters())
        controlnet_branch_params = 0
        if getattr(model, "control_branch", None) is not None:
            controlnet_branch_params = sum(p.numel() for p in model.control_branch.parameters())
        resources = {
            "epochs_completed": len(epoch_times),
            "epoch_times_sec": epoch_times,
            "sec_per_epoch_mean": float(sum(epoch_times) / len(epoch_times)) if epoch_times else None,
            "sec_per_epoch_max": float(max(epoch_times)) if epoch_times else None,
            "sec_per_epoch_min": float(min(epoch_times)) if epoch_times else None,
            "peak_gpu_mem_gb": (torch.cuda.max_memory_allocated() / 1e9) if torch.cuda.is_available() else None,
            "param_count_total": sum(p.numel() for p in model.parameters()),
            "param_count_conditioning_crossslide": cond_params,
            "param_count_control_branch_self": controlnet_branch_params,
        }
        try:
            with open(os.path.join(val_save_dir, "resources.json"), "w") as f:
                json.dump(resources, f, indent=2)
        except Exception as _e:
            print(f"WARN: failed to write resources.json: {_e}")

    final_state = {'model': model.state_dict(), 'epoch': epoch}
    if phi_e_module is not None:
        final_state['phi_e'] = phi_e_module.state_dict()
    torch.save(final_state, os.path.join(checkpoint_save_dir, 'last.pth'))

    if best_val_dict is None or best_epoch is None:
        return {
            'pearson_mean': float('nan'),
            'pearson_std': float('nan'),
            'l2_error_q1': float('nan'),
            'l2_error_q2': float('nan'),
            'l2_error_q3': float('nan'),
            'r2_score_q1': float('nan'),
            'r2_score_q2': float('nan'),
            'r2_score_q3': float('nan'),
        }

    return best_val_dict["all"]


def run(args):
    # get train/test splits
    split_dir = os.path.join(args.source_dataroot, args.dataset, 'splits')
    splits = sorted(os.listdir(split_dir))
    all_split_results = []

    if args.split_indices is not None:
        if len(args.split_indices) == 0:
            raise ValueError('--split-indices provided but empty')
        requested_splits = sorted(set(args.split_indices))
        max_requested = requested_splits[-1]
    else:
        requested_splits = None
        max_requested = None

    def _resolve_split_path(kind: str, idx: int) -> Optional[str]:
        candidates = [
            os.path.join(split_dir, f"{kind}_{idx}.csv"),
            os.path.join(split_dir, f"{kind}_{idx}_back1.csv"),
        ]
        for path in candidates:
            if os.path.exists(path):
                return path
        return None

    i = 0
    while True:
        train_path = _resolve_split_path('train', i)
        test_path = _resolve_split_path('test', i)
        if train_path is None  or test_path is None:
            if requested_splits is None:
                if i == 0:
                    raise FileNotFoundError(
                        f"No train/test split files found in {split_dir}. "
                        f"Tried train_{i}.csv/train_{i}_back1.csv and test_{i}.csv/test_{i}_back1.csv."
                    )
                break
            if i in requested_splits:
                raise FileNotFoundError(
                    f"Requested split index {i} but no train/test CSV found in {split_dir}."
                )
            if max_requested is not None and i >= max_requested:
                break
            i += 1
            continue

        if requested_splits is not None and i not in requested_splits:
            if max_requested is not None and i >= max_requested:
                break
            i += 1
            continue

        print(f"Running dataset {args.dataset} split {i}")

        train_df = pd.read_csv(train_path)
        test_df = pd.read_csv(test_path)

        train_sample_ids = train_df['sample_id'].tolist()
        test_sample_ids = test_df['sample_id'].tolist()

        kfold_save_dir = os.path.join(args.save_dir, f'split{i}')
        os.makedirs(kfold_save_dir, exist_ok=True)
        checkpoint_save_dir = os.path.join(kfold_save_dir, 'checkpoints')
        os.makedirs(checkpoint_save_dir, exist_ok=True)

        results = main(args, i, train_sample_ids, test_sample_ids, kfold_save_dir, checkpoint_save_dir)
        all_split_results.append(results)

        if requested_splits is not None and i == max_requested:
            break

        i += 1

    kfold_results = merge_fold_results(all_split_results)
    with open(os.path.join(args.save_dir, 'results_kfold.json'), 'w') as f:
        p_corrs = kfold_results['pearson_corrs']
        p_corrs = sorted(p_corrs, key=itemgetter('mean'), reverse=True)
        kfold_results['pearson_corrs'] = p_corrs
        json.dump(_to_native(kfold_results), f, sort_keys=True, indent=4)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Train / evaluate HoloTea.")
    # data
    parser.add_argument('--datasets', nargs='+', required=True, help="Dataset folder name(s) under --source_dataroot, e.g. EMBRYO")
    parser.add_argument('--source_dataroot', type=str, required=True, help="Contains <dataset>/{adata,splits,<gene_list>}")
    parser.add_argument('--embed_dataroot', type=str, required=True, help="Contains <dataset>/<feature_encoder>/fp32/<sample_id>.h5")
    parser.add_argument('--gene_list', type=str, default='var_150genes.json', help="Gene list JSON in <source_dataroot>/<dataset>/")
    parser.add_argument('--n_genes', type=int, default=150)
    parser.add_argument('--feature_encoder', type=str, default='uni_v1_official', help="uni_v1_official | gigapath | ciga")
    parser.add_argument('--normalize_method', type=str, default="log1p")
    parser.add_argument('--split-indices', '--split_indices', dest='split_indices', type=int, nargs='*', default=None,
                        help='Split IDs to run (splits/train_<k>.csv, test_<k>.csv). Runs all splits when omitted.')
    # output
    parser.add_argument('--save_dir', type=str, required=True)
    parser.add_argument('--exp_code', type=str, default="holotea", help="Run name; outputs go to <save_dir>/<exp_code>/<dataset>/split<k>/")
    parser.add_argument('--use_wandb', action='store_true')
    parser.add_argument('--log_resources', action='store_true',
                        help='Write resources.json (peak GPU memory, sec/epoch, parameter counts).')

    # training
    parser.add_argument('--seed', type=int, default=1)
    parser.add_argument('--device', type=int, default=0)
    parser.add_argument('--sample_times', type=int, default=10, help='Number of times to sample patches from each image per epoch')
    parser.add_argument('--batch_size', type=int, default=2)
    parser.add_argument('--lr', type=float, default=5e-4)
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--clip_norm', type=float, default=1.)
    parser.add_argument('--save_step', type=int, default=-1, help='Save a checkpoint every N epochs (-1: only best/last)')
    parser.add_argument('--eval_step', type=int, default=1)
    parser.add_argument('--early_stop_patience', type=int, default=20,
                        help='Evaluations without improvement before stopping; <=0 disables early stopping')
    parser.add_argument('--num_workers', type=int, default=1)
    parser.add_argument('--loss_func', type=str, default='mse', help="mse | mae | pearson")
    parser.add_argument('--patch_distribution', type=str, default='uniform')

    # flow matching / prior
    parser.add_argument('--n_sample_steps', type=int, default=5)
    parser.add_argument('--prior_sampler', type=str, default="zinb",
                        choices=['gaussian', 'zero', 'zinb', 'spatial_empirical', 'learned_zinb'],
                        help="Start distribution of the flow; HoloTea uses learned_zinb (requires --phi_e_ckpt)")
    parser.add_argument('--phi_e_ckpt', type=str, default=None, help='Pretrained learned ZINB prior (scripts/train_phi_e.py)')
    parser.add_argument('--prior_phi_hidden', type=int, default=64)
    parser.add_argument('--prior_phi_embed', type=int, default=64)
    parser.add_argument('--prior_phi_dropout', type=float, default=0.1)
    parser.add_argument('--zinb_logits', type=float, default=0.1, help='Fixed ZINB prior (--prior_sampler zinb)')
    parser.add_argument('--zinb_total_count', type=float, default=1)
    parser.add_argument('--zinb_zi_logits', type=float, default=0.)
    parser.add_argument('--empirical_k_neighbors', type=int, default=128, help='spatial_empirical prior only')
    parser.add_argument('--neighbor_mode', choices=['spatial', 'cosine'], default='spatial', help='spatial_empirical prior only')
    parser.add_argument('--cross_slide_neighbors', type=int, default=2, help='spatial_empirical prior only')
    parser.add_argument('--neighbor_candidate_count', type=int, default=64, help='spatial_empirical prior only')
    parser.add_argument('--neighbor_alpha', type=float, default=0.5, help='spatial_empirical prior only')

    # backbone
    parser.add_argument('--backbone', type=str, default="spatial_transformer")
    parser.add_argument('--hidden_dim', type=int, default=128)
    parser.add_argument('--pairwise_hidden_dim', type=int, default=128)
    parser.add_argument('--n_layers', type=int, default=4)
    parser.add_argument('--dropout', type=float, default=0.2)
    parser.add_argument('--attn_dropout', type=float, default=0.2)
    parser.add_argument('--n_neighbors', type=int, default=8)
    parser.add_argument('--n_heads', type=int, default=4)
    parser.add_argument('--norm', type=str, default='layer', help="batch | layer")
    parser.add_argument('--activation', type=str, default='swiglu', help="relu | gelu | swiglu")
    parser.add_argument('--use_inducing', action='store_true', help='Global set attention (GSA) with inducing tokens')
    parser.add_argument('--n_inducing', type=int, default=32, help='Number of inducing tokens m')
    parser.add_argument('--use_hf_branch', action='store_true', help='High-frequency (local detail) branch')

    # x_t ControlNet branch
    parser.add_argument('--enable_controlnet', action='store_true', help='ControlNet branch on the current state x_t')
    parser.add_argument('--controlnet_channels', type=int, default=32)
    parser.add_argument('--controlnet_grid_size', type=int, default=64)
    parser.add_argument('--controlnet_intermediate_channels', type=int, default=64)
    parser.add_argument('--controlnet_scale', type=float, default=1.0)
    parser.add_argument('--controlnet_warmup_epochs', type=int, default=0)
    parser.add_argument('--controlnet_variant', choices=['standard', 'sharp'], default='standard')

    # cross-slide branches (z-1 / z+1)
    parser.add_argument('--enable_crossslide_condition', action='store_true',
                        help='Condition on the matched expression of the closest profiled sections above and below')
    parser.add_argument('--crossslide_scale', type=float, default=1.0)
    parser.add_argument('--crossslide_warmup_epochs', type=int, default=0)

    # evaluation / imputation
    parser.add_argument('--eval_only', action='store_true',
                        help='Skip training; load --eval_checkpoint and impute the sections in the test split')
    parser.add_argument('--eval_checkpoint', type=str, default=None)
    parser.add_argument('--save_per_slide_h5ad', action='store_true', help='Write imputed expression per section (.h5ad)')
    parser.add_argument('--slide_output_dir', type=str, default=None, help='Where to write the .h5ad files (default: <split dir>/slides)')
    parser.add_argument('--spot-metrics', action='store_true', help='Also report spot-level PCC/MSE/MAE (keys suffixed _spot)')

    args = parser.parse_args()

    args.feature_dim = {
        "uni_v1_official": 1024,
        "gigapath": 1536,
        "ciga": 512,
    }[args.feature_encoder]

    set_random_seed(args.seed)

    save_dir = os.path.join(args.save_dir, args.exp_code)
    os.makedirs(save_dir, exist_ok=True)

    if args.use_wandb:
        import wandb
        wandb.init(project="holotea", name=args.exp_code)
        wandb.config.update(args)

    print(f"Save dir: {save_dir}")
    print(args)

    for dataset in args.datasets:
        args.dataset = dataset
        args.save_dir = os.path.join(save_dir, dataset)
        os.makedirs(args.save_dir, exist_ok=True)

        with open(os.path.join(args.save_dir, 'config.json'), 'w') as f:
            json.dump(vars(args), f, sort_keys=True, indent=4)

        run(args)

    if args.use_wandb:
        wandb.finish()
