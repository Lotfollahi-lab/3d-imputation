import torch
import numpy as np
from pathlib import Path
from typing import Dict
from scipy.stats import pearsonr
import scanpy as sc
from scvi.distributions import ZeroInflatedNegativeBinomial

from stflow.model.phi_zinb import build_spot_features
from stflow.model.zinb_fallback import prepare_count_tensor
from .crossslide import prepare_crossslide_condition


def metric_func(
    preds_all: np.ndarray,
    y_test: np.ndarray,
    genes: list,
    spot_metrics: bool = False,
    spot_suffix: str = "_spot",
):
    errors = []
    r2_scores = []
    pearson_corrs = []
    pearson_genes = []
    
    n_nan_genes = 0
    for i, target in enumerate(range(y_test.shape[1])):
        preds = preds_all[:, target]
        target_vals = y_test[:, target]

        errors.append(float(np.mean((preds - target_vals) ** 2)))
        r2_scores.append(
            float(
                1
                - np.sum((target_vals - preds) ** 2)
                / np.sum((target_vals - np.mean(target_vals)) ** 2)
            )
        )
        pearson_corr, _ = pearsonr(target_vals, preds)
        pearson_corrs.append(pearson_corr)

        if np.isnan(pearson_corr):
            n_nan_genes += 1

        score_dict = {
            'name': genes[i],
            'pearson_corr': pearson_corr,
        }
        pearson_genes.append(score_dict)

    if n_nan_genes > 0:
        print(f"Warning: {n_nan_genes} genes have NaN Pearson correlation")

    result = {
        'l2_errors': list(errors), 
        'r2_scores': list(r2_scores),
        'pearson_corrs': pearson_genes,
        'pearson_mean': float(np.mean(pearson_corrs)),
        'pearson_std': float(np.std(pearson_corrs)),
        'l2_error_q1': float(np.percentile(errors, 25)),
        'l2_error_q2': float(np.median(errors)),
        'l2_error_q3': float(np.percentile(errors, 75)),
        'r2_score_q1': float(np.percentile(r2_scores, 25)),
        'r2_score_q2': float(np.median(r2_scores)),
        'r2_score_q3': float(np.percentile(r2_scores, 75))
    }
    if spot_metrics:
        result.update(_spot_level_metrics(preds_all, y_test, suffix=spot_suffix))
    return result


def _spot_level_metrics(preds_all: np.ndarray, y_test: np.ndarray, suffix: str = "_spot") -> Dict[str, float]:
    if preds_all.size == 0 or y_test.size == 0:
        return {}

    diff = preds_all - y_test
    mse_per_spot = np.mean(diff ** 2, axis=1)
    mae_per_spot = np.mean(np.abs(diff), axis=1)

    x = preds_all - preds_all.mean(axis=1, keepdims=True)
    y = y_test - y_test.mean(axis=1, keepdims=True)
    denom = np.sqrt(np.sum(x ** 2, axis=1) * np.sum(y ** 2, axis=1) + 1e-8)
    # Prevent invalid divisions when denom becomes zero
    denom = np.where(denom == 0, 1e-8, denom)
    spot_pcc = np.sum(x * y, axis=1) / denom

    return {
        f"pcc_mean{suffix}": float(np.mean(spot_pcc)),
        f"pcc_std{suffix}": float(np.std(spot_pcc)),
        f"mse_mean{suffix}": float(np.mean(mse_per_spot)),
        f"mse_std{suffix}": float(np.std(mse_per_spot)),
        f"mae_mean{suffix}": float(np.mean(mae_per_spot)),
        f"mae_std{suffix}": float(np.std(mae_per_spot)),
    }


@torch.no_grad()
def test(
    args,
    diffusier,
    model,
    loader_list,
    slide_repo=None,
    return_all=False,
    fallback_helper=None,
    save_per_slide_h5ad: bool = False,
    output_dir: str | None = None,
):
    model.eval()
    device = torch.device(f"cuda:{args.device}" if torch.cuda.is_available() else "cpu")
    phi_module = getattr(getattr(diffusier.prior_sampler, 'prior_sampler', None), 'phi_e', None)
    if phi_module is not None:
        phi_module.eval()
    all_pred, all_gt = [], []
    res_dict = {}
    total_mle_pred = 0.0
    total_mle_gt = 0.0
    total_mle_count = 0

    for loader in loader_list:
        cur_pred, cur_gt = [], []
        cur_coords = []
        cur_coords_original = []
        cur_barcodes = []
        slide_name = None
        mle_pred_sum = 0.0
        mle_gt_sum = 0.0
        mle_count = 0

        for batch in loader:
            (
                img_features,
                coords,
                labels,
                coords_original,
                z_indices,
                detected_counts,
                spot_valid_mask,
                sample_names,
                barcodes,
            ) = batch

            if slide_name is None:
                slide_name = sample_names[0] if len(sample_names) > 0 else loader.dataset.name

            if args.prior_sampler == "learned_zinb":
                if fallback_helper is None:
                    raise RuntimeError("learned_zinb prior requires a fallback_helper during evaluation.")
                counts_for_phi = prepare_count_tensor(
                    sample_names=sample_names,
                    coords_original=coords_original,
                    raw_counts=labels,
                    valid_mask=spot_valid_mask,
                    fallback_helper=fallback_helper,
                    force_fallback=True,
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
                    n_genes=labels.size(-1),
                    dtype=labels.dtype,
                    device=device,
                )

            img_features = img_features.to(device)
            coords = coords.to(device)
            labels = labels.to(device)
            coords_original = coords_original.to(device)
            z_indices = z_indices.to(device)
            detected_counts = detected_counts.to(device)
            if spot_mask is not None:
                spot_features = spot_features.to(device)
                spot_mask = spot_mask.to(device)
            
            assert img_features.shape[0] == 1, "Batch size must be 1 for inference"

            batch_valid_mask = spot_valid_mask[0].to(device=device, dtype=torch.bool)

            if args.prior_sampler == "spatial_empirical":
                k_neighbors = args.empirical_k_neighbors
                prior_kwargs = dict(
                    z_indices=z_indices,
                    coords_original=coords_original,
                    k_neighbors=k_neighbors,
                    detected_counts=detected_counts,
                    spot_features=spot_features,
                    spot_mask=spot_mask,
                )
                exp_t1 = diffusier.sample_from_prior(
                    labels.shape,
                    **prior_kwargs,
                ).to(device)
            else:
                prior_kwargs = dict(
                    detected_counts=detected_counts,
                    spot_features=spot_features,
                    spot_mask=spot_mask,
                )
                exp_t1 = diffusier.sample_from_prior(
                    labels.shape,
                    **prior_kwargs,
                ).to(device)
            
            ts = torch.linspace(
                0.01, 1.0, args.n_sample_steps
            )[:, None].expand(args.n_sample_steps, exp_t1.shape[0]).to(device)

            for step, (t1, t2) in enumerate(zip(ts[:-1], ts[1:])):
                if crossslide_inputs is not None and getattr(model, "crossslide_branches", None) is not None:
                    pred = model.inference(
                        exp_t1,
                        img_features,
                        coords,
                        t1,
                        predict=True,
                        crossslide_expr=crossslide_inputs,
                    )
                else:
                    pred = model.inference(
                        exp_t1,
                        img_features,
                        coords,
                        t1,
                        predict=True,
                    )
                d_t = t2 - t1

                if step == args.n_sample_steps - 2:
                    break
                else:
                    exp_t1 = diffusier.denoise(pred, exp_t1, t1, d_t)

            sample = pred
            mask_np = batch_valid_mask.cpu().numpy().astype(bool)
            cur_pred.append(sample.squeeze(0).cpu().numpy()[mask_np])
            cur_gt.append(labels.squeeze(0).cpu().numpy()[mask_np])
            cur_coords.append(coords.squeeze(0).cpu().numpy()[mask_np])
            cur_coords_original.append(coords_original.squeeze(0).cpu().numpy()[mask_np])

            if len(barcodes) > 0:
                barcode_slice = barcodes[0]
                cur_barcodes.append([barcode_slice[i] for i, keep in enumerate(mask_np) if keep])
            else:
                cur_barcodes.append([])

            if phi_module is not None and args.prior_sampler == "learned_zinb":
                with torch.no_grad():
                    mu, theta, pi = phi_module(spot_features, mask=spot_mask)
                    log_pred, count = _zinb_log_likelihood(mu, theta, pi, sample, spot_mask)
                    log_gt, _ = _zinb_log_likelihood(mu, theta, pi, labels, spot_mask)
                    mle_pred_sum += log_pred
                    mle_gt_sum += log_gt
                    mle_count += count
        
        cur_pred = np.concatenate(cur_pred, axis=0)
        cur_gt = np.concatenate(cur_gt, axis=0)
        cur_coords = np.concatenate(cur_coords, axis=0)
        cur_coords_original = np.concatenate(cur_coords_original, axis=0)
        cur_barcodes_flat = [bc for chunk in cur_barcodes for bc in chunk]

        cur_res_dict = metric_func(
            cur_pred,
            cur_gt,
            loader.dataset.gene_list,
            spot_metrics=getattr(args, "spot_metrics", False),
        )
        cur_res_dict.update({'n_test': len(cur_gt)})
        res_dict[loader.dataset.name] = cur_res_dict
        if phi_module is not None and args.prior_sampler == "learned_zinb" and mle_count > 0:
            res_dict[loader.dataset.name]['mle_pred'] = mle_pred_sum / mle_count
            res_dict[loader.dataset.name]['mle_ground_truth'] = mle_gt_sum / mle_count
            total_mle_pred += mle_pred_sum
            total_mle_gt += mle_gt_sum
            total_mle_count += mle_count

        if save_per_slide_h5ad and output_dir is not None:
            save_slide_h5ad(
                predictions=cur_pred,
                ground_truth=cur_gt,
                coords_original=cur_coords_original,
                gene_names=loader.dataset.gene_list,
                slide_name=slide_name,
                output_dir=output_dir,
                source_h5ad=getattr(loader.dataset, 'h5ad_path', None),
                barcodes=cur_barcodes_flat if cur_barcodes_flat else None,
            )

        all_pred.append(cur_pred)
        all_gt.append(cur_gt)
    
    all_pred = np.concatenate(all_pred, axis=0)
    all_gt = np.concatenate(all_gt, axis=0)
    cur_res_dict = metric_func(
        all_pred,
        all_gt,
        loader_list[0].dataset.gene_list,
        spot_metrics=getattr(args, "spot_metrics", False),
    )
    cur_res_dict.update({'n_test': len(all_gt)})
    res_dict["all"] = cur_res_dict
    if phi_module is not None and args.prior_sampler == "learned_zinb" and total_mle_count > 0:
        res_dict["all"]["mle_pred"] = total_mle_pred / total_mle_count
        res_dict["all"]["mle_ground_truth"] = total_mle_gt / total_mle_count
    if return_all:
        return res_dict, {'preds_all': all_pred, 'targets_all': all_gt}
    return res_dict


def save_slide_h5ad(
    predictions: np.ndarray,
    ground_truth: np.ndarray,
    coords_original: np.ndarray,
    gene_names: list[str],
    slide_name: str,
    output_dir: str | Path,
    source_h5ad: str | None = None,
    barcodes: list[str] | None = None,
):
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    adata = sc.AnnData(X=predictions)
    adata.var_names = gene_names

    n_spots = predictions.shape[0]
    if barcodes is not None and len(barcodes) == n_spots:
        adata.obs['spot_id'] = np.array(barcodes)
    else:
        adata.obs['spot_id'] = [f"spot_{i}" for i in range(n_spots)]
    adata.obs['sample'] = slide_name

    adata.layers['ground_truth'] = ground_truth
    adata.layers['prediction'] = predictions

    adata.obsm['spatial'] = coords_original[:, :2]
    adata.obs['z_index'] = coords_original[:, 2]

    if source_h5ad is not None:
        try:
            base = sc.read_h5ad(source_h5ad)
            if barcodes is not None:
                barcode_set = set(barcodes)
                keep = [bc in barcode_set for bc in base.obs_names]
                if any(keep):
                    base = base[keep]
            if 'spatial' in base.obsm:
                adata.obsm['spatial'] = np.asarray(base.obsm['spatial'])[:, :2]
            if 'spatial' in base.uns:
                adata.uns['spatial'] = {}
                base_spatial = base.uns['spatial']
                if isinstance(base_spatial, dict) and len(base_spatial) >= 1:
                    first_key = next(iter(base_spatial.keys()))
                    lib_block = base_spatial[first_key]
                    images = lib_block.get('images', {})
                    scalefactors = lib_block.get('scalefactors', {})
                    images_np = {}
                    for key, value in images.items():
                        try:
                            images_np[key] = np.asarray(value)
                        except Exception:
                            pass
                    adata.uns['spatial'][slide_name] = {
                        'images': images_np,
                        'scalefactors': scalefactors,
                    }
            adata.obs['library_id'] = slide_name
        except Exception as exc:
            print(f"[WARN] Could not attach spatial data from {source_h5ad}: {exc}")

    out_file = output_path / f"{slide_name}.h5ad"
    adata.write_h5ad(out_file)
    print(f"Saved per-slide predictions to: {out_file}")


def _zinb_log_likelihood(mu: torch.Tensor, theta: torch.Tensor, pi: torch.Tensor,
                         values: torch.Tensor, mask: torch.Tensor | None) -> tuple[float, int]:
    eps = 1e-6
    theta_exp = theta.view(1, 1, -1).expand_as(mu)
    nb_logits = torch.log(theta_exp + eps) - torch.log(mu + eps)
    zi_logits = torch.log(pi + eps) - torch.log1p(-pi + eps)
    dist = ZeroInflatedNegativeBinomial(total_count=theta_exp, logits=nb_logits, zi_logits=zi_logits)
    values = values.clamp_min(0.0)
    log_prob = dist.log_prob(values)
    if mask is not None:
        valid = mask.unsqueeze(-1)
        log_prob = log_prob.masked_fill(~valid, 0.0)
        count = int(mask.sum().item() * mu.shape[-1])
    else:
        count = int(log_prob.numel())
    return float(log_prob.sum().item()), count
