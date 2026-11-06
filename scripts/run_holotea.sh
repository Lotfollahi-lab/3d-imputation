#!/bin/bash
# HoloTea end to end: learned prior -> training -> imputation of the held-out sections.
# Edit the paths below, then run from the repository root:  bash scripts/run_holotea.sh
set -euo pipefail
export PYTHONPATH="$(pwd)${PYTHONPATH:+:$PYTHONPATH}"

DATA_ROOT=/path/to/data          # contains $DATASET/adata/, $DATASET/splits/, $DATASET/$GENES
EMBED_ROOT=/path/to/embeddings   # contains $DATASET/uni_v1_official/fp32/<sample_id>.h5
DATASET=EMBRYO
GENES=var_150genes.json
N_GENES=150
SPLIT=0                          # uses splits/train_$SPLIT.csv and splits/test_$SPLIT.csv
OUT=/path/to/results
SEED=1

# 1) Learned ZINB prior, trained on the profiled (training) sections only.
python stflow/scripts/train_phi_e.py \
  --adata_dir "$DATA_ROOT/$DATASET/adata" \
  --train_ids_csv "$DATA_ROOT/$DATASET/splits/train_$SPLIT.csv" \
  --gene_list "$DATA_ROOT/$DATASET/$GENES" \
  --save_dir "$OUT/prior" \
  --epochs 50 --batch_size 4 --lr 1e-3 --seed 42 --device 0

# 2) HoloTea training (paper configuration).
COMMON=(
  --datasets "$DATASET" --source_dataroot "$DATA_ROOT" --embed_dataroot "$EMBED_ROOT"
  --gene_list "$GENES" --n_genes "$N_GENES" --feature_encoder uni_v1_official
  --split-indices "$SPLIT" --seed "$SEED" --spot-metrics
  --prior_sampler learned_zinb --phi_e_ckpt "$OUT/prior/phi_e.pt"
  --enable_crossslide_condition --crossslide_warmup_epochs 10
  --enable_controlnet --controlnet_warmup_epochs 10
  --use_inducing --n_inducing 64 --use_hf_branch
  --n_layers 4 --n_sample_steps 5
)
python stflow/app/flow/train.py "${COMMON[@]}" \
  --batch_size 2 --lr 5e-4 --epochs 100 --early_stop_patience 0 \
  --save_dir "$OUT" --exp_code holotea

# 3) Impute the sections listed in splits/test_$SPLIT.csv with the last checkpoint.
python stflow/app/flow/train.py "${COMMON[@]}" \
  --eval_only --eval_checkpoint "$OUT/holotea/$DATASET/split$SPLIT/checkpoints/last.pth" \
  --save_per_slide_h5ad --save_dir "$OUT" --exp_code holotea_impute

echo "Metrics:         $OUT/holotea_impute/$DATASET/split$SPLIT/all_results.json"
echo "Imputed sections: $OUT/holotea_impute/$DATASET/split$SPLIT/slides/"
