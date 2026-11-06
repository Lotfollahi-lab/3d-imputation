# HoloTea

**3D-Guided Scalable Flow Matching for Generating Volumetric Tissue Spatial Transcriptomics from Serial Histology** (MLCB 2026)

In a stack of serial sections, H&E images are usually available for every section, but spatial transcriptomics (ST) only for some of them. HoloTea imputes spot-level gene expression for the sections without ST, from their H&E images and the ST of the profiled sections next to them. It is a flow-matching model built on [STFlow](https://github.com/Graph-and-Geometric-Learning/STFlow). On top of STFlow it adds:

| Component | What it does | Flag |
|---|---|---|
| Cross-slide branches | Condition each spot on the expression of the nearest spot, by (a, b), on the closest profiled section above (z+1) and below (z−1) | `--enable_crossslide_condition` |
| x_t ControlNet branch | ControlNet-style branch on the current state x_t of the flow | `--enable_controlnet` |
| Global set attention (GSA) | Slide-wide context through m inducing tokens, O(N·m) | `--use_inducing --n_inducing 64` |
| Learned ZINB prior ψ_η | Start distribution of the flow, predicted from the counts of the matched spot on the closest profiled section | `--prior_sampler learned_zinb --phi_e_ckpt …` |

## Installation

```bash
git clone https://github.com/Lotfollahi-lab/3d-imputation.git
cd 3d-imputation
python -m venv .venv && source .venv/bin/activate     # Python 3.10
pip install torch torchvision                          # build matching your CUDA version
pip install -r requirements.txt
export PYTHONPATH=$(pwd)                               # run all commands from the repository root
```

A GPU is needed for embedding the H&E tiles and for training.

## Data layout

```
<DATA_ROOT>/<DATASET>/
    adata/<sample_id>.h5ad        one file per section; raw counts in X, spot barcodes as obs_names
    splits/train_<k>.csv          column sample_id: profiled sections (used for training)
    splits/test_<k>.csv           column sample_id: sections to impute / evaluate
    var_150genes.json             {"genes": [...]}: the gene panel
<EMBED_ROOT>/<DATASET>/uni_v1_official/fp32/<sample_id>.h5
                                  embeddings [N, D], coords [N, 3] = (a, b, z), barcodes [N]
```

The sections are ordered along the stack by the z value in `coords`. The sections in `test_<k>.csv` are imputed from the profiled sections in `train_<k>.csv`. Their ST is used only to compute metrics. To impute a section that has no ST, give it an `.h5ad` with its spot barcodes and zero counts, and ignore its metrics.

## 1. Preprocess and embed

`stflow/preprocess_and_embed_custom.py` builds this layout from a single AnnData with all sections and their H&E images:

```bash
python stflow/preprocess_and_embed_custom.py \
    --adata all_sections.h5ad \
    --ome_tiff sections.ome.tif \
    --dataset_name embryo \
    --output_dir $DATA_ROOT/EMBRYO \
    --embed_dir $EMBED_ROOT \
    --encoder uni_v1_official --weights_root /path/to/weights \
    --n_genes 150 \
    --test_samples 1,3,5 \
    --coord_scale 1.0
```

- **Input AnnData:** raw counts in `X`, the section ID in `obs['sample']`, and spot centres in `obsm['spatial']`. `--coord_scale` multiplies these to give pixel coordinates in the image.
- **Section order:** the sorted section IDs define the order along z (section i gets z = i), so name the sections so that they sort in stack order.
- **Images:** pass either an OME-TIFF with 3 pages (RGB) per section, in the same sorted order, or `--rgb_image` for a single section.
- **Encoder weights:** the [UNI](https://huggingface.co/MahmoodLab/UNI) weights are expected at `<weights_root>/uni/pytorch_model.bin`.
- **Test sections:** `--test_samples` takes 0-based positions in the sorted list.
- **Gene panel:** with `--n_genes`, the panel is selected automatically (see `--gene_selection`). To use your own panel, pass a JSON with `--gene_list_override`.

`stflow/preprocess_and_embed_st_raw.py` does the same for the ST-Data format (HE_*.jpg, *_stdata.tsv.gz, spots_*.csv.gz).

## 2. Train the learned prior ψ_η

Use the training sections only:

```bash
python stflow/scripts/train_phi_e.py \
    --adata_dir $DATA_ROOT/EMBRYO/adata \
    --train_ids_csv $DATA_ROOT/EMBRYO/splits/train_0.csv \
    --gene_list $DATA_ROOT/EMBRYO/var_150genes.json \
    --save_dir $OUT/prior \
    --epochs 50 --batch_size 4 --lr 1e-3 --seed 42 --device 0
```

This writes `$OUT/prior/phi_e.pt`.

## 3. Train HoloTea

```bash
python stflow/app/flow/train.py \
    --datasets EMBRYO --source_dataroot $DATA_ROOT --embed_dataroot $EMBED_ROOT \
    --gene_list var_150genes.json --n_genes 150 --feature_encoder uni_v1_official \
    --split-indices 0 --seed 1 --spot-metrics \
    --prior_sampler learned_zinb --phi_e_ckpt $OUT/prior/phi_e.pt \
    --enable_crossslide_condition --crossslide_warmup_epochs 10 \
    --enable_controlnet --controlnet_warmup_epochs 10 \
    --use_inducing --n_inducing 64 --use_hf_branch \
    --n_layers 4 --n_sample_steps 5 \
    --batch_size 2 --lr 5e-4 --epochs 100 --early_stop_patience 0 \
    --save_dir $OUT --exp_code holotea
```

Checkpoints are written to `$OUT/holotea/EMBRYO/split0/checkpoints/`:
- `last.pth` is the final epoch.
- `best.pth` is the epoch with the highest PCC on the sections in `test_0.csv`, which are evaluated after every epoch.

To select a checkpoint without looking at the test sections, put validation sections in `test_0.csv` for training. Then impute the test sections from a second split, `train_1.csv` (same training sections) and `test_1.csv`, using `--split-indices 1` in step 4.

## 4. Impute held-out sections

Use the model flags of step 3 with `--eval_only`:

```bash
python stflow/app/flow/train.py \
    --datasets EMBRYO --source_dataroot $DATA_ROOT --embed_dataroot $EMBED_ROOT \
    --gene_list var_150genes.json --n_genes 150 --feature_encoder uni_v1_official \
    --split-indices 0 --seed 1 --spot-metrics \
    --prior_sampler learned_zinb --phi_e_ckpt $OUT/prior/phi_e.pt \
    --enable_crossslide_condition --enable_controlnet \
    --use_inducing --n_inducing 64 --use_hf_branch \
    --n_layers 4 --n_sample_steps 5 \
    --eval_only --eval_checkpoint $OUT/holotea/EMBRYO/split0/checkpoints/last.pth \
    --save_per_slide_h5ad --save_dir $OUT --exp_code holotea_impute
```

Outputs in `$OUT/holotea_impute/EMBRYO/split0/`:
- `all_results.json` and `<sample_id>_results.json`. `pearson_mean` is the mean per-gene PCC and `l2_error_q2` the median per-gene MSE; keys ending in `_spot` are the per-spot metrics.
- `slides/<sample_id>.h5ad`: the imputed expression in `X` (log1p), the measured expression in `layers['ground_truth']`, and the coordinates in `obsm['spatial']`.

`scripts/run_holotea.sh` runs steps 2–4 in one go.

## Baseline and ablations

- **STFlow:** `--prior_sampler zinb`, without the cross-slide, ControlNet and GSA flags.
- **Ablations:** remove the flag of the component you want to drop (see the table above).

## Citation

```bibtex
@inproceedings{sanian2026holotea,
  title     = {3D-Guided Scalable Flow Matching for Generating Volumetric Tissue Spatial Transcriptomics from Serial Histology},
  author    = {Sanian, Mohammad Vali and Hemmat, Arshia and Vahidi, Amirhossein and Maaskola, Jonas and Lee, Jimmy Tsz Hang and Prete, Martin and Makarchuk, Stanislaw and Demirci, Yeliz and Chipampe, Nana-Jane and Haniffa, Muzlifah and Bayraktar, Omer and Paavolainen, Lassi and Lotfollahi, Mohammad},
  booktitle = {Machine Learning in Computational Biology (MLCB)},
  year      = {2026},
  note      = {arXiv:2511.14613}
}
```

## Acknowledgements

The flow-matching backbone, data loading and encoder utilities are adapted from [STFlow](https://github.com/Graph-and-Geometric-Learning/STFlow) (Huang et al., 2025) and [HEST](https://github.com/mahmoodlab/HEST).
