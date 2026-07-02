# PSSD

PSSD: **P**rogressive **S**patial-**S**emantic **D**ecoupling for Flow-Based Gene Expression Prediction from Histology Images.

This repository contains the official implementation of PSSD, together with the notebooks used for data preprocessing and evaluation.

## Project Structure

```
PSSD/
├── models_fm_gene.py           # Flow-matching models (PSSDFMProgressive, PSSDFMDecoupled, PSSDFM)
├── models_fm.py                # Base flow-matching model
├── models.py                   # Diffusion-based baseline model
├── flow_matching.py            # Flow matching & ODE solver
├── graph_proj.py               # Gene graph projection with GO ontology
├── diffusion/                  # Gaussian diffusion utilities
├── train_helper.py             # Training utilities
├── pssd_train_flow_gene.py     # Training script (gene-aware flow matching)
├── pssd_sample_flow_gene.py    # Sampling / inference script
├── dataset_preprocess.ipynb    # Notebook: build UNI/CONCH embeddings + gene lists
├── eval.ipynb                  # Notebook: PCC and downstream visualisation
├── environment.yml             # Conda environment (name: stem_env)
├── LICENSE
└── README.md
```

All scripts assume they are invoked from the **parent directory of `PSSD/`** (so that `from PSSD.xxx import ...` resolves correctly).

## 1. Installation

```bash
# Clone the repository
git clone https://github.com/ChyaZhang/PSSD.git
cd PSSD

# Create the conda environment
conda env create -f environment.yml
conda activate stem_env
```

External components required by `dataset_preprocess.ipynb` for building histology embeddings:

- **UNI** foundation model: https://github.com/mahmoodlab/UNI
- **CONCH** foundation model: https://github.com/mahmoodlab/CONCH

Optional foundation models used in the comparisons (referenced only in `dataset_preprocess.ipynb`):

- **H-optimus-0**: https://huggingface.co/bioptimus/H-optimus-0
- **Virchow2**: https://huggingface.co/paige-ai/Virchow2

## 2. Data Preparation

### 2.1 Download raw spatial transcriptomics data

We use the **HEST-1k** benchmark. Follow the official HEST-1k download workflow: https://github.com/mahmoodlab/HEST

Each dataset used in the paper corresponds to a subset of HEST-1k slides:

| Dataset         | HEST-1k slide IDs                             |
| --------------- | --------------------------------------------- |
| DLPFC           | `MISC1` – `MISC12`                            |
| BC (her2st)     | 12 breast-cancer slides                       |
| cSCC            | 12 cutaneous SCC slides                       |
| ccRCC           | `INT13` – `INT24`                             |
| PAAD            | `NCBI569`, `NCBI570`, `NCBI571`, `NCBI572`    |
| COAD            | `MISC62` – `MISC73`                           |
| COAD (Visium HD)| `TENX154`, `TENX155`, `TENX156`               |

Place each dataset under `./hest1k_datasets/<DATASET>/` following the HEST layout:

```
./hest1k_datasets/<DATASET>/
├── st/                          # <slide>.h5ad from HEST-1k
├── wsis/                        # Whole-slide images
└── processed_data/              # Produced by dataset_preprocess.ipynb (see 2.2)
```

You also need the two auxiliary files shared across datasets:

- Gene2Vec pretrained embeddings: `./hest1k_datasets/gene2vec_dim_200_iter_9_w2v.txt`  
  (available at https://github.com/jingcheng-du/Gene2vec)
- Gene Ontology OBO file: `./hest1k_datasets/go-basic.obo`  
  (available at https://geneontology.org/docs/download-ontology/)

### 2.2 Preprocess with `dataset_preprocess.ipynb`

Open and run `dataset_preprocess.ipynb` end-to-end. For each dataset it will:

1. Compute per-spot patch embeddings with UNI (`1spot_uni_ebd/`) and, optionally, CONCH (`1spot_conch_ebd/`) with augmented variants.
2. Select the top 50 Highly Variable Genes (HVGs) and top 50 Highly Expressed Genes (HEGs) across all slides.
3. Write outputs into `./hest1k_datasets/<DATASET>/processed_data/`:

```
processed_data/
├── 1spot_uni_ebd/            # <slide>_uni.pt   (N_spots, 1024)
├── 1spot_uni_ebd_aug/        # Augmented UNI embeddings
├── 1spot_conch_ebd/          # <slide>_conch.pt (optional)
├── 1spot_conch_ebd_aug/
├── all_slide_lst.txt         # One slide ID per line
├── selected_gene_list_hvg.txt
└── selected_gene_list_heg.txt
```

Update the `data_path` variable inside the notebook to point to `./hest1k_datasets/<DATASET>/`.

## 3. Directory Structure at Run Time

```
<workspace>/
├── PSSD/                             # This repository
├── hest1k_datasets/
│   ├── gene2vec_dim_200_iter_9_w2v.txt
│   ├── go-basic.obo
│   ├── DLPFC/
│   ├── BC/
│   ├── CSCC/
│   ├── ccRCC/
│   ├── PAAD/
│   ├── COAD/
│   └── COAD-HD/
└── results/                          # Training / sampling outputs (created automatically)
    └── <expr_name>_results/
        └── runs/
            └── 000/
                ├── checkpoints/
                └── samples/
```

## 4. Training

Training uses leave-one-out cross-validation: `--slide_out` selects the held-out test slide, and all remaining slides are used for training.

```bash
torchrun --nnodes=1 --nproc_per_node=1 PSSD/pssd_train_flow_gene.py \
    --expr_name DLPFC_PSSD_hvg \
    --data_path ./hest1k_datasets/DLPFC/ \
    --results_dir ./results/DLPFC_PSSD_hvg_results/runs/ \
    --slide_out MISC1 \
    --folder_list_filename all_slide_lst.txt \
    --gene_list_filename selected_gene_list_hvg.txt \
    --gene_embedding_file ./hest1k_datasets/gene2vec_dim_200_iter_9_w2v.txt \
    --go_obo_path ./hest1k_datasets/go-basic.obo \
    --model PSSDFMProgressive \
    --DiT_num_blocks 12 --hidden_size 384 --num_heads 6 \
    --lr 1e-4 --global_batch_size 128 \
    --total_epochs 40000 --ckpt_every 50000
```

Key arguments:

| Argument                | Description                                                                   |
| ----------------------- | ----------------------------------------------------------------------------- |
| `--expr_name`           | Experiment tag; controls the run subdirectory name.                           |
| `--data_path`           | Root of one HEST-1k dataset (contains `st/` and `processed_data/`).           |
| `--results_dir`         | Where checkpoints and logs are written.                                       |
| `--slide_out`           | Held-out test slide ID. Comma-separate multiple slides for multi-holdout.     |
| `--gene_list_filename`  | `selected_gene_list_hvg.txt` or `selected_gene_list_heg.txt`.                 |
| `--model`               | `PSSDFMProgressive` (default), `PSSDFMDecoupled`, or `PSSDFM`.                |
| `--total_epochs`        | Total training iterations.                                                    |
| `--ckpt_every`          | Save a checkpoint every N iterations.                                         |
| `--resume`              | Resume training: `auto`, a path to a `.pt` file, or an experiment directory.  |

To reproduce leave-one-out benchmarks, run one training job per slide with different `--slide_out` (e.g. `MISC1` … `MISC12` for DLPFC).

## 5. Sampling / Inference

Once a checkpoint is available (`./results/<expr_name>_results/runs/000/checkpoints/0200000.pt` in the default layout), draw 20 samples per spot for the held-out slide:

```bash
python PSSD/pssd_sample_flow_gene.py \
    --data_path ./hest1k_datasets/DLPFC/ \
    --ckpt ./results/DLPFC_PSSD_hvg_results/runs/000/checkpoints/0200000.pt \
    --save_path ./results/DLPFC_PSSD_hvg_results/runs/000/samples/ \
    --slide_out MISC1 \
    --gene_list_filename selected_gene_list_hvg.txt \
    --gene_embedding_file ./hest1k_datasets/gene2vec_dim_200_iter_9_w2v.txt \
    --go_obo_path ./hest1k_datasets/go-basic.obo \
    --model PSSDFMProgressive \
    --sample_num_per_cond 20 --num_sampling_steps 100 \
    --ode_method heun --sampling_batch_size 2000 \
    --device cuda:0
```

The output tensor `generated_samples_<step>_20sample.pt` has shape `(N_spots * 20, 1, N_genes)`. Reshape as `(N_spots, 20, N_genes)` and average over the 20-sample axis to obtain the final predicted expression matrix.

## 6. Evaluation

Open `eval.ipynb` for the reference evaluation pipeline. Point the notebook to your dataset and prediction paths (e.g. `./hest1k_datasets/CSCC/` and `./results/CSCC_PSSD_hvg_results/runs/011/samples/generated_samples_0200000_20sample.pt`) and run the cells end-to-end. It computes:

- Pearson correlation coefficient (PCC) between predicted and measured expression, averaged over the top HVGs / HEGs;
- Gene-variation curves and marker-gene spatial visualisations for the test slide.

## Acknowledgement

Built upon [Stem](https://github.com/SichenZhu/Stem) (ICLR 2025). We also thank the authors of [HEST-1k](https://github.com/mahmoodlab/HEST), [UNI](https://github.com/mahmoodlab/UNI), and [CONCH](https://github.com/mahmoodlab/CONCH) for the datasets and pretrained models used in this work.
