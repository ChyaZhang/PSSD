# PSSD

Diffusion Generative Modeling for Spatially Resolved Gene Expression Inference from Histology Images.

## Project Structure

```
PSSD/
├── models_fm_gene.py          # Flow matching models (PSSDFMProgressive, PSSDFMDecoupled, PSSDFM)
├── models_fm.py               # Base flow matching model
├── models.py                  # Diffusion-based model
├── flow_matching.py           # Flow matching & ODE solver
├── graph_proj.py              # Gene graph projection with GO ontology
├── diffusion/                 # Gaussian diffusion utilities
├── train_helper.py            # Training utilities
├── pssd_train_flow_gene.py    # Training script (gene-aware flow matching)
└── pssd_sample_flow_gene.py   # Sampling script (gene-aware flow matching)
```

## Training

```bash
torchrun --nnodes=1 --nproc_per_node=1 pssd_train_flow_gene.py \
  --expr_name $DATASETNAME \
  --data_path ./hest1k_datasets/$DATASETNAME/ \
  --results_dir ./results/ \
  --model PSSDFMProgressive
```

## Sampling

```bash
python pssd_sample_flow_gene.py \
  --ckpt /path/to/checkpoint.pt \
  --data_path ./hest1k_datasets/$DATASETNAME/ \
  --save_path ./results/samples/ \
  --model PSSDFMProgressive
```

## Acknowledgement

Built upon [Stem](https://github.com/SichenZhu/Stem) (ICLR 2025).
