import torch
# the first flag below was False when we tested this script but True makes A100 training a lot faster:
torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler
from torch.distributed import init_process_group, destroy_process_group
from torchvision.datasets import ImageFolder
from torchvision import transforms
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

import numpy as np
from collections import OrderedDict
from PIL import Image
from copy import deepcopy
from glob import glob
import time as tm
import argparse
import logging
import os
import pandas as pd
import json
import matplotlib.pyplot as plt
from tqdm import tqdm
import random
import anndata
import scprep as scp
import scanpy as sc

import sys
sys.path.append("./PSSD")
from PSSD.models_fm_gene import PSSDFM_models
from PSSD.train_helper import *


from PSSD.flow_matching import FlowMatcher 
from PSSD.train_helper import *

class CustomDataset(Dataset):
    def __init__(self, x, y, coordinates):
        self.data = x
        self.label = y
        self.coordinates = coordinates

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx], self.label[idx], self.coordinates[idx]

def load_gene_embeddings(gene_embedding_file):
    gene_embeddings = {}  
    with open(gene_embedding_file, 'r') as f:
        next(f)
        for line in f:
            parts = line.strip().split()
            gene_name = parts[0]
            embedding = [float(x) for x in parts[1:]]
            gene_embeddings[gene_name] = np.array(embedding) 
    return gene_embeddings

def create_gene_embedding_matrix(selected_genes, gene_embeddings_dict, embedding_dim):
    embedding_matrix = []
    valid_genes = []
    missing_genes = []
    
    for gene in selected_genes:
        if gene in gene_embeddings_dict:
            embedding_matrix.append(gene_embeddings_dict[gene])
            valid_genes.append(gene)
        else:
            missing_genes.append(gene)
    
    if len(embedding_matrix) == 0:
        raise ValueError("No genes found in the embedding file! Please check your gene list and embedding file.")
    
    embedding_matrix = torch.from_numpy(np.array(embedding_matrix)).float()
    return embedding_matrix, valid_genes, missing_genes

class Trainer:
    def __init__(
        self,
        model: torch.nn.Module,
        train_data: DataLoader,
        rank: int,
        gpu_id: int,
        model_args: argparse.Namespace,
        resume_checkpoint: str = None,
    ) -> None:
        self.rank = rank
        self.gpu_id = gpu_id
        self.train_data = train_data
        self.args = model_args
        
        self.model = model
        self.ema = deepcopy(model).cpu().to(gpu_id)
        requires_grad(self.ema, False)
        self.model = DDP(self.model.to(gpu_id), device_ids=[self.gpu_id])
        
        self.flow_matcher = FlowMatcher(sigma=self.args.fm_sigma, path_type=self.args.fm_path_type)

        self.optimizer = torch.optim.AdamW(self.model.parameters(), 
                                           lr=self.args.lr, weight_decay=0)
        update_ema(self.ema, self.model.module, decay=0)
        self.args.logger.info(f"Rank {rank} - Initializing Trainer... PSSD Parameters: {sum(p.numel() for p in model.parameters()):,}")

        self.train_steps=0
        self.start_epoch = 0
        self.current_epoch = 0
        self.log_steps=0
        self.running_loss=0
        self.best_loss = float('inf')

        # Resume from checkpoint if provided
        if resume_checkpoint and os.path.exists(resume_checkpoint):
            self.resume_from_checkpoint(resume_checkpoint)
        else:
            # Initialize EMA from scratch
            update_ema(self.ema, self.model.module, decay=0)
            if resume_checkpoint:
                self.args.logger.warning(f"Checkpoint {resume_checkpoint} not found, starting from scratch")
        
        self.args.logger.info(
            f"Rank {rank} - Trainer initialized. "
            f"Starting from step {self.train_steps}, epoch {self.start_epoch}"
        )
        self.args.logger.info(
            f"Model Parameters: {sum(p.numel() for p in model.parameters()):,}"
        )

    def resume_from_checkpoint(self, checkpoint_path: str):
        """
        Resume training from a checkpoint
        """
        self.args.logger.info(f"Resuming from checkpoint: {checkpoint_path}")
        
        checkpoint = torch.load(checkpoint_path, map_location=f'cuda:{self.gpu_id}', weights_only=False)
        
        # Load model state
        self.model.module.load_state_dict(checkpoint['model'])
        
        # Load EMA state
        if 'ema' in checkpoint:
            self.ema.load_state_dict(checkpoint['ema'])
        
        # Load optimizer state
        if 'opt' in checkpoint:
            self.optimizer.load_state_dict(checkpoint['opt'])
        
        # Load training state
        if 'train_state' in checkpoint:
            train_state = checkpoint['train_state']
            self.train_steps = train_state.get('train_steps', 0)
            self.start_epoch = train_state.get('epoch', 0)
            self.best_loss = train_state.get('best_loss', float('inf'))
            self.running_loss = train_state.get('running_loss', 0)
            self.log_steps = train_state.get('log_steps', 0)
        else:
            # For backward compatibility with old checkpoints
            self.train_steps = checkpoint.get('step', 0)
            self.start_epoch = checkpoint.get('epoch', 0)
        
        self.args.logger.info(
            f"Resumed from step {self.train_steps}, epoch {self.start_epoch}, "
            f"best_loss {self.best_loss:.5f}"
        )

    def _save_checkpoint(self, epoch: int = None, is_best: bool = False):
        """
        Save checkpoint with training state
        """
        if epoch is None:
            epoch = self.current_epoch if hasattr(self, 'current_epoch') else self.start_epoch
            
        # Prepare checkpoint data
        checkpoint = {
            "model": self.model.module.state_dict(),
            "ema": self.ema.state_dict(),
            "opt": self.optimizer.state_dict(),
            "train_state": {
                "train_steps": self.train_steps,
                "epoch": epoch,
                "best_loss": self.best_loss,
                "running_loss": self.running_loss,
                "log_steps": self.log_steps,
            },
            "args": vars(self.args),  # Save training arguments
        }
        
        if not is_best:
        # Save regular checkpoint
            checkpoint_path = f"{self.args.checkpoint_dir}/{self.train_steps:07d}.pt"
            torch.save(checkpoint, checkpoint_path)
            self.args.logger.info(f"Saved checkpoint to {checkpoint_path}")
        
        # Save as latest checkpoint (for easy resume)
        latest_path = f"{self.args.checkpoint_dir}/latest.pt"
        torch.save(checkpoint, latest_path)
        
        # Save best checkpoint if applicable
        if is_best:
            best_path = f"{self.args.checkpoint_dir}/best.pt"
            torch.save(checkpoint, best_path)
            self.args.logger.info(f"Saved best checkpoint to {best_path}")
        
        # Save training state to JSON for easy inspection
        state_path = f"{self.args.checkpoint_dir}/training_state.json"
        with open(state_path, 'w') as f:
            json.dump({
                "train_steps": self.train_steps,
                "epoch": epoch if epoch is not None else self.start_epoch,
                "best_loss": self.best_loss,
                "timestamp": tm.strftime("%Y-%m-%d_%H:%M:%S"),
            }, f, indent=2)

    def _run_batch(self, x1, y, coords=None):
        """
        x1: real data (gene expression)
        y: condition (image embedding)
        coords: spatial coordinates
        """

        xt, t, ut = self.flow_matcher.get_train_tuple(x1=x1)
        t_discrete = t * 999 
        model_kwargs = dict(y=y)

        # v_pred = self.model(xt, t_discrete, coordinates=coords, **model_kwargs)
        if coords is not None:
            output = self.model(xt, t_discrete, coordinates=coords, **model_kwargs)
        else:
            output = self.model(xt, t_discrete, **model_kwargs)

        loss = F.mse_loss(output, ut) 

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        update_ema(self.ema, self.model.module)

        self.running_loss += loss.item()
        self.train_steps += 1
        self.log_steps += 1

        if self.log_steps % 500 == 0:
            torch.cuda.synchronize()
            avg_loss = torch.tensor(self.running_loss / self.log_steps, device=xt.device)
            dist.all_reduce(avg_loss, op=dist.ReduceOp.SUM)
            avg_loss = avg_loss.item() / dist.get_world_size()

            is_best = avg_loss < self.best_loss
            if is_best:
                self.best_loss = avg_loss

            self.args.logger.info(
                f"Step={self.train_steps:07d} | "
                f"Training Loss: {avg_loss:.5f} | "
                f"Best Loss: {self.best_loss:.5f}"
            )

            if is_best and self.rank == 0:
                self._save_checkpoint(epoch= self.current_epoch, is_best=True)

            self.running_loss = 0
            self.log_steps = 0

        if self.train_steps % self.args.ckpt_every == 0 and self.train_steps > 0:
            if self.rank == 0:
                self._save_checkpoint(epoch= self.current_epoch, is_best=False)
            dist.barrier()    

    def _run_epoch(self, epoch):
        b_sz = len(next(iter(self.train_data))[0])
        print(f"[GPU{self.gpu_id}] Epoch {epoch} | Batchsize: {b_sz} | Steps: {len(self.train_data)}")
        self.train_data.sampler.set_epoch(epoch)

        for x, y, coords in self.train_data:
            x = x.unsqueeze(1).to(self.gpu_id)  # (N, 1, NumGene)
            y = y.to(self.gpu_id)               # (N, NumEmbed)
            coords = coords.to(self.gpu_id)     # (N, 2)
            self._run_batch(x, y, coords)

    def train(self, max_epochs: int):
        ##
        self.model.train()
        self.ema.eval()
        ##
        for epoch in range(self.start_epoch, max_epochs):
            self.current_epoch = epoch
            self._run_epoch(epoch)
        
        if self.rank == 0:
            self.args.logger.info("Training completed!")
            self._save_checkpoint(epoch=max_epochs, is_best=False)

def find_latest_checkpoint(checkpoint_dir: str):
    """
    Find the latest checkpoint in the directory
    """
    # First check for 'latest.pt'
    latest_path = os.path.join(checkpoint_dir, 'latest.pt')
    if os.path.exists(latest_path):
        return latest_path
    
    # Otherwise, find the checkpoint with the highest step number
    checkpoint_files = glob(os.path.join(checkpoint_dir, '[0-9]*.pt'))
    if checkpoint_files:
        # Sort by step number
        checkpoint_files.sort(key=lambda x: int(os.path.basename(x).split('.')[0]))
        return checkpoint_files[-1]
    
    return None


def assemble_dataset(input_args):
    # load & assemble data
    # leave the test slide out
    slidename_lst = list(np.genfromtxt(input_args.data_path + "processed_data/" + input_args.folder_list_filename, dtype=str))
    for slide_out in input_args.slide_out.split(","):
        slidename_lst.remove(slide_out)
        input_args.logger.info(f"{slide_out} is held out for testing.")
    input_args.logger.info(f"Remaining {len(slidename_lst)} slides: {slidename_lst}")

    # load selected gene list
    selected_genes = list(np.genfromtxt(input_args.data_path + "processed_data/" + input_args.gene_list_filename, dtype=str))
    input_args.logger.info(f"Original selected genes filename: {input_args.gene_list_filename} | len: {len(selected_genes)}")

    # load gene embeddings
    if input_args.gene_embedding_file:
        input_args.logger.info(f"Loading gene embeddings from {input_args.gene_embedding_file}")
        gene_embeddings_dict = load_gene_embeddings(input_args.gene_embedding_file)
        
        gene_embedding_matrix, valid_genes, missing_genes = create_gene_embedding_matrix(
            selected_genes, gene_embeddings_dict, input_args.gene_embedding_dim
        )
        
        selected_genes = valid_genes
        input_args.valid_genes = valid_genes
        input_args.logger.info(f"Valid genes with embeddings: {len(valid_genes)}")
        
        if missing_genes:
            input_args.logger.warning(f"Excluded {len(missing_genes)} genes without embeddings: {missing_genes[:100]}...")
            if len(missing_genes) > 100:
                input_args.logger.warning(f"... and {len(missing_genes) - 10} more genes")
        
        input_args.gene_embedding_matrix = gene_embedding_matrix
        input_args.logger.info(f"Gene embedding matrix shape: {gene_embedding_matrix.shape}")
    else:
        input_args.gene_embedding_matrix = None
        input_args.logger.info("No gene embedding file provided, using learnable embeddings")
    
    input_args.input_gene_size = len(selected_genes)
    input_args.logger.info(f"Final selected genes count: {input_args.input_gene_size}")

    # load original patches
    first_slide = True
    all_img_ebd_ori = None
    all_count_mtx_ori = None
    input_args.logger.info("Loading original data...")
    for sni in range(len(slidename_lst)):
        sample_name = slidename_lst[sni]
        test_adata = anndata.read_h5ad(input_args.data_path + "st/" + sample_name + ".h5ad")
        sc.pp.normalize_total(test_adata, target_sum=1e3)
        sc.pp.log1p(test_adata)
        test_count_mtx = pd.DataFrame(test_adata[:, selected_genes].X.toarray(), 
                                      columns=selected_genes, 
                                      index=[sample_name + "_" + str(i) for i in range(test_adata.shape[0])])
        
        if first_slide:
            all_count_mtx_ori = test_count_mtx
            img_ebd_uni   = torch.load(input_args.data_path + "processed_data/1spot_uni_ebd/"   + sample_name + "_uni.pt",   map_location="cpu")
            img_ebd_conch = torch.load(input_args.data_path + "processed_data/1spot_conch_ebd/" + sample_name + "_conch.pt", map_location="cpu")
            all_img_ebd_ori = torch.cat([img_ebd_uni, img_ebd_conch], axis=1)
            input_args.logger.info(f"{sample_name} loaded, count_mtx shape: {all_count_mtx_ori.shape}  | img ebd shape: {all_img_ebd_ori.shape}")
            first_slide = False
            continue
        
        img_ebd_uni   = torch.load(input_args.data_path + "processed_data/1spot_uni_ebd/"   + sample_name + "_uni.pt",   map_location="cpu")
        img_ebd_conch = torch.load(input_args.data_path + "processed_data/1spot_conch_ebd/" + sample_name + "_conch.pt", map_location="cpu")
        slide_img_ebd = torch.cat([img_ebd_uni, img_ebd_conch], axis=1)
        all_img_ebd_ori = torch.cat([all_img_ebd_ori, slide_img_ebd], axis=0)
        all_count_mtx_ori = np.concatenate((all_count_mtx_ori, test_count_mtx), axis=0)
        input_args.logger.info(f"{sample_name} loaded, count_mtx shape: {all_count_mtx_ori.shape} | img ebd shape: {all_img_ebd_ori.shape}")
    input_args.cond_size = all_img_ebd_ori.shape[1]
    
    # load augmented patches
    first_slide = True
    all_img_ebd_aug = None
    input_args.logger.info(f"Augmentation data loading...")
    for sni in range(len(slidename_lst)):
        sample_name = slidename_lst[sni]

        if first_slide:
            img_ebd_uni   = torch.load(input_args.data_path + "processed_data/1spot_uni_ebd_aug/"   + sample_name + "_uni_aug.pt",   map_location="cpu")
            img_ebd_conch = torch.load(input_args.data_path + "processed_data/1spot_conch_ebd_aug/" + sample_name + "_conch_aug.pt", map_location="cpu")
            all_img_ebd_aug = torch.cat([img_ebd_uni, img_ebd_conch], axis=-1)
            input_args.logger.info(f"With augmentation {sample_name} loaded, img_ebd_mtx shape: {all_img_ebd_aug.shape}, all_img_ebd shape: {all_img_ebd_aug.shape}")
            first_slide = False
            continue
        
        img_ebd_uni   = torch.load(input_args.data_path + "processed_data/1spot_uni_ebd_aug/"   + sample_name + "_uni_aug.pt",   map_location="cpu")
        img_ebd_conch = torch.load(input_args.data_path + "processed_data/1spot_conch_ebd_aug/" + sample_name + "_conch_aug.pt", map_location="cpu")
        slide_img_ebd = torch.cat([img_ebd_uni, img_ebd_conch], axis=-1)
        all_img_ebd_aug = torch.cat([all_img_ebd_aug, slide_img_ebd], axis=0)
        input_args.logger.info(f"With augmentation {sample_name} loaded, img_ebd_mtx shape: {slide_img_ebd.shape}, all_img_ebd shape: {all_img_ebd_aug.shape}")
     
    # randomly select augmented patches according to the input augmentation ratio (int)
    num_aug_ratio = input_args.num_aug_ratio
    all_count_mtx_aug = np.repeat(np.copy(all_count_mtx_ori), num_aug_ratio, axis=0)             # generate count matrix for all augmented patches
    selected_img_ebd_aug = torch.zeros((all_count_mtx_aug.shape[0], all_img_ebd_aug.shape[2]))
    for i in range(all_img_ebd_aug.shape[0]):                                                    # randomly select augmented patches
        selected_transpose_idx = np.random.choice(all_img_ebd_aug.shape[1], num_aug_ratio, replace=False)
        selected_img_ebd_aug[i*num_aug_ratio:(i+1)*num_aug_ratio, :] = all_img_ebd_aug[i, selected_transpose_idx, :]

    all_img_ebd = torch.cat([all_img_ebd_ori, selected_img_ebd_aug], axis=0)
    all_count_mtx = np.concatenate((all_count_mtx_ori, all_count_mtx_aug), axis=0)
    input_args.logger.info(f"{num_aug_ratio}:1 augmentation. CONCH+UNI. final count_mtx shape: {all_count_mtx.shape} | final img_ebd shape: {all_img_ebd.shape}")
    
    ################################################
    all_count_mtx_df = pd.DataFrame(all_count_mtx, columns=selected_genes, index=list(range(all_count_mtx.shape[0])))
    # remove the spot with all NAN/zero in count mtx
    all_count_mtx_all_nan_spot_index = all_count_mtx_df.index[all_count_mtx_df.isnull().all(axis=1)]
    all_count_mtx_all_zero_spot_index = all_count_mtx_df.index[all_count_mtx_df.sum(axis=1) == 0]
    input_args.logger.info(f"All NAN spot index: {all_count_mtx_all_nan_spot_index}")
    input_args.logger.info(f"All zero spot index: {all_count_mtx_all_zero_spot_index}")
    spot_idx_to_remove = list(set(all_count_mtx_all_nan_spot_index) | set(all_count_mtx_all_zero_spot_index))
    spot_idx_to_keep = list(set(all_count_mtx_df.index) - set(spot_idx_to_remove))
    all_count_mtx = all_count_mtx_df.loc[spot_idx_to_keep, :]
    all_img_ebd = all_img_ebd[spot_idx_to_keep, :]
    input_args.logger.info(f"After exclude rows with all nan/zeros: {all_count_mtx.shape}, {all_img_ebd.shape}")
    
    # only normalized by log2(+1)
    all_count_mtx_selected_genes = all_count_mtx.loc[:, selected_genes].copy()
    # all_count_mtx_selected_genes = np.log1p(all_count_mtx.loc[:, selected_genes]).copy()
    

    input_args.logger.info(f"Selected genes count matrix shape: {all_count_mtx_selected_genes.shape}" )
    all_img_ebd.requires_grad_(False)

    all_coordinates_ori = []
    for sni in range(len(slidename_lst)):
        sample_name = slidename_lst[sni]
        test_adata = anndata.read_h5ad(input_args.data_path + "st/" + sample_name + ".h5ad")
        # 获取空间坐标
        if 'spatial' in test_adata.obsm:
            coords = test_adata.obsm['spatial']
        else:
            # 如果没有spatial，尝试其他常见的坐标键
            coords = test_adata.obs[['x', 'y']].values if 'x' in test_adata.obs else np.random.randn(test_adata.shape[0], 2)
        all_coordinates_ori.append(coords)
    all_coordinates_ori = np.vstack(all_coordinates_ori)

    # 对增强数据复制坐标
    all_coordinates_aug = np.repeat(all_coordinates_ori, num_aug_ratio, axis=0)
    all_coordinates = np.vstack([all_coordinates_ori, all_coordinates_aug])
    # 过滤掉nan/zero spots后的坐标
    all_coordinates = all_coordinates[spot_idx_to_keep]
    # 归一化坐标到[-1, 1]
    coords_min = all_coordinates.min(axis=0)
    coords_max = all_coordinates.max(axis=0)
    all_coordinates = 2 * (all_coordinates - coords_min) / (coords_max - coords_min + 1e-8) - 1
    input_args.logger.info(f"Coordinates shape: {all_coordinates.shape}")

    alldataset = CustomDataset(torch.from_numpy(all_count_mtx_selected_genes.values).float(), all_img_ebd.float(), torch.from_numpy(all_coordinates).float())    
    return alldataset, input_args


def load_train_objs(args):
    train_set, args = assemble_dataset(args)

    gene_names = None
    if args.gene_embedding_matrix is not None:
        selected_genes = list(np.genfromtxt(args.data_path + "processed_data/" + args.gene_list_filename, dtype=str))
        
        if args.valid_genes:
            gene_names = args.valid_genes
        else:
            gene_names = selected_genes
        
        args.logger.info(f"Using {len(gene_names)} genes for graph construction")
    
    model = PSSDFM_models[args.model](
        input_size=args.input_gene_size,
        depth= args.DiT_num_blocks,
        hidden_size=args.hidden_size, 
        num_heads=args.num_heads, 
        label_size=args.cond_size,
        gene_embedding_matrix=args.gene_embedding_matrix,
        gene_names = gene_names,
        go_obo_path = args.go_obo_path,
    )
    args.logger.info(f"Dataset contains {len(train_set):,} images ({args.data_path})")
    return train_set, model, args


def prepare_dataloader(args, dataset: Dataset, batch_size: int):
    return DataLoader(
        dataset,
        batch_size=batch_size,
        pin_memory=False,
        shuffle=False,
        sampler=DistributedSampler(dataset,
                                   shuffle=True,
                                   seed=args.global_seed),
        num_workers=args.num_workers,
        drop_last=True,
    )

def main(world_size: int, available_gpus: list, input_args):
    # Set up DDP
    dist.init_process_group(backend="nccl", world_size=world_size)
    rank = dist.get_rank()
    device = available_gpus[rank]
    seed = input_args.global_seed * dist.get_world_size() + rank
    print("Rank: ", rank, " | Device: ", device, " | Seed: ", seed)

    # set random seed
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.cuda.set_device(device)

    # set up output folder and logger
    if rank == 0:
        print("Rank 0 mkdir & set up logger...")

        # Handle resume vs new experiment
        if input_args.resume:
            if input_args.resume == "auto":
                # Auto-find the latest experiment
                existing_runs = glob(f"{input_args.results_dir}/[0-9]*")
                if existing_runs:
                    existing_runs.sort(key=lambda x: int(os.path.basename(x)))
                    input_args.experiment_dir = existing_runs[-1]
                    input_args.checkpoint_dir = f"{input_args.experiment_dir}/checkpoints"
                    print(f"Auto-resuming from experiment: {input_args.experiment_dir}")
                else:
                    print("No existing experiments found, starting new one")
                    input_args.resume = None
            elif os.path.isdir(input_args.resume):
                # Resume from specific experiment directory
                input_args.experiment_dir = input_args.resume
                input_args.checkpoint_dir = f"{input_args.experiment_dir}/checkpoints"
            elif os.path.isfile(input_args.resume):
                # Resume from specific checkpoint file
                input_args.experiment_dir = os.path.dirname(os.path.dirname(input_args.resume))
                input_args.checkpoint_dir = os.path.dirname(input_args.resume)

        if not input_args.resume:
            # Create new experiment
            os.makedirs(input_args.results_dir, exist_ok=True)  # Make results folder (holds all experiment subfolders)
            experiment_index = len(glob(f"{input_args.results_dir}/*"))
            input_args.experiment_dir = f"{input_args.results_dir}/{experiment_index:03d}"  # Create an experiment folder
            input_args.checkpoint_dir = f"{input_args.experiment_dir}/checkpoints"  # Stores saved model checkpoints
            os.makedirs(input_args.checkpoint_dir, exist_ok=True)
            os.makedirs(f"{input_args.experiment_dir}/samples", exist_ok=True)      # Store sampling results

        input_args.logger = create_logger(input_args.experiment_dir)
        input_args.logger.info(f"Experiment directory created at {input_args.experiment_dir}")

        # Find checkpoint to resume from
        resume_checkpoint = None
        if input_args.resume:
            if os.path.isfile(input_args.resume):
                resume_checkpoint = input_args.resume
            else:
                resume_checkpoint = find_latest_checkpoint(input_args.checkpoint_dir)
            
            if resume_checkpoint:
                input_args.logger.info(f"Will resume from checkpoint: {resume_checkpoint}")
            else:
                input_args.logger.warning("No checkpoint found to resume from, starting fresh")

    else:
        input_args.logger=create_logger(None)
        resume_checkpoint = None
        if input_args.resume:
            # Non-rank-0 processes also need to know the checkpoint
            if input_args.resume == "auto":
                existing_runs = glob(f"{input_args.results_dir}/[0-9]*")
                if existing_runs:
                    existing_runs.sort(key=lambda x: int(os.path.basename(x)))
                    checkpoint_dir = f"{existing_runs[-1]}/checkpoints"
                    resume_checkpoint = find_latest_checkpoint(checkpoint_dir)
            elif os.path.isdir(input_args.resume):
                checkpoint_dir = f"{input_args.resume}/checkpoints"
                resume_checkpoint = find_latest_checkpoint(checkpoint_dir)
            elif os.path.isfile(input_args.resume):
                resume_checkpoint = input_args.resume

    input_args.logger.info(f"Rank: {rank} | Device: {device} | Seed: {seed}")
    
    # set up training objects
    dataset, model, args = load_train_objs(input_args)
    input_args.logger.info(f"Dataset, model, and args finished loading.")

    train_data = prepare_dataloader(args, dataset, 
                                    int(args.global_batch_size // dist.get_world_size()))
    input_args.logger.info(f"Dataloader finished loading.")
    trainer = Trainer(model, train_data, 
                      rank, int(device.split(":")[-1]), 
                      args, resume_checkpoint=resume_checkpoint)
    input_args.logger.info(f"Trainer finished loading.")
    input_args.logger.info(f"Starting...")
    trainer.train(args.total_epochs)
    destroy_process_group()


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    # data related arguments
    parser.add_argument("--expr_name", type=str, default="CSCC_FLOW")
    parser.add_argument("--data_path", type=str, default="./hest1k_datasets/CSCC/", help="Dataset path")
    parser.add_argument("--results_dir", type=str, default="./CSCC_PSSD_heg_results/runs/", help="Path to hold runs")
    parser.add_argument("--slide_out", type=str, default="NCBI770", help="Test slide ID. Multiple slides separated by comma.") 
    parser.add_argument("--folder_list_filename", type=str, default="all_slide_lst.txt", help="A txt file listing file names for all training and testing slides in the dataset")
    parser.add_argument("--gene_list_filename", type=str, default="selected_gene_list_heg.txt", help="Selected gene list")
    parser.add_argument("--gene_embedding_file", type=str, default="/home/zcy/PSSD-main/hest1k_datasets/gene2vec_dim_200_iter_9_w2v.txt", help="Path to gene embedding txt file")
    parser.add_argument("--gene_embedding_dim", type=int, default=200, help="Dimension of gene embeddings")
    parser.add_argument("--go_obo_path", type=str, default="/home/zcy/PSSD-main/hest1k_datasets/go-basic.obo", help="Path to GO ontology OBO file")
    parser.add_argument("--num_aug_ratio", type=int, default=4, help="Image augmentation ratio (int)")
    
    # model related arguments
    parser.add_argument("--fm_sigma", type=float, default=0.1, help="Noise standard deviation for flow matching regularization")
    parser.add_argument("--fm_path_type", type=str, default="linear",choices=["linear", "optimal_transport", "variance_preserving"], help="Type of interpolation path for flow matching")
    parser.add_argument("--model", type=str, default="PSSDFMProgressive")
    parser.add_argument("--DiT_num_blocks", type=int, default=12, help="DiT depth")
    parser.add_argument("--hidden_size", type=int, default=384, help="DiT hidden dimension")
    parser.add_argument("--num_heads", type=int, default=6, help="DiT heads")

    # training related arguments
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--total_epochs", type=int, default=40000)
    parser.add_argument("--global_batch_size", type=int, default=128)
    parser.add_argument("--global_seed", type=int, default=42)
    parser.add_argument("--num_workers", type=int, default=1, help="Number of GPUs to run the job")
    parser.add_argument("--ckpt_every", type=int, default=50000, help="Number of iterations to save checkpoints.")
    parser.add_argument("--resume", type=str, default=None, help="Resume from checkpoint. Options: " "'auto' (find latest), " "path/to/checkpoint.pt (specific checkpoint), " "path/to/experiment_dir (specific experiment)")
    
    input_args = parser.parse_args()

    ## set up available gpus
    world_size = input_args.num_workers
    ## specify GPU id
    available_gpus = ["cuda:0"] 
    ## or use all available GPU
    # available_gpus = ["cuda:"+str(i) for i in range(world_size)]
    print("Available GPUs: ", available_gpus)
    main(world_size, available_gpus, input_args)