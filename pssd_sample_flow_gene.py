import torch
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
from torch.utils.data import DataLoader, Dataset
import sys
sys.path.append("/home/zcy/PSSD-main")
from PSSD.models_fm_gene import PSSDFM_models
import argparse
import pandas as pd
import numpy as np
import anndata
import os
from tqdm import tqdm
from PSSD.flow_matching import ODESolver


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

def find_model(model_name, device=""):
    assert os.path.isfile(model_name), f'Could not find checkpoint at {model_name}'
    if device == "":
        checkpoint = torch.load(model_name, map_location=lambda storage, loc: storage)
    else:
        checkpoint = torch.load(model_name, map_location=device, weights_only=False)
    if "ema" in checkpoint:
        checkpoint = checkpoint["ema"]
    return checkpoint

def main(args):
    # Setup PyTorch:
    torch.manual_seed(args.seed)
    torch.set_grad_enabled(False)
    device = args.device

    gene_embedding_matrix = None
    valid_genes = None
    if args.gene_embedding_file and os.path.exists(args.gene_embedding_file):
        print(f"Loading gene embeddings from {args.gene_embedding_file}")
        gene_embeddings_dict = load_gene_embeddings(args.gene_embedding_file)
        
        # 加载基因列表
        selected_genes = np.genfromtxt(args.data_path + "processed_data/" + args.gene_list_filename, dtype=str)
        print(f"Original gene count: {len(selected_genes)}")
        
        # 创建embedding矩阵，排除没有embedding的基因
        gene_embedding_matrix, valid_genes, missing_genes = create_gene_embedding_matrix(
            selected_genes, gene_embeddings_dict, args.gene_embedding_dim
        )
        
        if missing_genes:
            print(f"Warning: Excluded {len(missing_genes)} genes without embeddings: {missing_genes[:100]}...")
        
        print(f"Valid genes with embeddings: {len(valid_genes)}")
        print(f"Gene embedding matrix shape: {gene_embedding_matrix.shape}")
        
        # 更新基因数量
        args.input_gene_size = len(valid_genes)
        gene_names = valid_genes
        # 将valid_genes保存到args中供后续使用
        args.valid_genes = valid_genes
    else:
        print("No gene embedding file provided or file not found, using learnable embeddings")
        # 加载原始基因列表
        selected_genes = np.genfromtxt(args.data_path + "processed_data/" + args.gene_list_filename, dtype=str)
        args.input_gene_size = len(selected_genes)
        gene_names = selected_genes.tolist()
        args.valid_genes = None

    model = PSSDFM_models[args.model](
        input_size=args.input_gene_size,
        depth= args.DiT_num_blocks,
        hidden_size=args.hidden_size, 
        num_heads=args.num_heads, 
        label_size=args.cond_size,
        gene_embedding_matrix=gene_embedding_matrix,
        gene_names=gene_names,
        go_obo_path=args.go_obo_path,
    )   
    
    ckpt_path = args.ckpt
    state_dict = find_model(ckpt_path, device=args.device)
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()
    
    solver = ODESolver(method=args.ode_method)
    
    print(f"Using ODE solver: {args.ode_method}")
    print(f"Number of ODE steps: {args.num_sampling_steps}")

    loader = DataLoader(args.dataset, batch_size=args.sampling_batch_size, shuffle=False)
    all_samples = []
    
    print("Generating samples...")
    for _, y_batch, coords_batch in loader:

        z = torch.randn(y_batch.shape[0], 1, args.input_gene_size, device=device)
        
        # Generate samples using selected ODE solver
        samples = solver.sample(
            model=model,
            z=z,
            y=y_batch,
            coordinates=coords_batch,
            num_steps=args.num_sampling_steps,
            device=device
        )
        # samples = ode_sampler(model, z, y_batch, num_steps=args.num_sampling_steps, device=device)
        all_samples.append(samples.cpu())

    all_samples_tensor = torch.cat(all_samples, dim=0)
    save_filename = args.save_path + "generated_samples_flowmatch_" + args.ckpt.split("/")[-1].split(".")[0] + "_" + str(args.sample_num_per_cond) + "_" + str(args.ode_method) + "sample.pt"
    torch.save(all_samples_tensor, save_filename)
    print(f"All samples saved to {save_filename}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # model parameters
    parser.add_argument("--model", type=str, choices=list(PSSDFM_models.keys()), default="PSSDFMProgressive")
    parser.add_argument("--DiT_num_blocks", type=int, default=12, help="DiT depth")
    parser.add_argument("--hidden_size", type=int, default=384, help="DiT hidden dimension")
    parser.add_argument("--num_heads", type=int, default=6, help="DiT heads")

    # test slide & gene list
    parser.add_argument("--slide_out", type=str, default="MISC1", help="Test slide ID")
    parser.add_argument("--gene_list_filename", type=str, default="selected_gene_list_hvg.txt")
    parser.add_argument("--gene_embedding_file", type=str, default="/home/zcy/PSSD-main/hest1k_datasets/gene2vec_dim_200_iter_9_w2v.txt", help="Path to gene embedding txt file")
    parser.add_argument("--gene_embedding_dim", type=int, default=200, help="Dimension of gene embeddings")
    parser.add_argument("--go_obo_path", type=str, default="/home/zcy/PSSD-main/hest1k_datasets/go-basic.obo", help="Path to GO ontology OBO file")
    
    # sampling parameter
    parser.add_argument("--ode_method", type=str, default="heun", choices=["euler", "midpoint", "rk4", "heun"], help="ODE solver method to use")
    parser.add_argument("--sample_num_per_cond", type=int, default=20, help="Number of samples generated for each input condition")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--sampling_batch_size", type=int, default=2000, help="Batch size when sampling. Reduce if GPU memory is limited")
    parser.add_argument("--num_sampling_steps", type=int, default=100, help="Number of steps for ODE sampler")
    
    parser.add_argument("--save_path", type=str, default="/home/zcy/PSSD-main/DLPFC_PSSD_hvg_results/runs/000/samples/") # TODO set to path like: ./PRAD_results/runs/000/samples/
    parser.add_argument("--ckpt", type=str, default="/home/zcy/PSSD-main/DLPFC_PSSD_hvg_results/runs/000/checkpoints/0200000.pt") # TODO set to ckpt path like: ./PRAD_results/runs/000/checkpoints/0300000.pt
    parser.add_argument("--data_path", type=str, default="/home/zcy/PSSD-main/hest1k_datasets/DLPFC/")
    
    parser.add_argument("--device", type=str, default="cuda:3")
    
    args = parser.parse_args()

    # load image patches
    data_path = args.data_path
    img_ebd_uni   = torch.load(data_path + "processed_data/1spot_uni_ebd/"   + args.slide_out + "_uni.pt")
    img_ebd_conch = torch.load(data_path + "processed_data/1spot_conch_ebd/" + args.slide_out + "_conch.pt")
    all_img_ebd = torch.cat([img_ebd_uni, img_ebd_conch], dim=1)
    args.raw_cond = all_img_ebd
    args.cond_size = all_img_ebd.shape[1]


    test_adata = anndata.read_h5ad(args.data_path + "st/" + args.slide_out + ".h5ad")
    if 'spatial' in test_adata.obsm:
        coordinates = test_adata.obsm['spatial']
    else:
        if 'x' in test_adata.obs and 'y' in test_adata.obs:
            coordinates = test_adata.obs[['x', 'y']].values
        else:
            print("Warning: No spatial coordinates found, using random coordinates")
            coordinates = np.random.randn(test_adata.shape[0], 2)
    coords_min = coordinates.min(axis=0)
    coords_max = coordinates.max(axis=0)
    coordinates = 2 * (coordinates - coords_min) / (coords_max - coords_min + 1e-8) - 1
    coordinates = torch.from_numpy(coordinates).float()

    args.coordinates = torch.zeros((args.raw_cond.shape[0] * args.sample_num_per_cond, 2))
    for i in range(args.sample_num_per_cond):
        args.coordinates[i::args.sample_num_per_cond] = coordinates

    # create condition matrix
    print("Image patches shape: ", args.raw_cond.shape)
    args.cond = torch.zeros_like(args.raw_cond.repeat((args.sample_num_per_cond, 1)))
    print("Total number of samples to generate: ", args.cond.shape)
    for i in range(args.sample_num_per_cond):
        args.cond[i::args.sample_num_per_cond] = args.raw_cond.clone()

    # create dataset
    args.dataset = CustomDataset(args.cond, args.cond, args.coordinates)
    print(len(args.dataset))
    
    main(args)