import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, GATConv, GraphConv
from torch_geometric.data import Data
import numpy as np
from scipy.spatial.distance import cdist
from scipy.optimize import linear_sum_assignment
from collections import defaultdict
import mygene
from goatools.obo_parser import GODag
from typing import Dict, List, Tuple, Optional
import os, pickle


class GeneGraphProjection(nn.Module):
    """
    基于图神经网络的基因embedding投影层
    使用GO本体构建基因相似性图
    """
    
    def __init__(self, 
                 input_dim: int, 
                 output_dim: int, 
                 gene_names: List[str],
                 go_obo_path: str = None,
                 gnn_type: str = 'gcn',
                 num_layers: int = 2,
                 hidden_dim: int = None):
        """
        Args:
            input_dim: 输入embedding维度
            output_dim: 输出embedding维度
            gene_names: 基因名称列表
            go_obo_path: GO本体OBO文件路径
            similarity_threshold: 相似性阈值，低于此值的边会被过滤
            gnn_type: GNN类型 ('gcn', 'gat', 'sage')
            num_layers: GNN层数
            hidden_dim: 隐藏层维度
        """
        super().__init__()
        
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.gene_names = gene_names
        self.num_genes = len(gene_names)
        # self.similarity_threshold = similarity_threshold
        
        hidden_dim = hidden_dim or max(input_dim, output_dim)
        
        # 构建基因相似性图
        print("Building gene similarity graph...")
        self.edge_index, self.edge_weight = self._build_gene_graph(go_obo_path)
        
        # 初始化GNN层
        if gnn_type == 'gcn':
            self.gnn_layers = nn.ModuleList([
                GCNConv(input_dim if i == 0 else hidden_dim, 
                       hidden_dim if i < num_layers - 1 else output_dim)
                for i in range(num_layers)
            ])
        elif gnn_type == 'gat':
            self.gnn_layers = nn.ModuleList([
                GATConv(input_dim if i == 0 else hidden_dim * 4,
                       hidden_dim if i < num_layers - 1 else output_dim,
                       heads=4 if i < num_layers - 1 else 1,
                       concat=i < num_layers - 1)
                for i in range(num_layers)
            ])
        else:  # 默认使用GraphConv
            self.gnn_layers = nn.ModuleList([
                GraphConv(input_dim if i == 0 else hidden_dim,
                         hidden_dim if i < num_layers - 1 else output_dim)
                for i in range(num_layers)
            ])
        
        self.dropout = nn.Dropout(0.1)
        
    def _build_gene_graph(self, go_obo_path: str) -> Tuple[torch.Tensor, torch.Tensor]:
        """构建基因相似性图"""
        cache_path = "/home/zcy/PSSD-main/hest1k_datasets/DLPFC/gene_graph_cache_hvg.pkl"  # 可以改成你想要的路径

        # 1. 如果缓存存在，优先读取
        if os.path.exists(cache_path):
            try:
                with open(cache_path, "rb") as f:
                    cache = pickle.load(f)
                cached_genes = cache["gene_names"]
                if cached_genes == self.gene_names:  # 基因列表一致，直接返回
                    print(f"Loaded cached gene graph from {cache_path}")
                    return cache["edge_index"], cache["edge_weight"]
                else:
                    print("Cached gene graph gene list differs, will recompute...")
            except Exception as e:
                print(f"Failed to load cache: {e}, recomputing...")

        # 如果提供了GO文件，使用GO相似性；否则使用单位矩阵
        if go_obo_path and os.path.exists(go_obo_path):
            similarity_matrix = self._compute_go_similarity_matrix(go_obo_path)
        else:
            print("GO file not found, using identity matrix")
            similarity_matrix = np.eye(self.num_genes)
        
        # 转换为图的边表示
        edge_indices = []
        edge_weights = []
        
        for i in range(self.num_genes):
            sims = similarity_matrix[i].copy()
            sims[i] = 0.0  

            # 取前10个最大相似度
            topk = min(10, self.num_genes - 1)
            topk_idx = np.argsort(sims)[-topk:]
            threshold = sims[topk_idx].mean()

            for j in range(i, self.num_genes):  # 只考虑上三角矩阵，因为对称
                # if similarity_matrix[i, j] > self.similarity_threshold:
                if similarity_matrix[i, j] > threshold:
                    edge_indices.extend([[i, j], [j, i]])  # 添加双向边
                    edge_weights.extend([similarity_matrix[i, j], similarity_matrix[i, j]])
        
        edge_index = torch.tensor(edge_indices, dtype=torch.long).t()
        edge_weight = torch.tensor(edge_weights, dtype=torch.float)
        
        print(f"Graph built: {self.num_genes} nodes, {len(edge_weights)} edges")
        try:
            with open(cache_path, "wb") as f:
                pickle.dump({
                    "gene_names": self.gene_names,
                    "edge_index": edge_index,
                    "edge_weight": edge_weight
                }, f)
            print(f"Saved gene graph cache to {cache_path}")
        except Exception as e:
            print(f"Failed to save cache: {e}")
        return edge_index, edge_weight
    
    def _compute_go_similarity_matrix(self, go_obo_path: str) -> np.ndarray:
        """计算基于GO的基因相似性矩阵"""
        
        # 加载GO本体
        go_dag = GODag(go_obo_path)
        
        # 获取基因的GO注释
        gene_go_mapping = self._get_gene_go_annotations(self.gene_names)
        
        # 计算IC值
        ic_values = self._calculate_ic_values(go_dag, gene_go_mapping)
        
        # 计算基因间相似性
        similarity_matrix = np.zeros((self.num_genes, self.num_genes))
        
        for i in range(self.num_genes):
            for j in range(i, self.num_genes):
                if i == j:
                    similarity_matrix[i, j] = 1.0
                else:
                    sim = self._compute_gene_similarity(
                        gene_go_mapping.get(self.gene_names[i], []),
                        gene_go_mapping.get(self.gene_names[j], []),
                        go_dag, ic_values
                    )
                    similarity_matrix[i, j] = sim
                    similarity_matrix[j, i] = sim
        
        return similarity_matrix
    
    def _get_gene_go_annotations(self, gene_names: List[str], cache_path: str = "/home/zcy/PSSD-main/hest1k_datasets/DLPFC/gene_go_cache_hvg.pkl") -> Dict[str, List[str]]:
        """获取基因的GO注释"""
        if os.path.exists(cache_path):
            print(f"Loading cached GO annotations from {cache_path}")
            with open(cache_path, "rb") as f:
                gene_go_mapping = pickle.load(f)
            # 保证只返回需要的gene_names子集
            return {g: gene_go_mapping.get(g, []) for g in gene_names}

        mg = mygene.MyGeneInfo()
        gene_go_mapping = defaultdict(list)
        
        print("Fetching GO annotations...")
        try:
            # 批量查询基因信息
            results = mg.querymany(gene_names, scopes='symbol', fields='go', species='human')
            
            for result in results:
                gene_name = result.get('query', '')
                go_data = result.get('go', {})
                
                go_terms = []
                for category in ['BP', 'MF', 'CC']:
                    terms = go_data.get(category, [])
                    if isinstance(terms, dict):
                        go_terms.append(terms['id'])
                    elif isinstance(terms, list):
                        go_terms.extend(term['id'] for term in terms if 'id' in term)
                
                gene_go_mapping[gene_name] = go_terms
            
            with open(cache_path, "wb") as f:
                pickle.dump(dict(gene_go_mapping), f)
            print(f"Saved GO annotations cache to {cache_path}")

        except Exception as e:
            print(f"Error fetching GO annotations: {e}")
            print("Using empty GO annotations")
        
        return gene_go_mapping
    
    def _calculate_ic_values(self, go_dag: GODag, gene_go_mapping: Dict[str, List[str]]) -> Dict[str, float]:
        """计算GO术语的信息含量(IC)"""
        
        # 统计每个GO术语的频率
        go_counts = defaultdict(int)
        total_annotations = 0
        
        for go_terms in gene_go_mapping.values():
            for go_term in go_terms:
                if go_term in go_dag:
                    # 计算该术语及其所有祖先的频率
                    ancestors = go_dag[go_term].get_all_parents() | {go_term}
                    for ancestor in ancestors:
                        go_counts[ancestor] += 1
                    total_annotations += 1
        
        # 计算IC值: IC(t) = -log(freq(t) / total)
        ic_values = {}
        for go_term, count in go_counts.items():
            if total_annotations > 0:
                freq = count / total_annotations
                ic_values[go_term] = -np.log(freq) if freq > 0 else 0
            else:
                ic_values[go_term] = 0
        
        return ic_values
    
    def _compute_gene_similarity(self, 
                               go_terms1: List[str], 
                               go_terms2: List[str],
                               go_dag: GODag,
                               ic_values: Dict[str, float]) -> float:
        """计算两个基因间的相似性"""
        
        if not go_terms1 or not go_terms2:
            return 0.0
        
        # 过滤有效的GO术语
        valid_terms1 = [t for t in go_terms1 if t in go_dag]
        valid_terms2 = [t for t in go_terms2 if t in go_dag]
        
        if not valid_terms1 or not valid_terms2:
            return 0.0
        
        # 计算术语间的Lin相似性矩阵
        sim_matrix = np.zeros((len(valid_terms1), len(valid_terms2)))
        
        for i, term1 in enumerate(valid_terms1):
            for j, term2 in enumerate(valid_terms2):
                sim_matrix[i, j] = self._lin_similarity(term1, term2, go_dag, ic_values)
        
        # 使用Earth Mover's Distance的近似 - 最大匹配
        if sim_matrix.size > 0:
            # 使用匈牙利算法找到最优匹配
            row_ind, col_ind = linear_sum_assignment(-sim_matrix)  # 负号因为要最大化
            similarity = sim_matrix[row_ind, col_ind].mean()
        else:
            similarity = 0.0
        
        return similarity
    
    def _lin_similarity(self, 
                       term1: str, 
                       term2: str, 
                       go_dag: GODag,
                       ic_values: Dict[str, float]) -> float:
        """计算两个GO术语的Lin相似性"""
        
        if term1 == term2:
            return 1.0
        
        if term1 not in go_dag or term2 not in go_dag:
            return 0.0
        
        # 确保术语在同一命名空间
        if go_dag[term1].namespace != go_dag[term2].namespace:
            return 0.0
        
        # 找到最低公共祖先(LCA)
        ancestors1 = go_dag[term1].get_all_parents() | {term1}
        ancestors2 = go_dag[term2].get_all_parents() | {term2}
        common_ancestors = ancestors1 & ancestors2
        
        if not common_ancestors:
            return 0.0
        
        # 找到IC值最大的公共祖先作为LCA
        lca = max(common_ancestors, key=lambda x: ic_values.get(x, 0))
        
        # 计算Lin相似性
        ic_lca = ic_values.get(lca, 0)
        ic_term1 = ic_values.get(term1, 0)
        ic_term2 = ic_values.get(term2, 0)
        
        if ic_term1 + ic_term2 == 0:
            return 0.0
        
        similarity = 2 * ic_lca / (ic_term1 + ic_term2)
        return similarity
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        前向传播
        Args:
            x: 输入基因embeddings (num_genes, input_dim)
        Returns:
            输出基因embeddings (num_genes, output_dim)
        """
        
        # 确保edge_index和edge_weight在正确的设备上
        edge_index = self.edge_index.to(x.device)
        edge_weight = self.edge_weight.to(x.device)
        
        # 通过GNN层
        for i, layer in enumerate(self.gnn_layers):
            if hasattr(layer, 'heads'):  # GAT层
                x = layer(x, edge_index)
            else:  # GCN或GraphConv层
                x = layer(x, edge_index, edge_weight)
            
            if i < len(self.gnn_layers) - 1:  # 不在最后一层应用激活和dropout
                x = F.relu(x)
                x = self.dropout(x)
        
        return x