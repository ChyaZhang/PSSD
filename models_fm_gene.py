import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math
from timm.models.vision_transformer import PatchEmbed, Attention, Mlp

from .graph_proj import GeneGraphProjection


def modulate(x, shift, scale):
    return x * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1)

class TimestepEmbedder(nn.Module):
    """
    Embeds scalar timesteps into vector representations.
    """
    def __init__(self, hidden_size, frequency_embedding_size=256):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(frequency_embedding_size, hidden_size, bias=True),
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size, bias=True),
        )
        self.frequency_embedding_size = frequency_embedding_size

    @staticmethod
    def timestep_embedding(t, dim, max_period=10000):
        """
        Create sinusoidal timestep embeddings.
        :param t: a 1-D Tensor of N indices, one per batch element.
                          These may be fractional.
        :param dim: the dimension of the output.
        :param max_period: controls the minimum frequency of the embeddings.
        :return: an (N, D) Tensor of positional embeddings.
        """

        half = dim // 2
        freqs = torch.exp(
            -math.log(max_period) * torch.arange(start=0, end=half, dtype=torch.float32) / half
        ).to(device=t.device)
        args = t[:, None].float() * freqs[None]
        embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
        if dim % 2:
            embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1)
        return embedding

    def forward(self, t):
        t_freq = self.timestep_embedding(t, self.frequency_embedding_size)
        t_emb = self.mlp(t_freq)
        return t_emb


class GeneJointEmbedding(nn.Module):
    def __init__(self, input_size, hidden_dim, gene_embedding_matrix=None, gene_names=None, go_obo_path=None):
        """
        input_size: number of genes in input
        hidden_dim: num hidden dimension
        gene_embedding_matrix
        """
        super().__init__()
        self.hidden_dim = hidden_dim
        self.input_size = input_size

        self.gene_count_ebd = nn.Sequential(
            nn.Linear(1, hidden_dim, bias=True),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim, bias=True),            
        )
        torch.nn.init.xavier_uniform_(self.gene_count_ebd[0].weight)
        torch.nn.init.xavier_uniform_(self.gene_count_ebd[2].weight)

        if gene_embedding_matrix is not None:
            # 使用预训练的基因embedding
            embedding_dim = gene_embedding_matrix.shape[1]
            self.gene_pretrained_ebd = nn.Parameter(gene_embedding_matrix.clone(), requires_grad=True)
            
            # 如果使用GO图神经网络
            if embedding_dim != hidden_dim:
                if go_obo_path is not None:
                    print("Using Graph Neural Network for gene embedding projection")
                    self.gene_embedding_proj = GeneGraphProjection(
                        input_dim=embedding_dim,
                        output_dim=hidden_dim,
                        gene_names=gene_names,
                        go_obo_path=go_obo_path,
                        gnn_type='gcn',
                        num_layers=2
                    )
                else:
                    print("GO not provided, using linear projection")
                    self.gene_embedding_proj = nn.Linear(embedding_dim, hidden_dim, bias=True)
                    torch.nn.init.xavier_uniform_(self.gene_embedding_proj.weight)
            else:
                self.gene_embedding_proj = None
            
            self.use_pretrained = True
        else:
            # 如果没有预训练embedding，只使用可训练的embedding
            self.gene_name_ebd = nn.Parameter(torch.empty((input_size, hidden_dim)), requires_grad=True)
            torch.nn.init.kaiming_uniform_(self.gene_name_ebd, a=math.sqrt(5))
            self.use_pretrained = False

    def forward(self, x):
        """
        x: (N, NumGene) tensor of inputs

        """
        gene_count_ebd = self.gene_count_ebd(x.squeeze(1).unsqueeze(2))      # (N, NumGene, hidden_dim)

        if self.use_pretrained:
            # 处理预训练的基因embedding
            pretrained_ebd = self.gene_pretrained_ebd  # (NumGene, embedding_dim)
            if self.gene_embedding_proj is not None:
                if isinstance(self.gene_embedding_proj, GeneGraphProjection):
                    # 使用图神经网络进行投影
                    pretrained_ebd = self.gene_embedding_proj(pretrained_ebd)  # (NumGene, hidden_dim)
                else:
                    # 使用线性投影
                    pretrained_ebd = self.gene_embedding_proj(pretrained_ebd)  # (NumGene, hidden_dim)
            gene_name_ebd = pretrained_ebd
        else:
            gene_name_ebd = self.gene_name_ebd                               # (NumGene, hidden_dim)

        gene_joint_ebd = torch.add(gene_count_ebd, gene_name_ebd)            # (N, NumGene, hidden_dim)

        return gene_joint_ebd                                                # (N, NumGene, hidden_dim)


class SpatialEncoder(nn.Module):
    """编码空间信息"""
    def __init__(self, hidden_dim, coord_dim=2):
        super().__init__()
        # 空间坐标编码
        self.coord_mlp = nn.Sequential(
            nn.Linear(coord_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )
        # 空间关系编码
        self.distance_encoder = nn.Sequential(
            nn.Linear(1, hidden_dim // 2),
            nn.SiLU(),
            nn.Linear(hidden_dim // 2, hidden_dim)
        )
        
    def forward(self, coordinates, k=6):
        """
        coordinates: (N, 2) spatial coordinates
        返回: (N, hidden_dim) spatial encoding
        """
        # 坐标编码
        coord_emb = self.coord_mlp(coordinates.float())
        
        # 计算k近邻的平均距离作为局部密度特征
        dist_matrix = torch.cdist(coordinates.float(), coordinates.float())
        topk_dist, _ = torch.topk(dist_matrix, k=min(k+1, dist_matrix.size(1)), dim=1, largest=False)
        mean_dist = topk_dist[:, 1:].mean(dim=1, keepdim=True)  # 排除自身
        dist_emb = self.distance_encoder(mean_dist)
        
        return coord_emb + dist_emb

class SemanticEncoder(nn.Module):
    """编码基因语义信息"""
    def __init__(self, input_size, hidden_dim, gene_embedding_matrix=None, gene_names=None, go_obo_path=None):
        super().__init__()
        self.gene_joint_embed = GeneJointEmbedding(
            input_size, hidden_dim, 
            gene_embedding_matrix=gene_embedding_matrix,
            gene_names=gene_names,
            go_obo_path=go_obo_path
        )
        # 全局基因关系建模
        # self.gene_relation = nn.TransformerEncoder(
        #     nn.TransformerEncoderLayer(hidden_dim, 4, hidden_dim*2, dropout=0.1, batch_first=True),
        #     num_layers=2
        # )
        
    def forward(self, x):
        """
        x: (N, 1, NumGene) gene expression
        返回: (N, NumGene, hidden_dim) semantic encoding
        """
        x_emb = self.gene_joint_embed(x)  # (N, NumGene, hidden_dim)
        # 建模基因间关系
        # x_semantic = self.gene_relation(x_emb.transpose(0, 1)).transpose(0, 1)
        return x_emb


#################################################################################
#                                 Core Model                                    #
#################################################################################

class DiTBlock(nn.Module):
    """
    A DiT block with adaptive layer norm zero (adaLN-Zero) conditioning.
    """
    def __init__(self, hidden_size, num_heads, mlp_ratio=4.0, **block_kwargs):
        super().__init__()
        self.norm1 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.attn = Attention(hidden_size, num_heads=num_heads, qkv_bias=True, **block_kwargs)
        self.norm2 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        mlp_hidden_dim = int(hidden_size * mlp_ratio)
        approx_gelu = lambda: nn.GELU(approximate="tanh")
        self.mlp = Mlp(in_features=hidden_size, 
                       hidden_features=mlp_hidden_dim, 
                       act_layer=approx_gelu, drop=0)
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(hidden_size, 6 * hidden_size, bias=True)
        )

    def forward(self, x, c):
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = self.adaLN_modulation(c).chunk(6, dim=1)
        x = x + gate_msa.unsqueeze(1) * self.attn(modulate(self.norm1(x), shift_msa, scale_msa))
        x = x + gate_mlp.unsqueeze(1) * self.mlp(modulate(self.norm2(x), shift_mlp, scale_mlp))
        return x


class FinalLayer(nn.Module):
    """
    The final layer of DiT for Flow Matching - outputs velocity directly.
    """
    def __init__(self, hidden_size):
        super().__init__()
        self.norm_final = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.linear = nn.Linear(hidden_size, 1, bias=True)  # Output velocity directly (no sigma prediction)
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(hidden_size, 2 * hidden_size, bias=True)
        )

    def forward(self, x, c):
        shift, scale = self.adaLN_modulation(c).chunk(2, dim=1)
        x = modulate(self.norm_final(x), shift, scale)
        x = torch.permute(self.linear(x), (0, 2, 1))  # (N, NumGene, hidden_size) -> (N, 1, NumGene)
        return x  # (N, NumGene) - velocity field


class PSSDFMModel(nn.Module):
    def __init__(self, 
        input_size=200,
        hidden_size=1152,
        depth=28,
        num_heads=16,
        mlp_ratio=4.0,
        label_size=512,
        gene_embedding_matrix=None,
        gene_names=None,
        go_obo_path=None,
    ):
        super().__init__()

        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        
        # gene name + gene count joint embedding
        self.gene_joint_embed = GeneJointEmbedding(
            self.input_size, 
            self.hidden_size, 
            gene_embedding_matrix=gene_embedding_matrix,
            gene_names=gene_names,
            go_obo_path=go_obo_path
        )
        # time step embedding
        self.time_embed = TimestepEmbedder(self.hidden_size)
        # label embedding (input label is already in embedding form, here just reorganize the size using linear layer)
        self.label_embed = nn.Sequential(
            nn.Linear(label_size, label_size, bias=True),
            nn.SiLU(),
            nn.Linear(label_size, hidden_size, bias=True),
        )

        self.blocks = nn.ModuleList([
            DiTBlock(hidden_size, num_heads, mlp_ratio=mlp_ratio) for _ in range(depth)
        ])
        
        self.final_layer = FinalLayer(self.hidden_size)
        self.initialize_weights()

    def initialize_weights(self):
        # Initialize transformer layers:
        def _basic_init(module):
            if isinstance(module, nn.Linear):
                torch.nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
        self.apply(_basic_init)

        # Initialize timestep embedding MLP:
        nn.init.normal_(self.time_embed.mlp[0].weight, std=0.02)
        nn.init.normal_(self.time_embed.mlp[2].weight, std=0.02)
        nn.init.normal_(self.label_embed[0].weight, std=0.02)
        nn.init.normal_(self.label_embed[2].weight, std=0.02)

        # Zero-out adaLN modulation layers in DiT blocks:
        for block in self.blocks:
            nn.init.constant_(block.adaLN_modulation[-1].weight, 0)
            nn.init.constant_(block.adaLN_modulation[-1].bias, 0)

        # Zero-out output layers:
        nn.init.constant_(self.final_layer.adaLN_modulation[-1].weight, 0)
        nn.init.constant_(self.final_layer.adaLN_modulation[-1].bias, 0)
        nn.init.constant_(self.final_layer.linear.weight, 0)
        nn.init.constant_(self.final_layer.linear.bias, 0)

    def forward(self, x, t, y):
        """
        Forward pass of Flow Matching DiT.
        x: (N, NumGene) tensor of inputs
        t: (N,) tensor of time steps in [0, 1]
        y: (N, label_size) tensor of conditions
        """
        x = self.gene_joint_embed(x)             # , NumGene, hidden_dim) [gene_joint_ebd]
        t = self.time_embed(t)                   # (N, hidden_dim) [time_ebd]
        y = self.label_embed(y)                  # (N, hidden_dim) [label_ebd]
        c = t + y                                # (N, hidden_dim) [condition]
        for block in self.blocks:
            x = block(x, c)                      # (N, NumGene, hidden_dim)
        x = self.final_layer(x, c)               # (N, NumGene) - velocity field
        return x


class DecoupledDiTBlock(nn.Module):
    """解耦的DiT block"""
    def __init__(self, hidden_size, num_heads, mlp_ratio=4.0):
        super().__init__()
        # 空间流处理
        self.spatial_norm1 = nn.LayerNorm(hidden_size, elementwise_affine=False)
        self.spatial_attn = Attention(hidden_size, num_heads=num_heads, qkv_bias=True)
        self.spatial_norm2 = nn.LayerNorm(hidden_size, elementwise_affine=False)
        self.spatial_mlp = Mlp(hidden_size, int(hidden_size * mlp_ratio))
        
        # 语义流处理
        self.semantic_norm1 = nn.LayerNorm(hidden_size, elementwise_affine=False)
        self.semantic_attn = Attention(hidden_size, num_heads=num_heads, qkv_bias=True)
        self.semantic_norm2 = nn.LayerNorm(hidden_size, elementwise_affine=False) 
        self.semantic_mlp = Mlp(hidden_size, int(hidden_size * mlp_ratio))
        
        # 耦合机制
        self.coupling = nn.MultiheadAttention(hidden_size, num_heads, dropout=0.1, batch_first=True)
        self.coupling_norm = nn.LayerNorm(hidden_size)
        
        # 调制
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(hidden_size * 2, 12 * hidden_size, bias=True)  # 空间+语义条件
        )
        
    def forward(self, x_spatial, x_semantic, c_spatial, c_semantic):
        # 获取调制参数
        c_combined = torch.cat([c_spatial, c_semantic], dim=-1)
        modulation = self.adaLN_modulation(c_combined).chunk(12, dim=-1)
        
        # 空间流
        shift_sp1, scale_sp1, gate_sp1, shift_sp2, scale_sp2, gate_sp2 = modulation[:6]
        x_sp = x_spatial + gate_sp1.unsqueeze(1) * self.spatial_attn(
            modulate(self.spatial_norm1(x_spatial), shift_sp1, scale_sp1)
        )
        x_sp = x_sp + gate_sp2.unsqueeze(1) * self.spatial_mlp(
            modulate(self.spatial_norm2(x_sp), shift_sp2, scale_sp2)
        )
        
        # 语义流
        shift_se1, scale_se1, gate_se1, shift_se2, scale_se2, gate_se2 = modulation[6:]
        x_se = x_semantic + gate_se1.unsqueeze(1) * self.semantic_attn(
            modulate(self.semantic_norm1(x_semantic), shift_se1, scale_se1)
        )
        x_se = x_se + gate_se2.unsqueeze(1) * self.semantic_mlp(
            modulate(self.semantic_norm2(x_se), shift_se2, scale_se2)
        )
        
        # 耦合：语义受空间引导
        x_coupled, _ = self.coupling(
            query=self.coupling_norm(x_se),
            key=x_sp,
            value=x_sp
        )
        x_semantic_new = x_se + 0.5 * x_coupled  # 残差连接
        
        return x_sp, x_semantic_new

class DecoupledFinalLayer(nn.Module):
    """解耦的最终层"""
    def __init__(self, hidden_size):
        super().__init__()
        self.norm_final = nn.LayerNorm(hidden_size * 2, elementwise_affine=False)
        self.linear = nn.Linear(hidden_size * 2, 1, bias=True)
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(hidden_size * 2, hidden_size * 4, bias=True)
        )
        
    def forward(self, x_spatial, x_semantic, c):
        # 融合空间和语义
        x_combined = torch.cat([x_spatial, x_semantic], dim=-1)
        shift, scale = self.adaLN_modulation(c).chunk(2, dim=-1)
        x = modulate(self.norm_final(x_combined), shift, scale)
        x = torch.permute(self.linear(x), (0, 2, 1))
        return x

class PSSDFMDecoupledModel(nn.Module):
    def __init__(self,
        input_size=200,
        hidden_size=384,
        depth=12,
        num_heads=6,
        mlp_ratio=4.0,
        label_size=512,
        gene_embedding_matrix=None,
        gene_names=None,
        go_obo_path=None,
    ):
        super().__init__()
        
        self.input_size = input_size
        self.hidden_size = hidden_size
        
        # 解耦编码器
        self.spatial_encoder = SpatialEncoder(hidden_size)
        self.semantic_encoder = SemanticEncoder(
            input_size, hidden_size,
            gene_embedding_matrix=gene_embedding_matrix,
            gene_names=gene_names,
            go_obo_path=go_obo_path
        )
        
        # 时间和条件编码
        self.time_embed = TimestepEmbedder(hidden_size)
        self.label_embed = nn.Sequential(
            nn.Linear(label_size, label_size, bias=True),
            nn.SiLU(),
            nn.Linear(label_size, hidden_size, bias=True),
        )
        
        # 解耦的DiT blocks
        self.blocks = nn.ModuleList([
            DecoupledDiTBlock(hidden_size, num_heads, mlp_ratio)
            for _ in range(depth)
        ])
        
        self.final_layer = DecoupledFinalLayer(hidden_size)
        self.initialize_weights()
        
    def initialize_weights(self):
        def _basic_init(module):
            if isinstance(module, nn.Linear):
                torch.nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
        self.apply(_basic_init)
        
    def forward(self, x, t, y, coordinates):
        """
        x: (N, 1, NumGene) gene expression
        t: (N,) time steps
        y: (N, label_size) image conditions
        coordinates: (N, 2) spatial coordinates
        """
            
        x_spatial_init = self.spatial_encoder(coordinates)  # (N, hidden_dim)
        x_semantic = self.semantic_encoder(x)  # (N, NumGene, hidden_dim)
        
        # 扩展空间编码到基因维度
        x_spatial = x_spatial_init.unsqueeze(1).expand(-1, self.input_size, -1)
        
        # 条件编码
        t_emb = self.time_embed(t)
        y_emb = self.label_embed(y)
        
        # 空间和语义条件分离
        c_spatial = t_emb + y_emb 
        c_semantic = t_emb + y_emb  
        
        # 通过解耦blocks
        for block in self.blocks:
            x_spatial, x_semantic = block(x_spatial, x_semantic, c_spatial, c_semantic)
            
        # 最终预测
        c_combined = torch.cat([c_spatial, c_semantic], dim=-1)
        x = self.final_layer(x_spatial, x_semantic, c_combined)
        return x


class LightweightDecoupledBlock(nn.Module):
    def __init__(self, hidden_size, num_heads, mlp_ratio=4.0):
        super().__init__()
        # 共享的主要组件
        self.norm1 = nn.LayerNorm(hidden_size, elementwise_affine=False)
        self.attn = Attention(hidden_size, num_heads=num_heads, qkv_bias=True)
        self.norm2 = nn.LayerNorm(hidden_size, elementwise_affine=False)
        mlp_hidden_dim = int(hidden_size * mlp_ratio)
        approx_gelu = lambda: nn.GELU(approximate="tanh")
        self.mlp = Mlp(hidden_size, mlp_hidden_dim, act_layer=approx_gelu)
        
        # 轻量级的模态特定适配器
        self.spatial_adapter = nn.Linear(hidden_size, hidden_size)
        self.semantic_adapter = nn.Linear(hidden_size, hidden_size)
        
        #融合门控
        # self.fusion_gate = nn.Sequential(
        #     nn.Linear(hidden_size * 2, hidden_size),
        #     nn.SiLU(),
        #     nn.Linear(hidden_size, 2),
        #     nn.Softmax(dim=-1)
        # )

        # 改进的融合门控 - 输入包含模态差异信息
        self.fusion_gate = nn.Sequential(
            nn.Linear(hidden_size * 4, hidden_size),  # 增加输入维度
            nn.SiLU(),
            nn.Linear(hidden_size, 2),
            nn.Softmax(dim=-1)
        )
        
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(hidden_size, 6 * hidden_size, bias=True)
        )
        
    def forward(self, x_spatial, x_semantic, c):
        """
        x_spatial: (N, NumGene, hidden_dim) - 空间流
        x_semantic: (N, NumGene, hidden_dim) - 语义流
        c: (N, hidden_dim) - 条件
        """
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = self.adaLN_modulation(c).chunk(6, dim=1)
        
        # 共享attention计算
        x_spatial_attn = self.attn(modulate(self.norm1(x_spatial), shift_msa, scale_msa))
        x_semantic_attn = self.attn(modulate(self.norm1(x_semantic), shift_msa, scale_msa))
        
        # 轻量级适配
        # x_spatial_attn = x_spatial_attn + 0.1 * self.spatial_adapter(x_spatial_attn)
        # x_semantic_attn = x_semantic_attn + 0.1 * self.semantic_adapter(x_semantic_attn)

        x_spatial_attn = x_spatial_attn + self.spatial_adapter(x_spatial_attn)
        x_semantic_attn = x_semantic_attn + self.semantic_adapter(x_semantic_attn)
        
        # 残差连接
        x_spatial = x_spatial + gate_msa.unsqueeze(1) * x_spatial_attn
        x_semantic = x_semantic + gate_msa.unsqueeze(1) * x_semantic_attn
        
        # MLP处理（共享）
        x_spatial = x_spatial + gate_mlp.unsqueeze(1) * self.mlp(modulate(self.norm2(x_spatial), shift_mlp, scale_mlp))
        x_semantic = x_semantic + gate_mlp.unsqueeze(1) * self.mlp(modulate(self.norm2(x_semantic), shift_mlp, scale_mlp))
        
        # 轻量级交叉更新
        # gate_weights = self.fusion_gate(torch.cat([c, c], dim=-1))

        # # 改进的fusion gate - 包含模态差异信息
        x_spatial_mean = x_spatial.mean(dim=1)  # (N, hidden_dim)
        x_semantic_mean = x_semantic.mean(dim=1)  # (N, hidden_dim)
        x_diff = (x_spatial_mean - x_semantic_mean)  # 模态差异
        gate_input = torch.cat([c, x_spatial_mean, x_semantic_mean, x_diff], dim=-1)
        gate_weights = self.fusion_gate(gate_input)  # (N, 2)

        # x_spatial_new = x_spatial + 0.1 * gate_weights[:, 0:1].unsqueeze(1) * x_semantic
        # x_semantic_new = x_semantic + 0.1 * gate_weights[:, 1:2].unsqueeze(1) * x_spatial

        x_spatial_new = x_spatial + gate_weights[:, 0:1].unsqueeze(1) * x_semantic
        x_semantic_new = x_semantic + gate_weights[:, 1:2].unsqueeze(1) * x_spatial
        
        return x_spatial_new, x_semantic_new

class PSSDFMProgressiveModel(nn.Module):
    def __init__(self,
        input_size=200,
        hidden_size=384,
        depth=12,
        num_heads=6,
        mlp_ratio=4.0,
        label_size=512,
        gene_embedding_matrix=None,
        gene_names=None,
        go_obo_path=None,
    ):
        super().__init__()
        
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        
        # 基因编码（保持原有）
        self.gene_joint_embed = GeneJointEmbedding(
            self.input_size, 
            self.hidden_size,
            gene_embedding_matrix=gene_embedding_matrix,
            gene_names=gene_names,
            go_obo_path=go_obo_path
        )
        
        # 时间和标签编码
        self.time_embed = TimestepEmbedder(self.hidden_size)
        self.label_embed = nn.Sequential(
            nn.Linear(label_size, label_size, bias=True),
            nn.SiLU(),
            nn.Linear(label_size, hidden_size, bias=True),
        )
        
        # 空间位置编码（轻量级）
        self.spatial_encoder = nn.Sequential(
            nn.Linear(2, hidden_size // 4),
            nn.SiLU(),
            nn.Linear(hidden_size // 4, hidden_size)
        )
        
        # 三阶段blocks
        stage1_depth = depth // 3
        stage2_depth = depth // 3
        stage3_depth = depth - stage1_depth - stage2_depth
        
        # Stage 1: 共享编码blocks
        self.shared_blocks = nn.ModuleList([
            DiTBlock(hidden_size, num_heads, mlp_ratio) for _ in range(stage1_depth)
        ])
        
        # Stage 2: 轻量解耦blocks
        self.decoupled_blocks = nn.ModuleList([
            LightweightDecoupledBlock(hidden_size, num_heads, mlp_ratio) for _ in range(stage2_depth)
        ])
        
        # Stage 3: 融合blocks
        self.fusion_blocks = nn.ModuleList([
            DiTBlock(hidden_size, num_heads, mlp_ratio) for _ in range(stage3_depth)
        ])

        # 改进的融合层 - per-spot gating
        # self.fusion_network = nn.Sequential(
        #     nn.Linear(hidden_size * 3, hidden_size * 2),
        #     nn.SiLU(),
        #     nn.Linear(hidden_size * 2, hidden_size),
        #     nn.SiLU(),
        #     nn.Linear(hidden_size, 1),
        #     nn.Sigmoid()
        # )
        
        self.final_layer = FinalLayer(hidden_size)
        self.initialize_weights()
        
    def initialize_weights(self):
        def _basic_init(module):
            if isinstance(module, nn.Linear):
                torch.nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
        self.apply(_basic_init)
        
        # 初始化时间和标签embedding
        nn.init.normal_(self.time_embed.mlp[0].weight, std=0.02)
        nn.init.normal_(self.time_embed.mlp[2].weight, std=0.02)
        nn.init.normal_(self.label_embed[0].weight, std=0.02)
        nn.init.normal_(self.label_embed[2].weight, std=0.02)
        
        # Zero-out modulation layers
        for block in self.shared_blocks:
            nn.init.constant_(block.adaLN_modulation[-1].weight, 0)
            nn.init.constant_(block.adaLN_modulation[-1].bias, 0)
        for block in self.fusion_blocks:
            nn.init.constant_(block.adaLN_modulation[-1].weight, 0)
            nn.init.constant_(block.adaLN_modulation[-1].bias, 0)
        
        nn.init.constant_(self.final_layer.adaLN_modulation[-1].weight, 0)
        nn.init.constant_(self.final_layer.adaLN_modulation[-1].bias, 0)
        nn.init.constant_(self.final_layer.linear.weight, 0)
        nn.init.constant_(self.final_layer.linear.bias, 0)

    # def compute_decorrelation_loss(self, x_spatial, x_semantic):
    #     """计算解耦正则化损失"""
    #     # 计算特征的协方差
    #     x_sp_flat = x_spatial.reshape(x_spatial.size(0), -1)
    #     x_se_flat = x_semantic.reshape(x_semantic.size(0), -1)
        
    #     # 归一化
    #     x_sp_norm = F.normalize(x_sp_flat, dim=1)
    #     x_se_norm = F.normalize(x_se_flat, dim=1)
        
    #     # 计算相关性
    #     correlation = torch.abs(torch.sum(x_sp_norm * x_se_norm, dim=1)).mean()
        
    #     return correlation * 0.1  # 缩放系数

    def forward(self, x, t, y, coordinates):
        """
        x: (N, 1, NumGene) 基因表达
        t: (N,) 时间步
        y: (N, label_size) 图像条件
        coordinates: (N, 2) 空间坐标（可选）
        """
        # 基因编码
        x = self.gene_joint_embed(x)  # (N, NumGene, hidden_dim)
        
        # 条件编码
        t_emb = self.time_embed(t)
        y_emb = self.label_embed(y)
        c = t_emb + y_emb
        
        # Stage 1: 共享编码
        for block in self.shared_blocks:
            x = block(x, c)
        
        # Stage 2: 轻量解耦处理
        spatial_emb = self.spatial_encoder(coordinates)  # (N, hidden_dim)
        x_spatial = x + spatial_emb.unsqueeze(1) * 0.5  
        x_semantic = x

        # 解耦处理
        # decorr_losses = []
        for block in self.decoupled_blocks:
            x_spatial, x_semantic = block(x_spatial, x_semantic, c)
            # if self.training:
            #     decorr_loss = self.compute_decorrelation_loss(x_spatial, x_semantic)
            #     decorr_losses.append(decorr_loss)
        
        # Stage 3: 融合
        # 自适应融合权重
        if coordinates is not None:
            # 基于空间位置的融合权重
            spatial_weight = torch.sigmoid(spatial_emb.mean(dim=-1, keepdim=True)).unsqueeze(1)
            x = spatial_weight * x_spatial + (1 - spatial_weight) * x_semantic
        else:
            x = (x_spatial + x_semantic) / 2
        
        for block in self.fusion_blocks:
            x = block(x, c)
        
        # 最终输出
        x = self.final_layer(x, c)
        # if self.training and decorr_losses:
        #     return x, sum(decorr_losses) / len(decorr_losses)
        return x




def PSSDFM(**kwargs):
    return PSSDFMModel(**kwargs)

def PSSDFMDecoupled(**kwargs):
    return PSSDFMDecoupledModel(**kwargs)

def PSSDFMProgressive(**kwargs):
    return PSSDFMProgressiveModel(**kwargs)



# Model registry
PSSDFM_models = {"PSSDFM": PSSDFM, "PSSDFMDecoupled": PSSDFMDecoupled, "PSSDFMProgressive": PSSDFMProgressive}