import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, ConcatDataset
import numpy as np
import pandas as pd
import scanpy as sc
import scipy.sparse
from pathlib import Path
from typing import Dict, Tuple, List, Optional, Any
import glob
import random
import logging
import math
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor
from pytorch_lightning.loggers import CSVLogger, TensorBoardLogger
import os
import fire
import json

# 设置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# ==========================================
# 1. 配置与超参数
# ==========================================
CONFIG = {
    # 数据相关
    "data_dir": "/gpfs/hybrid/data/public/TEDD/5.link_cells/",
    "output_dir": "bench/output/larger_data", # Output directory
    "n_files_to_load": 50,        # 随机加载的文件数量
    "train_ratio": 0.8,          # 训练集比例
    "gestation_period": 280.0,   # 人类孕期 (天)
    
    # 模型相关 (DiT)
    # DiT-S/2: hidden=384, depth=12, heads=6
    # DiT-B/2: hidden=768, depth=12, heads=12
    # DiT-L/2: hidden=1024, depth=24, heads=16
    "hidden_size": 1024,
    "depth": 24,
    "num_heads": 16,
    "mlp_ratio": 4.0,
    "class_dropout_prob": 0.1,
    
    # 训练相关
    "batch_size": 256,
    "lr": 1e-4,
    "epochs": 10,
    "seed": 42,
    "accelerator": "auto",       # Lighting accelerator
    "devices": 1,
}

# 设定随机种子
pl.seed_everything(CONFIG['seed'])


# ==========================================
# 2. DiT Backbone 实现
# ==========================================
def modulate(x, shift, scale):
    return x * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1)

class TimestepEmbedder(nn.Module):
    """
    Embeds scalar timesteps into vector representations.
    """
    def __init__(self, hidden_size, frequency_embedding_size=256):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(frequency_embedding_size, hidden_size),
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size),
        )
        self.frequency_embedding_size = frequency_embedding_size

    @staticmethod
    def timestep_embedding(t, dim, max_period=10000):
        """
        Create sinusoidal timestep embeddings.
        """
        # 关键修正：将输入时间 t 缩放 1000 倍
        # 因为我们的 t 通常在 [0, 1] 范围，而标准位置编码是为整数序列设计的。
        # 不缩放的话，所有频率分量都在低频区，变化太小。
        t = t * 1000.0 
        
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

class RotaryEmbedding(nn.Module):
    def __init__(self, dim, max_position_embeddings=2048, base=10000, device=None):
        super().__init__()
        inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2).float().to(device) / dim))
        self.register_buffer("inv_freq", inv_freq)
        self.max_seq_len_cached = max_position_embeddings
        t = torch.arange(self.max_seq_len_cached, device=device, dtype=self.inv_freq.dtype)
        freqs = torch.einsum("i,j->ij", t, self.inv_freq)
        emb = torch.cat((freqs, freqs), dim=-1)
        self.register_buffer("cos_cached", emb.cos()[None, None, :, :], persistent=False)
        self.register_buffer("sin_cached", emb.sin()[None, None, :, :], persistent=False)

    def forward(self, x, seq_len=None):
        if seq_len > self.max_seq_len_cached:
            self.max_seq_len_cached = seq_len
            t = torch.arange(self.max_seq_len_cached, device=x.device, dtype=self.inv_freq.dtype)
            freqs = torch.einsum("i,j->ij", t, self.inv_freq)
            emb = torch.cat((freqs, freqs), dim=-1)
            self.register_buffer("cos_cached", emb.cos()[None, None, :, :], persistent=False)
            self.register_buffer("sin_cached", emb.sin()[None, None, :, :], persistent=False)
        return (
            self.cos_cached[:, :, :seq_len, ...].to(dtype=x.dtype),
            self.sin_cached[:, :, :seq_len, ...].to(dtype=x.dtype),
        )

def rotate_half(x):
    x1, x2 = x.chunk(2, dim=-1)
    return torch.cat((-x2, x1), dim=-1)

def apply_rotary_pos_emb(q, k, cos, sin):
    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    return q_embed, k_embed

class Attention(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, use_rope=False):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim ** -0.5
        self.use_rope = use_rope

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.proj = nn.Linear(dim, dim)
        
        if use_rope:
            self.rope = RotaryEmbedding(head_dim)

    def forward(self, x):
        # x: (B, L, D)
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2] # (B, H, N, head_dim)

        if self.use_rope:
            cos, sin = self.rope(v, seq_len=N)
            q, k = apply_rotary_pos_emb(q, k, cos, sin)

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        return x, attn

class DiTBlock(nn.Module):
    """
    A DiT block with adaptive layer norm zero (adaLN-Zero) conditioning.
    """
    def __init__(self, hidden_size, num_heads, mlp_ratio=4.0, use_rope=False, **block_kwargs):
        super().__init__()
        self.norm1 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        # Replace nn.MultiheadAttention with custom Attention to support RoPE
        self.attn = Attention(hidden_size, num_heads=num_heads, qkv_bias=True, use_rope=use_rope)
        self.norm2 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        mlp_hidden_dim = int(hidden_size * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(hidden_size, mlp_hidden_dim),
            nn.GELU(approximate="tanh"),
            nn.Linear(mlp_hidden_dim, hidden_size),
        )
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(hidden_size, 6 * hidden_size, bias=True)
        )

    def forward(self, x, c):
        # x: (N, L, D), c: (N, D)
        
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = self.adaLN_modulation(c).chunk(6, dim=1)
        
        # Self-Attention
        x_norm = modulate(self.norm1(x), shift_msa, scale_msa)
        
        attn_out, _ = self.attn(x_norm) 
        x = x + gate_msa.unsqueeze(1) * attn_out
        
        # MLP
        x = x + gate_mlp.unsqueeze(1) * self.mlp(modulate(self.norm2(x), shift_mlp, scale_mlp))
        return x

class FinalLayer(nn.Module):
    """
    The final layer of DiT.
    """
    def __init__(self, hidden_size, out_channels):
        super().__init__()
        self.norm_final = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.linear = nn.Linear(hidden_size, out_channels, bias=True)
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(hidden_size, 2 * hidden_size, bias=True)
        )

    def forward(self, x, c):
        shift, scale = self.adaLN_modulation(c).chunk(2, dim=1)
        x = modulate(self.norm_final(x), shift, scale)
        x = self.linear(x)
        return x

class DiT(nn.Module):
    """
    Diffusion Transformer for Single Cell Flow Matching.
    Input: x (Cell State)
    Condition: t, tissue, celltype, x_current
    """
    def __init__(
        self,
        input_dim,
        hidden_size=384,
        depth=6,
        num_heads=6,
        n_tissues=10,
        n_celltypes=20,
        cond_dropout=0.1
    ):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        
        # 1. Input Embedding (Project to hidden_size)
        self.x_embedder = nn.Linear(input_dim, hidden_size)
        self.t_embedder = TimestepEmbedder(hidden_size)
        
        # 2. Condition Embeddings
        self.tissue_emb = nn.Embedding(n_tissues + 1, hidden_size) # +1 for null cond
        self.celltype_emb = nn.Embedding(n_celltypes + 1, hidden_size)
        # Condition from previous cell state
        self.x_curr_embedder = nn.Linear(input_dim, hidden_size)
        # Condition from time scalar (unified absolute time)
        self.abs_time_embedder = TimestepEmbedder(hidden_size)
        # Condition from delta time
        self.dt_embedder = TimestepEmbedder(hidden_size)

        # 3. Transformer Blocks
        self.blocks = nn.ModuleList([
            DiTBlock(hidden_size, num_heads) for _ in range(depth)
        ])
        
        # 4. Final Layer
        self.final_layer = FinalLayer(hidden_size, input_dim)
        
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
        nn.init.normal_(self.t_embedder.mlp[0].weight, std=0.02)
        nn.init.normal_(self.t_embedder.mlp[2].weight, std=0.02)

        # Zero-out adaLN modulation layers in DiT blocks:
        for block in self.blocks:
            nn.init.constant_(block.adaLN_modulation[-1].weight, 0)
            nn.init.constant_(block.adaLN_modulation[-1].bias, 0)

        # Zero-out output layers:
        nn.init.constant_(self.final_layer.adaLN_modulation[-1].weight, 0)
        nn.init.constant_(self.final_layer.adaLN_modulation[-1].bias, 0)
        nn.init.constant_(self.final_layer.linear.weight, 0)
        nn.init.constant_(self.final_layer.linear.bias, 0)

    def forward(self, x, t, cond_data):
        """
        x: (N, input_dim) - Noisy state
        t: (N,) - Flow timestep
        cond_data: Dict containing 'x_curr', 'time', 'dt', 'tissue', 'celltype'
        """
        # 1. Embed Input -> (N, 1, D)
        x = self.x_embedder(x).unsqueeze(1)
        
        # 2. Embed Conditions -> Sum them up to form the global conditioning vector 'c'
        t_emb = self.t_embedder(t)
        
        c_tissue = self.tissue_emb(cond_data['tissue'])
        c_ctype = self.celltype_emb(cond_data['celltype'])
        c_xcurr = self.x_curr_embedder(cond_data['x_curr'])
        c_abs_time = self.abs_time_embedder(cond_data['time'])
        c_dt = self.dt_embedder(cond_data['dt'])
        
        # Combine conditions (Simple additive conditioning)
        # You can also use cross-attention if you treat conditions as sequence
        c = t_emb + c_tissue + c_ctype + c_xcurr + c_abs_time + c_dt
        
        # 3. Transformer Blocks
        for block in self.blocks:
            x = block(x, c)
            
        # 4. Final Layer
        x = self.final_layer(x, c)
        
        return x.squeeze(1) # (N, input_dim)

class FlowMatchingLightning(pl.LightningModule):
    def __init__(self, input_dim, config, tissue_map, celltype_map):
        super().__init__()
        self.save_hyperparameters(ignore=['tissue_map', 'celltype_map']) # Save config for logging
        self.config = config
        
        self.model = DiT(
            input_dim=input_dim,
            hidden_size=config['hidden_size'],
            depth=config['depth'],
            num_heads=config['num_heads'],
            n_tissues=len(tissue_map),
            n_celltypes=len(celltype_map)
        )

    def forward(self, x, t, cond_data):
        return self.model(x, t, cond_data)

    def configure_optimizers(self):
        return torch.optim.AdamW(self.model.parameters(), lr=self.config['lr'])

    def training_step(self, batch, batch_idx):
        x_next = batch['x_next']
        cond_data = batch['cond_meta']
        cond_data['x_curr'] = batch['x_curr']
        
        # Mask handling
        gene_mask = cond_data.get('gene_mask', None) # (B, D)
        
        B = x_next.shape[0]
        
        # Flow Matching Loss
        t = torch.rand(B, device=self.device)
        x_0 = torch.randn_like(x_next)
        
        # Interpolation (Source=Noise, Target=Next)
        t_expand = t.view(B, 1)
        z_t = t_expand * x_next + (1 - t_expand) * x_0
        v_target = x_next - x_0
        
        v_pred = self.model(z_t, t, cond_data)
        
        # Masked Loss Calculation
        if gene_mask is not None:
             # Expand mask to match B if it was (D,) - but DataLoader handles batching so it should be (B, D)
             # We only compute loss on valid genes
             squared_error = F.mse_loss(v_pred, v_target, reduction='none')
             masked_error = squared_error * gene_mask.float()
             # Avoid division by zero
             loss = masked_error.sum() / (gene_mask.float().sum() + 1e-8)
        else:
             loss = F.mse_loss(v_pred, v_target)
        
        self.log('train_loss', loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        return loss

    def validation_step(self, batch, batch_idx):
        x_next = batch['x_next']
        cond_data = batch['cond_meta']
        cond_data['x_curr'] = batch['x_curr']
        gene_mask = cond_data.get('gene_mask', None)
        
        t = torch.rand(x_next.shape[0], device=self.device)
        x_0 = torch.randn_like(x_next)
        t_expand = t.view(-1, 1)
        z_t = t_expand * x_next + (1 - t_expand) * x_0
        v_target = x_next - x_0
        
        v_pred = self.model(z_t, t, cond_data)
        
        if gene_mask is not None:
             squared_error = F.mse_loss(v_pred, v_target, reduction='none')
             masked_error = squared_error * gene_mask.float()
             loss = masked_error.sum() / (gene_mask.float().sum() + 1e-8)
        else:
             loss = F.mse_loss(v_pred, v_target)
        
        self.log('val_loss', loss, on_step=False, on_epoch=True, prog_bar=True, logger=True)
        return loss

# ==========================================
# 3. 数据处理核心逻辑
# ==========================================
def load_gene_list(path: str) -> List[str]:
    """加载目标基因列表"""
    logger.info(f"Loading gene list from {path}...")
    try:
        df = pd.read_csv(path, sep='\t', header=None)
        # 假设第一列是基因名
        genes = df[0].astype(str).values.tolist()
        logger.info(f"Loaded {len(genes)} target genes.")
        return genes
    except Exception as e:
        logger.error(f"Failed to load gene list: {e}")
        raise e

def unify_time(adata: sc.AnnData, gestation_days: float = 280.0) -> np.ndarray:
    """统一时间轴"""
    dataset_stage = adata.uns.get('Stage', 'Unknown')
    if 'Time' in adata.obs.columns:
        raw_time = adata.obs['Time'].values.astype(np.float32)
    elif 'time' in adata.obs.columns: # 处理小写 time
        raw_time = adata.obs['time'].values.astype(np.float32)
    elif 'timepoint' in adata.obs.columns: # 处理列名变体
         raw_time = pd.to_numeric(adata.obs['timepoint'], errors='coerce').fillna(0).values.astype(np.float32)
    else:
        logger.warning(f"Warning: No Time column found. Using zeros.")
        raw_time = np.zeros(adata.n_obs, dtype=np.float32)

    unified_time = np.zeros_like(raw_time)
    
    is_embryonic_or_fetal = ('Embryonic' in dataset_stage) or ('Fetal' in dataset_stage) or (raw_time.max() < 100) # Heuristic
    
    if is_embryonic_or_fetal:
        unified_time = raw_time
    else:
        unified_time = raw_time + gestation_days
    
    return unified_time

def align_genes(adata: sc.AnnData, target_genes: List[str]) -> sc.AnnData:
    """
    对齐 AnnData 到目标基因列表 (Reindex with zero padding).
    不再取交集，而是强制对齐到 target_genes。
    """
    target_genes_arr = np.array(target_genes)
    current_genes_arr = np.array(adata.var_names)
    
    # 1. 检查是否完全一致
    if np.array_equal(target_genes_arr, current_genes_arr):
        return adata
        
    # 2. 建立索引映射
    gene_to_idx = {g: i for i, g in enumerate(current_genes_arr)}
    
    src_indices = []
    dst_indices = []
    
    # 找出存在的基因并记录索引
    for dst_i, gene in enumerate(target_genes):
        if gene in gene_to_idx:
            src_indices.append(gene_to_idx[gene])
            dst_indices.append(dst_i)
            
    # 3. 构建新的数据矩阵
    N = adata.n_obs
    M = len(target_genes)
    
    # 转换为 Dense (为了安全和简单，Flow Matching 最终也需要 Tensor)
    if scipy.sparse.issparse(adata.X):
        X_old = adata.X.toarray()
    else:
        X_old = adata.X
        
    X_new = np.zeros((N, M), dtype=X_old.dtype)
    
    # 填入存在的基因数据
    if src_indices:
        X_new[:, dst_indices] = X_old[:, src_indices]
        
    # 4. 创建新 AnnData
    new_adata = sc.AnnData(X=X_new, obs=adata.obs)
    new_adata.var_names = target_genes
    new_adata.uns = adata.uns
    
    return new_adata

def load_data_files(
    data_dir: str,
    target_genes: List[str],
    n_files: Optional[int] = None,
    tissue_map: Optional[Dict[str, int]] = None,
    celltype_map: Optional[Dict[str, int]] = None,
) -> Tuple[List[sc.AnnData], Dict, Dict]:
    """加载并处理多个数据文件，并对齐到 target_genes"""
    all_files = glob.glob(str(Path(data_dir) / "*.link.h5ad"))
    
    # 过滤掉非人类数据
    human_files = [f for f in all_files if not (Path(f).name.startswith("mm_") or Path(f).name.startswith("oc_") or Path(f).name.startswith("dr_"))]
    logger.info(f"过滤后剩余人类数据文件: {len(human_files)} / {len(all_files)}")
    
    if n_files is not None:
        if n_files > len(human_files):
             logger.warning(f"请求加载 {n_files} 个文件，但只有 {len(human_files)} 个。")
             target_files = list(human_files)
             random.shuffle(target_files)
        else:
            target_files = random.sample(human_files, n_files)
    else:
        target_files = list(human_files)
        random.shuffle(target_files)
    
    logger.info(f"计划加载 {len(target_files)} 个文件...")
    
    processed_adatas = []
    all_tissues = set()
    all_celltypes = set()
    
    # 临时存储，为了统一 Label 编码
    temp_adatas = [] 
    
    # 1. 加载、对齐、筛选
    for fpath in target_files:
        try:
            logger.info(f"Loading {Path(fpath).name}...")
            adata = sc.read_h5ad(fpath)
            
            # 检查物种
            species = adata.uns.get('Species', 'Unknown')
            if species != 'Unknown' and species != 'Homo sapiens':
                continue

            # 检查必要列
            required_cols = ['Tissue', 'Celltype', 'next_cell_id']
            if not all(col in adata.obs.columns for col in required_cols):
                continue
                
            # === STEP A: 基因对齐 ===
            adata = align_genes(adata, target_genes)
            
            # === STEP B: 处理 Next Cell ID (Smart Hybrid) ===
            # 策略 1: 尝试作为字符串 ID 映射 (优先)
            # 强制转换为字符串
            next_cell_id_str = adata.obs['next_cell_id'].astype(str)
            # 建立当前文件内的 ID -> Index 映射
            current_ids = adata.obs.index.astype(str)
            cell_id_to_idx = {cid: i for i, cid in enumerate(current_ids)}
            
            next_cell_indices_map = next_cell_id_str.map(cell_id_to_idx).fillna(-1).astype(int).values
            valid_mask_map = (next_cell_indices_map != -1) & (next_cell_indices_map < len(adata))
            
            # 策略 2: 尝试作为直接数值索引 (Fallback)
            next_cell_indices_direct = np.full(len(adata), -1, dtype=int)
            # 检查是否为数字类型，或者是看似数字的字符串
            is_numeric_col = pd.api.types.is_numeric_dtype(adata.obs['next_cell_id'])
            
            if is_numeric_col:
                 next_cell_indices_direct = adata.obs['next_cell_id'].fillna(-1).astype(int).values
            
            valid_mask_direct = (next_cell_indices_direct >= 0) & (next_cell_indices_direct < len(adata))
            
            # 决策逻辑
            # 如果字符串映射成功数量显著多，用字符串
            # 否则如果直接索引成功数量多，用直接索引
            if valid_mask_map.sum() >= valid_mask_direct.sum():
                 raw_next_indices = next_cell_indices_map
                 valid_mask_source = valid_mask_map
            else:
                 raw_next_indices = next_cell_indices_direct
                 valid_mask_source = valid_mask_direct
            
            # === STEP C: 索引重映射 (关键修复) ===
            # raw_next_indices 是相对于原始 adata 的索引
            # 我们需要过滤掉那些 "Target细胞被过滤掉" 的样本
            # 并且将 raw_next_indices 转换为相对于 adata_valid 的新索引
            
            # 1. 既然我们要过滤，首先确定哪些行是保留的
            # 目前的策略是：只要 next_cell_id 有效且在范围内，就保留该 Source
            # 但是，如果 Target 所在的行本身因为某些原因（比如它自己没有 next）被丢弃了怎么办？
            # 在 Flow Matching 中，Target 只需要作为 x1 存在即可，不需要 Target 也有 next。
            # 所以只要 Target 在原始 adata 中存在且由我们读取进来了，就可以。
            # 但是，align_genes 之后 adata 还是完整的。
            
            # 这里的隐患是：如果我们后续还做了其他过滤（目前没有），或者 target 索引超出了 len(adata)。
            # 目前 valid_mask_source 保证了 target_idx < len(adata)。
            
            # 唯一的问题是：如果我们在下面做了 `adata_valid = adata[valid_mask]`
            # 那么 `adata_valid` 的行号变了。
            # raw_next_indices 指向的是旧行号。
            
            # 修正方案：
            # Target 必须也在 adata_valid 中吗？
            # 不一定。Target 只需要有表达量数据 x_next。
            # 只要 adata_valid 保留了所有作为 Source 的细胞，
            # 而 x_next 我们可以在构建 Dataset 时直接从原始 adata 获取吗？
            # 不行，SingleCellDataset只接收一个 adata。
            
            # 所以，通常的做法是：
            # 1. 找出所有涉及到的细胞（作为 Source 或 Target）。
            # 2. 构建一个新的 adata，包含所有这些细胞。
            # 3. 重映射索引。
            
            # 为了简化，假设所有细胞都在 adata 中。
            # 如果我们只取 valid_mask 的子集，那么 Target 细胞必须也包含在这个子集中，
            # 否则 x_next 就找不到了（因为 Dataset 只存了 subset 的 X）。
            
            # 让我们统计一下，有多少 Source 指向的 Target 并不在 valid_mask 中？
            # 如果 Target 自己不是有效的 Source（即它没有 next），它就会被 valid_mask 排除。
            # 这样会导致丢失很多数据（链条断裂）。
            
            # 正确的做法：
            # 不要根据 valid_mask 物理切分 adata (不要 `adata[valid_mask]`)。
            # 而是保留完整 adata，但在 Dataset __getitem__ 中只对 valid_indices 进行采样。
            
            # 修改策略：
            # 1. 保留完整 adata (仅做基因对齐)。
            # 2. 在 adata.obs 中标记 'is_valid_source'。
            # 3. 将正确的 next_cell_idx (相对于完整 adata) 存入 obs。
            # 4. 在 SingleCellDataset 中，只对 is_valid_source 的索引进行迭代。
            
            if valid_mask_source.sum() < 10:
                logger.warning(f"File {Path(fpath).name} has too few valid transitions ({valid_mask_source.sum()}), skipping.")
                continue
                
            # 不再切分 adata，而是标记
            # adata_valid = adata[valid_mask].copy() 
            
            # 存入校正后的索引 (其实就是 raw，因为我们不切分了)
            adata.obs['next_cell_idx'] = raw_next_indices
            adata.obs['is_valid_transition'] = valid_mask_source
            
            # Record filename for tracking split
            adata.uns['filename'] = Path(fpath).name
            
            # 收集 Labels
            # 注意：这里我们收集所有细胞的 label，还是只收集 valid source 的？
            # 为了安全，收集 valid source 的，因为只有它们会参与训练 conditioning
            valid_subset = adata[valid_mask_source]
            all_tissues.update(valid_subset.obs['Tissue'].astype(str).unique())
            all_celltypes.update(valid_subset.obs['Celltype'].astype(str).unique())
            
            temp_adatas.append(adata)
            
        except Exception as e:
            logger.error(f"Failed to load {fpath}: {e}")
            
    if not temp_adatas:
        raise ValueError("没有成功加载任何数据文件。")

    # 2. 建立全局映射
    # 如果外部提供了训练时的 mapping（用于评估/复现），就直接复用它；
    # 否则根据当前加载数据动态构建。
    if tissue_map is None:
        tissue_map = {t: i for i, t in enumerate(sorted(list(all_tissues)))}
    if celltype_map is None:
        celltype_map = {c: i for i, c in enumerate(sorted(list(all_celltypes)))}
    
    logger.info(f"Total Tissues: {len(tissue_map)}, Total Celltypes: {len(celltype_map)}")
    
    # 3. 最终处理：编码 Labels 和 时间
    for adata in temp_adatas:
        # Encode Labels
        adata.obs['tissue_code'] = adata.obs['Tissue'].astype(str).map(tissue_map).fillna(0).astype(int)
        adata.obs['celltype_code'] = adata.obs['Celltype'].astype(str).map(celltype_map).fillna(0).astype(int)
        
        # Unify Time
        unified_t = unify_time(adata, CONFIG['gestation_period'])
        adata.obs['unified_time'] = unified_t / 1000.0
        
        # 确保 Dense (align_genes 已经做了，但为了保险)
        if scipy.sparse.issparse(adata.X):
            adata.X = adata.X.toarray()
            
        processed_adatas.append(adata)
        
    return processed_adatas, tissue_map, celltype_map

class SingleCellDataset(Dataset):
    def __init__(self, adata):
        self.X = torch.tensor(adata.X, dtype=torch.float32)
        
        # 加载 Mask (如果存在)
        if 'gene_mask' in adata.uns:
             self.gene_mask = torch.tensor(adata.uns['gene_mask'], dtype=torch.bool)
        else:
             self.gene_mask = torch.ones(self.X.shape[1], dtype=torch.bool)

        # 获取有效样本的掩码
        if 'is_valid_transition' in adata.obs.columns:
            self.valid_indices = np.where(adata.obs['is_valid_transition'].values)[0]
        else:
            self.valid_indices = np.arange(len(self.X))
            
        # 即使只训练 valid 部分，我们也需要完整的 metadata 数组以便通过索引访问
        self.next_idxs = adata.obs['next_cell_idx'].values
        self.times = torch.tensor(adata.obs['unified_time'].values, dtype=torch.float32)
        self.tissues = torch.tensor(adata.obs['tissue_code'].values, dtype=torch.long)
        self.celltypes = torch.tensor(adata.obs['celltype_code'].values, dtype=torch.long)
        
    def __len__(self):
        return len(self.valid_indices)
    
    def __getitem__(self, idx):
        # 映射到真实的全局索引
        curr_idx = self.valid_indices[idx]
        
        x_curr = self.X[curr_idx]
        
        # 获取 Target 索引 (相对于完整 X)
        next_i = self.next_idxs[curr_idx]
        x_next = self.X[next_i]
        
        t_curr = self.times[curr_idx]
        t_next = self.times[next_i]
        dt = t_next - t_curr
        
        return {
            "x_curr": x_curr,
            "x_next": x_next,
            "cond_meta": {
                "time": t_curr,
                "dt": dt,
                "tissue": self.tissues[curr_idx],
                "celltype": self.celltypes[curr_idx],
                "gene_mask": self.gene_mask # Pass mask
            }
        }

# ==========================================
# 4. 主程序
# ==========================================
def main():
    # 1. Load Gene List
    gene_list_path = "data/assets/gene_order.tsv"
    try:
        target_genes = load_gene_list(gene_list_path)
    except Exception:
        logger.error("Could not load gene list. Exiting.")
        return

    # 2. Load Data
    adatas, tissue_map, celltype_map = load_data_files(
        CONFIG['data_dir'], 
        target_genes=target_genes,
        n_files=CONFIG['n_files_to_load']
    )

    # 保存映射，保证 eval 时可复现训练时的类别编码
    os.makedirs(CONFIG['output_dir'], exist_ok=True)
    with open(os.path.join(CONFIG['output_dir'], "tissue_map.json"), "w") as f:
        json.dump(tissue_map, f, indent=2, sort_keys=True)
    with open(os.path.join(CONFIG['output_dir'], "celltype_map.json"), "w") as f:
        json.dump(celltype_map, f, indent=2, sort_keys=True)
    logger.info(f"Saved tissue/celltype maps to {CONFIG['output_dir']}")
    
    if not adatas:
        logger.error("No valid data loaded.")
        return

    # Split Train/Test by CELLS (Random Split)
    # 之前按文件划分容易导致 OOD (Out of Distribution) 问题
    full_ds = ConcatDataset([SingleCellDataset(a) for a in adatas])
    
    total_size = len(full_ds)
    train_size = int(total_size * CONFIG['train_ratio'])
    val_size = total_size - train_size
    
    # Random split
    generator = torch.Generator().manual_seed(CONFIG['seed'])
    train_ds, val_ds = torch.utils.data.random_split(full_ds, [train_size, val_size], generator=generator)
    
    logger.info(f"Total Cells: {total_size} | Train: {train_size} | Val: {val_size}")

    train_loader = DataLoader(train_ds, batch_size=CONFIG['batch_size'], shuffle=True, num_workers=4, persistent_workers=True)
    val_loader = DataLoader(val_ds, batch_size=CONFIG['batch_size'], shuffle=False, num_workers=4, persistent_workers=True)
        
    input_dim = len(target_genes)
    logger.info(f"Input Dimension (Genes): {input_dim}")
    
    # 3. Init Lightning Module
    model = FlowMatchingLightning(
        input_dim=input_dim,
        config=CONFIG,
        tissue_map=tissue_map,
        celltype_map=celltype_map
    )
    
    # 4. Init Trainer
    # Ensure output directory exists
    os.makedirs(CONFIG['output_dir'], exist_ok=True)
    
    logger_csv = CSVLogger(CONFIG['output_dir'], name="dit_flow_logs")
    checkpoint_callback = ModelCheckpoint(
        dirpath=os.path.join(CONFIG['output_dir'], "checkpoints"),
        filename="dit_flow-{epoch:02d}-{val_loss:.2f}",
        save_top_k=3,
        monitor="val_loss",
        mode="min",
        save_last=True
    )
    lr_monitor = LearningRateMonitor(logging_interval='step')
    
    trainer = pl.Trainer(
        default_root_dir=CONFIG['output_dir'],
        max_epochs=CONFIG['epochs'],
        accelerator=CONFIG['accelerator'],
        devices=CONFIG['devices'],
        logger=logger_csv,
        callbacks=[checkpoint_callback, lr_monitor],
        log_every_n_steps=10
    )
    
    logger.info("Start Training DiT Flow Matching with PyTorch Lightning...")
    
    # 5. Train
    if val_loader:
        trainer.fit(model, train_dataloaders=train_loader, val_dataloaders=val_loader)
    else:
        trainer.fit(model, train_dataloaders=train_loader)
    
    # 6. Save Final Model (Manual)
    final_path = os.path.join(CONFIG['output_dir'], "dit_flow_model_final.pt")
    torch.save(model.model.state_dict(), final_path)
    logger.info(f"Final model saved to {final_path}")

    # 强制保存 Lightning ckpt（即使 monitor 没触发也要落盘一个）
    ckpt_dir = os.path.join(CONFIG['output_dir'], "checkpoints")
    os.makedirs(ckpt_dir, exist_ok=True)
    last_ckpt_path = os.path.join(ckpt_dir, "last.ckpt")
    trainer.save_checkpoint(last_ckpt_path)
    logger.info(f"Lightning checkpoint saved to {last_ckpt_path}")

if __name__ == "__main__":
    main()