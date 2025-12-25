#!/usr/bin/env python
"""
Embryo数据潜空间提取与UMAP可视化脚本

功能：
1. 加载训练好的 AE 模型
2. 读取 h5ad 格式的单细胞数据
3. 将基因对齐到训练数据的基因空间
4. 通过 AE 编码到 Latent Space
5. 计算 UMAP 并生成可视化图片
6. 保存潜空间表示到 h5ad 文件

使用方法：
python scripts/embryo/extract_latent_umap.py \
    --ckpt_path logs/ae_stage1/runs/2025-12-17_06-48-59-ae_large/checkpoints/last.ckpt \
    --input_h5ad /gpfs/flash/home/jcw/projects/research/Embryo/data/hue11-99457A.h5ad \
    --output_dir outputs/embryo \
    --batch_size 2048
"""

import os
import sys
import argparse
from pathlib import Path

# 添加项目根目录到 Python 路径
PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

import torch
import numpy as np
import pandas as pd
import scanpy as sc
import anndata as ad
import matplotlib.pyplot as plt
import tiledbsoma
from tqdm import tqdm
from omegaconf import OmegaConf
from hydra.utils import instantiate

from src.models.ae_module import AELitModule


def get_training_genes(data_dir: str) -> list:
    """
    从训练数据目录获取基因列表

    Args:
        data_dir: TileDB 数据目录

    Returns:
        genes: 基因列表（按顺序）
    """
    shards = sorted([d for d in os.listdir(data_dir) if os.path.isdir(os.path.join(data_dir, d))])
    if len(shards) == 0:
        raise ValueError(f"No shards found in {data_dir}")

    shard_uri = os.path.join(data_dir, shards[0])
    ctx = tiledbsoma.SOMATileDBContext()
    with tiledbsoma.Experiment.open(shard_uri, context=ctx) as exp:
        var = exp.ms['RNA'].var.read().concat().to_pandas()
        genes = list(var['var_id'])

    return genes


def align_genes_to_training(adata: ad.AnnData, training_genes: list) -> np.ndarray:
    """
    将 AnnData 的基因对齐到训练数据的基因空间

    对于训练数据中存在但输入数据中不存在的基因，填充 0

    Args:
        adata: 输入的 AnnData 对象
        training_genes: 训练数据的基因列表（按顺序）

    Returns:
        X_aligned: 对齐后的表达矩阵 (n_cells, n_training_genes)
    """
    input_genes = set(adata.var_names)
    n_cells = adata.shape[0]
    n_training_genes = len(training_genes)

    # 获取输入数据的表达矩阵
    X = adata.X
    if hasattr(X, 'toarray'):
        X = X.toarray()

    # 创建输入基因到索引的映射
    input_gene_to_idx = {gene: i for i, gene in enumerate(adata.var_names)}

    # 创建对齐后的矩阵
    X_aligned = np.zeros((n_cells, n_training_genes), dtype=np.float32)

    # 统计
    matched_genes = 0
    missing_genes = 0

    for i, gene in enumerate(training_genes):
        if gene in input_gene_to_idx:
            X_aligned[:, i] = X[:, input_gene_to_idx[gene]]
            matched_genes += 1
        else:
            # 基因不存在，保持为 0
            missing_genes += 1

    print(f"Gene alignment: {matched_genes}/{n_training_genes} matched, {missing_genes} missing (filled with 0)")

    return X_aligned


def load_ae_model(ckpt_path: str, device: torch.device):
    """加载 AE 模型"""
    print(f"Loading AE model from: {ckpt_path}")

    # 找到对应的 hydra config
    ckpt_dir = Path(ckpt_path).parent.parent
    config_path = ckpt_dir / ".hydra" / "config.yaml"

    if config_path.exists():
        print(f"Loading config from: {config_path}")
        cfg = OmegaConf.load(config_path)
        # 实例化网络
        net = instantiate(cfg.model.net)
        ae_model = AELitModule.load_from_checkpoint(
            ckpt_path,
            net=net,
            optimizer=None,
            scheduler=None,
            map_location=device
        )
        # 返回 config 以获取 data_dir
        return ae_model, cfg
    else:
        # fallback: 直接加载（可能会失败）
        print(f"Warning: Config not found at {config_path}, trying direct load...")
        ae_model = AELitModule.load_from_checkpoint(ckpt_path, map_location=device)
        return ae_model, None


def extract_latents(
    X: np.ndarray,
    encoder,
    device: torch.device,
    batch_size: int = 2048,
) -> np.ndarray:
    """
    从表达矩阵提取潜空间表示

    Args:
        X: 表达矩阵 (n_cells, n_genes)
        encoder: AE 编码器
        device: 计算设备
        batch_size: 批次大小

    Returns:
        latents: (n_cells, latent_dim) 的 numpy 数组
    """
    n_cells = X.shape[0]
    n_batches = (n_cells + batch_size - 1) // batch_size

    print(f"Extracting latents for {n_cells} cells...")

    latents = []

    for i in tqdm(range(n_batches), desc="Encoding"):
        start = i * batch_size
        end = min(start + batch_size, n_cells)

        # 转换为 Tensor
        X_batch = torch.from_numpy(X[start:end]).float().to(device)

        # 编码
        with torch.no_grad():
            z_batch = encoder(X_batch)

        latents.append(z_batch.cpu().numpy())

    # 合并所有 batch
    latents = np.concatenate(latents, axis=0)
    print(f"Latent shape: {latents.shape}")

    return latents


def compute_umap(adata: ad.AnnData, use_rep: str = 'X_latent', n_neighbors: int = 30):
    """计算 UMAP"""
    print(f"Computing neighbors using {use_rep}...")
    sc.pp.neighbors(adata, use_rep=use_rep, n_neighbors=n_neighbors)

    print("Computing UMAP...")
    sc.tl.umap(adata)

    return adata


def plot_umap_comparison(
    adata: ad.AnnData,
    output_path: str,
    color_by: list = None,
    figsize: tuple = (8, 6),
    dpi: int = 150
):
    """
    绘制真实数据 vs 模型潜空间的 UMAP 对比图

    每个 color_by 列生成一张图，包含两个子图：
    - 左：基于原始数据 PCA 的 UMAP (X_umap_raw)
    - 右：基于模型潜空间的 UMAP (X_umap_latent)

    Args:
        adata: AnnData 对象（需要已计算两种 UMAP）
        output_path: 输出路径（不含扩展名）
        color_by: 用于着色的列名列表
        figsize: 单个子图大小
        dpi: 图片分辨率
    """
    if color_by is None:
        # 尝试自动检测可用的着色列
        available_cols = []
        for col in ['cell_type', 'celltype', 'leiden', 'louvain', 'batch', 'sample', 'day', 'time', 'stage', 'Type', 'Types']:
            if col in adata.obs.columns:
                available_cols.append(col)

        if len(available_cols) == 0:
            color_by = [None]
        else:
            color_by = available_cols[:4]  # 最多4个

    print(f"Plotting UMAP comparison for: {color_by}")

    output_dir = os.path.dirname(output_path)
    base_name = os.path.basename(output_path).replace('.png', '')

    for col in color_by:
        fig, axes = plt.subplots(1, 2, figsize=(figsize[0] * 2, figsize[1]))

        # 左图：原始数据 UMAP
        adata.obsm['X_umap'] = adata.obsm['X_umap_raw']
        sc.pl.umap(adata, color=col, ax=axes[0], show=False, frameon=False, title=f'Raw Data UMAP - {col}')

        # 右图：模型潜空间 UMAP
        adata.obsm['X_umap'] = adata.obsm['X_umap_latent']
        sc.pl.umap(adata, color=col, ax=axes[1], show=False, frameon=False, title=f'Latent Space UMAP - {col}')

        plt.tight_layout()

        col_name = col if col else 'default'
        save_path = os.path.join(output_dir, f"{base_name}_{col_name}.png")
        plt.savefig(save_path, dpi=dpi, bbox_inches='tight')
        plt.close()

        print(f"  Saved: {save_path}")


def main():
    parser = argparse.ArgumentParser(description="Embryo数据潜空间提取与UMAP可视化")
    parser.add_argument("--ckpt_path", type=str, required=True, help="AE 模型权重路径")
    parser.add_argument("--input_h5ad", type=str, required=True, help="输入 h5ad 文件路径")
    parser.add_argument("--output_dir", type=str, default="outputs/embryo", help="输出目录")
    parser.add_argument("--batch_size", type=int, default=2048, help="编码批次大小")
    parser.add_argument("--device", type=str, default=None, help="计算设备 (cuda:0, cpu, auto)")
    parser.add_argument("--n_neighbors", type=int, default=30, help="UMAP 邻居数")
    parser.add_argument("--color_by", type=str, nargs='+', default=None, help="UMAP 着色列")
    parser.add_argument("--save_latent", action="store_true", default=True, help="保存潜空间到 h5ad")
    parser.add_argument("--training_data_dir", type=str, default=None,
                        help="训练数据目录（用于基因对齐），如不指定则从 config 读取")

    args = parser.parse_args()

    # 1. 检查输入文件
    if not os.path.exists(args.input_h5ad):
        raise ValueError(f"Input file not found: {args.input_h5ad}")

    # 2. 创建输出目录
    os.makedirs(args.output_dir, exist_ok=True)
    print(f"Output directory: {args.output_dir}")

    # 3. 设置设备
    if args.device is None or args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)
    print(f"Using device: {device}")

    # 4. 加载 AE 模型
    ae_model, cfg = load_ae_model(args.ckpt_path, device)
    ae_model.eval()
    ae_model.to(device)
    encoder = ae_model.net.encode
    print(f"Model loaded on {device}")

    # 5. 获取训练数据的基因列表
    if args.training_data_dir:
        training_data_dir = args.training_data_dir
    elif cfg is not None and 'data' in cfg and 'data_dir' in cfg.data:
        training_data_dir = cfg.data.data_dir
    else:
        training_data_dir = "/fast/data/scTFM/ae/tiledb_all/"

    print(f"Loading training gene list from: {training_data_dir}")
    training_genes = get_training_genes(training_data_dir)
    print(f"Training genes: {len(training_genes)}")

    # 6. 加载 h5ad 数据
    print(f"Loading h5ad: {args.input_h5ad}")
    adata = sc.read_h5ad(args.input_h5ad)
    print(f"Loaded: {adata.shape[0]} cells, {adata.shape[1]} genes")
    print(f"Obs columns: {list(adata.obs.columns)}")

    # 7. 基因对齐
    print("Aligning genes to training data...")
    X_aligned = align_genes_to_training(adata, training_genes)

    # 8. 提取潜空间
    latents = extract_latents(
        X=X_aligned,
        encoder=encoder,
        device=device,
        batch_size=args.batch_size,
    )

    # 9. 存储潜空间到 adata
    adata.obsm['X_latent'] = latents

    # 10. 计算原始数据的 UMAP（基于 PCA）
    print("\n--- Computing Raw Data UMAP (PCA-based) ---")
    # 先做 PCA
    sc.pp.pca(adata, n_comps=50)
    sc.pp.neighbors(adata, use_rep='X_pca', n_neighbors=args.n_neighbors)
    sc.tl.umap(adata)
    # 保存原始数据的 UMAP
    adata.obsm['X_umap_raw'] = adata.obsm['X_umap'].copy()

    # 11. 计算模型潜空间的 UMAP
    print("\n--- Computing Latent Space UMAP ---")
    sc.pp.neighbors(adata, use_rep='X_latent', n_neighbors=args.n_neighbors)
    sc.tl.umap(adata)
    # 保存潜空间的 UMAP
    adata.obsm['X_umap_latent'] = adata.obsm['X_umap'].copy()

    # 12. 绘制对比 UMAP（每个 color_by 一张图，左右对比）
    input_name = Path(args.input_h5ad).stem
    umap_path = os.path.join(args.output_dir, f"{input_name}_umap.png")
    plot_umap_comparison(adata, umap_path, color_by=args.color_by)

    # 13. 保存带有潜空间的 h5ad
    if args.save_latent:
        output_h5ad = os.path.join(args.output_dir, f"{input_name}_with_latent.h5ad")
        print(f"Saving h5ad with latent to: {output_h5ad}")
        adata.write_h5ad(output_h5ad)

    # 14. 总结
    print("\n" + "=" * 50)
    print("Done!")
    print(f"  - Latent dim: {latents.shape[1]}")
    print(f"  - UMAP comparison saved to: {args.output_dir}")
    if args.save_latent:
        print(f"  - H5AD saved: {output_h5ad}")
    print("=" * 50)


if __name__ == "__main__":
    main()
