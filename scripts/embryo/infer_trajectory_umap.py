#!/usr/bin/env python
"""
Embryo 发育轨迹推断与 UMAP 可视化

功能：
1. 加载 AE 模型提取 Embryo 数据的潜空间
2. 以最早期的 CS 阶段细胞作为起点
3. 使用 RTF 模型逐步推断后续 CS 阶段
4. 生成两张对比图：
   - 真实数据的 UMAP（按 CS 阶段着色）
   - 模型推断的 UMAP（按 CS 阶段着色）

使用方法：
python scripts/embryo/infer_trajectory_umap.py \
    --ae_ckpt logs/ae_stage1/runs/2025-12-17_06-48-59-ae_large/checkpoints/last.ckpt \
    --rtf_ckpt logs/rtf_train_cfg/runs/2025-12-18_17-50-03_cfg1.0/checkpoints/last.ckpt \
    --input_h5ad /gpfs/flash/home/jcw/projects/research/Embryo/data/hue11-99457A.h5ad \
    --output_dir outputs/embryo_trajectory \
    --delta_t 2.0
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
from src.models.flow_module import FlowLitModule


def get_training_genes(data_dir: str) -> list:
    """从训练数据目录获取基因列表"""
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
    """将 AnnData 的基因对齐到训练数据的基因空间"""
    n_cells = adata.shape[0]
    n_training_genes = len(training_genes)

    X = adata.X
    if hasattr(X, 'toarray'):
        X = X.toarray()

    input_gene_to_idx = {gene: i for i, gene in enumerate(adata.var_names)}
    X_aligned = np.zeros((n_cells, n_training_genes), dtype=np.float32)

    matched_genes = 0
    for i, gene in enumerate(training_genes):
        if gene in input_gene_to_idx:
            X_aligned[:, i] = X[:, input_gene_to_idx[gene]]
            matched_genes += 1

    print(f"Gene alignment: {matched_genes}/{n_training_genes} matched")
    return X_aligned


def load_ae_model(ckpt_path: str, device: torch.device):
    """加载 AE 模型"""
    print(f"Loading AE model from: {ckpt_path}")
    ckpt_dir = Path(ckpt_path).parent.parent
    config_path = ckpt_dir / ".hydra" / "config.yaml"

    if config_path.exists():
        cfg = OmegaConf.load(config_path)
        net = instantiate(cfg.model.net)
        ae_model = AELitModule.load_from_checkpoint(
            ckpt_path, net=net, optimizer=None, scheduler=None, map_location=device
        )
        return ae_model, cfg
    else:
        raise ValueError(f"Config not found: {config_path}")


def load_rtf_model(ckpt_path: str, device: torch.device):
    """加载 RTF 模型"""
    print(f"Loading RTF model from: {ckpt_path}")
    ckpt_dir = Path(ckpt_path).parent.parent
    config_path = ckpt_dir / ".hydra" / "config.yaml"

    if config_path.exists():
        cfg = OmegaConf.load(config_path)
        net = instantiate(cfg.model.net)
        rtf_model = FlowLitModule.load_from_checkpoint(
            ckpt_path, net=net, optimizer=None, scheduler=None, map_location=device
        )
        return rtf_model, cfg
    else:
        raise ValueError(f"Config not found: {config_path}")


def extract_latents(X: np.ndarray, encoder, device: torch.device, batch_size: int = 2048) -> np.ndarray:
    """提取潜空间"""
    n_cells = X.shape[0]
    n_batches = (n_cells + batch_size - 1) // batch_size
    latents = []

    for i in tqdm(range(n_batches), desc="Encoding"):
        start = i * batch_size
        end = min(start + batch_size, n_cells)
        X_batch = torch.from_numpy(X[start:end]).float().to(device)
        with torch.no_grad():
            z_batch = encoder(X_batch)
        latents.append(z_batch.cpu().numpy())

    return np.concatenate(latents, axis=0)


def infer_next_stage(
    rtf_model,
    x_curr: np.ndarray,
    delta_t: float,
    device: torch.device,
    batch_size: int = 2048,
    steps: int = 50,
    cfg_scale: float = 1.0
) -> np.ndarray:
    """
    使用 RTF 模型推断下一个阶段的细胞状态

    Args:
        rtf_model: RTF 模型
        x_curr: 当前阶段的 latent (n_cells, latent_dim)
        delta_t: 时间间隔
        device: 设备
        batch_size: 批次大小
        steps: ODE 求解步数
        cfg_scale: CFG 强度

    Returns:
        x_next: 下一个阶段的 latent (n_cells, latent_dim)
    """
    n_cells = x_curr.shape[0]
    n_batches = (n_cells + batch_size - 1) // batch_size
    x_next_list = []

    for i in tqdm(range(n_batches), desc="Inferring"):
        start = i * batch_size
        end = min(start + batch_size, n_cells)
        batch_size_curr = end - start

        x_batch = torch.from_numpy(x_curr[start:end]).float().to(device)

        # 准备条件数据
        cond_data = {
            'x_curr': x_batch,
            'delta_t': torch.tensor([delta_t] * batch_size_curr).float().to(device),
        }

        # 从噪声开始采样
        x0 = torch.randn_like(x_batch)

        with torch.no_grad():
            x_next_batch = rtf_model.flow.sample(x0, cond_data, steps=steps, cfg_scale=cfg_scale)

        x_next_list.append(x_next_batch.cpu().numpy())

    return np.concatenate(x_next_list, axis=0)


def main():
    parser = argparse.ArgumentParser(description="Embryo 发育轨迹推断")
    parser.add_argument("--ae_ckpt", type=str, required=True, help="AE 模型路径")
    parser.add_argument("--rtf_ckpt", type=str, required=True, help="RTF 模型路径")
    parser.add_argument("--input_h5ad", type=str, required=True, help="输入 h5ad 文件")
    parser.add_argument("--output_dir", type=str, default="outputs/embryo_trajectory", help="输出目录")
    parser.add_argument("--delta_t", type=float, default=2.0, help="时间间隔（每个 CS 阶段）")
    parser.add_argument("--batch_size", type=int, default=2048, help="批次大小")
    parser.add_argument("--device", type=str, default="cuda:0", help="设备")
    parser.add_argument("--steps", type=int, default=50, help="ODE 求解步数")
    parser.add_argument("--cfg_scale", type=float, default=1.0, help="CFG 强度")
    parser.add_argument("--training_data_dir", type=str, default="/fast/data/scTFM/ae/tiledb_all/",
                        help="训练数据目录（用于基因对齐）")

    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    device = torch.device(args.device)

    # 1. 加载模型
    ae_model, ae_cfg = load_ae_model(args.ae_ckpt, device)
    ae_model.eval().to(device)
    encoder = ae_model.net.encode
    decoder = ae_model.net.decode

    rtf_model, rtf_cfg = load_rtf_model(args.rtf_ckpt, device)
    rtf_model.eval().to(device)

    # 2. 加载数据
    print(f"\nLoading h5ad: {args.input_h5ad}")
    adata = sc.read_h5ad(args.input_h5ad)
    print(f"Loaded: {adata.shape[0]} cells, {adata.shape[1]} genes")

    # 获取 stage 信息
    if 'stage' not in adata.obs.columns:
        raise ValueError("'stage' column not found in adata.obs")

    stages = sorted(adata.obs['stage'].unique())
    print(f"Found stages: {stages}")

    # 3. 基因对齐
    print("\nAligning genes...")
    training_genes = get_training_genes(args.training_data_dir)
    X_aligned = align_genes_to_training(adata, training_genes)

    # 4. 提取所有细胞的 latent
    print("\nExtracting latents for all cells...")
    latents_real = extract_latents(X_aligned, encoder, device, args.batch_size)
    adata.obsm['X_latent'] = latents_real

    # 5. 按 stage 分组
    stage_to_indices = {}
    for stage in stages:
        stage_to_indices[stage] = np.where(adata.obs['stage'] == stage)[0]
        print(f"  {stage}: {len(stage_to_indices[stage])} cells")

    # 6. 轨迹推断：从最早期开始，逐步推断
    print(f"\n--- Trajectory Inference (delta_t={args.delta_t}) ---")

    # 初始化：使用第一个阶段的真实 latent
    first_stage = stages[0]
    latents_inferred = np.zeros_like(latents_real)

    # 第一个阶段使用真实值
    latents_inferred[stage_to_indices[first_stage]] = latents_real[stage_to_indices[first_stage]]
    print(f"Stage {first_stage}: using real data ({len(stage_to_indices[first_stage])} cells)")

    # 逐步推断后续阶段
    for i in range(1, len(stages)):
        prev_stage = stages[i - 1]
        curr_stage = stages[i]

        prev_indices = stage_to_indices[prev_stage]
        curr_indices = stage_to_indices[curr_stage]

        print(f"\nInferring {curr_stage} from {prev_stage}...")

        # 获取前一阶段的 latent（使用推断值）
        x_prev = latents_inferred[prev_indices]

        # 推断当前阶段
        # 注意：这里我们随机采样前一阶段的细胞来推断当前阶段
        # 因为细胞数量可能不同，我们采样相同数量
        n_curr = len(curr_indices)
        if len(x_prev) >= n_curr:
            # 从前一阶段随机采样
            sample_idx = np.random.choice(len(x_prev), n_curr, replace=False)
            x_prev_sampled = x_prev[sample_idx]
        else:
            # 前一阶段细胞数不够，允许重复采样
            sample_idx = np.random.choice(len(x_prev), n_curr, replace=True)
            x_prev_sampled = x_prev[sample_idx]

        x_inferred = infer_next_stage(
            rtf_model, x_prev_sampled, args.delta_t, device,
            args.batch_size, args.steps, args.cfg_scale
        )

        latents_inferred[curr_indices] = x_inferred
        print(f"  {curr_stage}: inferred {len(curr_indices)} cells")

    # 保存推断的 latent
    adata.obsm['X_latent_inferred'] = latents_inferred

    # 7. 计算 UMAP
    print("\n--- Computing UMAPs ---")

    # 真实数据 UMAP
    print("Computing UMAP for real latents...")
    sc.pp.neighbors(adata, use_rep='X_latent', n_neighbors=30)
    sc.tl.umap(adata)
    adata.obsm['X_umap_real'] = adata.obsm['X_umap'].copy()

    # 推断数据 UMAP
    print("Computing UMAP for inferred latents...")
    sc.pp.neighbors(adata, use_rep='X_latent_inferred', n_neighbors=30)
    sc.tl.umap(adata)
    adata.obsm['X_umap_inferred'] = adata.obsm['X_umap'].copy()

    # 8. 绘制对比图
    print("\n--- Plotting ---")
    input_name = Path(args.input_h5ad).stem

    # 绘制按 stage 着色的对比图
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    # 左图：真实数据
    adata.obsm['X_umap'] = adata.obsm['X_umap_real']
    sc.pl.umap(adata, color='stage', ax=axes[0], show=False, frameon=False,
               title='Real Data (AE Latent)')

    # 右图：推断数据
    adata.obsm['X_umap'] = adata.obsm['X_umap_inferred']
    sc.pl.umap(adata, color='stage', ax=axes[1], show=False, frameon=False,
               title=f'Inferred Data (RTF, delta_t={args.delta_t})')

    plt.tight_layout()
    save_path = os.path.join(args.output_dir, f"{input_name}_trajectory_stage.png")
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {save_path}")

    # 绘制按 Type 着色的对比图
    if 'Type' in adata.obs.columns:
        fig, axes = plt.subplots(1, 2, figsize=(16, 6))

        adata.obsm['X_umap'] = adata.obsm['X_umap_real']
        sc.pl.umap(adata, color='Type', ax=axes[0], show=False, frameon=False,
                   title='Real Data (AE Latent)', legend_loc='on data', legend_fontsize=4)

        adata.obsm['X_umap'] = adata.obsm['X_umap_inferred']
        sc.pl.umap(adata, color='Type', ax=axes[1], show=False, frameon=False,
                   title=f'Inferred Data (RTF, delta_t={args.delta_t})', legend_loc='on data', legend_fontsize=4)

        plt.tight_layout()
        save_path = os.path.join(args.output_dir, f"{input_name}_trajectory_type.png")
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"Saved: {save_path}")

    # 9. 保存带有推断结果的 h5ad
    output_h5ad = os.path.join(args.output_dir, f"{input_name}_with_inference.h5ad")
    print(f"\nSaving h5ad to: {output_h5ad}")
    adata.write_h5ad(output_h5ad)

    print("\n" + "=" * 50)
    print("Done!")
    print(f"  - Output dir: {args.output_dir}")
    print("=" * 50)


if __name__ == "__main__":
    main()
