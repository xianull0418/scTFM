"""
RTF (Rectified Flow) 模型批量测评脚本

功能：
1. 自动扫描指定目录下的 RTF 训练运行
2. 加载模型并在测试集上评估
3. 计算指标：MSE, Pearson Correlation
4. 生成可视化：
   - Correlation 分布图
   - PCA 轨迹图
   - UMAP 对比图（Ground Truth vs Prediction）
5. 上传到 W&B

使用方式：
    python bench/benchmark_rtf.py --dir logs/rtf_stage2 --wandb_project scTime-RTF-bench
"""

import argparse
import os
import sys
import glob
import json
import yaml
import torch
import torch.nn.functional as F
import tiledbsoma
import numpy as np
import pandas as pd
import scanpy as sc
import matplotlib.pyplot as plt
import seaborn as sns
import wandb
from omegaconf import OmegaConf
from hydra.utils import instantiate
from tqdm import tqdm
from torch.utils.data import DataLoader
from sklearn.decomposition import PCA
import logging
import umap

# 确保项目根目录在路径中
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../"))
sys.path.append(project_root)
os.environ["PROJECT_ROOT"] = project_root

from src.models.flow_module import FlowLitModule
from src.data.components.rtf_dataset import (
    SomaRTFDataset,
    normalize_time,
    normalize_delta_t,
    get_stage_map,
    MAX_TIME_DAYS,
)

# 设置绘图风格
sns.set_theme(style="whitegrid", context="paper", font_scale=1.2)
plt.rcParams['figure.dpi'] = 150
plt.rcParams['savefig.bbox'] = 'tight'
plt.rcParams['font.family'] = 'DejaVu Sans'
plt.rcParams['pdf.fonttype'] = 42
plt.rcParams['ps.fonttype'] = 42

# 配置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Register dummy hydra resolver
try:
    OmegaConf.register_new_resolver("hydra", lambda *args: "hydra_placeholder")
except Exception:
    pass


def find_runs(base_dir):
    """
    扫描目录寻找包含 .hydra/config.yaml 的运行目录。
    """
    runs = []
    if os.path.exists(os.path.join(base_dir, ".hydra", "config.yaml")):
        runs.append(base_dir)

    for root, dirs, files in os.walk(base_dir):
        for d in dirs:
            run_path = os.path.join(root, d)
            if os.path.exists(os.path.join(run_path, ".hydra", "config.yaml")):
                runs.append(run_path)
    return sorted(list(set(runs)))


def get_best_checkpoint(run_dir):
    """
    寻找最佳 checkpoint（优先找 val_loss 最低的，否则用最新的）。
    """
    ckpt_dir = os.path.join(run_dir, "checkpoints")
    if not os.path.exists(ckpt_dir):
        return None

    ckpts = glob.glob(os.path.join(ckpt_dir, "*.ckpt"))
    if not ckpts:
        return None

    # 优先选择包含 val_loss 的 checkpoint
    val_ckpts = [c for c in ckpts if "val_loss" in c]
    if val_ckpts:
        # 从文件名中提取 val_loss 值，选择最低的
        def extract_val_loss(path):
            try:
                # 格式: epoch_XXX-val_loss_X.XXXX.ckpt
                name = os.path.basename(path)
                loss_str = name.split("val_loss_")[-1].replace(".ckpt", "")
                # 处理可能的下划线替换点的情况
                loss_str = loss_str.replace("_", ".")
                return float(loss_str)
            except:
                return float('inf')

        val_ckpts.sort(key=extract_val_loss)
        return val_ckpts[0]

    # Fallback: 使用最新的
    ckpts.sort(key=os.path.getmtime)
    return ckpts[-1]


def compute_correlation_rowwise(x, y):
    """
    计算逐行 Pearson 相关系数。
    """
    x_mean = x - x.mean(dim=1, keepdim=True)
    y_mean = y - y.mean(dim=1, keepdim=True)

    x_norm = x_mean.norm(dim=1, keepdim=True)
    y_norm = y_mean.norm(dim=1, keepdim=True)

    x_norm = torch.clamp(x_norm, min=1e-8)
    y_norm = torch.clamp(y_norm, min=1e-8)

    correlation = (x_mean * y_mean).sum(dim=1, keepdim=True) / (x_norm * y_norm)
    return correlation.squeeze()


def setup_dataloader(data_dir, split_label=3, batch_size=256, num_workers=4, stage_info_path=None, use_log_time=True):
    """
    设置 RTF 测试 DataLoader。

    Split Labels:
    0: Train (ID)
    1: Val (ID)
    2: Test (ID)
    3: Test (OOD) - 整个 shard 作为 OOD，默认使用
    """
    logger.info(f"Setting up RTF dataloader from {data_dir}...")
    logger.info(f"Split label: {split_label} ({'OOD' if split_label == 3 else 'ID'})")

    # 预扫描 shards
    preloaded_sub_uris = sorted([
        os.path.join(data_dir, d)
        for d in os.listdir(data_dir)
        if os.path.isdir(os.path.join(data_dir, d))
    ])
    logger.info(f"Found {len(preloaded_sub_uris)} shards")

    dataset = SomaRTFDataset(
        root_dir=data_dir,
        split_label=split_label,
        io_chunk_size=8192,
        batch_size=batch_size,
        direction="forward",
        preloaded_sub_uris=preloaded_sub_uris,
        stage_info_path=stage_info_path,
        use_log_time=use_log_time,
    )

    loader = DataLoader(
        dataset,
        batch_size=None,  # Dataset 内部已经 batch
        num_workers=num_workers,
        pin_memory=True,
        persistent_workers=True if num_workers > 0 else False,
    )

    return loader


def sample_with_model(model, x_curr, cond_data, steps=50, method='euler', cfg_scale=1.0):
    """
    使用 RTF 模型采样预测下一时刻的细胞状态。

    Args:
        model: FlowLitModule
        x_curr: 当前细胞表达 [B, D]
        cond_data: 条件数据字典
        steps: ODE 求解步数
        method: 'euler' 或 'rk4'
        cfg_scale: Classifier-Free Guidance 强度
                   1.0 = 普通采样
                   > 1.0 = 增强条件影响（推荐 1.5 ~ 3.0）

    Returns:
        x_pred: 预测的下一时刻表达 [B, D]
    """
    # 从噪声开始采样
    x0 = torch.randn_like(x_curr)

    # 添加当前细胞作为条件
    cond_data['x_curr'] = x_curr

    # 调用 RectifiedFlow.sample（支持 CFG）
    x_pred = model.flow.sample(x0, cond_data, steps=steps, method=method, cfg_scale=cfg_scale)

    return x_pred


def evaluate_model(model, dataloader, device, sample_steps=50, max_batches=100, desc="Test", cfg_scale=1.0):
    """
    在给定 DataLoader 上评估 RTF 模型。
    """
    model.eval()

    all_corrs = []
    all_mse = []

    # 存储用于可视化的数据
    vis_data = {
        'x_curr': [],
        'x_next_true': [],
        'x_next_pred': [],
        'time_curr': [],
        'time_next': [],
        'delta_t': [],
        'stage': [],
    }

    max_vis_samples = 5000  # 最多保存多少样本用于可视化

    with torch.no_grad():
        batch_count = 0
        for batch in tqdm(dataloader, desc=f"Evaluating {desc}", total=max_batches):
            if batch_count >= max_batches:
                break
            batch_count += 1

            x_curr = batch['x_curr'].to(device)
            x_next_true = batch['x_next'].to(device)
            cond_meta = {k: v.to(device) if isinstance(v, torch.Tensor) else v
                        for k, v in batch['cond_meta'].items()}

            # 采样预测（支持 CFG）
            x_next_pred = sample_with_model(model, x_curr, cond_meta, steps=sample_steps, cfg_scale=cfg_scale)

            # 计算指标
            corr = compute_correlation_rowwise(x_next_true, x_next_pred)
            mse = F.mse_loss(x_next_pred, x_next_true, reduction='none').mean(dim=1)

            all_corrs.append(corr.cpu())
            all_mse.append(mse.cpu())

            # 保存可视化数据
            current_vis_count = sum(len(v) for v in vis_data['x_curr'])
            if current_vis_count < max_vis_samples:
                vis_data['x_curr'].append(x_curr.cpu().numpy())
                vis_data['x_next_true'].append(x_next_true.cpu().numpy())
                vis_data['x_next_pred'].append(x_next_pred.cpu().numpy())
                vis_data['time_curr'].append(cond_meta['time_curr'].cpu().numpy())
                vis_data['time_next'].append(cond_meta['time_next'].cpu().numpy())
                vis_data['delta_t'].append(cond_meta['delta_t'].cpu().numpy())
                vis_data['stage'].append(cond_meta['stage'].cpu().numpy())

    if len(all_corrs) == 0:
        logger.warning(f"No data found for {desc}")
        return None

    all_corrs = torch.cat(all_corrs).numpy()
    all_mse = torch.cat(all_mse).numpy()

    # 合并可视化数据
    for key in vis_data:
        if vis_data[key]:
            vis_data[key] = np.concatenate(vis_data[key], axis=0)[:max_vis_samples]

    return {
        'mse': float(np.mean(all_mse)),
        'mse_std': float(np.std(all_mse)),
        'corr': float(np.mean(all_corrs)),
        'corr_std': float(np.std(all_corrs)),
        'all_corrs': all_corrs,
        'all_mse': all_mse,
        'vis_data': vis_data,
    }


def plot_correlation_distribution(all_corrs, save_path=None):
    """绘制相关系数分布图"""
    plt.figure(figsize=(8, 6))
    sns.histplot(all_corrs, bins=50, kde=True, color='steelblue')
    plt.axvline(x=np.mean(all_corrs), color='red', linestyle='--',
                label=f'Mean: {np.mean(all_corrs):.3f}')
    plt.title("Distribution of Prediction Correlation")
    plt.xlabel("Pearson Correlation")
    plt.ylabel("Count")
    plt.legend()
    plt.grid(True, alpha=0.3)

    if save_path:
        plt.savefig(save_path)
        plt.close()
    return plt.gcf()


def plot_pca_trajectory(vis_data, save_path=None, n_samples=500):
    """绘制 PCA 空间中的轨迹预测图"""
    x_curr = vis_data['x_curr'][:n_samples]
    x_true = vis_data['x_next_true'][:n_samples]
    x_pred = vis_data['x_next_pred'][:n_samples]
    times = vis_data['time_curr'][:n_samples]

    # 合并所有数据进行 PCA
    all_data = np.vstack([x_curr, x_true, x_pred])
    pca = PCA(n_components=2)
    all_pca = pca.fit_transform(all_data)

    n = len(x_curr)
    pca_curr = all_pca[:n]
    pca_true = all_pca[n:2*n]
    pca_pred = all_pca[2*n:]

    fig, ax = plt.subplots(figsize=(12, 10))

    # 颜色映射
    norm_times = (times - times.min()) / (times.max() - times.min() + 1e-8)
    colors = plt.cm.viridis(norm_times)

    # 绘制
    for i in range(min(200, n)):  # 限制箭头数量
        # 起点
        ax.scatter(pca_curr[i, 0], pca_curr[i, 1], c=[colors[i]], s=40,
                  alpha=0.7, marker='o', edgecolors='black', linewidths=0.5, zorder=3)
        # 真实终点
        ax.scatter(pca_true[i, 0], pca_true[i, 1], c=[colors[i]], s=25,
                  alpha=0.4, marker='s', zorder=2)
        # 预测终点
        ax.scatter(pca_pred[i, 0], pca_pred[i, 1], c='red', s=30,
                  alpha=0.6, marker='^', edgecolors='darkred', linewidths=0.5, zorder=4)
        # 预测箭头
        ax.annotate('', xy=(pca_pred[i, 0], pca_pred[i, 1]),
                   xytext=(pca_curr[i, 0], pca_curr[i, 1]),
                   arrowprops=dict(arrowstyle='->', color='red', alpha=0.3, lw=0.8))

    # 图例
    from matplotlib.lines import Line2D
    legend_elements = [
        Line2D([0], [0], marker='o', color='w', markerfacecolor='gray',
               markeredgecolor='black', markersize=10, label='Start Cell (t)'),
        Line2D([0], [0], marker='s', color='w', markerfacecolor='gray',
               markersize=8, alpha=0.5, label='True Next (t+1)'),
        Line2D([0], [0], marker='^', color='w', markerfacecolor='red',
               markeredgecolor='darkred', markersize=10, label='Predicted (t+1)'),
    ]
    ax.legend(handles=legend_elements, loc='upper right', fontsize=10)

    # Colorbar
    sm = plt.cm.ScalarMappable(cmap='viridis', norm=plt.Normalize(vmin=times.min(), vmax=times.max()))
    sm.set_array([])
    cbar = plt.colorbar(sm, ax=ax, shrink=0.6)
    cbar.set_label('Time', fontsize=12)

    ax.set_xlabel('PC1', fontsize=12)
    ax.set_ylabel('PC2', fontsize=12)
    ax.set_title('Cell Trajectory Prediction in PCA Space', fontsize=14)
    ax.grid(True, linestyle='--', alpha=0.3)

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150)
        plt.close()
    return fig


def plot_umap_comparison(vis_data, save_path=None, n_samples=2000):
    """
    绘制 UMAP 对比图：左边 Ground Truth，右边 Prediction
    """
    x_curr = vis_data['x_curr'][:n_samples]
    x_true = vis_data['x_next_true'][:n_samples]
    x_pred = vis_data['x_next_pred'][:n_samples]
    times = vis_data['time_curr'][:n_samples]

    logger.info(f"Computing UMAP for {len(x_curr)} samples...")

    # 合并所有数据计算 UMAP（确保坐标一致）
    all_data = np.vstack([x_curr, x_true, x_pred])

    # 使用 UMAP 降维
    reducer = umap.UMAP(n_neighbors=30, min_dist=0.3, metric='euclidean', random_state=42)
    all_umap = reducer.fit_transform(all_data)

    n = len(x_curr)
    umap_curr = all_umap[:n]
    umap_true = all_umap[n:2*n]
    umap_pred = all_umap[2*n:]

    # 创建 2x1 子图
    fig, axes = plt.subplots(1, 2, figsize=(16, 7))

    # 归一化时间用于着色
    norm_times = (times - times.min()) / (times.max() - times.min() + 1e-8)

    # 左图：Ground Truth (Current -> True Next)
    ax = axes[0]
    # 绘制当前细胞
    scatter1 = ax.scatter(umap_curr[:, 0], umap_curr[:, 1], c=norm_times,
                         cmap='viridis', s=15, alpha=0.6, marker='o', label='Current (t)')
    # 绘制真实下一时刻
    ax.scatter(umap_true[:, 0], umap_true[:, 1], c=norm_times,
              cmap='viridis', s=15, alpha=0.6, marker='s', label='True Next (t+1)')

    # 绘制箭头（稀疏采样）
    arrow_idx = np.random.choice(n, min(200, n), replace=False)
    for i in arrow_idx:
        ax.annotate('', xy=(umap_true[i, 0], umap_true[i, 1]),
                   xytext=(umap_curr[i, 0], umap_curr[i, 1]),
                   arrowprops=dict(arrowstyle='->', color='blue', alpha=0.2, lw=0.5))

    ax.set_xlabel('UMAP1', fontsize=12)
    ax.set_ylabel('UMAP2', fontsize=12)
    ax.set_title('Ground Truth Trajectory', fontsize=14)
    ax.legend(loc='upper right', fontsize=9)
    ax.grid(True, linestyle='--', alpha=0.2)

    # 右图：Prediction (Current -> Predicted Next)
    ax = axes[1]
    # 绘制当前细胞
    ax.scatter(umap_curr[:, 0], umap_curr[:, 1], c=norm_times,
              cmap='viridis', s=15, alpha=0.6, marker='o', label='Current (t)')
    # 绘制预测下一时刻
    ax.scatter(umap_pred[:, 0], umap_pred[:, 1], c=norm_times,
              cmap='viridis', s=15, alpha=0.6, marker='^', label='Predicted Next (t+1)')

    # 绘制箭头
    for i in arrow_idx:
        ax.annotate('', xy=(umap_pred[i, 0], umap_pred[i, 1]),
                   xytext=(umap_curr[i, 0], umap_curr[i, 1]),
                   arrowprops=dict(arrowstyle='->', color='red', alpha=0.2, lw=0.5))

    ax.set_xlabel('UMAP1', fontsize=12)
    ax.set_ylabel('UMAP2', fontsize=12)
    ax.set_title('Model Prediction Trajectory', fontsize=14)
    ax.legend(loc='upper right', fontsize=9)
    ax.grid(True, linestyle='--', alpha=0.2)

    # 共享 colorbar
    cbar = fig.colorbar(scatter1, ax=axes, shrink=0.8, aspect=30)
    cbar.set_label('Time', fontsize=12)

    plt.suptitle('UMAP Trajectory Comparison: Ground Truth vs Prediction', fontsize=16, y=1.02)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()

    return fig


def plot_vector_field(vis_data, save_path=None, n_samples=500):
    """绘制向量场对比图"""
    x_curr = vis_data['x_curr'][:n_samples]
    x_true = vis_data['x_next_true'][:n_samples]
    x_pred = vis_data['x_next_pred'][:n_samples]
    times = vis_data['time_curr'][:n_samples]

    # PCA
    all_data = np.vstack([x_curr, x_true, x_pred])
    pca = PCA(n_components=2)
    all_pca = pca.fit_transform(all_data)

    n = len(x_curr)
    pca_curr = all_pca[:n]
    pca_true = all_pca[n:2*n]
    pca_pred = all_pca[2*n:]

    # 计算位移向量
    true_displacement = pca_true - pca_curr
    pred_displacement = pca_pred - pca_curr

    fig, ax = plt.subplots(figsize=(12, 10))

    # 绘制向量场
    ax.quiver(pca_curr[:, 0], pca_curr[:, 1],
             pred_displacement[:, 0], pred_displacement[:, 1],
             color='red', alpha=0.5, scale=1, scale_units='xy', angles='xy',
             width=0.003, headwidth=4, label='Predicted Direction')

    ax.quiver(pca_curr[:, 0], pca_curr[:, 1],
             true_displacement[:, 0], true_displacement[:, 1],
             color='blue', alpha=0.3, scale=1, scale_units='xy', angles='xy',
             width=0.002, headwidth=3, label='True Direction')

    # 起点着色
    scatter = ax.scatter(pca_curr[:, 0], pca_curr[:, 1], c=times,
                        cmap='viridis', s=50, alpha=0.8, edgecolors='black',
                        linewidths=0.5, zorder=5)

    cbar = plt.colorbar(scatter, ax=ax, shrink=0.6)
    cbar.set_label('Time', fontsize=12)

    ax.set_xlabel('PC1', fontsize=12)
    ax.set_ylabel('PC2', fontsize=12)
    ax.set_title('Cell Trajectory Vector Field (Predicted vs True)', fontsize=14)
    ax.legend(loc='upper right', fontsize=10)
    ax.grid(True, linestyle='--', alpha=0.3)

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150)
        plt.close()
    return fig


def plot_mse_distribution(all_mse, save_path=None):
    """绘制 MSE 分布图"""
    plt.figure(figsize=(8, 6))
    sns.histplot(all_mse, bins=50, kde=True, color='coral')
    plt.axvline(x=np.mean(all_mse), color='red', linestyle='--',
                label=f'Mean: {np.mean(all_mse):.4f}')
    plt.axvline(x=np.median(all_mse), color='blue', linestyle=':',
                label=f'Median: {np.median(all_mse):.4f}')
    plt.title("Distribution of Prediction MSE")
    plt.xlabel("Mean Squared Error")
    plt.ylabel("Count")
    plt.legend()
    plt.grid(True, alpha=0.3)

    if save_path:
        plt.savefig(save_path)
        plt.close()
    return plt.gcf()


def plot_correlation_by_time(vis_data, save_path=None, n_bins=10):
    """绘制按时间分组的相关系数箱线图"""
    times = vis_data['time_curr']
    x_true = vis_data['x_next_true']
    x_pred = vis_data['x_next_pred']

    # 计算每个样本的相关系数
    corrs = []
    for i in range(len(x_true)):
        corr = np.corrcoef(x_true[i], x_pred[i])[0, 1]
        if np.isfinite(corr):
            corrs.append(corr)
        else:
            corrs.append(0)
    corrs = np.array(corrs)

    # 将时间分成 bins
    unique_times = np.unique(times)
    if len(unique_times) > n_bins:
        # 按分位数分组
        time_bins = np.percentile(times, np.linspace(0, 100, n_bins + 1))
        time_labels = [f'{time_bins[i]:.1f}-{time_bins[i+1]:.1f}' for i in range(n_bins)]
        bin_indices = np.digitize(times, time_bins[1:-1])
    else:
        # 使用唯一时间点
        time_labels = [f'{t:.1f}' for t in unique_times]
        bin_indices = np.searchsorted(unique_times, times)
        n_bins = len(unique_times)

    # 按组收集数据
    corr_by_bin = [[] for _ in range(n_bins)]
    for i, bin_idx in enumerate(bin_indices):
        if bin_idx < n_bins:
            corr_by_bin[bin_idx].append(corrs[i])

    # 过滤空组
    valid_data = [(label, data) for label, data in zip(time_labels, corr_by_bin) if len(data) > 0]
    if not valid_data:
        return None

    labels, data = zip(*valid_data)

    fig, ax = plt.subplots(figsize=(12, 6))

    # 箱线图
    bp = ax.boxplot(data, labels=labels, patch_artist=True)
    colors = plt.cm.viridis(np.linspace(0, 1, len(data)))
    for patch, color in zip(bp['boxes'], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)

    ax.set_xlabel('Time', fontsize=12)
    ax.set_ylabel('Pearson Correlation', fontsize=12)
    ax.set_title('Prediction Accuracy by Timepoint', fontsize=14)
    ax.axhline(y=0.5, color='gray', linestyle='--', alpha=0.5, label='r=0.5')
    ax.axhline(y=np.mean(corrs), color='red', linestyle='--', alpha=0.7,
              label=f'Overall Mean: {np.mean(corrs):.3f}')
    ax.set_ylim(-0.2, 1.1)
    ax.grid(True, axis='y', linestyle='--', alpha=0.3)
    ax.legend(loc='lower right')
    plt.xticks(rotation=45, ha='right')

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150)
        plt.close()
    return fig


def plot_correlation_summary(vis_data, all_corrs, save_path=None):
    """绘制相关系数综合统计图（3合1）"""
    times = vis_data['time_curr']
    x_true = vis_data['x_next_true']
    x_pred = vis_data['x_next_pred']

    # 计算每个样本的相关系数
    sample_corrs = []
    for i in range(len(x_true)):
        corr = np.corrcoef(x_true[i], x_pred[i])[0, 1]
        if np.isfinite(corr):
            sample_corrs.append(corr)
        else:
            sample_corrs.append(0)
    sample_corrs = np.array(sample_corrs)

    fig, axes = plt.subplots(1, 3, figsize=(16, 5))

    # 1. 相关系数分布
    ax = axes[0]
    sns.histplot(all_corrs, bins=40, kde=True, ax=ax, color='steelblue')
    ax.axvline(x=np.mean(all_corrs), color='red', linestyle='--',
              label=f'Mean: {np.mean(all_corrs):.3f}')
    ax.axvline(x=np.median(all_corrs), color='orange', linestyle=':',
              label=f'Median: {np.median(all_corrs):.3f}')
    ax.set_xlabel('Pearson Correlation', fontsize=11)
    ax.set_ylabel('Count', fontsize=11)
    ax.set_title('Correlation Distribution', fontsize=12)
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

    # 2. 相关系数 vs 时间散点图
    ax = axes[1]
    scatter = ax.scatter(times, sample_corrs, c=times, cmap='viridis',
                        s=10, alpha=0.5)
    # 添加趋势线
    z = np.polyfit(times, sample_corrs, 1)
    p = np.poly1d(z)
    time_range = np.linspace(times.min(), times.max(), 100)
    ax.plot(time_range, p(time_range), 'r--', lw=2, label=f'Trend (slope={z[0]:.4f})')
    ax.axhline(y=np.mean(sample_corrs), color='orange', linestyle=':',
              alpha=0.7, label=f'Mean: {np.mean(sample_corrs):.3f}')
    ax.set_xlabel('Time', fontsize=11)
    ax.set_ylabel('Pearson Correlation', fontsize=11)
    ax.set_title('Correlation vs Time', fontsize=12)
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    ax.set_ylim(-0.2, 1.1)

    # 3. MSE vs Correlation 散点图
    ax = axes[2]
    # 计算每个样本的 MSE
    sample_mse = np.mean((x_true - x_pred) ** 2, axis=1)
    ax.scatter(sample_corrs, sample_mse, c=times, cmap='viridis', s=10, alpha=0.5)
    ax.set_xlabel('Pearson Correlation', fontsize=11)
    ax.set_ylabel('MSE', fontsize=11)
    ax.set_title('MSE vs Correlation', fontsize=12)
    ax.grid(True, alpha=0.3)

    # 添加 colorbar
    cbar = fig.colorbar(scatter, ax=axes[2], shrink=0.8)
    cbar.set_label('Time', fontsize=10)

    plt.suptitle('Prediction Performance Summary', fontsize=14, y=1.02)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
    return fig


def plot_expression_scatter(vis_data, save_path=None, n_samples=1000, n_genes=3):
    """绘制真实值 vs 预测值的表达散点图（选择高变异基因）"""
    x_true = vis_data['x_next_true'][:n_samples]
    x_pred = vis_data['x_next_pred'][:n_samples]

    # 选择高变异基因
    gene_var = np.var(x_true, axis=0)
    top_gene_indices = np.argsort(gene_var)[-n_genes:]

    fig, axes = plt.subplots(1, n_genes, figsize=(5 * n_genes, 5))
    if n_genes == 1:
        axes = [axes]

    for i, gene_idx in enumerate(top_gene_indices):
        ax = axes[i]
        true_vals = x_true[:, gene_idx]
        pred_vals = x_pred[:, gene_idx]

        ax.scatter(true_vals, pred_vals, s=10, alpha=0.3, c='steelblue')

        # 添加对角线
        max_val = max(true_vals.max(), pred_vals.max())
        min_val = min(true_vals.min(), pred_vals.min())
        ax.plot([min_val, max_val], [min_val, max_val], 'r--', lw=2, alpha=0.7)

        # 计算相关系数
        corr = np.corrcoef(true_vals, pred_vals)[0, 1]
        ax.set_xlabel('True Expression', fontsize=11)
        ax.set_ylabel('Predicted Expression', fontsize=11)
        ax.set_title(f'Gene {gene_idx} (r={corr:.3f})', fontsize=12)
        ax.grid(True, alpha=0.3)

    plt.suptitle('Expression Prediction: High Variance Genes', fontsize=14, y=1.02)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
    return fig


def plot_mean_expression_comparison(vis_data, save_path=None):
    """绘制平均表达量对比图"""
    x_true = vis_data['x_next_true']
    x_pred = vis_data['x_next_pred']

    mean_true = np.mean(x_true, axis=0)
    mean_pred = np.mean(x_pred, axis=0)

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # 1. Mean Expression Scatter
    ax = axes[0]
    ax.scatter(mean_true, mean_pred, s=5, alpha=0.5, c='steelblue')
    max_val = max(mean_true.max(), mean_pred.max())
    ax.plot([0, max_val], [0, max_val], 'r--', lw=2, alpha=0.7)

    corr = np.corrcoef(mean_true, mean_pred)[0, 1]
    ax.set_xlabel('True Mean Expression', fontsize=12)
    ax.set_ylabel('Predicted Mean Expression', fontsize=12)
    ax.set_title(f'Mean Expression Comparison (r={corr:.4f})', fontsize=13)
    ax.grid(True, alpha=0.3)

    # 2. Variance Comparison
    ax = axes[1]
    var_true = np.var(x_true, axis=0)
    var_pred = np.var(x_pred, axis=0)

    ax.scatter(var_true, var_pred, s=5, alpha=0.5, c='coral')
    max_val = max(var_true.max(), var_pred.max())
    ax.plot([0, max_val], [0, max_val], 'r--', lw=2, alpha=0.7)

    corr_var = np.corrcoef(var_true, var_pred)[0, 1]
    ax.set_xlabel('True Variance', fontsize=12)
    ax.set_ylabel('Predicted Variance', fontsize=12)
    ax.set_title(f'Variance Comparison (r={corr_var:.4f})', fontsize=13)
    ax.grid(True, alpha=0.3)

    plt.suptitle('Gene Statistics: True vs Predicted', fontsize=14, y=1.02)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
    return fig


def plot_per_cell_metrics(vis_data, all_corrs, all_mse, save_path=None):
    """绘制每个细胞的指标分布图"""
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))

    # 1. Correlation Histogram
    ax = axes[0, 0]
    sns.histplot(all_corrs, bins=50, kde=True, ax=ax, color='steelblue')
    ax.axvline(x=np.mean(all_corrs), color='red', linestyle='--',
              label=f'Mean: {np.mean(all_corrs):.3f}')
    ax.set_xlabel('Pearson Correlation')
    ax.set_ylabel('Count')
    ax.set_title('Per-Cell Correlation Distribution')
    ax.legend()

    # 2. MSE Histogram
    ax = axes[0, 1]
    sns.histplot(all_mse, bins=50, kde=True, ax=ax, color='coral')
    ax.axvline(x=np.mean(all_mse), color='red', linestyle='--',
              label=f'Mean: {np.mean(all_mse):.4f}')
    ax.set_xlabel('MSE')
    ax.set_ylabel('Count')
    ax.set_title('Per-Cell MSE Distribution')
    ax.legend()

    # 3. Correlation CDF
    ax = axes[1, 0]
    sorted_corrs = np.sort(all_corrs)
    cdf = np.arange(1, len(sorted_corrs) + 1) / len(sorted_corrs)
    ax.plot(sorted_corrs, cdf, 'b-', lw=2)
    ax.axvline(x=0.5, color='gray', linestyle='--', alpha=0.7)
    ax.axhline(y=0.5, color='gray', linestyle='--', alpha=0.7)
    # 标记关键分位数
    for q in [0.25, 0.5, 0.75, 0.9]:
        q_val = np.percentile(all_corrs, q * 100)
        ax.axvline(x=q_val, color='red', linestyle=':', alpha=0.5)
        ax.text(q_val, q + 0.02, f'{q_val:.2f}', fontsize=9, ha='center')
    ax.set_xlabel('Pearson Correlation')
    ax.set_ylabel('Cumulative Probability')
    ax.set_title('Correlation CDF')
    ax.grid(True, alpha=0.3)

    # 4. Summary Statistics Table
    ax = axes[1, 1]
    ax.axis('off')
    stats_text = f"""
    === Correlation Statistics ===
    Mean:     {np.mean(all_corrs):.4f}
    Std:      {np.std(all_corrs):.4f}
    Median:   {np.median(all_corrs):.4f}
    Q25:      {np.percentile(all_corrs, 25):.4f}
    Q75:      {np.percentile(all_corrs, 75):.4f}
    Min:      {np.min(all_corrs):.4f}
    Max:      {np.max(all_corrs):.4f}

    === MSE Statistics ===
    Mean:     {np.mean(all_mse):.6f}
    Std:      {np.std(all_mse):.6f}
    Median:   {np.median(all_mse):.6f}
    Q25:      {np.percentile(all_mse, 25):.6f}
    Q75:      {np.percentile(all_mse, 75):.6f}

    === Sample Info ===
    N Samples: {len(all_corrs)}
    % Corr > 0.5: {100 * np.mean(all_corrs > 0.5):.1f}%
    % Corr > 0.7: {100 * np.mean(all_corrs > 0.7):.1f}%
    % Corr > 0.9: {100 * np.mean(all_corrs > 0.9):.1f}%
    """
    ax.text(0.1, 0.95, stats_text, transform=ax.transAxes, fontsize=11,
           verticalalignment='top', fontfamily='monospace',
           bbox=dict(boxstyle='round', facecolor='lightgray', alpha=0.3))
    ax.set_title('Summary Statistics')

    plt.suptitle('Per-Cell Prediction Metrics', fontsize=14, y=1.02)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
    return fig


def load_shard_data_for_trajectory(
    data_dir: str,
    shard_name: str,
    stage_info_path: str = None,
    use_log_time: bool = True,
    max_cells: int = 500
):
    """
    加载单个 shard 的数据用于时序轨迹可视化。

    返回按时间排序的细胞数据。
    """
    import tiledbsoma

    shard_path = os.path.join(data_dir, shard_name)
    if not os.path.exists(shard_path):
        raise ValueError(f"Shard not found: {shard_path}")

    ctx = tiledbsoma.SOMATileDBContext()

    with tiledbsoma.Experiment.open(shard_path, context=ctx) as exp:
        # 读取 obs 元数据
        obs_df = exp.obs.read().concat().to_pandas()
        if len(obs_df) == 0:
            return None

        # 读取 X 数据
        n_vars = exp.ms["RNA"].var.count
        x_uri = os.path.join(shard_path, "ms", "RNA", "X", "data")

        # 获取所有 soma_joinid
        soma_joinids = obs_df['soma_joinid'].values
        soma_joinids_sorted = np.sort(soma_joinids)

        # 读取稀疏数据
        X_dense = np.zeros((len(soma_joinids_sorted), n_vars), dtype=np.float32)

        with tiledbsoma.open(x_uri, mode='r', context=ctx) as X:
            data = X.read(coords=(soma_joinids_sorted, slice(None))).tables().concat()
            row_indices = data["soma_dim_0"].to_numpy()
            col_indices = data["soma_dim_1"].to_numpy()
            values = data["soma_data"].to_numpy()

            local_rows = np.searchsorted(soma_joinids_sorted, row_indices)
            X_dense[local_rows, col_indices] = values

        # 构建 joinid -> 行索引映射
        joinid_to_row = {jid: i for i, jid in enumerate(soma_joinids_sorted)}

        # 获取原始时间并排序
        obs_df['raw_time'] = obs_df['time'].astype(float)
        obs_df = obs_df.sort_values('raw_time')

        # 获取唯一时间点
        unique_times = obs_df['raw_time'].unique()
        unique_times = np.sort(unique_times)

        # 按时间分组
        time_groups = {}
        for t in unique_times:
            cells_at_t = obs_df[obs_df['raw_time'] == t]
            if len(cells_at_t) > 0:
                # 随机采样（避免数据过多）
                if len(cells_at_t) > max_cells:
                    cells_at_t = cells_at_t.sample(n=max_cells, random_state=42)

                joinids = cells_at_t['soma_joinid'].values
                rows = [joinid_to_row[jid] for jid in joinids]
                time_groups[t] = {
                    'X': X_dense[rows],
                    'time_raw': t,
                    'time_norm': normalize_time(t, use_log_time),
                    'n_cells': len(rows),
                }

    return {
        'shard_name': shard_name,
        'unique_times': unique_times,
        'time_groups': time_groups,
        'n_vars': n_vars,
    }


def generate_autoregressive_trajectory(
    model,
    initial_cells: np.ndarray,
    time_points: list,
    stage_id: int,
    device,
    sample_steps: int = 50,
    use_log_time: bool = True,
):
    """
    从初始时间点开始，自回归生成后续时间点的细胞状态。

    Args:
        model: FlowLitModule
        initial_cells: 初始时间点的真实细胞 [N, D]
        time_points: 原始时间点列表（天）
        stage_id: 发育阶段 ID
        device: 设备
        sample_steps: ODE 步数
        use_log_time: 是否使用 log-scale 时间

    Returns:
        Dict[time -> generated_cells]
    """
    model.eval()

    generated = {}
    current_cells = torch.tensor(initial_cells, dtype=torch.float32, device=device)
    n_cells = current_cells.shape[0]

    # 第一个时间点使用真实数据
    generated[time_points[0]] = initial_cells.copy()

    with torch.no_grad():
        for i in range(len(time_points) - 1):
            time_curr_raw = time_points[i]
            time_next_raw = time_points[i + 1]
            delta_t_raw = time_next_raw - time_curr_raw

            # 归一化时间
            time_curr_norm = normalize_time(time_curr_raw, use_log_time)
            time_next_norm = normalize_time(time_next_raw, use_log_time)
            delta_t_norm = normalize_delta_t(delta_t_raw, use_log_time)

            # 构建条件
            cond_data = {
                'time_curr': torch.full((n_cells,), time_curr_norm, dtype=torch.float32, device=device),
                'time_next': torch.full((n_cells,), time_next_norm, dtype=torch.float32, device=device),
                'delta_t': torch.full((n_cells,), delta_t_norm, dtype=torch.float32, device=device),
                'stage': torch.full((n_cells,), stage_id, dtype=torch.long, device=device),
                'x_curr': current_cells,
            }

            # 从噪声采样
            x0 = torch.randn_like(current_cells)
            x_next = model.flow.sample(x0, cond_data, steps=sample_steps, method='euler')

            generated[time_next_raw] = x_next.cpu().numpy()
            current_cells = x_next  # 下一轮的输入

    return generated


def plot_temporal_trajectory_comparison(
    shard_data: dict,
    generated_data: dict,
    save_path: str = None,
    n_samples_per_time: int = 200,
):
    """
    绘制时序轨迹对比图：
    - 上图：真实数据按时间的 UMAP 分布
    - 下图：生成数据（初始真实 + 后续自回归生成）的 UMAP 分布

    Args:
        shard_data: load_shard_data_for_trajectory 返回的数据
        generated_data: generate_autoregressive_trajectory 返回的数据
        save_path: 保存路径
        n_samples_per_time: 每个时间点最多显示的样本数
    """
    time_groups = shard_data['time_groups']
    unique_times = sorted(time_groups.keys())

    # 收集所有数据用于 UMAP
    all_real = []
    all_gen = []
    real_time_labels = []
    gen_time_labels = []

    for t in unique_times:
        if t not in time_groups:
            continue

        # 真实数据
        real_X = time_groups[t]['X']
        n_real = min(len(real_X), n_samples_per_time)
        all_real.append(real_X[:n_real])
        real_time_labels.extend([t] * n_real)

        # 生成数据
        if t in generated_data:
            gen_X = generated_data[t]
            n_gen = min(len(gen_X), n_samples_per_time)
            all_gen.append(gen_X[:n_gen])
            gen_time_labels.extend([t] * n_gen)

    if len(all_real) == 0 or len(all_gen) == 0:
        logger.warning("No data for trajectory visualization")
        return None

    all_real = np.vstack(all_real)
    all_gen = np.vstack(all_gen)
    real_time_labels = np.array(real_time_labels)
    gen_time_labels = np.array(gen_time_labels)

    # 合并计算 UMAP（确保坐标一致）
    logger.info(f"Computing UMAP for trajectory visualization ({len(all_real)} real + {len(all_gen)} gen cells)...")
    all_data = np.vstack([all_real, all_gen])
    reducer = umap.UMAP(n_neighbors=30, min_dist=0.3, metric='euclidean', random_state=42)
    all_umap = reducer.fit_transform(all_data)

    n_real_total = len(all_real)
    umap_real = all_umap[:n_real_total]
    umap_gen = all_umap[n_real_total:]

    # 创建图
    fig, axes = plt.subplots(1, 2, figsize=(16, 7))

    # 颜色映射：时间 -> 颜色
    time_min, time_max = min(unique_times), max(unique_times)
    norm = plt.Normalize(vmin=time_min, vmax=time_max)
    cmap = plt.cm.viridis

    # 左图：真实数据
    ax = axes[0]
    scatter1 = ax.scatter(
        umap_real[:, 0], umap_real[:, 1],
        c=real_time_labels, cmap=cmap, norm=norm,
        s=15, alpha=0.6, edgecolors='none'
    )
    ax.set_xlabel('UMAP1', fontsize=12)
    ax.set_ylabel('UMAP2', fontsize=12)
    ax.set_title(f'Real Data ({shard_data["shard_name"]})', fontsize=14)
    ax.grid(True, linestyle='--', alpha=0.2)

    # 右图：生成数据
    ax = axes[1]

    # 标记初始时间点（真实）和后续时间点（生成）
    first_time = unique_times[0]
    is_initial = gen_time_labels == first_time

    # 初始时间点用方形标记（真实）
    if np.any(is_initial):
        ax.scatter(
            umap_gen[is_initial, 0], umap_gen[is_initial, 1],
            c=gen_time_labels[is_initial], cmap=cmap, norm=norm,
            s=20, alpha=0.7, marker='s', edgecolors='black', linewidths=0.3,
            label='Real (initial)'
        )

    # 后续时间点用圆形标记（生成）
    if np.any(~is_initial):
        ax.scatter(
            umap_gen[~is_initial, 0], umap_gen[~is_initial, 1],
            c=gen_time_labels[~is_initial], cmap=cmap, norm=norm,
            s=15, alpha=0.6, marker='o', edgecolors='none',
            label='Generated (autoregressive)'
        )

    ax.set_xlabel('UMAP1', fontsize=12)
    ax.set_ylabel('UMAP2', fontsize=12)
    ax.set_title('RTF Autoregressive Generation', fontsize=14)
    ax.legend(loc='upper right', fontsize=9)
    ax.grid(True, linestyle='--', alpha=0.2)

    # 共享 colorbar（转换为年龄）
    cbar = fig.colorbar(scatter1, ax=axes, shrink=0.8, aspect=30)
    # 将天数转换为年（用于显示）
    tick_locs = cbar.get_ticks()
    tick_labels = [f'{t/365.25:.1f}y' for t in tick_locs]
    cbar.set_ticklabels(tick_labels)
    cbar.set_label('Age (years)', fontsize=12)

    plt.suptitle('Temporal Trajectory: Real vs Autoregressive Generation', fontsize=16, y=1.02)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()

    return fig


def plot_temporal_trajectory_stripplot(
    shard_data: dict,
    generated_data: dict,
    save_path: str = None,
    n_samples_per_time: int = 100,
):
    """
    绘制时序轨迹的 strip plot：
    - X 轴：时间（年龄）
    - Y 轴：PCA 第一主成分（代表发育程度）

    对比真实数据 vs 生成数据
    """
    from sklearn.decomposition import PCA

    time_groups = shard_data['time_groups']
    unique_times = sorted(time_groups.keys())

    # 收集数据
    all_real = []
    all_gen = []
    real_times = []
    gen_times = []
    real_is_initial = []
    gen_is_initial = []

    first_time = unique_times[0]

    for t in unique_times:
        if t not in time_groups:
            continue

        # 真实数据
        real_X = time_groups[t]['X']
        n = min(len(real_X), n_samples_per_time)
        all_real.append(real_X[:n])
        real_times.extend([t] * n)
        real_is_initial.extend([t == first_time] * n)

        # 生成数据
        if t in generated_data:
            gen_X = generated_data[t]
            n_gen = min(len(gen_X), n_samples_per_time)
            all_gen.append(gen_X[:n_gen])
            gen_times.extend([t] * n_gen)
            gen_is_initial.extend([t == first_time] * n_gen)

    if len(all_real) == 0:
        return None

    all_real = np.vstack(all_real)
    all_gen = np.vstack(all_gen) if all_gen else np.array([])
    real_times = np.array(real_times)
    gen_times = np.array(gen_times) if gen_times else np.array([])

    # PCA 降维
    all_data = np.vstack([all_real, all_gen]) if len(all_gen) > 0 else all_real
    pca = PCA(n_components=2)
    all_pca = pca.fit_transform(all_data)

    n_real = len(all_real)
    pca_real = all_pca[:n_real]
    pca_gen = all_pca[n_real:] if len(all_gen) > 0 else np.array([])

    # 创建图
    fig, axes = plt.subplots(2, 1, figsize=(14, 10), sharex=True)

    # 转换时间为年
    real_years = real_times / 365.25
    gen_years = gen_times / 365.25 if len(gen_times) > 0 else np.array([])

    # 上图：真实数据
    ax = axes[0]
    scatter1 = ax.scatter(
        real_years, pca_real[:, 0],
        c=real_times, cmap='viridis', s=20, alpha=0.6
    )
    ax.set_ylabel('PC1 (Developmental Axis)', fontsize=12)
    ax.set_title('Real Data', fontsize=14)
    ax.grid(True, linestyle='--', alpha=0.3)

    # 添加时间点的 boxplot 轮廓
    unique_years = sorted(set(real_years))
    for year in unique_years:
        mask = real_years == year
        pc1_vals = pca_real[mask, 0]
        if len(pc1_vals) > 5:
            q25, q75 = np.percentile(pc1_vals, [25, 75])
            ax.plot([year, year], [q25, q75], 'k-', lw=2, alpha=0.5)

    # 下图：生成数据
    ax = axes[1]
    if len(pca_gen) > 0:
        gen_is_initial = np.array(gen_is_initial)

        # 初始点（真实）- 方形
        mask_init = gen_is_initial
        if np.any(mask_init):
            ax.scatter(
                gen_years[mask_init], pca_gen[mask_init, 0],
                c=gen_times[mask_init], cmap='viridis', s=30, alpha=0.7,
                marker='s', edgecolors='black', linewidths=0.5,
                label='Real (initial)'
            )

        # 生成点 - 圆形
        mask_gen = ~gen_is_initial
        if np.any(mask_gen):
            ax.scatter(
                gen_years[mask_gen], pca_gen[mask_gen, 0],
                c=gen_times[mask_gen], cmap='viridis', s=20, alpha=0.6,
                marker='o', label='Generated'
            )

        ax.legend(loc='upper left', fontsize=10)

    ax.set_xlabel('Age (years)', fontsize=12)
    ax.set_ylabel('PC1 (Developmental Axis)', fontsize=12)
    ax.set_title('RTF Autoregressive Generation', fontsize=14)
    ax.grid(True, linestyle='--', alpha=0.3)

    # Colorbar
    cbar = fig.colorbar(scatter1, ax=axes, shrink=0.8, aspect=40)
    tick_locs = cbar.get_ticks()
    tick_labels = [f'{t/365.25:.1f}y' for t in tick_locs]
    cbar.set_ticklabels(tick_labels)
    cbar.set_label('Age (years)', fontsize=12)

    plt.suptitle(f'Temporal Development: {shard_data["shard_name"]}', fontsize=16, y=1.02)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()

    return fig


def run_trajectory_visualization(
    model,
    data_dir: str,
    device,
    stage_info_path: str = None,
    use_log_time: bool = True,
    sample_steps: int = 50,
    max_shards: int = 3,
    save_dir: str = None,
):
    """
    运行时序轨迹可视化。

    选择包含多个时间点的 shards，展示自回归生成效果。
    """
    # 获取 Stage 映射
    stage_map = get_stage_map(stage_info_path)

    # 扫描 shards
    shard_names = sorted([
        d for d in os.listdir(data_dir)
        if os.path.isdir(os.path.join(data_dir, d))
    ])

    logger.info(f"Scanning {len(shard_names)} shards for trajectory visualization...")

    # 选择包含多个时间点的 shards
    selected_shards = []
    for shard_name in shard_names:
        try:
            shard_data = load_shard_data_for_trajectory(
                data_dir, shard_name,
                stage_info_path=stage_info_path,
                use_log_time=use_log_time,
                max_cells=200
            )
            if shard_data is None:
                continue

            n_time_points = len(shard_data['unique_times'])
            if n_time_points >= 3:  # 至少 3 个时间点
                selected_shards.append({
                    'name': shard_name,
                    'n_times': n_time_points,
                    'data': shard_data,
                })
                logger.info(f"  {shard_name}: {n_time_points} time points")

        except Exception as e:
            logger.warning(f"  Skip {shard_name}: {e}")
            continue

        if len(selected_shards) >= max_shards:
            break

    if len(selected_shards) == 0:
        logger.warning("No suitable shards found for trajectory visualization")
        return []

    # 生成并绘图
    figures = []
    for shard_info in selected_shards:
        shard_name = shard_info['name']
        shard_data = shard_info['data']
        unique_times = shard_data['unique_times']

        logger.info(f"Generating trajectory for {shard_name}...")

        # 获取 stage ID
        stage_id = stage_map.get(shard_name, 0)

        # 获取初始时间点的真实细胞
        first_time = unique_times[0]
        initial_cells = shard_data['time_groups'][first_time]['X']

        # 自回归生成
        generated = generate_autoregressive_trajectory(
            model,
            initial_cells,
            list(unique_times),
            stage_id,
            device,
            sample_steps=sample_steps,
            use_log_time=use_log_time,
        )

        # 绘制 UMAP 对比图
        if save_dir:
            save_path_umap = os.path.join(save_dir, f"trajectory_umap_{shard_name}.png")
            save_path_strip = os.path.join(save_dir, f"trajectory_strip_{shard_name}.png")
        else:
            save_path_umap = f"trajectory_umap_{shard_name}.png"
            save_path_strip = f"trajectory_strip_{shard_name}.png"

        fig_umap = plot_temporal_trajectory_comparison(
            shard_data, generated,
            save_path=save_path_umap
        )
        fig_strip = plot_temporal_trajectory_stripplot(
            shard_data, generated,
            save_path=save_path_strip
        )

        figures.append({
            'shard_name': shard_name,
            'fig_umap': fig_umap,
            'fig_strip': fig_strip,
            'save_path_umap': save_path_umap,
            'save_path_strip': save_path_strip,
        })

    return figures


def run_benchmark(run_dir, wandb_project, run_idx=1, total_runs=1, sample_steps=50, max_batches=100, cfg_scale=1.0):
    """
    对单个运行进行评测。

    Args:
        run_dir: 训练运行目录
        wandb_project: WandB 项目名称
        run_idx: 当前运行索引
        total_runs: 总运行数
        sample_steps: ODE 采样步数
        max_batches: 最大评估批次数
        cfg_scale: Classifier-Free Guidance 强度（1.0 = 不使用 CFG）
    """
    logger.info(f"{'='*80}")
    logger.info(f"Processing run [{run_idx}/{total_runs}]: {run_dir}")
    if cfg_scale != 1.0:
        logger.info(f"Using CFG with scale={cfg_scale}")
    logger.info(f"{'='*80}")

    # 1. 加载配置
    cfg_path = os.path.join(run_dir, ".hydra", "config.yaml")
    cfg = OmegaConf.load(cfg_path)

    # 2. 获取 checkpoint
    ckpt_path = get_best_checkpoint(run_dir)
    if not ckpt_path:
        logger.warning(f"No checkpoint found in {run_dir}, skipping.")
        return
    logger.info(f"Found checkpoint: {ckpt_path}")

    # 3. 加载模型
    logger.info("Loading model...")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # [关键] 如果是 latent 模式，需要推断正确的 input_dim
    # 因为 hydra 配置保存的可能是默认值
    mode = cfg.model.get("mode", "raw")
    ae_ckpt_path = cfg.model.get("ae_ckpt_path")

    # 先尝试从 checkpoint 本身推断 input_dim
    checkpoint = torch.load(ckpt_path, map_location='cpu', weights_only=False)
    state_dict = checkpoint.get('state_dict', checkpoint)

    # 从 x_embedder.weight 推断 input_dim
    x_embedder_key = 'flow.backbone.x_embedder.weight'
    if x_embedder_key in state_dict:
        # shape: [hidden_size, input_dim]
        inferred_input_dim = state_dict[x_embedder_key].shape[1]
        logger.info(f"Inferred input_dim={inferred_input_dim} from checkpoint")
        OmegaConf.set_struct(cfg, False)
        cfg.model.net.input_dim = inferred_input_dim
        OmegaConf.set_struct(cfg, True)
    elif mode == "latent" and ae_ckpt_path:
        # 回退：尝试从 AE checkpoint 的 hydra 配置读取 latent_dim
        from pathlib import Path
        ae_ckpt_dir = Path(ae_ckpt_path).parent.parent
        ae_hydra_config = ae_ckpt_dir / ".hydra" / "config.yaml"

        if ae_hydra_config.exists():
            ae_cfg = OmegaConf.load(ae_hydra_config)
            latent_dim = ae_cfg.get('model', {}).get('net', {}).get('latent_dim')
            if latent_dim:
                logger.info(f"Inferred input_dim={latent_dim} from AE config")
                OmegaConf.set_struct(cfg, False)
                cfg.model.net.input_dim = latent_dim
                OmegaConf.set_struct(cfg, True)

    del checkpoint  # 释放内存

    # 实例化 backbone net
    net_cfg = cfg.model.net
    net = instantiate(net_cfg)

    # 加载模型
    model = FlowLitModule.load_from_checkpoint(
        ckpt_path,
        net=net,
        optimizer=None,
        scheduler=None,
        map_location=device
    )
    model.to(device)
    model.eval()
    logger.info(f"Model loaded on {device}")

    # 4. 准备数据
    logger.info("Preparing dataloader...")

    # [关键] 根据 mode 确定正确的数据目录
    raw_data_dir = cfg.data.get("raw_data_dir")
    if raw_data_dir is None:
        # 向后兼容：尝试使用 data_dir
        raw_data_dir = cfg.data.get("data_dir")

    # 根据 mode 决定实际数据目录
    if mode == "latent" and raw_data_dir:
        data_dir = raw_data_dir + "_latents"
        logger.info(f"Latent mode: using latent data from {data_dir}")
    else:
        data_dir = raw_data_dir
        logger.info(f"Raw mode: using raw data from {data_dir}")

    if not data_dir or not os.path.exists(data_dir):
        logger.error(f"Data directory not found: {data_dir}")
        return

    batch_size = cfg.data.get("batch_size", 256)
    num_workers = cfg.data.get("num_workers", 4)

    # 获取 stage_info_path 和 use_log_time
    stage_info_path = cfg.data.get("stage_info_path")
    use_log_time = cfg.data.get("use_log_time", True)

    # 使用 OOD（split_label=3）进行评测，整个 shard 作为 OOD
    loader = setup_dataloader(
        data_dir, split_label=3, batch_size=batch_size, num_workers=num_workers,
        stage_info_path=stage_info_path, use_log_time=use_log_time
    )

    # 5. 初始化 W&B
    logger.info("Initializing W&B...")
    run_name = f"bench_{os.path.basename(run_dir)}"

    # 清理配置（移除无法解析的插值和不需要的字段）
    config_dict = {}
    try:
        # 只提取关键配置，避免插值解析失败
        if 'model' in cfg:
            model_cfg = cfg.model
            config_dict['model'] = {
                'mode': model_cfg.get('mode'),
                'flow_type': model_cfg.get('flow_type'),
                'ae_ckpt_path': model_cfg.get('ae_ckpt_path'),
            }
            if 'net' in model_cfg:
                config_dict['model']['net'] = {
                    'input_dim': model_cfg.net.get('input_dim'),
                    'hidden_size': model_cfg.net.get('hidden_size'),
                    'depth': model_cfg.net.get('depth'),
                    'num_heads': model_cfg.net.get('num_heads'),
                }
        if 'data' in cfg:
            config_dict['data'] = {
                'batch_size': cfg.data.get('batch_size'),
                'direction': cfg.data.get('direction'),
            }
        config_dict['task_name'] = cfg.get('task_name')
        config_dict['seed'] = cfg.get('seed')
    except Exception as e:
        logger.warning(f"Failed to extract config: {e}")
        config_dict = {'run_dir': run_dir}

    # 添加 CFG 相关配置到 wandb
    config_dict['cfg_scale'] = cfg_scale

    wandb_run = wandb.init(
        project=wandb_project,
        name=run_name,
        config=config_dict,
        job_type="benchmark",
        tags=["rtf", "benchmark"] + (["cfg"] if cfg_scale != 1.0 else []),
        reinit=True
    )
    logger.info(f"W&B run: {wandb_run.name}")

    # 6. 评估
    logger.info(f"Evaluating (sample_steps={sample_steps}, max_batches={max_batches}, cfg_scale={cfg_scale})...")
    results = evaluate_model(model, loader, device, sample_steps=sample_steps, max_batches=max_batches, cfg_scale=cfg_scale)

    if results is None:
        logger.warning("No evaluation results, skipping.")
        wandb.finish()
        return

    # 记录指标
    wandb.log({
        "eval/mse": results['mse'],
        "eval/mse_std": results['mse_std'],
        "eval/corr": results['corr'],
        "eval/corr_std": results['corr_std'],
        "eval/corr_median": float(np.median(results['all_corrs'])),
        "eval/pct_corr_gt_0.5": float(100 * np.mean(results['all_corrs'] > 0.5)),
        "eval/pct_corr_gt_0.7": float(100 * np.mean(results['all_corrs'] > 0.7)),
        "eval/pct_corr_gt_0.9": float(100 * np.mean(results['all_corrs'] > 0.9)),
    })
    logger.info(f"Results - MSE: {results['mse']:.6f} +/- {results['mse_std']:.6f}")
    logger.info(f"Results - Corr: {results['corr']:.4f} +/- {results['corr_std']:.4f}")
    logger.info(f"Results - Corr > 0.5: {100 * np.mean(results['all_corrs'] > 0.5):.1f}%")

    # 7. 生成可视化
    logger.info("Generating visualizations...")

    # 7.1 相关系数分布
    fig_path = "corr_dist.png"
    plot_correlation_distribution(results['all_corrs'], save_path=fig_path)
    wandb.log({"plots/correlation_distribution": wandb.Image(fig_path)})
    os.remove(fig_path)

    # 7.2 MSE 分布
    fig_path = "mse_dist.png"
    plot_mse_distribution(results['all_mse'], save_path=fig_path)
    wandb.log({"plots/mse_distribution": wandb.Image(fig_path)})
    os.remove(fig_path)

    # 7.3 PCA 轨迹图
    fig_path = "pca_trajectory.png"
    plot_pca_trajectory(results['vis_data'], save_path=fig_path)
    wandb.log({"plots/pca_trajectory": wandb.Image(fig_path)})
    os.remove(fig_path)

    # 7.4 向量场图
    fig_path = "vector_field.png"
    plot_vector_field(results['vis_data'], save_path=fig_path)
    wandb.log({"plots/vector_field": wandb.Image(fig_path)})
    os.remove(fig_path)

    # 7.5 按时间分组的相关系数箱线图
    fig_path = "corr_by_time.png"
    plot_correlation_by_time(results['vis_data'], save_path=fig_path)
    wandb.log({"plots/correlation_by_time": wandb.Image(fig_path)})
    os.remove(fig_path)

    # 7.6 相关系数综合统计图（3合1）
    fig_path = "corr_summary.png"
    plot_correlation_summary(results['vis_data'], results['all_corrs'], save_path=fig_path)
    wandb.log({"plots/correlation_summary": wandb.Image(fig_path)})
    os.remove(fig_path)

    # 7.7 高变异基因表达散点图
    fig_path = "expression_scatter.png"
    plot_expression_scatter(results['vis_data'], save_path=fig_path)
    wandb.log({"plots/expression_scatter": wandb.Image(fig_path)})
    os.remove(fig_path)

    # 7.8 平均表达量和方差对比
    fig_path = "mean_var_comparison.png"
    plot_mean_expression_comparison(results['vis_data'], save_path=fig_path)
    wandb.log({"plots/mean_var_comparison": wandb.Image(fig_path)})
    os.remove(fig_path)

    # 7.9 每细胞指标汇总（4合1）
    fig_path = "per_cell_metrics.png"
    plot_per_cell_metrics(results['vis_data'], results['all_corrs'], results['all_mse'], save_path=fig_path)
    wandb.log({"plots/per_cell_metrics": wandb.Image(fig_path)})
    os.remove(fig_path)

    # 7.10 UMAP 对比图（Ground Truth vs Prediction）
    logger.info("Generating UMAP comparison plot...")
    fig_path = "umap_comparison.png"
    plot_umap_comparison(results['vis_data'], save_path=fig_path)
    wandb.log({"plots/umap_comparison": wandb.Image(fig_path)})
    os.remove(fig_path)

    # 7.11 时序轨迹可视化（自回归生成）
    logger.info("Generating temporal trajectory visualizations...")
    try:
        trajectory_figs = run_trajectory_visualization(
            model,
            data_dir,
            device,
            stage_info_path=stage_info_path,
            use_log_time=use_log_time,
            sample_steps=sample_steps,
            max_shards=3,
            save_dir=None,
        )
        for fig_info in trajectory_figs:
            shard_name = fig_info['shard_name']
            # 上传 UMAP 图
            if fig_info['save_path_umap'] and os.path.exists(fig_info['save_path_umap']):
                wandb.log({f"plots/trajectory_umap_{shard_name}": wandb.Image(fig_info['save_path_umap'])})
                os.remove(fig_info['save_path_umap'])
            # 上传 Strip plot 图
            if fig_info['save_path_strip'] and os.path.exists(fig_info['save_path_strip']):
                wandb.log({f"plots/trajectory_strip_{shard_name}": wandb.Image(fig_info['save_path_strip'])})
                os.remove(fig_info['save_path_strip'])
        logger.info(f"Generated trajectory visualizations for {len(trajectory_figs)} shards")
    except Exception as e:
        logger.warning(f"Failed to generate trajectory visualizations: {e}")

    logger.info(f"Benchmark complete for run [{run_idx}/{total_runs}]")
    wandb.finish()


def main():
    parser = argparse.ArgumentParser(description="RTF 模型批量测评脚本")
    parser.add_argument("--dir", type=str, required=True, help="包含运行日志的目录")
    parser.add_argument("--wandb_project", type=str, default="rtf-cross-bench-cfg", help="W&B 项目名称")
    parser.add_argument("--sample_steps", type=int, default=50, help="ODE 采样步数")
    parser.add_argument("--max_batches", type=int, default=100, help="最大评估批次数")
    parser.add_argument("--cfg_scale", type=float, default=1.0,
                       help="Classifier-Free Guidance 强度 (1.0=不使用, >1.0=增强条件)")

    args = parser.parse_args()

    if not os.path.exists(args.dir):
        logger.error(f"Directory not found: {args.dir}")
        sys.exit(1)

    runs = find_runs(args.dir)
    logger.info(f"Found {len(runs)} runs to benchmark.")
    if args.cfg_scale != 1.0:
        logger.info(f"Using CFG with scale={args.cfg_scale}")
    logger.info(f"{'='*80}\n")

    for idx, run in enumerate(runs, start=1):
        try:
            run_benchmark(
                run,
                args.wandb_project,
                run_idx=idx,
                total_runs=len(runs),
                sample_steps=args.sample_steps,
                max_batches=args.max_batches,
                cfg_scale=args.cfg_scale
            )
            logger.info("\n")
        except Exception as e:
            logger.error(f"Failed to benchmark run {run}: {e}", exc_info=True)
            logger.info("\n")

    logger.info(f"{'='*80}")
    logger.info(f"All benchmarks completed! Total runs: {len(runs)}")
    logger.info(f"{'='*80}")


if __name__ == "__main__":
    main()
