import argparse
import os
import sys
import glob
import json
import yaml
import torch
import tiledb
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
import logging

# 确保项目根目录在路径中
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../"))
sys.path.append(project_root)
os.environ["PROJECT_ROOT"] = project_root # Fix for ${oc.env:PROJECT_ROOT} interpolation

from src.models.ae_module import AELitModule
from src.data.components.tiledb_dataset import TileDBDataset, TileDBCollator

# 设置绘图风格 (使用英文，美观)
sns.set_theme(style="whitegrid", context="paper", font_scale=1.2)
plt.rcParams['figure.dpi'] = 300
plt.rcParams['savefig.bbox'] = 'tight'

# 配置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Register dummy hydra resolver to handle ${hydra:...} interpolations
try:
    OmegaConf.register_new_resolver("hydra", lambda *args: "hydra_placeholder")
except Exception:
    pass

def find_runs(base_dir):
    """
    扫描目录寻找包含 .hydra/config.yaml 的运行目录。
    支持递归查找 multiruns。
    """
    runs = []
    # 检查 base_dir 本身是否是一个运行目录
    if os.path.exists(os.path.join(base_dir, ".hydra", "config.yaml")):
        runs.append(base_dir)
    
    # 递归查找
    for root, dirs, files in os.walk(base_dir):
        for d in dirs:
            run_path = os.path.join(root, d)
            if os.path.exists(os.path.join(run_path, ".hydra", "config.yaml")):
                runs.append(run_path)
    
    # 去重
    return sorted(list(set(runs)))

def get_best_checkpoint(run_dir):
    """
    寻找最佳 checkpoint。
    通常在 checkpoints/ 目录下，优先选 epoch 最大的或者是 best model。
    """
    ckpt_dir = os.path.join(run_dir, "checkpoints")
    if not os.path.exists(ckpt_dir):
        return None
    
    # 获取所有 ckpt
    ckpts = glob.glob(os.path.join(ckpt_dir, "*.ckpt"))
    if not ckpts:
        return None
    
    # 简单策略：选择最后修改的文件，或者根据文件名中的 epoch 排序
    # 假设文件名类似 "epoch_010.ckpt"
    # 这里我们简单地按修改时间排序取最新的，通常是最优或最后的
    ckpts.sort(key=os.path.getmtime)
    return ckpts[-1]

def setup_dataloaders(data_dir, batch_size=1024, seed=42):
    """
    重新构建数据集划分逻辑，获取 ID-Validation 和 OOD-Test 数据集。
    基于预计算的 split_label 进行划分，确保与训练时一致。
    """
    counts_uri = os.path.join(data_dir, "counts")
    meta_uri = os.path.join(data_dir, "cell_metadata")
    
    # TileDB 配置
    tiledb_cfg = {
        "sm.compute_concurrency_level": "2",
        "sm.io_concurrency_level": "2",
        "vfs.file.enable_filelocks": "false",
    }
    ctx = tiledb.Ctx(tiledb.Config(tiledb_cfg))
    
    # 1. 获取基因数量
    with tiledb.open(counts_uri, mode='r', ctx=ctx) as A:
        n_genes = A.schema.domain.dim("gene_index").domain[1] + 1
        
    # 2. 读取 split_label (替代旧的 is_ood 和动态划分)
    # 尝试读取 metadata.json 获取 total_cells
    total_cells = None
    meta_json_path = os.path.join(data_dir, "metadata.json")
    if os.path.exists(meta_json_path):
        try:
            with open(meta_json_path, 'r') as f:
                meta_info = json.load(f)
                total_cells = meta_info.get('total_cells')
        except Exception:
            pass
            
    with tiledb.open(meta_uri, mode='r', ctx=ctx) as A:
        if total_cells is not None:
            split_labels = A.query(attrs=["split_label"])[0:total_cells]["split_label"]
        else:
            split_labels = A.query(attrs=["split_label"])[:]["split_label"]
            
    # 3. 根据标签划分
    # 0: Train (ID)
    # 1: Val (ID) -> 训练过程中的验证
    # 2: Test (ID) -> 独立的 ID 测试集 (Benchmark 用)
    # 3: Test (OOD) -> 独立的 OOD 测试集 (Benchmark 用)
    
    # Benchmark 脚本应优先使用 Test Set (Label=2)
    test_id_idxs = np.where(split_labels == 2)[0]
    ood_idxs = np.where(split_labels == 3)[0]
    
    # 容错：如果没有 Test Set (Label=2)，则回退到 Validation (Label=1) 并给出警告
    if len(test_id_idxs) == 0:
         logger.warning("未找到 Test ID (Label=2) 数据，回退使用 Validation (Label=1) 进行测评。")
         test_id_idxs = np.where(split_labels == 1)[0]
    
    logger.info(f"Data Split for Benchmark - ID (Test): {len(test_id_idxs)}, OOD (Test): {len(ood_idxs)}")
    
    # 创建 Datasets
    ds_test_id = TileDBDataset(counts_uri, test_id_idxs, n_genes)
    ds_ood = TileDBDataset(counts_uri, ood_idxs, n_genes)
    
    collator = TileDBCollator(counts_uri, n_genes, ctx_cfg=tiledb_cfg)
    
    loader_test_id = DataLoader(ds_test_id, batch_size=batch_size, shuffle=False, collate_fn=collator, num_workers=4, persistent_workers=True, multiprocessing_context='spawn')
    
    # OOD 可能为空
    loader_ood = None
    if len(ood_idxs) > 0:
        loader_ood = DataLoader(ds_ood, batch_size=batch_size, shuffle=False, collate_fn=collator, num_workers=4, persistent_workers=True, multiprocessing_context='spawn')
        
    return loader_test_id, loader_ood

def evaluate_model(model, dataloader, device, desc="ID"):
    """
    在给定 DataLoader 上评估模型。
    返回:
        mse: 均方误差
        corrs: 基因/细胞相关性 (可选)
        latents: 潜在向量
        recons: 重构数据 (部分，为了省内存)
        original: 原始数据 (部分)
    """
    model.eval()
    mse_sum = 0
    n_elements = 0
    n_total_cells = 0
    
    # Metrics accumulators
    cell_corr_sum = 0.0
    
    # For global gene mean correlation
    # We need to sum up all vectors to get the global mean later
    sum_orig = None
    sum_recon = None
    
    # For variance and dropout calculation
    sum_sq_orig = None
    sum_sq_recon = None
    sum_gt0_orig = None
    sum_gt0_recon = None
    
    latents = []
    
    # 为了绘图，我们只保留前 N 个样本的重构结果
    # Increase to 4096 for better distribution/correlation estimation on subset
    n_keep = 4096
    kept_recon = []
    kept_orig = []
    
    with torch.no_grad():
        for batch in tqdm(dataloader, desc=f"Evaluating {desc}"):
            x, _ = batch
            x = x.to(device)
            
            recon_x, z = model(x)
            
            # 1. MSE
            loss = torch.nn.functional.mse_loss(recon_x, x, reduction='sum')
            mse_sum += loss.item()
            n_elements += x.numel()
            n_total_cells += x.size(0)
            
            # 2. Per-Cell Pearson Correlation
            # Center the data per cell
            vx = x - x.mean(dim=1, keepdim=True)
            vy = recon_x - recon_x.mean(dim=1, keepdim=True)
            # Covariance and variances
            # Add epsilon 1e-8 to avoid division by zero
            cost = (vx * vy).sum(dim=1) / (torch.sqrt((vx ** 2).sum(dim=1)) * torch.sqrt((vy ** 2).sum(dim=1)) + 1e-8)
            cell_corr_sum += cost.sum().item()
            
            # 3. Accumulate sums for Global Statistics
            # Initialize if first batch
            if sum_orig is None:
                # Use float64 for accumulation to prevent numerical instability
                sum_orig = torch.zeros(x.size(1), device=device, dtype=torch.float64)
                sum_recon = torch.zeros(x.size(1), device=device, dtype=torch.float64)
                sum_sq_orig = torch.zeros(x.size(1), device=device, dtype=torch.float64)
                sum_sq_recon = torch.zeros(x.size(1), device=device, dtype=torch.float64)
                sum_gt0_orig = torch.zeros(x.size(1), device=device, dtype=torch.float64)
                sum_gt0_recon = torch.zeros(x.size(1), device=device, dtype=torch.float64)
            
            x_64 = x.to(torch.float64)
            recon_64 = recon_x.to(torch.float64)
            
            sum_orig += x_64.sum(dim=0)
            sum_recon += recon_64.sum(dim=0)
            
            sum_sq_orig += (x_64 ** 2).sum(dim=0)
            sum_sq_recon += (recon_64 ** 2).sum(dim=0)
            
            sum_gt0_orig += (x_64 > 0).float().sum(dim=0)
            sum_gt0_recon += (recon_64 > 0).float().sum(dim=0)
            
            latents.append(z.cpu().numpy())
            
            if len(kept_recon) < n_keep:
                kept_recon.append(recon_x.cpu().numpy())
                kept_orig.append(x.cpu().numpy())
                
    mse = mse_sum / n_elements
    mean_cell_corr = cell_corr_sum / n_total_cells
    
    # Calculate Global Gene Statistics
    # Mean
    mean_orig = (sum_orig / n_total_cells).float()
    mean_recon = (sum_recon / n_total_cells).float()
    
    # Variance: E[X^2] - (E[X])^2
    var_orig = ((sum_sq_orig / n_total_cells) - (mean_orig.double() ** 2)).float()
    var_recon = ((sum_sq_recon / n_total_cells) - (mean_recon.double() ** 2)).float()
    
    # Dropout (Fraction of Zeros)
    # sum_gt0 is count of > 0, so dropout is 1 - (count / n)
    dropout_orig = 1.0 - (sum_gt0_orig / n_total_cells).float()
    dropout_recon = 1.0 - (sum_gt0_recon / n_total_cells).float()
    
    # Gene Mean Correlation
    vm_orig = mean_orig - mean_orig.mean()
    vm_recon = mean_recon - mean_recon.mean()
    gene_mean_corr = (vm_orig * vm_recon).sum() / (torch.sqrt((vm_orig**2).sum()) * torch.sqrt((vm_recon**2).sum()) + 1e-8)
    gene_mean_corr = gene_mean_corr.item()
    
    latents = np.concatenate(latents, axis=0)
    
    # 拼接保留的样本
    if kept_recon:
        kept_recon = np.concatenate(kept_recon, axis=0)
        kept_orig = np.concatenate(kept_orig, axis=0)
        
        # 截断到 n_keep
        if kept_recon.shape[0] > n_keep:
            kept_recon = kept_recon[:n_keep]
            kept_orig = kept_orig[:n_keep]
            
    return {
        "mse": mse,
        "mean_cell_corr": mean_cell_corr,
        "gene_mean_corr": gene_mean_corr,
        "latents": latents,
        "recon_sample": kept_recon,
        "orig_sample": kept_orig,
        "stats": {
            "mean_orig": mean_orig.cpu().numpy(),
            "mean_recon": mean_recon.cpu().numpy(),
            "var_orig": var_orig.cpu().numpy(),
            "var_recon": var_recon.cpu().numpy(),
            "dropout_orig": dropout_orig.cpu().numpy(),
            "dropout_recon": dropout_recon.cpu().numpy()
        }
    }

def plot_reconstruction(orig, recon, title_suffix="", save_path=None):
    """
    绘制重构散点图 (Mean Expression)。
    """
    # 计算每个基因的平均表达量
    mean_orig = np.mean(orig, axis=0)
    mean_recon = np.mean(recon, axis=0)
    
    plt.figure(figsize=(6, 6))
    plt.scatter(mean_orig, mean_recon, s=5, alpha=0.5, c='b')
    
    # 绘制对角线
    max_val = max(mean_orig.max(), mean_recon.max())
    plt.plot([0, max_val], [0, max_val], 'r--', alpha=0.7)
    
    plt.title(f"Reconstruction: Mean Expression ({title_suffix})")
    plt.xlabel("Original Mean Expression")
    plt.ylabel("Reconstructed Mean Expression")
    
    # 计算 R2 或 Correlation
    corr = np.corrcoef(mean_orig, mean_recon)[0, 1]
    plt.text(0.05, 0.95, f"R = {corr:.3f}", transform=plt.gca().transAxes, fontsize=12, verticalalignment='top')
    
    if save_path:
        plt.savefig(save_path)
        plt.close()
    else:
        plt.show()

def plot_metric_scatter(val_orig, val_recon, metric_name, title_suffix="", save_path=None):
    """
    绘制基因指标散点图 (如 Variance, Dropout)。
    """
    plt.figure(figsize=(6, 6))
    plt.scatter(val_orig, val_recon, s=5, alpha=0.5, c='purple')
    
    # 绘制对角线
    max_val = max(val_orig.max(), val_recon.max())
    min_val = min(val_orig.min(), val_recon.min())
    plt.plot([min_val, max_val], [min_val, max_val], 'r--', alpha=0.7)
    
    plt.title(f"Gene {metric_name} ({title_suffix})")
    plt.xlabel(f"Original {metric_name}")
    plt.ylabel(f"Reconstructed {metric_name}")
    
    # 计算 R
    # Handle NaNs if any
    mask = np.isfinite(val_orig) & np.isfinite(val_recon)
    if mask.sum() > 1:
        corr = np.corrcoef(val_orig[mask], val_recon[mask])[0, 1]
        plt.text(0.05, 0.95, f"R = {corr:.3f}", transform=plt.gca().transAxes, fontsize=12, verticalalignment='top')
    
    if save_path:
        plt.savefig(save_path)
        plt.close()

def plot_gene_distributions(orig_data, recon_data, title_suffix="", save_path=None, n_genes=3):
    """
    绘制 Top Variable Genes 的分布对比。
    """
    # 找到变异最大的基因 (在原始数据中)
    vars = np.var(orig_data, axis=0)
    top_indices = np.argsort(vars)[-n_genes:][::-1]
    
    fig, axes = plt.subplots(1, n_genes, figsize=(4 * n_genes, 4))
    if n_genes == 1: axes = [axes]
    
    for i, idx in enumerate(top_indices):
        ax = axes[i]
        sns.kdeplot(orig_data[:, idx], label='Original', fill=True, alpha=0.3, ax=ax, clip=(0, None))
        sns.kdeplot(recon_data[:, idx], label='Recon', fill=True, alpha=0.3, ax=ax, clip=(0, None))
        ax.set_title(f"Gene {idx} Distribution")
        ax.set_xlabel("Expression")
        if i == 0:
            ax.legend()
            
    plt.suptitle(f"Top Variable Genes Distributions ({title_suffix})")
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path)
        plt.close()

def plot_gene_corr_scatter(orig_data, recon_data, title_suffix="", save_path=None):
    """
    绘制基因间相关性矩阵的散点图对比。
    使用 subset 数据计算 Top 500 HVGs 的相关性矩阵。
    """
    # 1. 筛选 HVGs
    vars = np.var(orig_data, axis=0)
    n_top = min(500, orig_data.shape[1])
    top_indices = np.argsort(vars)[-n_top:]
    
    # 2. 提取数据
    sub_orig = orig_data[:, top_indices]
    sub_recon = recon_data[:, top_indices]
    
    # 3. 计算相关性矩阵
    corr_orig = np.corrcoef(sub_orig, rowvar=False)
    corr_recon = np.corrcoef(sub_recon, rowvar=False)
    
    # 4. 展平并取上三角 (排除对角线)
    triu_idx = np.triu_indices_from(corr_orig, k=1)
    flat_orig = corr_orig[triu_idx]
    flat_recon = corr_recon[triu_idx]
    
    # 5. 下采样以绘图 (如果点太多)
    max_points = 10000
    if len(flat_orig) > max_points:
        idx = np.random.choice(len(flat_orig), max_points, replace=False)
        flat_orig = flat_orig[idx]
        flat_recon = flat_recon[idx]
        
    plt.figure(figsize=(6, 6))
    plt.scatter(flat_orig, flat_recon, s=5, alpha=0.3, c='green')
    plt.plot([-1, 1], [-1, 1], 'r--', alpha=0.7)
    
    plt.title(f"Gene-Gene Correlation ({title_suffix})\n(Top {n_top} HVGs)")
    plt.xlabel("Original Correlation")
    plt.ylabel("Reconstructed Correlation")
    
    corr_of_corr = np.corrcoef(flat_orig, flat_recon)[0, 1]
    plt.text(0.05, 0.95, f"R = {corr_of_corr:.3f}", transform=plt.gca().transAxes, fontsize=12, verticalalignment='top')
    
    if save_path:
        plt.savefig(save_path)
        plt.close()

def run_benchmark(run_dir, wandb_base_cfg):
    """
    对单个运行目录执行测评。
    """
    logger.info(f"Processing run: {run_dir}")
    
    # 1. 加载 Config
    cfg_path = os.path.join(run_dir, ".hydra", "config.yaml")
    cfg = OmegaConf.load(cfg_path)
    
    # Patch paths to avoid interpolation errors
    if "paths" in cfg:
        # Force resolve root_dir to current project root
        project_root = os.environ.get("PROJECT_ROOT", os.path.abspath(os.path.join(os.path.dirname(__file__), "../")))
        
        # Flatten all path interpolations to static strings
        cfg.paths.root_dir = project_root
        cfg.paths.data_dir = os.path.join(project_root, "data")
        cfg.paths.log_dir = os.path.join(project_root, "logs")
        cfg.paths.output_dir = run_dir
        cfg.paths.work_dir = run_dir
            
    # 2. 寻找 Checkpoint
    ckpt_path = get_best_checkpoint(run_dir)
    if not ckpt_path:
        logger.warning(f"No checkpoint found in {run_dir}, skipping.")
        return
    
    logger.info(f"Found checkpoint: {ckpt_path}")
    
    # 3. 加载模型
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    net_cfg = cfg.model.net
    net = instantiate(net_cfg)
    model = AELitModule.load_from_checkpoint(ckpt_path, net=net, optimizer=None, scheduler=None, map_location=device)
    model.to(device)
    model.eval()
    
    # 4. 准备数据 (使用 Config 中的 data_dir)
    data_dir = cfg.data.data_dir
    loader_id, loader_ood = setup_dataloaders(data_dir, batch_size=2048) # 推理可以使用大一点的 batch
    
    # 5. 初始化 WandB
    # 合并 wandb 配置
    # 我们创建一个新的 run 用于 benchmark，或者 resume？
    # 为了清晰，建议创建新 run，带上 tags
    wandb_cfg = OmegaConf.load(wandb_base_cfg)
    
    run_name = f"bench_{os.path.basename(run_dir)}_{cfg.model.net.get('_target_', 'ae').split('.')[-1]}"
    
    # 提取关键参数用于记录
    # 移除不需要的且可能包含破坏性插值的配置部分
    for key in ["callbacks", "trainer", "logger", "hydra"]:
        if key in cfg:
            with open(os.devnull, "w") as f: # Suppress potential read-only errors if any, though DictConfig is usually mutable
                pass
            try:
                del cfg[key]
            except Exception:
                pass

    config_dict = OmegaConf.to_container(cfg, resolve=True)
    
    run = wandb.init(
        project=wandb_cfg.get("project", "scTime-AE-bench_1210"),
        name=run_name,
        config=config_dict,
        job_type="benchmark",
        tags=["benchmark", "ood"],
        reinit=True
    )
    
    # 6. 执行评估 (ID)
    res_id = evaluate_model(model, loader_id, device, desc="ID Test")
    wandb.log({
        "eval/id_mse": res_id['mse'],
        "eval/id_cell_corr": res_id['mean_cell_corr'],
        "eval/id_gene_corr": res_id['gene_mean_corr']
    })
    logger.info(f"ID MSE: {res_id['mse']:.6f}, Cell Corr: {res_id['mean_cell_corr']:.4f}, Gene Mean Corr: {res_id['gene_mean_corr']:.4f}")
    
    # 7. 执行评估 (OOD)
    res_ood = None
    if loader_ood:
        res_ood = evaluate_model(model, loader_ood, device, desc="OOD Test")
        wandb.log({
            "eval/ood_mse": res_ood['mse'],
            "eval/ood_cell_corr": res_ood['mean_cell_corr'],
            "eval/ood_gene_corr": res_ood['gene_mean_corr']
        })
        logger.info(f"OOD MSE: {res_ood['mse']:.6f}, Cell Corr: {res_ood['mean_cell_corr']:.4f}, Gene Mean Corr: {res_ood['gene_mean_corr']:.4f}")
    
    # 8. 生成图表
    def generate_plots(res, suffix):
        stats = res['stats']
        
        # A. Mean Expression Scatter
        fig_mean = f"recon_mean_{suffix}.png"
        plot_reconstruction(res['orig_sample'], res['recon_sample'], title_suffix=suffix, save_path=fig_mean)
        wandb.log({f"plots/recon_mean_{suffix}": wandb.Image(fig_mean)})
        os.remove(fig_mean)
        
        # B. Variance Scatter
        fig_var = f"recon_var_{suffix}.png"
        plot_metric_scatter(stats['var_orig'], stats['var_recon'], "Variance", title_suffix=suffix, save_path=fig_var)
        wandb.log({f"plots/recon_var_{suffix}": wandb.Image(fig_var)})
        os.remove(fig_var)
        
        # C. Dropout Scatter
        fig_drop = f"recon_dropout_{suffix}.png"
        plot_metric_scatter(stats['dropout_orig'], stats['dropout_recon'], "Dropout Rate", title_suffix=suffix, save_path=fig_drop)
        wandb.log({f"plots/recon_dropout_{suffix}": wandb.Image(fig_drop)})
        os.remove(fig_drop)
        
        # D. Gene Distributions (Top Genes)
        fig_dist = f"gene_dist_{suffix}.png"
        plot_gene_distributions(res['orig_sample'], res['recon_sample'], title_suffix=suffix, save_path=fig_dist)
        wandb.log({f"plots/gene_dist_{suffix}": wandb.Image(fig_dist)})
        os.remove(fig_dist)
        
        # E. Gene-Gene Correlation Scatter
        fig_corr = f"gene_corr_{suffix}.png"
        plot_gene_corr_scatter(res['orig_sample'], res['recon_sample'], title_suffix=suffix, save_path=fig_corr)
        wandb.log({f"plots/gene_corr_{suffix}": wandb.Image(fig_corr)})
        os.remove(fig_corr)

    # Generate plots for ID
    generate_plots(res_id, "id")
    
    # Generate plots for OOD
    if res_ood:
        generate_plots(res_ood, "ood")
    
    # F. Latent Space Visualization (PCA/UMAP)
    # 合并 ID 和 OOD 的 latents 进行对比
    logger.info("Computing Latent Space Visualization...")
    latents_id = res_id['latents']
    labels = ["ID"] * len(latents_id)
    
    all_latents = latents_id
    if res_ood:
        latents_ood = res_ood['latents']
        all_latents = np.concatenate([latents_id, latents_ood], axis=0)
        labels += ["OOD"] * len(latents_ood)
    
    # 为了速度，如果数据量太大，下采样
    max_plot = 10000
    if len(all_latents) > max_plot:
        idx = np.random.choice(len(all_latents), max_plot, replace=False)
        all_latents = all_latents[idx]
        labels = np.array(labels)[idx]
    
    adata_vis = sc.AnnData(X=all_latents)
    adata_vis.obs['condition'] = labels
    
    # Compute PCA & UMAP
    sc.tl.pca(adata_vis)
    sc.pp.neighbors(adata_vis)
    sc.tl.umap(adata_vis)
    
    # Plot
    fig, ax = plt.subplots(figsize=(8, 8))
    sc.pl.umap(adata_vis, color='condition', ax=ax, show=False, title="Latent Space: ID vs OOD", frameon=False)
    fig_path_umap = "latent_umap.png"
    plt.savefig(fig_path_umap, bbox_inches='tight')
    plt.close()
    
    wandb.log({"plots/latent_umap": wandb.Image(fig_path_umap)})
    os.remove(fig_path_umap)
    
    wandb.finish()

def main():
    parser = argparse.ArgumentParser(description="AE 模型批量测评脚本")
    parser.add_argument("--dir", type=str, required=True, help="包含运行日志的目录 (例如 logs/ae_stage1/multiruns/...)")
    parser.add_argument("--wandb_config", type=str, default="configs/logger/wandb.yaml", help="WandB 配置文件路径")
    
    args = parser.parse_args()
    
    if not os.path.exists(args.dir):
        logger.error(f"Directory not found: {args.dir}")
        sys.exit(1)
        
    runs = find_runs(args.dir)
    logger.info(f"Found {len(runs)} runs to benchmark.")
    
    for run in runs:
        try:
            run_benchmark(run, args.wandb_config)
        except Exception as e:
            logger.error(f"Failed to benchmark run {run}: {e}", exc_info=True)

if __name__ == "__main__":
    main()
