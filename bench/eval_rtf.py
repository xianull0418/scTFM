
import sys
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import scanpy as sc
import fire
import pytorch_lightning as pl
from pathlib import Path
from tqdm import tqdm
from sklearn.decomposition import PCA
from torch.utils.data import DataLoader, ConcatDataset
import json
import types
import re

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

# Import from demo_rtf_flow (assuming it's in bench/)
try:
    from bench.demo_rtf_flow import (
        CONFIG, FlowMatchingLightning, load_gene_list, load_data_files, 
        SingleCellDataset, DiT
    )
except ImportError:
    # Fallback if running directly inside bench
    sys.path.append(os.path.join(os.getcwd(), 'bench'))
    from demo_rtf_flow import (
        CONFIG, FlowMatchingLightning, load_gene_list, load_data_files, 
        SingleCellDataset, DiT
    )

# Set English font
plt.rcParams['font.family'] = 'DejaVu Sans'
plt.rcParams['pdf.fonttype'] = 42
plt.rcParams['ps.fonttype'] = 42

def compute_correlation_rowwise(x, y):
    """
    Compute Pearson correlation row-wise between two tensors x and y.
    x, y: (batch_size, n_features)
    """
    # Centering
    x_mean = x - x.mean(dim=1, keepdim=True)
    y_mean = y - y.mean(dim=1, keepdim=True)
    
    # Normalization
    x_norm = x_mean.norm(dim=1, keepdim=True)
    y_norm = y_mean.norm(dim=1, keepdim=True)
    
    # Avoid division by zero
    x_norm = torch.clamp(x_norm, min=1e-8)
    y_norm = torch.clamp(y_norm, min=1e-8)
    
    # Correlation
    correlation = (x_mean * y_mean).sum(dim=1, keepdim=True) / (x_norm * y_norm)
    return correlation.squeeze()

def _torch_load(path: str, map_location="cpu", weights_only: bool | None = None):
    """
    torch.load 的兼容封装：
    - 对纯权重（state_dict）文件尽量用 weights_only=True，避免 FutureWarning
    - 老版本 torch 不支持 weights_only 参数时自动回退
    """
    try:
        if weights_only is None:
            return torch.load(path, map_location=map_location)
        return torch.load(path, map_location=map_location, weights_only=weights_only)
    except TypeError:
        return torch.load(path, map_location=map_location)


def _strip_known_prefixes(sd: dict) -> dict:
    """去掉 Lightning 常见前缀：'model.' 或 'model.model.'"""
    if not sd:
        return sd
    prefixes = ["model.model.", "model."]
    for p in prefixes:
        if any(k.startswith(p) for k in sd.keys()):
            return {k[len(p):]: v for k, v in sd.items() if k.startswith(p)}
    return sd


def _infer_depth_from_state_dict(sd: dict) -> int:
    idx = set()
    for k in sd.keys():
        if k.startswith("blocks."):
            parts = k.split(".")
            if len(parts) > 1:
                try:
                    idx.add(int(parts[1]))
                except Exception:
                    pass
    return (max(idx) + 1) if idx else 0


def _infer_mlp_ratio_from_state_dict(sd: dict, hidden_size: int) -> float:
    # blocks.0.mlp.0.weight: (mlp_hidden, hidden)
    for k, v in sd.items():
        if k.endswith("blocks.0.mlp.0.weight") and hasattr(v, "shape") and len(v.shape) == 2:
            mlp_hidden = int(v.shape[0])
            if hidden_size > 0:
                return float(mlp_hidden) / float(hidden_size)
    return 4.0


def _find_num_heads_from_hparams_yaml(root: Path) -> int | None:
    # 尝试从 lightning logger 的 hparams.yaml 里读 num_heads
    candidates = list(root.glob("dit_flow_logs/**/hparams.yaml"))
    if not candidates:
        return None
    # 取最新修改的
    candidates.sort(key=lambda p: p.stat().st_mtime, reverse=True)
    try:
        txt = candidates[0].read_text()
    except Exception:
        return None
    m = re.search(r"num_heads\\s*:\\s*(\\d+)", txt)
    if m:
        return int(m.group(1))
    return None


def _guess_num_heads(hidden_size: int, root_for_hparams: Path | None = None) -> int:
    # 优先从 hparams.yaml 推断
    if root_for_hparams is not None:
        nh = _find_num_heads_from_hparams_yaml(root_for_hparams)
        if nh is not None and nh > 0 and hidden_size % nh == 0:
            return nh
    # 兜底：常见配置
    for nh in (6, 8, 12, 16, 4, 2, 1):
        if hidden_size % nh == 0:
            return nh
    return 1


def modulate(x, shift, scale):
    return x * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1)


class LegacyTimestepEmbedder(nn.Module):
    """与 demo_rtf_flow.py 的 TimestepEmbedder 保持一致的命名结构（mlp.0 / mlp.2）。"""
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
        t = t * 1000.0
        half = dim // 2
        freqs = torch.exp(
            -np.log(max_period) * torch.arange(start=0, end=half, dtype=torch.float32) / half
        ).to(device=t.device)
        args = t[:, None].float() * freqs[None]
        emb = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
        if dim % 2:
            emb = torch.cat([emb, torch.zeros_like(emb[:, :1])], dim=-1)
        return emb

    def forward(self, t):
        t_freq = self.timestep_embedding(t, self.frequency_embedding_size)
        return self.mlp(t_freq)


class LegacyDiTBlock(nn.Module):
    """
    兼容旧权重：attn 使用 nn.MultiheadAttention（state_dict key: in_proj_weight/out_proj.*）
    """
    def __init__(self, hidden_size: int, num_heads: int, mlp_ratio: float = 4.0):
        super().__init__()
        self.norm1 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.attn = nn.MultiheadAttention(hidden_size, num_heads=num_heads, batch_first=True)
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
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = self.adaLN_modulation(c).chunk(6, dim=1)
        x_norm = modulate(self.norm1(x), shift_msa, scale_msa)
        attn_out, _ = self.attn(x_norm, x_norm, x_norm, need_weights=False)
        x = x + gate_msa.unsqueeze(1) * attn_out
        x = x + gate_mlp.unsqueeze(1) * self.mlp(modulate(self.norm2(x), shift_mlp, scale_mlp))
        return x


class LegacyFinalLayer(nn.Module):
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


class LegacyDiT(nn.Module):
    def __init__(
        self,
        input_dim: int,
        hidden_size: int,
        depth: int,
        num_heads: int,
        n_tissues: int,
        n_celltypes: int,
        mlp_ratio: float = 4.0,
    ):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.x_embedder = nn.Linear(input_dim, hidden_size)
        self.t_embedder = LegacyTimestepEmbedder(hidden_size)
        self.tissue_emb = nn.Embedding(n_tissues + 1, hidden_size)
        self.celltype_emb = nn.Embedding(n_celltypes + 1, hidden_size)
        self.x_curr_embedder = nn.Linear(input_dim, hidden_size)
        self.abs_time_embedder = LegacyTimestepEmbedder(hidden_size)
        self.dt_embedder = LegacyTimestepEmbedder(hidden_size)
        self.blocks = nn.ModuleList([LegacyDiTBlock(hidden_size, num_heads, mlp_ratio=mlp_ratio) for _ in range(depth)])
        self.final_layer = LegacyFinalLayer(hidden_size, input_dim)

    def forward(self, x, t, cond_data):
        x = self.x_embedder(x).unsqueeze(1)
        t_emb = self.t_embedder(t)
        c_tissue = self.tissue_emb(cond_data["tissue"])
        c_ctype = self.celltype_emb(cond_data["celltype"])
        c_xcurr = self.x_curr_embedder(cond_data["x_curr"])
        c_abs_time = self.abs_time_embedder(cond_data["time"])
        c_dt = self.dt_embedder(cond_data["dt"])
        c = t_emb + c_tissue + c_ctype + c_xcurr + c_abs_time + c_dt
        for blk in self.blocks:
            x = blk(x, c)
        x = self.final_layer(x, c)
        return x.squeeze(1)


def load_model(ckpt_path, device, target_genes, tissue_map, celltype_map):
    """
    兼容三种情况：
    - *.pt：纯 DiT state_dict（可能是旧版 MultiheadAttention，也可能是新版自定义 Attention）
    - *.ckpt：Lightning checkpoint（读取其中 state_dict；不依赖 load_from_checkpoint，避免结构变更导致的硬失败）
    - 误命名：把 state_dict 当 ckpt 传入
    返回对象仅需具备 `.model`（用于后续 ode_solve）。
    """
    print(f"Loading model from {ckpt_path}...")

    ckpt_p = Path(ckpt_path)
    root_for_hparams = ckpt_p.parent
    if ckpt_p.suffix == ".ckpt":
        # <output_dir>/checkpoints/last.ckpt -> root=<output_dir>
        root_for_hparams = ckpt_p.parent.parent

    # 1) 读文件
    loaded = None
    if ckpt_p.suffix == ".pt":
        loaded = _torch_load(str(ckpt_p), map_location="cpu", weights_only=True)
    else:
        loaded = _torch_load(str(ckpt_p), map_location="cpu", weights_only=None)

    # 2) 提取 state_dict
    if isinstance(loaded, dict) and "state_dict" in loaded:
        sd = _strip_known_prefixes(loaded["state_dict"])
        # Lightning 的 hyperparameters 可能包含 config（更可靠的 num_heads）
        hp_cfg = None
        try:
            hp_cfg = loaded.get("hyper_parameters", {}).get("config", None)
        except Exception:
            hp_cfg = None
    elif isinstance(loaded, dict) and loaded and all(isinstance(v, torch.Tensor) for v in loaded.values()):
        sd = loaded
        hp_cfg = None
    else:
        raise RuntimeError(f"Unsupported checkpoint format: {type(loaded)}")

    # 3) 推断结构
    hidden_size = int(sd["x_embedder.weight"].shape[0])
    input_dim = int(sd["x_embedder.weight"].shape[1])
    depth = _infer_depth_from_state_dict(sd)
    n_tissues = int(sd["tissue_emb.weight"].shape[0]) - 1
    n_celltypes = int(sd["celltype_emb.weight"].shape[0]) - 1
    mlp_ratio = _infer_mlp_ratio_from_state_dict(sd, hidden_size)

    is_legacy_attn = any(".attn.in_proj_weight" in k for k in sd.keys())

    if hp_cfg is not None and isinstance(hp_cfg, dict):
        nh = hp_cfg.get("num_heads", None)
        num_heads = int(nh) if nh is not None else _guess_num_heads(hidden_size, root_for_hparams=root_for_hparams)
    else:
        num_heads = _guess_num_heads(hidden_size, root_for_hparams=root_for_hparams)

    # 4) 构建模型并加载
    if is_legacy_attn:
        model = LegacyDiT(
            input_dim=input_dim,
            hidden_size=hidden_size,
            depth=depth,
            num_heads=num_heads,
            n_tissues=n_tissues,
            n_celltypes=n_celltypes,
            mlp_ratio=mlp_ratio,
        )
    else:
        # 新版 Attention（qkv/proj）走 demo_rtf_flow.py 的 DiT
        model = DiT(
            input_dim=input_dim,
            hidden_size=hidden_size,
            depth=depth,
            num_heads=num_heads,
            n_tissues=n_tissues,
            n_celltypes=n_celltypes,
        )

    model.load_state_dict(sd, strict=True)
    model.to(device)
    model.eval()
    return types.SimpleNamespace(model=model)

def ode_solve(model, x_curr, cond_data, steps=20, device='cuda'):
    """
    使用 Euler 方法求解 ODE 生成预测样本。
    z_0 ~ N(0, I) -> z_1 (Predicted x_next)
    """
    B = x_curr.shape[0]
    D = x_curr.shape[1]
    
    # 1. 采样初始噪声 z_0
    z = torch.randn(B, D, device=device)
    
    # 2. 准备条件
    # 确保 cond_data 里的 Tensor 都在 device 上且维度正确
    cond_input = {}
    cond_input['x_curr'] = x_curr.to(device)
    cond_input['time'] = cond_data['time'].to(device)
    cond_input['dt'] = cond_data['dt'].to(device)
    cond_input['tissue'] = cond_data['tissue'].to(device).long()
    cond_input['celltype'] = cond_data['celltype'].to(device).long()

    # 防御性处理：确保 embedding 索引不越界（CUDA 越界会触发 device-side assert，且常在后续算子才报出来）
    if hasattr(model, "tissue_emb") and hasattr(model.tissue_emb, "num_embeddings"):
        max_tissue = int(model.tissue_emb.num_embeddings) - 1
        cond_input["tissue"] = cond_input["tissue"].clamp(min=0, max=max_tissue)
    if hasattr(model, "celltype_emb") and hasattr(model.celltype_emb, "num_embeddings"):
        max_celltype = int(model.celltype_emb.num_embeddings) - 1
        cond_input["celltype"] = cond_input["celltype"].clamp(min=0, max=max_celltype)
    
    # 3. Euler 积分
    dt = 1.0 / steps
    traj = [z.cpu().numpy()] # 记录轨迹
    
    with torch.no_grad():
        for i in range(steps):
            t_scalar = torch.ones(B, device=device) * (i * dt)
            
            # 预测速度 v_t
            v_pred = model(z, t_scalar, cond_input)
            
            # 更新状态 z_{t+1} = z_t + v_t * dt
            z = z + v_pred * dt
            # traj.append(z.cpu().numpy())
            
    return z, traj

def run_benchmark(
    ckpt_path: str = "bench/output/checkpoints/last.ckpt",
    data_dir: str = None, # If None, use CONFIG default
    gene_list_path: str = "data/assets/gene_order.tsv",
    output_dir: str = "bench/output/eval",
    batch_size: int = 100,
    sample_steps: int = 20,
    vis_cells_per_time: int = 50,
    device: str = "cuda",
    force_unknown_cond: bool = True,
    seed: int = 42
):
    """
    Run benchmark for Flow Matching model (adapted from RTF Only benchmark).
    """
    pl.seed_everything(seed)
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    if data_dir is None:
        data_dir = CONFIG['data_dir']

    device = torch.device(device if torch.cuda.is_available() else "cpu")
    
    # 1. Load Data Headers (to reconstruct maps)
    print("Loading Data headers and splitting...")
    try:
        target_genes = load_gene_list(gene_list_path)
    except Exception as e:
        print(f"Error loading gene list: {e}")
        return

    # 如果使用 Lightning 的 ckpt，并且同目录保存了训练时的映射，则复用训练映射以保证编码一致
    train_tissue_map = None
    train_celltype_map = None
    ckpt_p = Path(ckpt_path)
    if ckpt_p.suffix == ".ckpt":
        # 约定：<output_dir>/checkpoints/last.ckpt 旁边的 <output_dir>/tissue_map.json
        cand_root = ckpt_p.parent.parent
        tissue_json = cand_root / "tissue_map.json"
        celltype_json = cand_root / "celltype_map.json"
        if tissue_json.exists() and celltype_json.exists():
            try:
                train_tissue_map = json.loads(tissue_json.read_text())
                train_celltype_map = json.loads(celltype_json.read_text())
                print(f"Using training maps from {cand_root}")
            except Exception as e:
                print(f"[WARN] Failed to load training maps from {cand_root}: {e}")

    # Reload data to get test set
    adatas, tissue_map, celltype_map = load_data_files(
        data_dir, 
        target_genes=target_genes, 
        n_files=CONFIG['n_files_to_load'],
        tissue_map=train_tissue_map,
        celltype_map=train_celltype_map
    )
    
    if not adatas:
        print("No data loaded.")
        return

    # 2. Resolve checkpoint path first (may update ckpt_path)
    if not os.path.exists(ckpt_path):
        import glob
        checkpoints = glob.glob(os.path.join(os.path.dirname(ckpt_path), "*.ckpt"))
        if checkpoints:
            ckpt_path = sorted(checkpoints)[-1]
            print(f"Checkpoint not found, using latest: {ckpt_path}")
        else:
            print(f"Checkpoint {ckpt_path} not found and no others found.")
            return

    # 如果是 *.pt（纯 state_dict），其组织/细胞类型 embedding 的尺寸必须与当前映射一致；
    # 若不一致，默认将条件全部置为 unknown(0)，以保证评估脚本可跑通（但会弱化条件信息）。
    # 重要：必须在构建 Dataset 之前做，否则 Dataset 内部缓存的 code tensor 不会更新，仍会触发越界。
    if str(ckpt_path).endswith(".pt"):
        sd = _torch_load(str(ckpt_path), map_location="cpu", weights_only=True)
        expected_tissues = int(sd["tissue_emb.weight"].shape[0]) - 1
        expected_celltypes = int(sd["celltype_emb.weight"].shape[0]) - 1
        if (len(tissue_map) != expected_tissues) or (len(celltype_map) != expected_celltypes):
            msg = (
                f"[WARN] checkpoint(.pt) embedding sizes mismatch: "
                f"expected n_tissues={expected_tissues}, n_celltypes={expected_celltypes}; "
                f"but data has n_tissues={len(tissue_map)}, n_celltypes={len(celltype_map)}."
            )
            if force_unknown_cond:
                print(msg + " -> Forcing tissue/celltype codes to 0 (unknown) for evaluation.")
                for a in adatas:
                    a.obs["tissue_code"] = 0
                    a.obs["celltype_code"] = 0
                tissue_map = {f"__UNK_TISSUE_{i}": i for i in range(expected_tissues)}
                celltype_map = {f"__UNK_CELLTYPE_{i}": i for i in range(expected_celltypes)}
            else:
                raise RuntimeError(
                    msg
                    + " Set --force_unknown_cond=True to ignore tissue/celltype conditioning, "
                    + "or evaluate with the same dataset/category maps used at training time."
                )

    # Reconstruct Split (after any code remapping)
    full_ds = ConcatDataset([SingleCellDataset(a) for a in adatas])
    total_size = len(full_ds)
    train_size = int(total_size * CONFIG['train_ratio'])
    val_size = total_size - train_size
    
    generator = torch.Generator().manual_seed(CONFIG['seed'])
    _, val_ds = torch.utils.data.random_split(full_ds, [train_size, val_size], generator=generator)
    
    print(f"Validation Set Size: {len(val_ds)}")

    # 3. Load Model
    model_system = load_model(ckpt_path, device, target_genes, tissue_map, celltype_map)
    model = model_system.model # The inner DiT model

    # 3. Compute Metrics
    print("\nComputing correlations on validation set...")
    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=4
    )

    all_corrs = []
    all_mse = []
    
    # Data for visualization
    vis_data_by_time = {} # {time_scalar: list of dicts}
    pca_training_data = []

    MAX_BATCHES = 100 # Limit for speed

    with torch.no_grad():
        for i, batch in enumerate(tqdm(val_loader, total=min(len(val_loader), MAX_BATCHES))):
            if i >= MAX_BATCHES:
                break

            x_curr = batch["x_curr"].to(device)
            x_next = batch["x_next"].to(device) # True Target
            cond_meta = batch["cond_meta"]
            
            # Mask handling
            gene_mask = cond_meta.get('gene_mask', torch.ones_like(x_next, dtype=torch.bool)).to(device)

            # Sample prediction
            x_pred, _ = ode_solve(model, x_curr, cond_meta, steps=sample_steps, device=device)
            
            x_next_np = x_next.cpu().numpy()
            x_pred_np = x_pred.cpu().numpy()
            mask_np = gene_mask.cpu().numpy().astype(bool)
            time_np = cond_meta['time'].cpu().numpy()
            
            for j in range(len(x_next_np)):
                # Filter by mask
                valid_mask = mask_np[j]
                if valid_mask.sum() < 10: continue # Skip if too few genes
                
                y_true = x_next_np[j][valid_mask]
                y_pred = x_pred_np[j][valid_mask]
                
                if np.std(y_true) < 1e-6 or np.std(y_pred) < 1e-6: continue
                
                corr = np.corrcoef(y_true, y_pred)[0, 1]
                mse = np.mean((y_true - y_pred)**2)
                
                if not np.isnan(corr):
                    all_corrs.append(corr)
                all_mse.append(mse)
                
                # Collect vis data (Reservoir sampling if too many)
                # Group by time (rounded)
                t_val = round(float(time_np[j]), 2)
                if t_val not in vis_data_by_time:
                    vis_data_by_time[t_val] = []
                
                if len(vis_data_by_time[t_val]) < vis_cells_per_time:
                    vis_data_by_time[t_val].append({
                        "x_start": x_curr[j].cpu().numpy(), # Full dimension
                        "x_true": x_next[j].cpu().numpy(),
                        "x_pred": x_pred[j].cpu().numpy(),
                        "corr": corr
                    })
                    # Add to PCA training set (only valid genes? No, PCA usually on all)
                    # Note: Zero-padding affects PCA. It's better to use all genes for global structure.
                    pca_training_data.extend([x_curr[j].cpu().numpy(), x_next[j].cpu().numpy(), x_pred[j].cpu().numpy()])


    print(f"Mean Correlation: {np.mean(all_corrs):.4f} +/- {np.std(all_corrs):.4f}")
    print(f"Mean Masked MSE: {np.mean(all_mse):.4f} +/- {np.std(all_mse):.4f}")
    
    # Save metrics
    df_metrics = pd.DataFrame({"Correlation": all_corrs, "MSE": all_mse})
    df_metrics.to_csv(output_path / "metrics.csv", index=False)
    
    # Plot Correlation Distribution
    plt.figure(figsize=(8, 6))
    sns.histplot(all_corrs, bins=50, kde=True)
    plt.title("Distribution of Predicted vs True Correlation")
    plt.xlabel("Pearson Correlation")
    plt.ylabel("Count")
    plt.savefig(output_path / "correlation_dist.png")
    plt.close()
    
    # 4. Visualization (Trajectory)
    if not pca_training_data:
        print("No data for visualization.")
        return

    print("\nComputing PCA...")
    pca_data_mat = np.array(pca_training_data)
    # Subsample if too large
    if len(pca_data_mat) > 10000:
        indices = np.random.choice(len(pca_data_mat), 10000, replace=False)
        pca_data_mat = pca_data_mat[indices]
        
    pca = PCA(n_components=2)
    pca.fit(pca_data_mat)
    
    # ========== Figure 1: 2D PCA Space Trajectory ==========
    print("Generating 2D PCA trajectory plot...")
    fig, ax = plt.subplots(figsize=(12, 10))
    
    # Sort times
    sorted_times = sorted(vis_data_by_time.keys())
    time_colors = plt.cm.viridis(np.linspace(0, 1, len(sorted_times)))
    time_color_map = {t: time_colors[i] for i, t in enumerate(sorted_times)}
    
    from matplotlib.lines import Line2D
    
    for t_val in sorted_times:
        cells = vis_data_by_time[t_val]
        color = time_color_map[t_val]
        
        for cell in cells:
            pc_start = pca.transform(cell["x_start"].reshape(1, -1))[0]
            pc_true = pca.transform(cell["x_true"].reshape(1, -1))[0]
            pc_pred = pca.transform(cell["x_pred"].reshape(1, -1))[0]
            
            # Plot start
            ax.scatter(pc_start[0], pc_start[1], c=[color], s=40, alpha=0.7, marker='o', edgecolors='black', linewidths=0.5, zorder=3)
            # Plot true next
            ax.scatter(pc_true[0], pc_true[1], c=[color], s=25, alpha=0.4, marker='s', zorder=2)
            # Plot predicted next
            ax.scatter(pc_pred[0], pc_pred[1], c='red', s=30, alpha=0.6, marker='^', edgecolors='darkred', linewidths=0.5, zorder=4)
            
            # Arrow
            ax.annotate('', xy=(pc_pred[0], pc_pred[1]), xytext=(pc_start[0], pc_start[1]),
                       arrowprops=dict(arrowstyle='->', color='red', alpha=0.4, lw=1))
            
            # Dashed line true
            ax.plot([pc_start[0], pc_true[0]], [pc_start[1], pc_true[1]], '--', color=color, alpha=0.3, lw=0.8)

    # Legend
    legend_elements = [
        Line2D([0], [0], marker='o', color='w', markerfacecolor='gray', markeredgecolor='black', markersize=10, label='Start Cell'),
        Line2D([0], [0], marker='s', color='w', markerfacecolor='gray', markersize=8, alpha=0.5, label='True Next'),
        Line2D([0], [0], marker='^', color='w', markerfacecolor='red', markeredgecolor='darkred', markersize=10, label='Predicted'),
        Line2D([0], [0], color='red', lw=2, alpha=0.5, label='Prediction'),
        Line2D([0], [0], linestyle='--', color='gray', lw=1, alpha=0.5, label='True Path')
    ]
    ax.legend(handles=legend_elements, loc='upper right')
    
    # Colorbar
    sm = plt.cm.ScalarMappable(cmap='viridis', norm=plt.Normalize(vmin=min(sorted_times), vmax=max(sorted_times)))
    sm.set_array([])
    cbar = plt.colorbar(sm, ax=ax, shrink=0.6)
    cbar.set_label('Normalized Time')
    
    ax.set_title('Cell Trajectory Prediction in PCA Space')
    plt.savefig(output_path / "trajectory_pca_2d.png", dpi=150)
    plt.close()

    # ========== Figure 2: Vector Field ==========
    print("Generating vector field visualization...")
    fig, ax = plt.subplots(figsize=(12, 10))
    
    all_starts = []
    all_preds = []
    all_trues = []
    all_times = []
    
    for t_val in sorted_times:
        for c in vis_data_by_time[t_val]:
            all_starts.append(pca.transform(c["x_start"].reshape(1, -1))[0])
            all_preds.append(pca.transform(c["x_pred"].reshape(1, -1))[0])
            all_trues.append(pca.transform(c["x_true"].reshape(1, -1))[0])
            all_times.append(t_val)
            
    all_starts = np.array(all_starts)
    all_preds = np.array(all_preds)
    all_trues = np.array(all_trues)
    
    pred_disp = all_preds - all_starts
    true_disp = all_trues - all_starts
    
    ax.quiver(all_starts[:,0], all_starts[:,1], pred_disp[:,0], pred_disp[:,1], color='red', alpha=0.5,  width=0.003, headwidth=4, label='Predicted')
    ax.quiver(all_starts[:,0], all_starts[:,1], true_disp[:,0], true_disp[:,1], color='blue', alpha=0.3, width=0.002, headwidth=3, label='True')
    
    scatter = ax.scatter(all_starts[:,0], all_starts[:,1], c=all_times, cmap='viridis', s=50, alpha=0.8, edgecolors='black', linewidths=0.5)
    plt.colorbar(scatter, ax=ax, shrink=0.6).set_label('Time')
    
    ax.legend()
    ax.set_title('Trajectory Vector Field')
    plt.savefig(output_path / "trajectory_vector_field.png", dpi=150)
    plt.close()
    
    print(f"Done. Results saved to {output_dir}")

if __name__ == "__main__":
    fire.Fire(run_benchmark)
