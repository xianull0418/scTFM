import tiledb
import torch
import numpy as np
import json
import os
from pytorch_lightning import LightningDataModule
from torch.utils.data import DataLoader
from src.data.components.tiledb_dataset import TileDBDataset, TileDBCollator

class SingleCellDataModule(LightningDataModule):
    def __init__(
        self,
        data_dir: str,
        batch_size: int = 1024,
        num_workers: int = 16,
        train_val_split: float = 0.95,
        pin_memory: bool = False,
        tile_cache_size: int = 4000000000, # Default 4GB
        tiledb_config: dict = None, # [新增] 接收外部 TileDB 配置
    ):
        super().__init__()
        self.save_hyperparameters(logger=False)
        
        self.data_train = None
        self.data_val = None
        # 使用传入配置或默认空字典
        self.tiledb_config = tiledb_config or {}

    def setup(self, stage=None):
        """
        在 DDP 模式下，setup 会在每张显卡的进程里都运行一次。
        如果有 8 张卡，就会有 8 个进程同时读取 GPFS。
        必须强制关闭 TileDB 文件锁，否则会发生死锁 (Deadlock)。
        """
        # 防止重复 setup
        if self.data_train and self.data_val:
            return

        counts_uri = f"{self.hparams.data_dir}/counts"
        meta_uri = f"{self.hparams.data_dir}/cell_metadata"
        
        # -----------------------------------------------------------------
        # 【关键修复】定义全局无锁 Context
        # 使用统一的 TileDB 配置 (与 Worker 保持一致)
        # -----------------------------------------------------------------
        # 如果有传入配置则使用，否则使用默认的安全配置
        cfg_dict = self.tiledb_config if self.tiledb_config else {
            "sm.compute_concurrency_level": "2",
            "sm.io_concurrency_level": "2",
            "vfs.file.enable_filelocks": "false",
            "sm.tile_cache_size": str(self.hparams.tile_cache_size),
        }
        ctx = tiledb.Ctx(tiledb.Config(cfg_dict))

        # 1. 读取 Schema 获取基因数量 (必须传入 ctx)
        try:
            with tiledb.open(counts_uri, mode='r', ctx=ctx) as A:
                n_genes = A.schema.domain.dim("gene_index").domain[1] + 1
        except Exception as e:
            raise FileNotFoundError(f"Could not open TileDB at {counts_uri}. Check path!") from e

        # 2. 读取 Metadata 进行过滤 (必须传入 ctx)
        print(f"Loading metadata from {meta_uri}...")
        
        # [Fix] 尝试从 metadata.json 获取真实的 total_cells
        # 对于 Dense Array，必须指定读取范围，不能使用 [:]，否则会读取整个 Domain (int64.max) 导致 OOM
        total_cells = None
        meta_json_path = os.path.join(self.hparams.data_dir, "metadata.json")
        if os.path.exists(meta_json_path):
            try:
                with open(meta_json_path, 'r') as f:
                    meta_info = json.load(f)
                    total_cells = meta_info.get('total_cells')
                    print(f"Found metadata.json: total_cells = {total_cells}")
            except Exception as e:
                print(f"Warning: Failed to read metadata.json: {e}")

        try:
            with tiledb.open(meta_uri, mode='r', ctx=ctx) as A:
                # 只读取需要的列，减少 IO
                if total_cells is not None:
                    # 精确读取有效范围 (Python 切片是左闭右开，所以使用 total_cells)
                    # 例如: total_cells=100, 索引 0..99, 切片 0:100 返回 100 个元素
                    is_ood = A.query(attrs=["is_ood"])[0:total_cells]["is_ood"]
                else:
                    # 备选方案: 如果没有 json (比如旧数据)，尝试读取非空域或直接全读 (可能有风险)
                    print("Warning: total_cells unknown, using full slice (risky for Dense Arrays)...")
                    # 对于 Dense Array，如果没有 non_empty_domain，这步可能会挂
                    # 但我们假设旧数据是 Sparse 的，所以 [:] 是安全的
                    is_ood = A.query(attrs=["is_ood"])[:]["is_ood"]
                    
        except Exception as e:
             raise FileNotFoundError(f"Could not read metadata at {meta_uri}") from e
        
        # 3. 筛选 In-Distribution 数据 (is_ood == 0)
        valid_indices = np.where(is_ood == 0)[0]
        total_valid = len(valid_indices)
        
        # 4. 固定种子 Shuffle 并划分 (使用 Chunked Shuffle 优化 GPFS 性能)
        print(f"Applying Chunked Shuffle optimization for GPFS...")
        
        # A. 确保物理顺序 (利用 TileDB 的空间局部性)
        valid_indices.sort()
        
        # B. 定义块大小 (Tile Extent 4096 的倍数)
        # 4096 * 20 ≈ 80k 细胞。在这个范围内随机，既保证了 Batch 的随机性，
        # 又保证了读取只会命中 ~20 个 Tile，极大提高 Cache 命中率并减少读放大。
        chunk_size = 4096 * 20 
        rng = np.random.default_rng(seed=42)
        
        if len(valid_indices) > chunk_size:
            n_chunks = len(valid_indices) // chunk_size
            
            # 分割主数据和剩余数据
            main_part = valid_indices[:n_chunks * chunk_size]
            rest_part = valid_indices[n_chunks * chunk_size:]
            
            # Reshape 成 (n_chunks, chunk_size)
            # 注意：创建副本以避免 View 问题
            chunks = main_part.reshape(n_chunks, chunk_size).copy()
            
            # C. 块间 Shuffle (宏观随机：决定先读哪一块 80k 细胞)
            rng.shuffle(chunks)
            
            # D. 块内 Shuffle (微观随机：块内 80k 细胞完全打乱)
            for i in range(n_chunks):
                rng.shuffle(chunks[i])
            
            shuffled_main = chunks.flatten()
            
            # 剩余部分也 shuffle
            rng.shuffle(rest_part)
            
            valid_indices = np.concatenate([shuffled_main, rest_part])
        else:
            # 数据量太小，直接全局 Shuffle
            rng.shuffle(valid_indices)
        
        n_train = int(total_valid * self.hparams.train_val_split)
        train_idxs = valid_indices[:n_train]
        val_idxs = valid_indices[n_train:]
        
        print(f"Dataset Setup: {total_valid} cells (Filtered OOD).")
        print(f"Train: {len(train_idxs)} | Val: {len(val_idxs)}")

        # 5. 实例化 Dataset
        # 注意：TileDBDataset 内部的 __getitem__ 也必须有关锁逻辑
        self.data_train = TileDBDataset(counts_uri, train_idxs, n_genes)
        self.data_val = TileDBDataset(counts_uri, val_idxs, n_genes)

        from omegaconf import OmegaConf
        
        # 6. 实例化 Collator (用于 Batch 读取 TileDB)
        # 将 YAML 中的配置透传给 Collator
        # [关键修复] 必须将 OmegaConf 对象转为原生 dict，否则 TileDB Ctx 会报错
        collator_cfg = OmegaConf.to_container(self.tiledb_config, resolve=True) if self.tiledb_config else None
        self.collator = TileDBCollator(counts_uri, n_genes, ctx_cfg=collator_cfg)

    def train_dataloader(self):
            # 1. 基础参数
            loader_args = dict(
                batch_size=self.hparams.batch_size,
                num_workers=self.hparams.num_workers,
                pin_memory=self.hparams.pin_memory,
                # [关键修复] 必须关闭全局 Shuffle！
                # 我们在 setup() 中已经做了 Chunked Shuffle (局部打乱)，
                # 开启全局 Shuffle 会导致随机读取整个磁盘，引发 GPFS I/O 崩溃 (D状态进程)。
                shuffle=True,
                drop_last=True,
                collate_fn=self.collator,  # <--- 使用自定义 Collator
            )
            
            # 2. 只有在有 Worker 的时候才启用 spawn 和持久化
            if self.hparams.num_workers > 0:
                loader_args['persistent_workers'] = True
                loader_args['multiprocessing_context'] = 'spawn'  # <--- 动态添加
                
            return DataLoader(self.data_train, **loader_args)

    def val_dataloader(self):
        loader_args = dict(
            batch_size=self.hparams.batch_size,
            num_workers=self.hparams.num_workers,
            pin_memory=self.hparams.pin_memory,
            shuffle=False,
            collate_fn=self.collator,  # <--- 使用自定义 Collator
        )
        
        if self.hparams.num_workers > 0:
            loader_args['persistent_workers'] = True
            loader_args['multiprocessing_context'] = 'spawn' # <--- 动态添加
            
        return DataLoader(self.data_val, **loader_args)