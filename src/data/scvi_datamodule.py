"""
scVI-style Single-Cell DataModule

使用 SomaSCVIDataset，返回:
- x: log1p(normalized) - Encoder 输入
- counts: normalized counts (未 log) - NB Loss 目标
- library_size: 每个细胞的 UMI 总数 - Decoder 缩放
"""

from typing import Optional, Dict
import os

import torch
from pytorch_lightning import LightningDataModule
from torch.utils.data import DataLoader

from src.data.components.scvi_dataset import SomaSCVIDataset


class SCVIDataModule(LightningDataModule):
    """
    scVI-style 单细胞数据 DataModule

    与 SingleCellDataModule 的区别:
    - 使用 SomaSCVIDataset (返回 dict 而非 tuple)
    - 包含 counts 和 library_size 用于 NB loss

    Split Labels:
    0: Train (ID) - 用于训练
    1: Val (ID)   - 用于验证
    2: Test (ID)  - 用于测试 (同分布)
    3: Test (OOD) - 用于测试 (外分布)
    """

    def __init__(
        self,
        data_dir: str = "data/",
        batch_size: int = 256,
        num_workers: int = 4,
        pin_memory: bool = True,
        io_chunk_size: int = 16384,
        prefetch_factor: int = 2,
        persistent_workers: bool = True,
        shard_assignment: Optional[Dict] = None,
        use_counts_layer: bool = False,  # 默认 False，用 expm1(X) 更快
    ):
        """
        Args:
            data_dir: 数据集根目录 (scVI 预处理后的 TileDB 目录)
            batch_size: 每个 batch 的大小
            num_workers: DataLoader 的 worker 数量
            pin_memory: 是否将数据锁在内存中
            io_chunk_size: TileDB 读取时的 chunk 大小
            prefetch_factor: 每个 worker 预加载的 batch 数量
            persistent_workers: 是否保持 workers 存活
            shard_assignment: 智能负载均衡的 shard 分配方案
            use_counts_layer: 是否从 counts layer 读取 (False 用 expm1 计算，更快)
        """
        super().__init__()

        self.save_hyperparameters(logger=False)

        self.data_train: Optional[SomaSCVIDataset] = None
        self.data_val: Optional[SomaSCVIDataset] = None

        self._cached_sub_uris: Optional[list] = None

    def setup(self, stage: Optional[str] = None):
        """加载数据集"""
        if not self.data_train and not self.data_val:
            # 预扫描 Shards
            if self._cached_sub_uris is None:
                print(f"🔍 [SCVIDataModule] Pre-scanning shards in {self.hparams.data_dir}...")
                self._cached_sub_uris = sorted([
                    os.path.join(self.hparams.data_dir, d)
                    for d in os.listdir(self.hparams.data_dir)
                    if os.path.isdir(os.path.join(self.hparams.data_dir, d))
                ])
                print(f"✅ [SCVIDataModule] Found {len(self._cached_sub_uris)} shards")

            # 训练集
            self.data_train = SomaSCVIDataset(
                root_dir=self.hparams.data_dir,
                split_label=0,
                io_chunk_size=self.hparams.io_chunk_size,
                batch_size=self.hparams.batch_size,
                preloaded_sub_uris=self._cached_sub_uris,
                shard_assignment=self.hparams.shard_assignment,
                use_counts_layer=self.hparams.use_counts_layer,
            )

            # 验证集
            self.data_val = SomaSCVIDataset(
                root_dir=self.hparams.data_dir,
                split_label=1,
                io_chunk_size=self.hparams.io_chunk_size,
                batch_size=self.hparams.batch_size,
                preloaded_sub_uris=self._cached_sub_uris,
                shard_assignment=None,
                use_counts_layer=self.hparams.use_counts_layer,
            )

    def train_dataloader(self):
        """返回训练集的 DataLoader"""
        return DataLoader(
            dataset=self.data_train,
            batch_size=None,  # Dataset 已经处理了 batching
            num_workers=self.hparams.num_workers,
            prefetch_factor=self.hparams.prefetch_factor,
            pin_memory=self.hparams.pin_memory,
            persistent_workers=self.hparams.persistent_workers,
        )

    def val_dataloader(self):
        """返回验证集的 DataLoader"""
        return DataLoader(
            dataset=self.data_val,
            batch_size=None,
            num_workers=self.hparams.num_workers,
            prefetch_factor=self.hparams.prefetch_factor,
            pin_memory=self.hparams.pin_memory,
            persistent_workers=self.hparams.persistent_workers,
        )

    def teardown(self, stage: Optional[str] = None):
        pass

    def state_dict(self):
        return {}

    def load_state_dict(self, state_dict):
        pass
