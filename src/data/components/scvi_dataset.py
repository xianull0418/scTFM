"""
scVI-style Dataset for Single-Cell Data (Optimized - Merged Matrix)

数据格式 (由 preprocess_tiledb_scvi.py 生成):
- X: [log1p(normalized) | raw_counts], shape (n_cells, 2 * n_genes)
  - 前 n_genes 列: log1p(normalized) - Encoder 输入
  - 后 n_genes 列: raw counts - NB Loss 目标
- obs['library_size']: 每个细胞的 UMI 总数

优化点:
- 只读取一个稀疏矩阵，然后在内存中 split
- 与 ae_dataset.py 相同的 I/O 效率
"""

import torch
from torch.utils.data import IterableDataset
import tiledbsoma
import numpy as np
import os
import random
import gc


class SomaSCVIDataset(IterableDataset):
    """
    scVI-style Dataset (Optimized - Merged Matrix)

    读取合并后的矩阵 [log1p_norm | raw_counts]，然后 split。

    Yields:
        dict with keys:
            - 'x': log1p(normalized) tensor, shape (batch, n_genes) - Encoder 输入
            - 'counts': raw counts tensor, shape (batch, n_genes) - NB Loss 目标
            - 'library_size': library size tensor, shape (batch,) - Decoder 缩放
    """

    def __init__(
        self,
        root_dir: str,
        split_label: int = 0,
        io_chunk_size: int = 16384,
        batch_size: int = 256,
        measurement_name: str = "RNA",
        preloaded_sub_uris: list = None,
        shard_assignment: dict = None,
        use_counts_layer: bool = True,  # 保留参数但不使用（兼容性）
    ):
        self.root_dir = root_dir
        self.split_label = split_label
        self.io_chunk_size = io_chunk_size
        self.batch_size = batch_size
        self.measurement_name = measurement_name
        self._n_vars = None  # 这是合并后的维度 (2 * n_genes)
        self._n_genes = None  # 实际基因数 (n_genes)
        self.shard_assignment = shard_assignment

        if preloaded_sub_uris is not None:
            self._sub_uris = preloaded_sub_uris
        else:
            self._sub_uris = None

        if not os.path.exists(root_dir):
            raise ValueError(f"Path does not exist: {root_dir}")

    @property
    def sub_uris(self):
        """延迟加载 Shards 列表"""
        if self._sub_uris is None:
            self._sub_uris = sorted([
                os.path.join(self.root_dir, d)
                for d in os.listdir(self.root_dir)
                if os.path.isdir(os.path.join(self.root_dir, d))
            ])

            if len(self._sub_uris) == 0:
                raise ValueError(f"No subdirectories found in {self.root_dir}")

        return self._sub_uris

    @property
    def n_vars(self):
        """延迟加载特征维度（合并后的维度 = 2 * n_genes）"""
        if self._n_vars is None:
            tmp_ctx = tiledbsoma.SOMATileDBContext()
            try:
                with tiledbsoma.Experiment.open(self.sub_uris[0], context=tmp_ctx) as exp:
                    self._n_vars = exp.ms[self.measurement_name].var.count
            except Exception:
                if len(self.sub_uris) > 1:
                    with tiledbsoma.Experiment.open(self.sub_uris[1], context=tmp_ctx) as exp:
                        self._n_vars = exp.ms[self.measurement_name].var.count
                else:
                    raise

        return self._n_vars

    @property
    def n_genes(self):
        """实际基因数量 = n_vars / 2"""
        if self._n_genes is None:
            self._n_genes = self.n_vars // 2
        return self._n_genes

    def _get_context(self):
        return tiledbsoma.SOMATileDBContext(tiledb_config={
            "py.init_buffer_bytes": 512 * 1024**2,
            "sm.memory_budget": 4 * 1024**3,
        })

    def __iter__(self):
        # 1. 获取 DDP 和 Worker 信息
        if torch.distributed.is_initialized():
            rank = torch.distributed.get_rank()
            world_size = torch.distributed.get_world_size()
        else:
            rank = 0
            world_size = 1

        worker_info = torch.utils.data.get_worker_info()
        if worker_info is None:
            worker_id = 0
            num_workers = 1
        else:
            worker_id = worker_info.id
            num_workers = worker_info.num_workers

        global_worker_id = rank * num_workers + worker_id

        # 2. 选择分片策略
        if self.shard_assignment is not None:
            assigned_shard_names = self.shard_assignment.get(str(global_worker_id), [])
            shard_name_to_uri = {os.path.basename(uri): uri for uri in self.sub_uris}
            my_worker_uris = [shard_name_to_uri[name] for name in assigned_shard_names if name in shard_name_to_uri]
        else:
            total_workers = world_size * num_workers
            global_uris = sorted(self.sub_uris)
            my_worker_uris = global_uris[global_worker_id::total_workers]

        if len(my_worker_uris) == 0:
            return

        random.shuffle(my_worker_uris)

        ctx = self._get_context()

        # 内存池 (合并后的矩阵，包含 x 和 counts)
        n_vars_total = self.n_vars  # 2 * n_genes
        n_genes = self.n_genes
        dense_buffer = np.zeros((self.io_chunk_size, n_vars_total), dtype=np.float32)

        try:
            for uri in my_worker_uris:
                try:
                    with tiledbsoma.Experiment.open(uri, context=ctx) as exp:
                        # 读取 soma_joinid
                        try:
                            query = exp.obs.read(
                                value_filter=f"split_label == {self.split_label}",
                                column_names=["soma_joinid"]
                            ).concat()
                            chunk_ids = query["soma_joinid"].to_numpy().copy()
                        except Exception:
                            continue

                        if len(chunk_ids) == 0:
                            continue

                        np.random.shuffle(chunk_ids)

                        x_uri = os.path.join(uri, "ms", self.measurement_name, "X", "data")

                        with tiledbsoma.open(x_uri, mode='r', context=ctx) as X:
                            for i in range(0, len(chunk_ids), self.io_chunk_size):
                                sub_ids = chunk_ids[i: i + self.io_chunk_size]
                                current_len = len(sub_ids)
                                read_ids = np.sort(sub_ids)

                                # 读取合并后的矩阵 [log1p_norm | raw_counts]
                                data = X.read(coords=(read_ids, slice(None))).tables().concat()
                                row_indices = data["soma_dim_0"].to_numpy()
                                col_indices = data["soma_dim_1"].to_numpy()
                                values = data["soma_data"].to_numpy()
                                local_rows = np.searchsorted(read_ids, row_indices)

                                active_buffer = dense_buffer[:current_len]
                                active_buffer.fill(0)
                                active_buffer[local_rows, col_indices] = values

                                # Batch 划分
                                perm = np.random.permutation(current_len)
                                num_batches = (current_len + self.batch_size - 1) // self.batch_size

                                for b in range(num_batches):
                                    start_idx = b * self.batch_size
                                    end_idx = min(start_idx + self.batch_size, current_len)
                                    batch_idx = perm[start_idx:end_idx]

                                    if len(batch_idx) <= 1:
                                        continue

                                    # Split: 前 n_genes 列是 log1p_norm，后 n_genes 列是 raw_counts
                                    batch_data = active_buffer[batch_idx]
                                    x_batch = batch_data[:, :n_genes].copy()
                                    counts_batch = batch_data[:, n_genes:].copy()

                                    # library_size: sum of raw counts per cell
                                    library_batch = counts_batch.sum(axis=1)
                                    library_batch = np.maximum(library_batch, 1.0)

                                    yield {
                                        'x': torch.from_numpy(x_batch),
                                        'counts': torch.from_numpy(counts_batch),
                                        'library_size': torch.from_numpy(library_batch.astype(np.float32)),
                                    }

                except Exception as e:
                    print(f"Warning: Error processing {os.path.basename(uri)}: {e}")
                    continue

        finally:
            del dense_buffer
            del ctx
            gc.collect()
