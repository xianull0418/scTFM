import torch
from torch.utils.data import Dataset
import tiledb
import numpy as np
from scipy.sparse import coo_matrix

class TileDBCollator:
    def __init__(self, tiledb_path: str, n_genes: int, ctx_cfg: dict = None):
        self.tiledb_path = tiledb_path
        self.n_genes = n_genes
        # Default config: 如果外部未传入，则使用最小可用配置
        self.ctx_cfg = ctx_cfg if ctx_cfg else {
            "sm.compute_concurrency_level": "1",
            "sm.io_concurrency_level": "1",
            "vfs.file.enable_filelocks": "false",
        }
        self._ctx = None
        self._array = None
        # [优化] 缓存 Schema 信息，避免每批次重复查询
        self._dim0_name = None
        self._dim1_name = None
        self._attr_name = None

    def _get_array(self):
        if self._ctx is None:
            self._ctx = tiledb.Ctx(self.ctx_cfg)
        if self._array is None:
            self._array = tiledb.open(self.tiledb_path, mode="r", ctx=self._ctx)
            
            # 初始化缓存
            if self._dim0_name is None:
                schema = self._array.schema
                self._dim0_name = schema.domain.dim(0).name
                self._dim1_name = schema.domain.dim(1).name
                self._attr_name = schema.attr(0).name
                
        return self._array

    def __call__(self, batch_indices):
        # batch_indices 是数据集中的整数索引列表
        if not batch_indices:
            return torch.empty(0, self.n_genes), torch.empty(0)
            
        # 确保索引是 numpy 数组 (处理 List[int] 或 Tensor)
        if isinstance(batch_indices, torch.Tensor):
            indices = batch_indices.numpy()
        else:
            indices = np.array(batch_indices, dtype=np.int64)
        
        A = self._get_array()
        
        # 使用 multi_index 进行批量读取 - 这是优化的关键
        # 它在一个 C++ 调用中获取所请求细胞的所有数据
        try:
            results = A.multi_index[indices, :]
        except Exception as e:
            # 如果出现问题（例如 context/handle 无效），尝试重新打开
            # 这是一个基本的恢复机制
            print(f"Warning: Re-opening TileDB array due to error: {e}")
            self._array = tiledb.open(self.tiledb_path, mode="r", ctx=self._ctx)
            results = self._array.multi_index[indices, :]

        # 使用缓存的维度名
        cell_coords = results[self._dim0_name]
        gene_coords = results[self._dim1_name]
        data_vals = results[self._attr_name]
        
        # --- [核心优化] 向量化映射全局索引到 Batch 行号 ---
        # 替代之前的 Python 字典循环: row_indices = np.array([idx_map[c] for c in cell_coords])
        
        # 1. 对请求的 indices 进行排序，记录排序后的原始位置 (sorter)
        sorter = np.argsort(indices)
        sorted_indices = indices[sorter]
        
        # 2. 在排序后的 indices 中查找 cell_coords 的位置
        # multi_index 返回的 cell_coords 保证在 indices 范围内，所以不会越界
        insert_pos = np.searchsorted(sorted_indices, cell_coords)
        
        # 3. 通过 sorter 映射回原始 batch 的行号
        # 解释: cell_coords[k] 对应 sorted_indices[insert_pos[k]]
        # 而 sorted_indices[m] 对应原始 indices[sorter[m]]
        # 所以原始行号为 sorter[insert_pos[k]]
        row_indices = sorter[insert_pos]
        
        # 构建稀疏矩阵然后转为密集矩阵
        # shape: (batch_size, n_genes)
        mat = coo_matrix(
            (data_vals, (row_indices, gene_coords)), 
            shape=(len(indices), self.n_genes)
        )
        
        # 转换为密集的 torch tensor
        # float32 是标准格式
        batch_x = torch.from_numpy(mat.toarray()).float()
        
        return batch_x, torch.from_numpy(indices)

    def __getstate__(self):
        """
        在 fork/spawn 时丢弃不可 pickle 的 TileDB 句柄。
        """
        state = self.__dict__.copy()
        state["_ctx"] = None
        state["_array"] = None
        return state

    def __setstate__(self, state):
        self.__dict__.update(state)
        # 句柄将被延迟重新创建

class TileDBDataset(Dataset):
    def __init__(self, tiledb_path: str, indices: np.ndarray, n_genes: int):
        super().__init__()
        self.tiledb_path = tiledb_path
        self.indices = indices
        self.n_genes = n_genes

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        # 仅返回全局索引。
        # 繁重的工作由 Collator 完成。
        return self.indices[idx]
