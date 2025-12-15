import torch
from torch.utils.data import IterableDataset
import tiledbsoma
import numpy as np
import math
import os
import random
import gc

class SomaCollectionDataset(IterableDataset):
    def __init__(self, root_dir, split_label=0, io_chunk_size=16384, batch_size=256, measurement_name="RNA"):
        self.root_dir = root_dir
        self.split_label = split_label
        self.io_chunk_size = io_chunk_size
        self.batch_size = batch_size
        self.measurement_name = measurement_name
        
        if not os.path.exists(root_dir):
             raise ValueError(f"❌ 路径不存在: {root_dir}")

        self.sub_uris = sorted([
            os.path.join(root_dir, d) 
            for d in os.listdir(root_dir) 
            if os.path.isdir(os.path.join(root_dir, d))
        ])
        
        print(f"🌍 [Init] 扫描路径: {root_dir}")
        print(f"   发现 {len(self.sub_uris)} 个子数据集 (Shards).")
        
        if len(self.sub_uris) == 0:
            raise ValueError(f"❌ 路径 {root_dir} 下没有发现子文件夹！")

        # 临时读取特征维度
        tmp_ctx = tiledbsoma.SOMATileDBContext()
        try:
            with tiledbsoma.Experiment.open(self.sub_uris[0], context=tmp_ctx) as exp:
                self.n_vars = exp.ms[measurement_name].var.count
            print(f"✅ 特征维度确认: {self.n_vars} genes")
        except Exception:
            if len(self.sub_uris) > 1:
                with tiledbsoma.Experiment.open(self.sub_uris[1], context=tmp_ctx) as exp:
                    self.n_vars = exp.ms[measurement_name].var.count
            else:
                raise

    def _get_context(self):
        return tiledbsoma.SOMATileDBContext(tiledb_config={
            "py.init_buffer_bytes": 512 * 1024**2,
            "sm.memory_budget": 4 * 1024**3,
        })

    def __iter__(self):
        worker_info = torch.utils.data.get_worker_info()
        
        if worker_info is None:
            my_uris = self.sub_uris.copy()
        else:
            per_worker = int(math.ceil(len(self.sub_uris) / worker_info.num_workers))
            start = worker_info.id * per_worker
            end = min(start + per_worker, len(self.sub_uris))
            my_uris = self.sub_uris[start:end].copy()

        if len(my_uris) == 0:
            return

        random.shuffle(my_uris)
        ctx = self._get_context()
        
        # 大块内存池 (复用)
        dense_buffer = np.zeros((self.io_chunk_size, self.n_vars), dtype=np.float32)

        try:
            for uri in my_uris:
                try:
                    with tiledbsoma.Experiment.open(uri, context=ctx) as exp:
                        try:
                            query = exp.obs.read(
                                value_filter=f"split_label == {self.split_label}",
                                column_names=["soma_joinid"]
                            ).concat()
                            chunk_ids = query["soma_joinid"].to_numpy().copy()
                        except Exception:
                            continue 
                        
                        if len(chunk_ids) == 0: continue
                        np.random.shuffle(chunk_ids)
                        
                        x_uri = os.path.join(uri, "ms", self.measurement_name, "X", "data")
                        
                        with tiledbsoma.open(x_uri, mode='r', context=ctx) as X:
                            for i in range(0, len(chunk_ids), self.io_chunk_size):
                                sub_ids = chunk_ids[i : i + self.io_chunk_size]
                                current_len = len(sub_ids)
                                read_ids = np.sort(sub_ids)
                                
                                data = X.read(coords=(read_ids, slice(None))).tables().concat()
                                
                                row_indices = data["soma_dim_0"].to_numpy()
                                col_indices = data["soma_dim_1"].to_numpy()
                                values = data["soma_data"].to_numpy()
                                
                                local_rows = np.searchsorted(read_ids, row_indices)
                                
                                # --- 🔥 修复点在这里 🔥 ---
                                # 必须先定义 active_buffer 是 dense_buffer 的一个切片
                                active_buffer = dense_buffer[:current_len]
                                
                                # 然后才能清零和赋值
                                active_buffer.fill(0)
                                active_buffer[local_rows, col_indices] = values
                                
                                perm = np.random.permutation(current_len)
                                num_batches = (current_len + self.batch_size - 1) // self.batch_size
                                
                                for b in range(num_batches):
                                    start_idx = b * self.batch_size
                                    end_idx = min(start_idx + self.batch_size, current_len)
                                    batch_perm_idx = perm[start_idx:end_idx]
                                    
                                    # [CRITICAL FIX] 检查最后一个 batch 是否太小
                                    # 如果太小 (比如 1)，BatchNorm 会崩溃
                                    if len(batch_perm_idx) <= 1:
                                        continue
                                    
                                    out_tensor = torch.from_numpy(active_buffer[batch_perm_idx].copy())
                                    out_labels = torch.zeros(len(out_tensor), dtype=torch.long)
                                    
                                    yield out_tensor, out_labels
                                    
                except Exception as e:
                    print(f"⚠️ Error processing {os.path.basename(uri)}: {e}")
                    continue
                    
        finally:
            del dense_buffer
            del ctx
            gc.collect()
