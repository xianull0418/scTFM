import os
import tiledbsoma
import math
import multiprocessing
from concurrent.futures import ProcessPoolExecutor
from typing import Tuple, List

def _count_one_shard(args: Tuple[str, int]) -> int:
    """
    单个 Worker 的任务：计算一个分片里符合 split_label 的细胞数
    必须是顶层函数，以便 pickle 序列化。
    """
    uri, split_label = args
    try:
        # 显式创建独立的 Context，避免多进程共享 Context 导致的 C++ 层死锁
        ctx = tiledbsoma.SOMATileDBContext()
        with tiledbsoma.Experiment.open(uri, context=ctx) as exp:
            query = exp.obs.read(
                value_filter=f"split_label == {split_label}",
                column_names=["soma_joinid"]
            ).concat()
            return len(query)
    except Exception:
        return 0

def get_dataset_stats(
    root_dir: str, 
    split_label: int, 
    batch_size: int, 
    num_workers: int = 16, 
    world_size: int = 1
) -> Tuple[int, int]:
    """
    多进程并行扫描 TileDB 数据集，计算总细胞数和步数。
    
    Args:
        root_dir: 数据集根目录
        split_label: 0=Train, 1=Val
        batch_size: 单卡 Batch Size
        num_workers: 并行扫描的进程数 (建议设为 CPU 核心数的一半)
        world_size: DDP 总 GPU 数 (用于计算 Global Batch Size)
        
    Returns:
        (total_cells, total_steps)
    """
    if not os.path.exists(root_dir):
        print(f"⚠️ [Stats] 路径不存在: {root_dir}")
        return 0, 0
        
    sub_uris = sorted([
        os.path.join(root_dir, d) 
        for d in os.listdir(root_dir) 
        if os.path.isdir(os.path.join(root_dir, d))
    ])
    
    if not sub_uris:
        return 0, 0
    
    print(f"📊 [Stats] 启动多进程扫描 {len(sub_uris)} 个 Shards (Split={split_label})...")
    
    # 准备任务参数
    tasks = [(uri, split_label) for uri in sub_uris]
    total_cells = 0
    
    # 动态调整 worker 数，不超过任务数也不超过 CPU 核心数
    max_workers = min(num_workers, len(tasks), os.cpu_count() or 1)

    # 使用 ProcessPoolExecutor 并行处理
    # TileDB 的 C++ 核心在 ThreadPool 下可能会有 GIL 或锁竞争问题，ProcessPool 更稳
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        results = executor.map(_count_one_shard, tasks)
        total_cells = sum(results)
    
    # 计算 DDP 环境下的 Global Batch Size
    global_batch_size = batch_size * world_size
    if global_batch_size == 0:
        return 0, 0
        
    total_steps = math.ceil(total_cells / global_batch_size)
    
    print(f"✅ [Stats] 完成: {total_cells} cells | Global Batch: {global_batch_size} | Epoch Steps: {total_steps}")
    
    return total_cells, total_steps
