"""
scVI-style Preprocessing for TileDB-SOMA (Optimized)

优化点：将 log1p(normalized) 和 raw counts 合并到单个矩阵中
- 前 n_genes 列: log1p(normalized) - Encoder 输入
- 后 n_genes 列: raw counts - NB Loss 目标

这样只需要读取一个稀疏矩阵，大幅提升 I/O 效率。

保存的数据结构:
- X: [log1p(normalized) | raw_counts], shape (n_cells, 2 * n_genes)
- obs['library_size']: 每个细胞的 UMI 总数
- obs['split_label']: 数据集划分标签

使用时:
- x = X[:, :n_genes]  # log1p normalized
- counts = X[:, n_genes:]  # raw counts
"""

import multiprocessing
import os

try:
    multiprocessing.set_start_method('spawn', force=True)
except RuntimeError:
    pass

import shutil
import pandas as pd
import numpy as np
import scanpy as sc
import anndata as ad
import scipy.sparse
import tiledbsoma
import tiledbsoma.io
import warnings
from concurrent.futures import ProcessPoolExecutor, as_completed
from tqdm import tqdm

warnings.filterwarnings('ignore')

# ============ 配置 ============
CSV_PATH = '/gpfs/flash/home/jcw/projects/research/cellTime/scTFM/data/ae_data_info.csv'
GENE_ORDER_PATH = "/gpfs/flash/home/jcw/projects/research/cellTime/scTFM/data/gene_order.tsv"
OUTPUT_BASE_URI = "/fast/data/scTFM/ae/tiledb_scvi_merged"  # 新的输出目录
NUM_SAMPLES = 16000
MAX_WORKERS = 16
TARGET_SUM = 1e4  # normalize_total 的目标值

# 全局变量
global_target_genes = None
global_target_gene_map = None


def worker_init(gene_list):
    """子进程初始化函数：加载目标基因列表"""
    global global_target_genes, global_target_gene_map
    global_target_genes = gene_list
    global_target_gene_map = {gene: i for i, gene in enumerate(gene_list)}


def process_single_file(row_data):
    """
    处理单个文件的核心逻辑

    保存:
    - X: [log1p(normalized) | raw_counts], shape (n_cells, 2 * n_genes)
    - obs['library_size']: 总 UMI 数
    - obs['split_label']: 数据集划分标签
    """
    idx, row = row_data
    file_path = row['file_path']
    is_full_val = row['full_validation_dataset']

    try:
        file_name = os.path.basename(file_path)
        sample_id = os.path.splitext(file_name)[0]
        soma_uri = os.path.join(OUTPUT_BASE_URI, sample_id)

        if not os.path.exists(file_path):
            return f"Missing: {file_path}"

        # 1. 读取数据
        adata = sc.read_h5ad(file_path)

        # 2. 准备变量名
        adata.var_names = adata.var['gene_symbols'].astype(str)
        adata.var_names_make_unique()

        # 3. 过滤低质量细胞
        sc.pp.filter_cells(adata, min_genes=200)

        if adata.n_obs == 0:
            return "Skipped (Low quality raw cells)"

        # 4. 基因对齐
        target_genes = global_target_genes
        target_n_vars = len(target_genes)
        target_gene_map = global_target_gene_map

        common_genes = [g for g in adata.var_names if g in target_gene_map]

        if len(common_genes) == 0:
            new_X = scipy.sparse.csr_matrix((adata.n_obs, target_n_vars), dtype=np.float32)
            adata = ad.AnnData(X=new_X, obs=adata.obs)
            adata.var_names = target_genes
        else:
            adata = adata[:, common_genes].copy()

            if not scipy.sparse.isspmatrix_csr(adata.X):
                adata.X = adata.X.tocsr()

            current_col_to_target_col = np.array(
                [target_gene_map[g] for g in adata.var_names],
                dtype=np.int32
            )
            new_indices = current_col_to_target_col[adata.X.indices]

            new_X = scipy.sparse.csr_matrix(
                (adata.X.data, new_indices, adata.X.indptr),
                shape=(adata.n_obs, target_n_vars)
            )
            new_X.sort_indices()

            adata = ad.AnnData(X=new_X, obs=adata.obs)
            adata.var_names = target_genes

        # ============ 关键：保存 raw counts ============

        # 5. 计算 library size (在任何归一化之前!)
        if scipy.sparse.issparse(adata.X):
            library_size = np.array(adata.X.sum(axis=1)).flatten()
        else:
            library_size = adata.X.sum(axis=1)

        adata.obs['library_size'] = library_size.astype(np.float32)

        # 6. 保存 raw counts (整数)
        if scipy.sparse.issparse(adata.X):
            raw_counts = adata.X.copy()
        else:
            raw_counts = scipy.sparse.csr_matrix(adata.X)

        # 确保是整数（原始 counts）
        raw_counts.data = np.round(raw_counts.data).astype(np.float32)

        # 7. 归一化 + log1p
        sc.pp.normalize_total(adata, target_sum=TARGET_SUM)
        sc.pp.log1p(adata)

        # 获取 log1p normalized 矩阵
        if scipy.sparse.issparse(adata.X):
            log1p_norm = adata.X.copy()
        else:
            log1p_norm = scipy.sparse.csr_matrix(adata.X)

        # ============ 合并两个矩阵 ============
        # 将 [log1p_norm | raw_counts] 水平拼接
        # 列 0 到 n_genes-1: log1p(normalized)
        # 列 n_genes 到 2*n_genes-1: raw_counts

        combined_X = scipy.sparse.hstack([log1p_norm, raw_counts], format='csr')
        combined_X = combined_X.astype(np.float32)

        # 创建新的 var names (doubled)
        combined_var_names = list(target_genes) + [f"{g}_counts" for g in target_genes]

        # 8. 打标签
        if is_full_val == 1:
            adata.obs['split_label'] = 3
        else:
            n_cells = adata.n_obs
            split_labels = np.random.choice(
                [0, 1, 2],
                size=n_cells,
                p=[0.9, 0.05, 0.05]
            )
            adata.obs['split_label'] = split_labels

        adata.obs['split_label'] = adata.obs['split_label'].astype(np.int32)

        # 创建新的 AnnData（合并后的矩阵）
        adata_combined = ad.AnnData(
            X=combined_X,
            obs=adata.obs
        )
        adata_combined.var_names = combined_var_names

        # ============ 保存为 TileDB-SOMA ============
        if os.path.exists(soma_uri):
            shutil.rmtree(soma_uri)

        tiledbsoma.io.from_anndata(
            experiment_uri=soma_uri,
            anndata=adata_combined,
            measurement_name="RNA"
        )
        return "Success"

    except Exception as e:
        import traceback
        return f"Error: {str(e)}\n{traceback.format_exc()}"


if __name__ == "__main__":
    # 1. 准备输出目录
    if not os.path.exists(OUTPUT_BASE_URI):
        os.makedirs(OUTPUT_BASE_URI)
        print(f"Created output directory: {OUTPUT_BASE_URI}")
    else:
        print(f"Output directory exists: {OUTPUT_BASE_URI}")

    print("\n" + "="*60)
    print("scVI-style Preprocessing (Merged Matrix)")
    print("="*60)
    print(f"Output: {OUTPUT_BASE_URI}")
    print(f"Matrix structure:")
    print(f"  - Columns 0 to n_genes-1: log1p(normalized) for Encoder")
    print(f"  - Columns n_genes to 2*n_genes-1: raw counts for NB Loss")
    print(f"  - obs['library_size']: total UMI counts")
    print("="*60 + "\n")

    # 2. 读取数据
    print("Loading gene order...")
    target_genes = pd.read_csv(GENE_ORDER_PATH, sep='\t', header=None)[0].values
    print(f"  Found {len(target_genes)} target genes")
    print(f"  Combined matrix will have {len(target_genes) * 2} columns")

    print("Loading and sampling CSV...")
    df = pd.read_csv(CSV_PATH)

    if len(df) > NUM_SAMPLES:
        sampled_df = df.sample(n=NUM_SAMPLES, random_state=42)
    else:
        sampled_df = df
        print(f"Warning: CSV only has {len(df)} rows, using all.")

    print(f"  Target sample size: {len(sampled_df)}")

    tasks = list(sampled_df.iterrows())

    # 3. 多进程处理
    print(f"\nStarting parallel processing with {MAX_WORKERS} workers...")

    results = {
        "Success": 0,
        "Skipped (Low quality raw cells)": 0,
        "Errors": 0
    }

    error_messages = []

    with ProcessPoolExecutor(max_workers=MAX_WORKERS, initializer=worker_init, initargs=(target_genes,)) as executor:
        future_to_file = {executor.submit(process_single_file, task): task[1]['file_path'] for task in tasks}

        pbar = tqdm(total=len(tasks), desc="Processing H5ADs")

        for future in as_completed(future_to_file):
            file_path = future_to_file[future]
            try:
                status = future.result()

                if status == "Success":
                    results["Success"] += 1
                elif status.startswith("Skipped"):
                    results["Skipped (Low quality raw cells)"] += 1
                else:
                    results["Errors"] += 1
                    error_messages.append(f"{file_path}: {status}")

            except Exception as exc:
                tqdm.write(f"Critical exception for {file_path}: {exc}")
                results["Errors"] += 1

            pbar.update(1)

        pbar.close()

    print("\n" + "="*60)
    print("Processing Complete")
    print("="*60)
    print(f"Success: {results['Success']}")
    print(f"Skipped: {results['Skipped (Low quality raw cells)']}")
    print(f"Errors : {results['Errors']}")
    print(f"\nData saved to: {OUTPUT_BASE_URI}/")

    if error_messages:
        print("\nFirst 5 errors:")
        for msg in error_messages[:5]:
            print(f"  {msg[:200]}...")

    print("="*60)
