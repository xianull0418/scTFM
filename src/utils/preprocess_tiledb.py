import multiprocessing
import os
# tiledb底层是c，用fork会导致每个进程都初始化实例
try:
    multiprocessing.set_start_method('spawn', force=True)
except RuntimeError:
    pass
import shutil
import random
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

CSV_PATH = '/gpfs/flash/home/jcw/projects/research/cellTime/scTFM/data/assets/ae_data_info.csv'
GENE_ORDER_PATH = "/gpfs/flash/home/jcw/projects/research/cellTime/scTFM/data/assets/gene_order.tsv"
OUTPUT_BASE_URI = "/fast/data/scTFM/ae/tile_4000_fix"
NUM_SAMPLES = 4000
MAX_WORKERS = 16  


# 定义全局变量
# 这样每个线程读取的时候共享同一个变量
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
    """
    idx, row = row_data
    file_path = row['file_path']
    is_full_val = row['full_validation_dataset']
    
    # 用文件名作为 ID，而不是随机 UUID
    try:
        file_name = os.path.basename(file_path)       # 获取文件名 "SRX21870170.h5ad"
        sample_id = os.path.splitext(file_name)[0]    # 去掉后缀 "SRX21870170"
        soma_uri = os.path.join(OUTPUT_BASE_URI, sample_id)
        
        if not os.path.exists(file_path):
            return f"Missing: {file_path}"

        # 1. 读取数据
        adata = sc.read_h5ad(file_path)
        
        # 2. 准备变量名
        adata.var_names = adata.var['gene_symbols'].astype(str)
        adata.var_names_make_unique()
        
        # 先过滤细胞质量，再切片
        # 只要原始数据中检测到 >200 个基因，就视为有效细胞
        sc.pp.filter_cells(adata, min_genes=200)
        
        if adata.n_obs == 0:
            return "Skipped (Low quality raw cells)"

        # 3. 极速对齐逻辑
        target_genes = global_target_genes
        target_n_vars = len(target_genes)
        target_gene_map = global_target_gene_map
        
        # 计算交集
        common_genes = [g for g in adata.var_names if g in target_gene_map]
        
        if len(common_genes) == 0:
            # 全零矩阵 (但保留了有效细胞的占位)
            new_X = scipy.sparse.csr_matrix((adata.n_obs, target_n_vars), dtype=np.float32)
            adata = ad.AnnData(X=new_X, obs=adata.obs)
            adata.var_names = target_genes
        else:
            # 先切片保留交集基因
            adata = adata[:, common_genes].copy()
            
            # 极速映射
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

        # 4. 后续处理
        sc.pp.normalize_total(adata, target_sum=1e4)
        sc.pp.log1p(adata)
        
        # 5. 打标签
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
        
        # 确保 float32
        if adata.X.dtype != np.float32:
            adata.X = adata.X.astype(np.float32)

        # 6. 写入 SOMA (如果目录已存在，先删除旧的，避免冲突)
        if os.path.exists(soma_uri):
            shutil.rmtree(soma_uri)
            
        tiledbsoma.io.from_anndata(
            experiment_uri=soma_uri,
            anndata=adata,
            measurement_name="RNA"
        )
        return "Success"

    except Exception as e:
        return f"Error: {str(e)}"


if __name__ == "__main__":
    # 1. 准备输出目录
    if not os.path.exists(OUTPUT_BASE_URI):
        os.makedirs(OUTPUT_BASE_URI)
        print(f"Created output directory: {OUTPUT_BASE_URI}")
    else:
        print(f"Output directory exists: {OUTPUT_BASE_URI}")
    
    # 2. 读取数据
    print("Loading gene order...")
    target_genes = pd.read_csv(GENE_ORDER_PATH, sep='\t', header=None)[0].values
    
    print("Loading and sampling CSV...")
    df = pd.read_csv(CSV_PATH)
    
    # 采样
    if len(df) > NUM_SAMPLES:
        sampled_df = df.sample(n=NUM_SAMPLES, random_state=42)
    else:
        sampled_df = df
        print(f"Warning: CSV only has {len(df)} rows, using all.")
    
    print(f"Target sample size: {len(sampled_df)}")
    
    tasks = list(sampled_df.iterrows())
    
    # 4. 多进程处理
    print(f"Starting parallel processing with {MAX_WORKERS} workers...")
    
    results = {
        "Success": 0,
        "Skipped (Low quality raw cells)": 0,
        "Errors": 0
    }
    
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
                    # 这里的 Key 要和上面 results 里的 key 对应
                    results["Skipped (Low quality raw cells)"] += 1
                else:
                    results["Errors"] += 1
                    
            except Exception as exc:
                tqdm.write(f"Critical exception for {file_path}: {exc}")
                results["Errors"] += 1
            
            pbar.update(1)
            
        pbar.close()

    print("\n" + "="*30)
    print("处理完成")
    print(f"Success: {results['Success']}")
    print(f"Skipped: {results['Skipped (Low quality raw cells)']}")
    print(f"Errors : {results['Errors']}")
    print(f"数据保存到目录: {OUTPUT_BASE_URI}/")
    print("="*30)