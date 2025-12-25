import torch
from torch.utils.data import IterableDataset
import tiledbsoma
import numpy as np
import pandas as pd
import math
import os
import random
import gc
from typing import Optional, Dict, List

# ============================================================================
# 时间编码常量和函数
# ============================================================================
# 时间范围（天）：从受精卵开始
# - 最小：~0 天（受精卵）
# - 最大：~36500 天（100岁）
# 使用 log-scale 归一化，压缩大范围同时保留早期分辨率

MAX_TIME_DAYS = 36500.0  # 100 岁，作为归一化上界
MAX_DELTA_DAYS = 36500.0  # delta_t 也用相同范围

def normalize_time(time_days: float, use_log: bool = True) -> float:
    """
    将绝对时间（天）归一化到 [0, 1] 范围。

    使用 log-scale 归一化的好处：
    - 压缩大范围（成人时间）
    - 保留早期发育的分辨率（胚胎/胎儿期）

    示例：
    - 0 天 (受精卵) -> 0.0
    - 7 天 (囊胚) -> ~0.18
    - 280 天 (出生) -> ~0.53
    - 365 天 (1岁) -> ~0.56
    - 3650 天 (10岁) -> ~0.78
    - 11000 天 (30岁) -> ~0.89
    - 36500 天 (100岁) -> 1.0
    """
    if use_log:
        # log(1 + t) / log(1 + max_t) 归一化到 [0, 1]
        return math.log(1.0 + time_days) / math.log(1.0 + MAX_TIME_DAYS)
    else:
        # 简单线性归一化
        return min(time_days / MAX_TIME_DAYS, 1.0)


def normalize_delta_t(delta_days: float, use_log: bool = True) -> float:
    """
    将时间差（天）归一化到 [-1, 1] 范围（支持负值，即 backward flow）。

    使用对称的 log-scale：sign(dt) * log(1 + |dt|) / log(1 + max_dt)
    """
    if use_log:
        sign = 1.0 if delta_days >= 0 else -1.0
        abs_delta = abs(delta_days)
        return sign * math.log(1.0 + abs_delta) / math.log(1.0 + MAX_DELTA_DAYS)
    else:
        return max(min(delta_days / MAX_DELTA_DAYS, 1.0), -1.0)


# ============================================================================
# 全局类别映射（根据 TEDD 数据集统计）
# ============================================================================
# 用于将字符串类别转换为整数 ID，供 DiT 的 Embedding 层使用
# ID 0 保留给 "Unknown" 或未知类别

# 发育阶段类别
STAGE_CATEGORIES = [
    "Unknown",      # ID 0 (保留)
    "Embryonic",    # ID 1: 胚胎期（受精后 ~2 周内）
    "Fetal",        # ID 2: 胎儿期（~2周 - 出生）
    "Newborn",      # ID 3: 新生儿期（出生后 ~1个月）
    "Paediatric",   # ID 4: 儿童期（~1个月 - 18岁）
    "Adult",        # ID 5: 成人期（18岁+）
]

STAGE_TO_ID = {name: idx for idx, name in enumerate(STAGE_CATEGORIES)}
# 添加一些常见的别名映射
STAGE_TO_ID.update({
    "unknown": 0,
    "embryonic": 1,
    "fetal": 2,
    "newborn": 3,
    "neonatal": 3,  # 别名
    "paediatric": 4,
    "pediatric": 4,  # 美式拼写
    "child": 4,      # 别名
    "adult": 5,
})

def encode_stage(stage_name: str) -> int:
    """将 Stage 名称编码为整数 ID"""
    if stage_name is None:
        return 0
    # 尝试精确匹配，然后尝试小写匹配
    stage_str = str(stage_name).strip()
    if stage_str in STAGE_TO_ID:
        return STAGE_TO_ID[stage_str]
    stage_lower = stage_str.lower()
    if stage_lower in STAGE_TO_ID:
        return STAGE_TO_ID[stage_lower]
    return 0  # Unknown


TISSUE_CATEGORIES = [
    "Unknown",  # ID 0 (保留)
    "Bone Marrow",
    "Epiblast",
    "Eye",
    "Intestine",
    "Kidney",
    "Placenta",
    "Primitive Endoderm",
    "Prostate",
    "Thymus",
    "Trophectoderm",
]

CELLTYPE_CATEGORIES = [
    "Unknown",  # ID 0 (保留)
    "AFP_ALB Positive Cell",
    "Amacrine Cell",
    "Antigen Presenting Cell",
    "Astrocyte",
    "B Cell",
    "Bipolar Cell",
    "Corneal and Conjunctival Epithelial Cell",
    "Dendritic Cell",
    "Endothelial Cell",
    "Endothelial Cell (Lymphatic)",
    "Endothelial Cell (Vascular)",
    "Eosinophil/Basophil/Mast Cell",
    "Epiblast",
    "Epithelial Cell",
    "Epithelial Cell (Basal)",
    "Epithelial Cell (Club)",
    "Epithelial Cell (Hillock)",
    "Epithelial Cell (Luminal)",
    "Erythroid Cell",
    "Erythroid Progenitor Cell",
    "Fibroblast",
    "Ganglion Cell",
    "Hematopoietic Stem Cell",
    "Horizontal Cell",
    "IGFBP1_DKK1 Positive Cell",
    "Lens Fibre Cell",
    "Lymphoid Cell",
    "Macrophage",
    "Mast Cell",
    "Megakaryocyte",
    "Mesangial Cell",
    "Mesodermal Killer Cell",
    "Mesothelial Cell",
    "Metanephric Cell",
    "Microglia",
    "Monocyte",
    "Myeloid Cell",
    "Myocyte (Skeletal Muscle)",
    "Myocyte (Smooth Muscle)",
    "Natural Killer Cell",
    "Natural Killer T Cell",
    "Neuroendocrine Cell",
    "Neuron",
    "Neutrophil",
    "PAEP_MECOM Positive Cell",
    "PDE11A_FAM19A2 Positive Cell",
    "Pericyte",
    "Pericyte/Myocyte (Smooth Muscle)",
    "Photoreceptor Cell",
    "Primitive Endoderm",
    "Proliferating T Cell",
    "Retinal Pigment Cell",
    "Retinal Progenitor and Muller Glia",
    "Schwann Cell",
    "Stroma",
    "Syncytiotrophoblasts and Villous Cytotrophoblasts",
    "T Cell",
    "Trophectoderm",
    "Trophoblast Giant Cell",
    "Ureteric Bud Cell",
]

# 构建快速查找字典
TISSUE_TO_ID = {name: idx for idx, name in enumerate(TISSUE_CATEGORIES)}
CELLTYPE_TO_ID = {name: idx for idx, name in enumerate(CELLTYPE_CATEGORIES)}

def encode_tissue(tissue_name: str) -> int:
    """将 tissue 名称编码为整数 ID"""
    return TISSUE_TO_ID.get(str(tissue_name), 0)  # 未知返回 0

def encode_celltype(celltype_name: str) -> int:
    """将 celltype 名称编码为整数 ID"""
    return CELLTYPE_TO_ID.get(str(celltype_name), 0)  # 未知返回 0


def load_stage_mapping(csv_path: str) -> Dict[str, int]:
    """
    从 tedd_info.csv 加载 Stage 映射。

    处理 ID 格式差异：
    - CSV 中的 ID: Tedd.1, Tedd.10_AdrenalGland_scrna
    - 处理后的目录名: Tedd_1, Tedd_10_AdrenalGland_scrna
    - 转换规则: 将 `.` 和 `-` 都替换为 `_`

    注意：有些数据集包含多个 Stage（如 "Fetal, Newborn, Adult"）
    我们取第一个 Stage 作为主要阶段。

    Returns:
        Dict[str, int]: 目录名 -> Stage ID 的映射
    """
    if not os.path.exists(csv_path):
        print(f"⚠️ Stage info CSV not found: {csv_path}")
        return {}

    try:
        df = pd.read_csv(csv_path)
        stage_map = {}

        for _, row in df.iterrows():
            original_id = str(row.get('ID', ''))
            stage_raw = row.get('Stage', 'Unknown')

            if not original_id or pd.isna(original_id):
                continue

            # 转换 ID 格式：将 `.` 和 `-` 都替换为 `_`
            normalized_id = original_id.replace('.', '_').replace('-', '_')

            # 处理多 Stage 情况：取第一个
            if pd.isna(stage_raw):
                stage = 'Unknown'
            else:
                stage = str(stage_raw).split(',')[0].strip()

            stage_id = encode_stage(stage)
            stage_map[normalized_id] = stage_id

            # 也保留原始 ID（以防万一）
            stage_map[original_id] = stage_id

        print(f"✅ Loaded Stage mapping for {len(stage_map)} entries")
        return stage_map

    except Exception as e:
        print(f"⚠️ Failed to load Stage mapping: {e}")
        return {}


# 默认的 Stage 信息路径
DEFAULT_STAGE_INFO_PATH = "/gpfs/flash/home/jcw/projects/research/cellTime/scTFM/data/tedd_info.csv"

# 全局缓存（避免重复加载）
_STAGE_MAP_CACHE: Optional[Dict[str, int]] = None


def get_stage_map(csv_path: Optional[str] = None) -> Dict[str, int]:
    """获取 Stage 映射（带缓存）"""
    global _STAGE_MAP_CACHE
    if _STAGE_MAP_CACHE is None:
        path = csv_path or DEFAULT_STAGE_INFO_PATH
        _STAGE_MAP_CACHE = load_stage_mapping(path)
    return _STAGE_MAP_CACHE


class SomaRTFDataset(IterableDataset):
    """
    用于 Rectified Flow 训练的 TileDB-SOMA 数据集（高效版本）。

    核心逻辑：
    1. 读取 TileDB shards，每个 shard 是一个独立的 Experiment
    2. 根据 obs 中的 'next_cell_id' 或 'prev_cell_id' 构建细胞对
    3. 返回格式: {x_curr, x_next, cond_meta}
    4. 支持 Latent 和 Raw 两种模式

    时间编码：
    - time_curr / time_next: 归一化到 [0, 1]（log-scale）
    - delta_t: 归一化到 [-1, 1]（对称 log-scale）
    - stage: 发育阶段 ID（从 tedd_info.csv 加载）

    优化特性（与 AE Dataset 一致）：
    - TileDB Context 内存优化配置
    - 预分配内存缓冲区复用
    - 分块读取 + 稀疏→稠密转换
    - SQL 风格过滤
    """

    def __init__(
        self,
        root_dir: str,
        split_label: int = 0,
        io_chunk_size: int = 16384,
        batch_size: int = 256,
        measurement_name: str = "RNA",
        latent_key: Optional[str] = None,
        direction: str = "forward",
        preloaded_sub_uris: Optional[List[str]] = None,
        shard_assignment: Optional[Dict[str, List[str]]] = None,
        stage_info_path: Optional[str] = None,
        use_log_time: bool = True,
    ):
        self.root_dir = root_dir
        self.split_label = split_label
        self.io_chunk_size = io_chunk_size
        self.batch_size = batch_size
        self.measurement_name = measurement_name
        self.latent_key = latent_key
        self.direction = direction
        self._n_vars = None
        self.shard_assignment = shard_assignment
        self.use_log_time = use_log_time

        # 加载 Stage 映射
        self.stage_map = get_stage_map(stage_info_path)

        if preloaded_sub_uris is not None:
            self._sub_uris = preloaded_sub_uris
        else:
            self._sub_uris = None

        if not os.path.exists(root_dir):
            raise ValueError(f"❌ 路径不存在: {root_dir}")

    @property
    def sub_uris(self):
        if self._sub_uris is None:
            self._sub_uris = sorted([
                os.path.join(self.root_dir, d)
                for d in os.listdir(self.root_dir)
                if os.path.isdir(os.path.join(self.root_dir, d))
            ])
            if len(self._sub_uris) == 0:
                raise ValueError(f"❌ 路径 {self.root_dir} 下没有发现子文件夹！")
        return self._sub_uris

    @property
    def n_vars(self):
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

    def _get_context(self):
        """返回优化后的 TileDB Context"""
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

        # 3. 预分配内存缓冲区（复用）
        n_vars = self.n_vars
        dense_buffer = np.zeros((self.io_chunk_size, n_vars), dtype=np.float32)

        try:
            for uri in my_worker_uris:
                try:
                    yield from self._process_shard(uri, ctx, dense_buffer, global_worker_id)
                except Exception as e:
                    print(f"⚠️ [Worker {global_worker_id}] 读取 Shard {os.path.basename(uri)} 失败: {e}")
                    continue
        finally:
            del dense_buffer
            gc.collect()

    def _process_shard(self, uri: str, ctx, dense_buffer: np.ndarray, worker_id: int):
        """处理单个 shard（高效版本）"""
        shard_name = os.path.basename(uri)  # 用于查找 Stage

        with tiledbsoma.Experiment.open(uri, context=ctx) as exp:
            # 1. 读取所有 obs 元数据（用于查找 next_cell）
            try:
                obs_all = exp.obs.read().concat().to_pandas()
            except Exception:
                return

            if len(obs_all) == 0:
                return

            # [关键] 确定使用哪个列作为细胞索引
            # 优先使用 obs_id，如果不存在或无法匹配 next_cell_id，则使用 new_index
            cell_id_col = None

            # 检查 next_cell_id 是否能匹配 obs_id 或 new_index
            if 'next_cell_id' in obs_all.columns:
                valid_next = obs_all[obs_all['next_cell_id'].notna()]['next_cell_id'].head(100)

                if 'obs_id' in obs_all.columns:
                    obs_id_set = set(obs_all['obs_id'].astype(str))
                    matches = sum(1 for x in valid_next if str(x) in obs_id_set)
                    if matches > len(valid_next) * 0.5:  # 超过 50% 匹配
                        cell_id_col = 'obs_id'

                if cell_id_col is None and 'new_index' in obs_all.columns:
                    new_index_set = set(obs_all['new_index'].astype(str))
                    matches = sum(1 for x in valid_next if str(x) in new_index_set)
                    if matches > len(valid_next) * 0.5:  # 超过 50% 匹配
                        cell_id_col = 'new_index'

            if cell_id_col is None:
                # 无法确定细胞索引列，跳过
                return

            # [关键] 使用确定的列作为索引
            obs_all = obs_all.set_index(cell_id_col)

            # 2. 筛选当前 split 的细胞作为"起点"
            obs_query = obs_all[obs_all['split_label'] == self.split_label]
            if len(obs_query) == 0:
                return

            # 3. 构建细胞对映射（允许 next_cell 来自任意 split）
            cell_pairs = self._build_pair_indices(obs_query, obs_all)
            if len(cell_pairs) == 0:
                return

            # 4. 获取需要读取的所有 cell IDs（包括 next_cell）
            all_cell_ids = set()
            for curr_idx, next_idx in cell_pairs:
                all_cell_ids.add(curr_idx)
                all_cell_ids.add(next_idx)

            # 获取对应的 soma_joinid
            soma_joinids = obs_all.loc[list(all_cell_ids), 'soma_joinid'].values
            soma_joinids = np.sort(soma_joinids)

            # 5. 分块读取 X 数据（高效稀疏读取）
            x_uri = os.path.join(uri, "ms", self.measurement_name, "X", "data")

            # 构建 soma_joinid -> 行索引 的映射
            joinid_to_row = {jid: i for i, jid in enumerate(soma_joinids)}
            obs_idx_to_row = {
                obs_idx: joinid_to_row[obs_all.loc[obs_idx, 'soma_joinid']]
                for obs_idx in all_cell_ids
            }

            n_cells = len(soma_joinids)
            n_vars = dense_buffer.shape[1]

            # 使用预分配的缓冲区（或扩展）
            if n_cells > dense_buffer.shape[0]:
                cell_buffer = np.zeros((n_cells, n_vars), dtype=np.float32)
            else:
                cell_buffer = dense_buffer[:n_cells]
                cell_buffer.fill(0)

            # 读取稀疏数据并填充
            with tiledbsoma.open(x_uri, mode='r', context=ctx) as X:
                data = X.read(coords=(soma_joinids, slice(None))).tables().concat()

                row_indices = data["soma_dim_0"].to_numpy()
                col_indices = data["soma_dim_1"].to_numpy()
                values = data["soma_data"].to_numpy()

                # 映射到本地行索引
                local_rows = np.searchsorted(soma_joinids, row_indices)
                cell_buffer[local_rows, col_indices] = values

            # 6. 构建细胞对并 yield（使用 obs_all 获取元数据）
            yield from self._yield_pairs(cell_pairs, obs_all, cell_buffer, obs_idx_to_row, shard_name)

    def _build_pair_indices(self, obs_query, obs_all) -> List[tuple]:
        """构建细胞对的索引列表

        Args:
            obs_query: 当前 split 的细胞（作为起点）
            obs_all: 所有细胞（next_cell 可以来自这里）
        """
        pairs = []
        all_indices = set(obs_all.index)  # next_cell 可以来自任意 split

        for obs_idx, row in obs_query.iterrows():
            if self.direction in ("forward", "both"):
                next_id = row.get('next_cell_id')
                if next_id is not None and next_id in all_indices:
                    pairs.append((obs_idx, next_id))

            if self.direction in ("backward", "both"):
                prev_id = row.get('prev_cell_id')
                if prev_id is not None and prev_id in all_indices:
                    pairs.append((prev_id, obs_idx))

        return pairs

    def _yield_pairs(self, cell_pairs, obs_df, cell_buffer, obs_idx_to_row, shard_name: str):
        """生成训练批次

        Args:
            cell_pairs: 细胞对列表 [(curr_idx, next_idx), ...]
            obs_df: obs DataFrame（以 obs_id 为索引）
            cell_buffer: 表达矩阵缓冲区
            obs_idx_to_row: obs_id -> buffer 行索引的映射
            shard_name: shard 名称（用于查找 Stage）
        """
        # 随机打乱
        random.shuffle(cell_pairs)

        n_pairs = len(cell_pairs)
        n_batches = math.ceil(n_pairs / self.batch_size)

        # 检查是否有 Celltype/Tissue 列
        has_celltype = 'Celltype' in obs_df.columns
        has_tissue = 'Tissue' in obs_df.columns

        # 获取该 shard 的 Stage ID
        shard_stage_id = self.stage_map.get(shard_name, 0)

        for b in range(n_batches):
            start = b * self.batch_size
            end = min(start + self.batch_size, n_pairs)
            batch_pairs = cell_pairs[start:end]

            if len(batch_pairs) <= 1:
                continue

            # 构建批次数据
            x_curr_list = []
            x_next_list = []
            time_curr_list = []
            time_next_list = []
            delta_t_list = []
            tissue_list = []
            celltype_list = []
            stage_list = []

            for curr_idx, next_idx in batch_pairs:
                curr_row = obs_df.loc[curr_idx]
                next_row = obs_df.loc[next_idx]

                x_curr_list.append(cell_buffer[obs_idx_to_row[curr_idx]])
                x_next_list.append(cell_buffer[obs_idx_to_row[next_idx]])

                # 获取原始时间（天）
                time_curr_raw = float(curr_row['time']) if curr_row['time'] is not None else 0.0
                time_next_raw = float(next_row['time']) if next_row['time'] is not None else 0.0
                delta_t_raw = time_next_raw - time_curr_raw

                # 归一化时间（使用 log-scale）
                time_curr_list.append(normalize_time(time_curr_raw, self.use_log_time))
                time_next_list.append(normalize_time(time_next_raw, self.use_log_time))
                delta_t_list.append(normalize_delta_t(delta_t_raw, self.use_log_time))

                tissue_list.append(encode_tissue(curr_row.get('Tissue', 'Unknown')) if has_tissue else 0)
                celltype_list.append(encode_celltype(curr_row.get('Celltype', 'Unknown')) if has_celltype else 0)
                stage_list.append(shard_stage_id)

            batch = {
                'x_curr': torch.tensor(np.stack(x_curr_list), dtype=torch.float32),
                'x_next': torch.tensor(np.stack(x_next_list), dtype=torch.float32),
                'cond_meta': {
                    'time_curr': torch.tensor(time_curr_list, dtype=torch.float32),
                    'time_next': torch.tensor(time_next_list, dtype=torch.float32),
                    'delta_t': torch.tensor(delta_t_list, dtype=torch.float32),
                    'tissue': torch.tensor(tissue_list, dtype=torch.long),
                    'celltype': torch.tensor(celltype_list, dtype=torch.long),
                    'stage': torch.tensor(stage_list, dtype=torch.long),
                }
            }

            yield batch

