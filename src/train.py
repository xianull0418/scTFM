import pyrootutils
import torch.multiprocessing as mp
import torch
import os

root = pyrootutils.setup_root(
    search_from=__file__,
    indicator=[".git", "requirements.txt"],
    pythonpath=True,
    dotenv=True,
)

import hydra
import wandb
from omegaconf import DictConfig, OmegaConf
import pytorch_lightning as pl
from pytorch_lightning import Callback, LightningDataModule, LightningModule, Trainer
from pytorch_lightning.loggers import Logger
from pytorch_lightning.loggers.wandb import WandbLogger
from typing import List, Optional

from src.utils.pylogger import get_pylogger
from src.utils.dataset_stats_utils import get_dataset_stats

log = get_pylogger(__name__)


def log_hyperparameters_to_wandb(
    cfg: DictConfig,
    loggers: List[Logger],
) -> None:
    """将Hydra配置记录到WandB。"""
    wandb_logger: Optional[WandbLogger] = None
    for logger in loggers:
        if isinstance(logger, WandbLogger):
            wandb_logger = logger
            break

    if wandb_logger is None:
        log.warning("No WandbLogger found, skipping hyperparameter logging.")
        return

    experiment = wandb_logger.experiment
    if not hasattr(experiment, 'config') or not hasattr(experiment.config, 'update'):
        return

    hparams = {}
    config_keys = ["model", "data", "trainer", "callbacks", "task_name", "seed", "train", "test"]
    for key in config_keys:
        if key in cfg:
            value = cfg[key]
            if OmegaConf.is_config(value):
                hparams[key] = OmegaConf.to_container(value, resolve=True)
            else:
                hparams[key] = value

    experiment.config.update(hparams, allow_val_change=True)
    log.info("Hyperparameters logged to WandB successfully.")

@hydra.main(version_base="1.3", config_path="../configs", config_name="train.yaml")
def main(cfg: DictConfig):
    # ---------------------------------------------------------------------------
    # [DDP 关键修复] 
    # 1. 只有主进程 (Rank 0) 负责计算统计信息，避免所有进程同时读 IO 导致死锁或拥塞
    # 2. 统计信息计算移到 seed 之前，防止 DDP 初始化后的干扰
    # ---------------------------------------------------------------------------
    
    # 检测是否是主进程 (在 Hydra/Lightning 初始化前比较 tricky，只能靠环境变量)
    # PyTorch Lightning DDP 启动时会设置 LOCAL_RANK
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    limit_train_batches = None
    
    if cfg.get("train") and local_rank == 0:
        try:
            data_dir = cfg.data.get("data_dir")
            batch_size = cfg.data.get("batch_size", 256)
            
            # 计算 World Size
            if cfg.trainer.get("devices") == "auto":
                world_size = torch.cuda.device_count()
            elif isinstance(cfg.trainer.get("devices"), list):
                world_size = len(cfg.trainer.get("devices"))
            elif isinstance(cfg.trainer.get("devices"), int):
                world_size = cfg.trainer.get("devices")
            else:
                world_size = 1
                
            log.info(f"Rank {local_rank}: Calculating total dataset size (World Size={world_size})...")
            
            # 使用多进程加速
            total_cells, total_steps = get_dataset_stats(
                root_dir=data_dir,
                split_label=0, 
                batch_size=batch_size,
                num_workers=32, # 可以开大一点，因为是 IO 密集型
                world_size=world_size
            )
            
            if total_steps > 0:
                limit_train_batches = total_steps
                log.info(f"Rank {local_rank}: Calculated limit_train_batches = {limit_train_batches}")
                
        except Exception as e:
            log.warning(f"Failed to calculate stats: {e}")

    # ---------------------------------------------------------------------------
    # [Hydra/Lightning 初始化]
    # ---------------------------------------------------------------------------

    # 1. Seed
    if cfg.get("seed"):
        pl.seed_everything(cfg.seed, workers=True)

    # 2. DataModule
    log.info(f"Instantiating datamodule <{cfg.data._target_}>")
    datamodule: LightningDataModule = hydra.utils.instantiate(cfg.data)

    # 3. 注入 limit_train_batches
    # 注意：在 DDP 中，必须确保所有进程的配置一致，否则会报错。
    # 但 limit_train_batches 如果只在 Rank 0 设置，其他 Rank 默认为 infinite，可能会不同步。
    # 不过 Lightning 内部通常会广播这个值，或者我们需要确保手动广播。
    # 更安全的做法是：在 rank 0 计算后，不通过 Hydra 注入，而是通过 Trainer 参数？
    # 不，Hydra 配置是各自加载的。
    # 既然我们无法简单地在进程间通信（还没初始化 DDP），我们只能让每个进程都算，或者接受 Rank 0 算完（稍微慢点），其他进程也算（并发 IO）。
    
    # 修正策略：
    # 上面的 local_rank == 0 策略在 DDP spawn 模式下可能不生效（因为还没 spawn）。
    # 如果是用 torchrun 启动，环境变量已有。如果是 Lightning 内部 spawn，则还没。
    # 考虑到我们用的是 srun 或 python train.py (单机多卡)，通常是 spawn。
    
    # 最稳妥的方式：
    # 直接在所有进程都算。但为了解决"慢"和"卡死"，我们采用了优化后的 ProcessPool。
    # ProcessPool 在 DDP 启动前运行是安全的。
    
    if cfg.get("train"):
        # 重新计算（这次是快速的多进程版本）
        try:
            data_dir = cfg.data.get("data_dir")
            batch_size = cfg.data.get("batch_size", 256)
             # 计算 World Size
            if cfg.trainer.get("devices") == "auto":
                world_size = torch.cuda.device_count()
            elif isinstance(cfg.trainer.get("devices"), list):
                world_size = len(cfg.trainer.get("devices"))
            elif isinstance(cfg.trainer.get("devices"), int):
                world_size = cfg.trainer.get("devices")
            else:
                world_size = 1
            
            total_cells, total_steps = get_dataset_stats(
                root_dir=data_dir,
                split_label=0, 
                batch_size=batch_size,
                num_workers=16, # 适度并发，避免打死文件系统
                world_size=world_size
            )
            
            if total_steps > 0:
                OmegaConf.set_struct(cfg, False)
                cfg.trainer.limit_train_batches = total_steps
                OmegaConf.set_struct(cfg, True)
                
        except Exception:
            pass

    # 4. Model
    log.info(f"Instantiating model <{cfg.model._target_}>")
    model: LightningModule = hydra.utils.instantiate(cfg.model)

    # 5. Callbacks
    callbacks: List[Callback] = []
    if cfg.get("callbacks"):
        for _, cb_conf in cfg.callbacks.items():
            if "_target_" in cb_conf:
                callbacks.append(hydra.utils.instantiate(cb_conf))

    # 6. Logger
    logger: List[Logger] = []
    if cfg.get("logger"):
        for _, lg_conf in cfg.logger.items():
            if "_target_" in lg_conf:
                logger.append(hydra.utils.instantiate(lg_conf))

    if logger:
        log_hyperparameters_to_wandb(cfg, logger)

    # 7. Trainer
    log.info(f"Instantiating trainer <{cfg.trainer._target_}>")
    trainer: Trainer = hydra.utils.instantiate(
        cfg.trainer,
        callbacks=callbacks,
        logger=logger,
    )

    # 8. Train
    if cfg.get("train"):
        log.info("Starting training!")
        trainer.fit(model=model, datamodule=datamodule, ckpt_path=cfg.get("ckpt_path"))

    # 9. Test
    if cfg.get("test"):
        log.info("Starting testing!")
        ckpt_path = trainer.checkpoint_callback.best_model_path
        if ckpt_path == "":
            log.warning("Best ckpt not found! Using current weights for testing...")
            ckpt_path = None
        trainer.test(model=model, datamodule=datamodule, ckpt_path=ckpt_path)

    wandb.finish()

if __name__ == "__main__":
    # [关键] 必须设置为 spawn，否则 ProcessPoolExecutor 和 Lightning DDP 都会出问题
    mp.set_start_method('spawn', force=True)
    main()
