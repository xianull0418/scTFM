import pyrootutils
import torch.multiprocessing as mp


root = pyrootutils.setup_root(
    search_from=__file__,
    indicator=[".git", "requirements.txt"],
    pythonpath=True,
    dotenv=True,
)

import hydra
from omegaconf import DictConfig, OmegaConf
import pytorch_lightning as pl
from pytorch_lightning import Callback, LightningDataModule, LightningModule, Trainer
from pytorch_lightning.loggers import Logger
from pytorch_lightning.loggers.wandb import WandbLogger
from typing import List, Optional

from src.utils.pylogger import get_pylogger

log = get_pylogger(__name__)


def log_hyperparameters_to_wandb(
    cfg: DictConfig,
    loggers: List[Logger],
) -> None:
    """将Hydra配置记录到WandB。

    记录内容包括:
    - model: 模型配置 (包括net子配置)
    - data: 数据配置
    - trainer: 训练器配置
    - callbacks: 回调配置
    - task_name, seed 等顶层配置
    """
    # 找到WandbLogger
    wandb_logger: Optional[WandbLogger] = None
    for logger in loggers:
        if isinstance(logger, WandbLogger):
            wandb_logger = logger
            break

    if wandb_logger is None:
        log.warning("No WandbLogger found, skipping hyperparameter logging.")
        return

    # DDP模式下，只有rank 0的进程有真正的wandb实例
    # 检查experiment是否是真正的wandb run对象
    experiment = wandb_logger.experiment
    if not hasattr(experiment, 'config') or not hasattr(experiment.config, 'update'):
        # 非主进程，跳过
        return

    # 将OmegaConf配置转换为可序列化的字典
    hparams = {}

    # 记录关键配置组
    config_keys = ["model", "data", "trainer", "callbacks", "task_name", "seed", "train", "test"]
    for key in config_keys:
        if key in cfg:
            value = cfg[key]
            # 检查是否为 OmegaConf 对象（DictConfig 或 ListConfig）
            if OmegaConf.is_config(value):
                # 使用 OmegaConf.to_container 转换为原生Python对象
                # resolve=True 会解析所有插值 (如 ${paths.output_dir})
                hparams[key] = OmegaConf.to_container(value, resolve=True)
            else:
                # 原始类型直接赋值
                hparams[key] = value

    # 记录到wandb
    experiment.config.update(hparams, allow_val_change=True)
    log.info("Hyperparameters logged to WandB successfully.")

@hydra.main(version_base="1.3", config_path="../configs", config_name="train.yaml")
def main(cfg: DictConfig):
    # 1. Seed
    if cfg.get("seed"):
        pl.seed_everything(cfg.seed, workers=True)

    # 2. DataModule
    log.info(f"Instantiating datamodule <{cfg.data._target_}>")
    datamodule: LightningDataModule = hydra.utils.instantiate(cfg.data)

    # 3. Model
    # 注意：这里的 cfg.model 已经包含了 net 的完整配置 (由 defaults 注入)
    log.info(f"Instantiating model <{cfg.model._target_}>")
    model: LightningModule = hydra.utils.instantiate(cfg.model)

    # 4. Callbacks
    callbacks: List[Callback] = []
    if cfg.get("callbacks"):
        for _, cb_conf in cfg.callbacks.items():
            if "_target_" in cb_conf:
                callbacks.append(hydra.utils.instantiate(cb_conf))

    # 5. Logger
    logger: List[Logger] = []
    if cfg.get("logger"):
        for _, lg_conf in cfg.logger.items():
            if "_target_" in lg_conf:
                logger.append(hydra.utils.instantiate(lg_conf))

    # 5.1 记录超参数到WandB
    if logger:
        log_hyperparameters_to_wandb(cfg, logger)

    # 6. Trainer
    log.info(f"Instantiating trainer <{cfg.trainer._target_}>")
    trainer: Trainer = hydra.utils.instantiate(
        cfg.trainer,
        callbacks=callbacks,
        logger=logger,
    )

    # 7. Train
    if cfg.get("train"):
        log.info("Starting training!")
        trainer.fit(model=model, datamodule=datamodule, ckpt_path=cfg.get("ckpt_path"))

    # 8. Test
    if cfg.get("test"):
        log.info("Starting testing!")
        ckpt_path = trainer.checkpoint_callback.best_model_path
        if ckpt_path == "":
            log.warning("Best ckpt not found! Using current weights for testing...")
            ckpt_path = None
        trainer.test(model=model, datamodule=datamodule, ckpt_path=ckpt_path)

    pass
if __name__ == "__main__":
    # 我们在读取数据的时候关锁了
    # TileDB 底层是复杂的 C++ 库。当 Python 进程被 fork 时，父进程中的 C++ 状态（如线程池、文件句柄）会被复制到子进程中，但这种复制往往是不安全的（Not Fork-Safe）。
    # 因此，我们需要使用 spawn 模式来创建子进程，以确保数据读取的安全性。
    mp.set_start_method('spawn', force=True)


    main()