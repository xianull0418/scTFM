from typing import Any, Dict, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from pytorch_lightning import LightningModule

class AELitModule(LightningModule):
    """
    Autoencoder 统一训练模块。
    
    支持多种变体：
    1. Vanilla AE: 仅 MSE Loss
    2. VAE: MSE + KL Divergence (网络需输出 mu, logvar)
    3. RAE (Regularized AE): MSE + L2 Regularization on Z (Latent)
    4. SAE (Sparse AE): MSE + L1 Regularization on Z (Latent)
    """

    def __init__(
        self,
        net: nn.Module,
        optimizer: torch.optim.Optimizer,
        scheduler: torch.optim.lr_scheduler,
        compile: bool = False,
        kld_weight: float = 0.00001, # VAE KL 权重
        reg_type: str = "none",      # 'none' | 'l2' (RAE) | 'l1' (SAE)
        reg_weight: float = 1e-4,    # 正则化权重
    ):
        """
        Args:
            net: 网络实例
            optimizer: 优化器
            scheduler: 调度器
            compile: 是否编译
            kld_weight: VAE KL 散度权重
            reg_type: 正则化类型 ('l2' 为 RAE, 'l1' 为 SAE)
            reg_weight: 正则化强度
        """
        super().__init__()

        self.save_hyperparameters(logger=False, ignore=["net"])
        self.net = net
        
        if compile and hasattr(torch, "compile"):
            self.net = torch.compile(self.net)

    def forward(self, x: torch.Tensor):
        return self.net(x)

    def on_train_start(self):
        self.log("val/loss", 0.0, sync_dist=True)

    def model_step(self, batch: Any):
        # 1. 获取数据
        x, _ = batch
        
        # 2. 前向传播
        outputs = self.forward(x)
        
        # 3. 计算 Loss
        # -----------------------------------------------
        # Case A: VAE (返回 4 个值: recon, z, mu, logvar)
        # -----------------------------------------------
        if len(outputs) == 4:
            recon_x, z, mu, logvar = outputs
            
            # Reconstruction
            loss_recon = F.mse_loss(recon_x, x)
            
            # KL Divergence
            loss_kl = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
            loss_kl = loss_kl / x.shape[0] # Batch mean
            
            loss = loss_recon + self.hparams.kld_weight * loss_kl
            
            self.log("train/recon_loss", loss_recon, on_step=False, on_epoch=True)
            self.log("train/kl_loss", loss_kl, on_step=False, on_epoch=True)

        # -----------------------------------------------
        # Case B: Deterministic AE (返回 2 个值: recon, z)
        # -----------------------------------------------
        elif len(outputs) == 2:
            recon_x, z = outputs
            loss_recon = F.mse_loss(recon_x, x)
            loss = loss_recon
            
            # RAE (L2 Regularization)
            if self.hparams.reg_type == 'l2':
                # L2 norm of latent code
                loss_reg = torch.mean(torch.sum(z ** 2, dim=1)) * self.hparams.reg_weight
                loss += loss_reg
                self.log("train/reg_loss_l2", loss_reg, on_step=False, on_epoch=True)
                
            # SAE (L1 Regularization - Sparsity)
            elif self.hparams.reg_type == 'l1':
                # L1 norm of latent code
                loss_reg = torch.mean(torch.sum(torch.abs(z), dim=1)) * self.hparams.reg_weight
                loss += loss_reg
                self.log("train/reg_loss_l1", loss_reg, on_step=False, on_epoch=True)
                
            self.log("train/recon_loss", loss_recon, on_step=False, on_epoch=True)
            
        else:
            raise ValueError(f"Unexpected output format: len={len(outputs)}")

        return loss, recon_x, z

    def training_step(self, batch: Any, batch_idx: int):
        loss, _, _ = self.model_step(batch)
        self.log("train/loss", loss, on_step=False, on_epoch=True, prog_bar=True)
        return loss

    def validation_step(self, batch: Any, batch_idx: int):
        loss, _, _ = self.model_step(batch)
        self.log("val/loss", loss, on_step=False, on_epoch=True, prog_bar=True)
        return loss

    def test_step(self, batch: Any, batch_idx: int):
        loss, _, _ = self.model_step(batch)
        self.log("test/loss", loss, on_step=False, on_epoch=True, prog_bar=True)
        return loss

    def configure_optimizers(self):
        optimizer = self.hparams.optimizer(params=self.parameters())
        if self.hparams.scheduler is not None:
            scheduler = self.hparams.scheduler(optimizer=optimizer)
            return {
                "optimizer": optimizer,
                "lr_scheduler": {
                    "scheduler": scheduler,
                    "monitor": "val/loss",
                    "interval": "epoch",
                    "frequency": 1,
                },
            }
        return {"optimizer": optimizer}
