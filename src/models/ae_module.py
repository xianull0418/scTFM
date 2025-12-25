"""
Autoencoder Lightning Module

支持多种自编码器变体:
1. Vanilla AE: 仅 MSE Loss
2. VAE: MSE + KL Divergence
3. RAE (Regularized AE): MSE + L2 Regularization on Z
4. SAE (Sparse AE): MSE + L1 Regularization on Z
5. SCAE: Single-Cell AE with distribution-specific loss (Gaussian/NB/ZINB)
6. SCVAE: Single-Cell VAE with distribution-specific loss + KL
7. SCVIAE: scVI-style AE with NB loss on raw counts (推荐)
8. SCVIVAE: scVI-style VAE with NB loss + KL divergence (推荐)
"""

from typing import Any, Dict, Tuple, Union, Literal, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from pytorch_lightning import LightningModule

from src.models.components.ae.losses import SCLoss, log_nb_positive, log_zinb_positive


class AELitModule(LightningModule):
    """
    Autoencoder 统一训练模块。

    支持多种变体：
    1. Vanilla AE: 仅 MSE Loss
    2. VAE: MSE + KL Divergence (网络需输出 mu, logvar)
    3. RAE (Regularized AE): MSE + L2 Regularization on Z (Latent)
    4. SAE (Sparse AE): MSE + L1 Regularization on Z (Latent)
    5. SCAE: Single-Cell AE with distribution-specific loss
    6. SCVAE: Single-Cell VAE with distribution-specific loss + KL
    7. SCVIAE: scVI-style AE (推荐) - 使用 library_size 分离技术噪声
    8. SCVIVAE: scVI-style VAE (推荐) - 同上 + KL 正则化

    新增功能:
    - 支持 Gaussian/NB/ZINB 输出分布
    - KL annealing (warmup)
    - scVI-style: NB loss 在 raw counts 上计算
    - 更好的损失记录
    """

    def __init__(
        self,
        net: nn.Module,
        optimizer: torch.optim.Optimizer,
        scheduler: torch.optim.lr_scheduler,
        compile: bool = False,
        # === Loss 类型 ===
        loss_type: Literal['mse', 'gaussian', 'nb', 'zinb'] = 'mse',
        # === VAE 参数 ===
        kld_weight: float = 0.001,       # VAE KL 权重 (提高默认值)
        kl_annealing: bool = False,      # 是否使用 KL annealing
        kl_warmup_epochs: int = 10,      # KL warmup epochs
        # === 正则化参数 ===
        reg_type: str = "none",          # 'none' | 'l2' (RAE) | 'l1' (SAE)
        reg_weight: float = 1e-4,        # 正则化强度
    ):
        """
        Args:
            net: 网络模型 (VanillaAE, VariationalAE, SCAE, SCVAE)
            optimizer: 优化器配置
            scheduler: 学习率调度器配置
            compile: 是否使用 torch.compile
            loss_type: 损失类型
                - 'mse': 均方误差（原始实现，向后兼容）
                - 'gaussian': 高斯负对数似然（可学习方差）
                - 'nb': 负二项分布（适用于 count 数据）
                - 'zinb': 零膨胀负二项（适用于稀疏 count）
            kld_weight: KL 散度权重
            kl_annealing: 是否使用 KL annealing (从 0 逐渐增加到 kld_weight)
            kl_warmup_epochs: KL annealing 的 warmup epochs
            reg_type: 正则化类型 ('none', 'l2', 'l1')
            reg_weight: 正则化权重
        """
        super().__init__()

        self.save_hyperparameters(logger=False, ignore=["net"])
        self.net = net

        # 检测网络类型
        self._is_scae = hasattr(net, 'distribution')
        self._is_scvi = hasattr(net, 'gene_likelihood')  # SCVIAE/SCVIVAE 标识
        self._is_vae = self._check_is_vae(net)

        # 单细胞损失函数
        if self._is_scvi:
            # scVI-style: 使用 NB/ZINB loss，需要 counts 和 library_size
            self.sc_loss = None  # 在 step 中手动计算
        elif self._is_scae:
            # SCAE/SCVAE: 使用网络定义的分布类型
            self.sc_loss = SCLoss(loss_type=net.distribution, reduction='mean')
        elif loss_type != 'mse':
            # 非 SCAE 但指定了特殊损失类型
            self.sc_loss = SCLoss(loss_type=loss_type, reduction='mean')
        else:
            self.sc_loss = None

        if compile and hasattr(torch, "compile"):
            self.net = torch.compile(self.net)

    def _check_is_vae(self, net: nn.Module) -> bool:
        """检测是否为 VAE 类型网络"""
        # 检查 forward 输出是否包含 mu, logvar
        if hasattr(net, 'fc_logvar') or hasattr(net, 'reparameterize'):
            return True
        # SCVAE 检测
        if hasattr(net, 'encode'):
            import inspect
            sig = inspect.signature(net.encode)
            # SCVAE.encode 返回 (mu, logvar)
            return True
        return False

    def forward(self, x: torch.Tensor):
        return self.net(x)

    def on_train_start(self):
        # 初始化验证损失，防止第一次 sanity check 前报错
        self.log("val/loss", 0.0, sync_dist=True)

    def get_kl_weight(self) -> float:
        """获取当前 KL 权重（支持 annealing）"""
        if not self.hparams.kl_annealing:
            return self.hparams.kld_weight

        # Linear annealing
        current_epoch = self.current_epoch
        warmup = self.hparams.kl_warmup_epochs

        if current_epoch >= warmup:
            return self.hparams.kld_weight
        else:
            return self.hparams.kld_weight * (current_epoch / warmup)

    def model_step(self, batch: Any) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        统一的模型步骤

        Returns:
            loss: 总损失
            recon_x: 重建结果
            z: 潜在表示
        """
        # 1. 获取数据 - 支持多种 batch 格式
        if isinstance(batch, dict):
            # scVI-style batch: {'x': log1p, 'counts': raw, 'library_size': lib}
            x = batch['x']
            counts = batch.get('counts', None)
            library_size = batch.get('library_size', None)
        elif isinstance(batch, (tuple, list)) and len(batch) == 2:
            # Legacy batch: (x, labels) 或 (x, _)
            x, _ = batch
            counts = None
            library_size = None
        else:
            x = batch
            counts = None
            library_size = None

        # 2. 根据网络类型选择计算方式
        if self._is_scvi:
            return self._scvi_step(x, counts, library_size)
        elif self._is_scae:
            outputs = self.forward(x)
            return self._scae_step(x, outputs)
        else:
            outputs = self.forward(x)
            return self._legacy_step(x, outputs)

    def _scvi_step(
        self,
        x: torch.Tensor,
        counts: torch.Tensor,
        library_size: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        scVI-style 损失计算

        核心设计:
        - 输入: log1p(normalized) -> Encoder
        - 输出: softmax(rho) × library_size = NB mean
        - Loss: -log NB(counts | mu, theta)

        Args:
            x: log1p(normalized), shape (batch, n_genes)
            counts: normalized counts (非 log), shape (batch, n_genes)
            library_size: total UMI per cell, shape (batch,)
        """
        # 检测是 SCVIAE 还是 SCVIVAE
        from src.models.components.ae.scvi_ae import SCVIVAE

        is_vae = isinstance(self.net, SCVIVAE)

        if is_vae:
            # SCVIVAE: forward 返回 (mu, z, q_mu, q_logvar, outputs)
            mu, z, q_mu, q_logvar, outputs = self.net(x, library_size)
        else:
            # SCVIAE: forward 返回 (mu, z, outputs)
            mu, z, outputs = self.net(x, library_size)

        # NB 参数
        theta = outputs['theta']

        # 如果没有提供 counts，从 x 反推 (fallback)
        if counts is None:
            counts = torch.expm1(x)  # log1p 的逆

        # NB 负对数似然
        if self.net.gene_likelihood == 'zinb':
            pi = outputs.get('pi', None)
            if pi is not None:
                log_likelihood = log_zinb_positive(counts, mu, theta, pi)
            else:
                log_likelihood = log_nb_positive(counts, mu, theta)
        else:
            log_likelihood = log_nb_positive(counts, mu, theta)

        # 负对数似然作为损失
        loss_recon = -log_likelihood.mean()

        self.log("train/recon_loss", loss_recon, on_step=False, on_epoch=True, sync_dist=True)

        loss = loss_recon

        # KL 散度 (SCVIVAE)
        if is_vae:
            loss_kl = SCVIVAE.kl_divergence(q_mu, q_logvar).mean()

            kl_weight = self.get_kl_weight()
            loss = loss + kl_weight * loss_kl

            self.log("train/kl_loss", loss_kl, on_step=False, on_epoch=True, sync_dist=True)
            self.log("train/kl_weight", kl_weight, on_step=False, on_epoch=True, sync_dist=True)

        # 正则化 (L1/L2)
        loss = self._add_regularization(loss, z)

        return loss, mu, z

    def _scae_step(self, x: torch.Tensor, outputs: tuple) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """SCAE/SCVAE 的损失计算"""

        # 判断是 SCAE 还是 SCVAE (根据输出数量)
        if len(outputs) == 5:
            # SCVAE: (recon_x, z, q_mu, q_logvar, output_dict)
            recon_x, z, q_mu, q_logvar, output_dict = outputs
            is_vae = True
        elif len(outputs) == 3:
            # SCAE: (recon_x, z, output_dict)
            recon_x, z, output_dict = outputs
            is_vae = False
        else:
            raise ValueError(f"Unexpected SCAE output format: len={len(outputs)}")

        # 重建损失 (使用分布特定的损失)
        loss_recon = self.sc_loss(x, output_dict)

        self.log("train/recon_loss", loss_recon, on_step=False, on_epoch=True, sync_dist=True)

        loss = loss_recon

        # KL 损失 (SCVAE)
        if is_vae:
            from src.models.components.ae.scvae import SCVAE
            loss_kl = SCVAE.kl_divergence(q_mu, q_logvar).mean()

            kl_weight = self.get_kl_weight()
            loss = loss + kl_weight * loss_kl

            self.log("train/kl_loss", loss_kl, on_step=False, on_epoch=True, sync_dist=True)
            self.log("train/kl_weight", kl_weight, on_step=False, on_epoch=True, sync_dist=True)

        # 正则化 (L1/L2)
        loss = self._add_regularization(loss, z)

        return loss, recon_x, z

    def _legacy_step(self, x: torch.Tensor, outputs: tuple) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """原始 AE/VAE 的损失计算（向后兼容）"""

        if len(outputs) == 4:
            # VAE: (recon_x, z, mu, logvar)
            recon_x, z, mu, logvar = outputs
            loss_recon = F.mse_loss(recon_x, x)
            loss_kl = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
            loss_kl = loss_kl / x.shape[0]

            kl_weight = self.get_kl_weight()
            loss = loss_recon + kl_weight * loss_kl

            self.log("train/recon_loss", loss_recon, on_step=False, on_epoch=True, sync_dist=True)
            self.log("train/kl_loss", loss_kl, on_step=False, on_epoch=True, sync_dist=True)
            self.log("train/kl_weight", kl_weight, on_step=False, on_epoch=True, sync_dist=True)

        elif len(outputs) == 2:
            # AE / RAE / SAE: (recon_x, z)
            recon_x, z = outputs

            if self.sc_loss is not None:
                # 使用特殊损失
                loss_recon = self.sc_loss(x, {'mu': recon_x})
            else:
                loss_recon = F.mse_loss(recon_x, x)

            loss = loss_recon
            self.log("train/recon_loss", loss_recon, on_step=False, on_epoch=True, sync_dist=True)

        else:
            raise ValueError(f"Unexpected output format: len={len(outputs)}")

        # 正则化
        loss = self._add_regularization(loss, z)

        return loss, recon_x, z

    def _add_regularization(self, loss: torch.Tensor, z: torch.Tensor) -> torch.Tensor:
        """添加正则化项"""
        if self.hparams.reg_type == 'l2':
            loss_reg = torch.mean(torch.sum(z ** 2, dim=1)) * self.hparams.reg_weight
            loss = loss + loss_reg
            self.log("train/reg_loss_l2", loss_reg, on_step=False, on_epoch=True, sync_dist=True)

        elif self.hparams.reg_type == 'l1':
            loss_reg = torch.mean(torch.sum(torch.abs(z), dim=1)) * self.hparams.reg_weight
            loss = loss + loss_reg
            self.log("train/reg_loss_l1", loss_reg, on_step=False, on_epoch=True, sync_dist=True)

        return loss

    def training_step(self, batch: Any, batch_idx: int):
        loss, _, _ = self.model_step(batch)
        self.log("train/loss", loss, on_step=False, on_epoch=True, prog_bar=True, sync_dist=True)
        return loss

    def validation_step(self, batch: Any, batch_idx: int):
        loss, _, _ = self.model_step(batch)
        self.log("val/loss", loss, on_step=False, on_epoch=True, prog_bar=True, sync_dist=True)
        return loss

    def test_step(self, batch: Any, batch_idx: int):
        loss, _, _ = self.model_step(batch)
        self.log("test/loss", loss, on_step=False, on_epoch=True, prog_bar=True, sync_dist=True)
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
