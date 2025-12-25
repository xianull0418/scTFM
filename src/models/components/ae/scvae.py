"""
Single-Cell Variational Autoencoder (SCVAE)

专为单细胞 RNA-seq 数据设计的变分自编码器：
1. 残差连接 (ResNet-style) 防止梯度消失
2. 支持多种输出分布 (Gaussian, NB, ZINB)
3. Pre-LayerNorm 架构，更稳定的训练
4. 可学习的 dispersion 参数
5. 支持 KL annealing

参考:
- scVI: https://docs.scvi-tools.org
- DCA: https://github.com/theislab/dca
- β-VAE: Higgins et al., 2017
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, List, Literal, Tuple

from .scae import ResidualEncoder, ResidualDecoder


class SCVAE(nn.Module):
    """
    Single-Cell Variational Autoencoder

    专为单细胞数据设计的 VAE：
    - 残差连接防止梯度消失
    - 支持多种输出分布: 'gaussian', 'nb', 'zinb'
    - 可学习的 gene-specific dispersion
    - 变分推断 with reparameterization trick
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dims: List[int] = [512, 256],
        latent_dim: int = 64,
        dropout_rate: float = 0.1,
        activation: str = "GELU",
        distribution: Literal['gaussian', 'nb', 'zinb'] = 'gaussian',
        dispersion: Literal['gene', 'gene-cell'] = 'gene',
        var_eps: float = 1e-4,
        **kwargs
    ):
        """
        Args:
            input_dim: 输入维度（基因数）
            hidden_dims: 隐藏层维度列表
            latent_dim: 潜在空间维度
            dropout_rate: Dropout 比例
            activation: 激活函数 ('GELU', 'SiLU', 'ReLU', 'LeakyReLU')
            distribution: 输出分布类型
                - 'gaussian': 高斯分布（适用于 log1p 数据）
                - 'nb': 负二项分布（适用于 count 数据）
                - 'zinb': 零膨胀负二项（适用于稀疏 count 数据）
            dispersion: dispersion 参数类型
                - 'gene': 每个基因一个 dispersion（推荐）
                - 'gene-cell': 每个细胞-基因对一个 dispersion
            var_eps: 最小方差（数值稳定性）
        """
        super().__init__()

        self.input_dim = input_dim
        self.latent_dim = latent_dim
        self.distribution = distribution
        self.dispersion_type = dispersion
        self.var_eps = var_eps

        # Encoder
        self.encoder = ResidualEncoder(
            input_dim=input_dim,
            hidden_dims=hidden_dims,
            latent_dim=latent_dim,
            dropout_rate=dropout_rate,
            activation=activation
        )

        # VAE: 两个投影头 (mu, logvar)
        encoder_out_dim = hidden_dims[-1]
        self.fc_mu = nn.Linear(encoder_out_dim, latent_dim)
        self.fc_logvar = nn.Linear(encoder_out_dim, latent_dim)

        # Decoder
        self.decoder = ResidualDecoder(
            latent_dim=latent_dim,
            hidden_dims=hidden_dims,
            output_dim=input_dim,
            dropout_rate=dropout_rate,
            activation=activation
        )

        decoder_out_dim = self.decoder.final_hidden_dim

        # Output heads (依赖于分布类型)
        self._build_output_heads(decoder_out_dim, input_dim, distribution, dispersion)

        # 初始化权重
        self._init_weights()

    def _build_output_heads(
        self,
        hidden_dim: int,
        output_dim: int,
        distribution: str,
        dispersion: str
    ):
        """构建输出头"""

        # Mean output (所有分布都需要)
        if distribution == 'gaussian':
            # Gaussian: 输出任意实数（适用于 log1p 数据）
            self.mean_head = nn.Linear(hidden_dim, output_dim)
            # 可学习的对数方差 (reconstruction)
            self.recon_log_var_head = nn.Linear(hidden_dim, output_dim)
        else:
            # NB/ZINB: 输出正数（rate parameter）
            self.mean_head = nn.Sequential(
                nn.Linear(hidden_dim, output_dim),
                nn.Softplus()  # 确保正数
            )

        # Dispersion (NB, ZINB)
        if distribution in ['nb', 'zinb']:
            if dispersion == 'gene':
                # Gene-specific dispersion (推荐，减少过拟合)
                self.log_theta = nn.Parameter(torch.zeros(output_dim))
            else:
                # Gene-cell specific dispersion
                self.theta_head = nn.Sequential(
                    nn.Linear(hidden_dim, output_dim),
                    nn.Softplus()
                )

        # Dropout probability (ZINB only)
        if distribution == 'zinb':
            self.pi_head = nn.Linear(hidden_dim, output_dim)

    def _init_weights(self):
        """初始化权重"""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight, gain=0.6)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

        # 特殊初始化: logvar 初始化为负值，使初始 std 较小
        nn.init.constant_(self.fc_logvar.weight, 0.0)
        nn.init.constant_(self.fc_logvar.bias, -2.0)

    def reparameterize(self, mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        """
        Reparameterization trick: z = mu + std * eps

        Args:
            mu: 均值
            logvar: 对数方差

        Returns:
            采样的 z
        """
        if self.training:
            # 限制 logvar 范围，防止数值问题
            logvar = logvar.clamp(min=-10, max=10)
            std = torch.exp(0.5 * logvar)
            eps = torch.randn_like(std)
            return mu + eps * std
        else:
            # 推理时返回均值
            return mu

    def encode(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        编码到潜在空间

        Returns:
            mu: 后验均值
            logvar: 后验对数方差
        """
        h = self.encoder(x)
        mu = self.fc_mu(h)
        logvar = self.fc_logvar(h)
        return mu, logvar

    def decode(self, z: torch.Tensor) -> dict:
        """
        从潜在空间解码

        Returns:
            dict 包含:
                - 'mu': 预测均值
                - 'log_var': 重建对数方差 (gaussian)
                - 'theta': dispersion (nb, zinb)
                - 'pi': dropout logits (zinb)
        """
        h = self.decoder(z)

        outputs = {}

        # Mean
        outputs['mu'] = self.mean_head(h)

        # Distribution-specific outputs
        if self.distribution == 'gaussian':
            outputs['log_var'] = self.recon_log_var_head(h)

        elif self.distribution == 'nb':
            if self.dispersion_type == 'gene':
                outputs['theta'] = torch.exp(self.log_theta).unsqueeze(0).expand(z.size(0), -1)
            else:
                outputs['theta'] = self.theta_head(h)

        elif self.distribution == 'zinb':
            if self.dispersion_type == 'gene':
                outputs['theta'] = torch.exp(self.log_theta).unsqueeze(0).expand(z.size(0), -1)
            else:
                outputs['theta'] = self.theta_head(h)
            outputs['pi'] = self.pi_head(h)

        return outputs

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, dict]:
        """
        前向传播

        Args:
            x: 输入数据 (batch, genes)

        Returns:
            recon_x: 重建值 (即 mu)
            z: 潜在表示（采样后）
            q_mu: 后验均值
            q_logvar: 后验对数方差
            outputs: 完整输出字典（用于损失计算）
        """
        # Encode
        q_mu, q_logvar = self.encode(x)

        # Reparameterize
        z = self.reparameterize(q_mu, q_logvar)

        # Decode
        outputs = self.decode(z)
        recon_x = outputs['mu']

        # 添加后验参数到 outputs
        outputs['q_mu'] = q_mu
        outputs['q_logvar'] = q_logvar

        return recon_x, z, q_mu, q_logvar, outputs

    def get_latent(self, x: torch.Tensor) -> torch.Tensor:
        """
        获取潜在表示（用于下游任务）

        Returns:
            后验均值 mu（确定性）
        """
        mu, _ = self.encode(x)
        return mu

    def sample(self, z: torch.Tensor) -> torch.Tensor:
        """
        从潜在空间采样生成

        Args:
            z: 潜在向量

        Returns:
            生成的样本
        """
        outputs = self.decode(z)
        return outputs['mu']

    @staticmethod
    def kl_divergence(mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        """
        计算 KL 散度: KL(q(z|x) || p(z))

        假设先验 p(z) = N(0, I)

        KL = -0.5 * sum(1 + logvar - mu^2 - exp(logvar))

        Args:
            mu: 后验均值
            logvar: 后验对数方差

        Returns:
            KL 散度（每个样本）
        """
        return -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), dim=1)
