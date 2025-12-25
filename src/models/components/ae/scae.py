"""
Single-Cell Autoencoder (SCAE)

专为单细胞 RNA-seq 数据设计的自编码器：
1. 残差连接 (ResNet-style) 防止梯度消失
2. 支持多种输出分布 (Gaussian, NB, ZINB)
3. Pre-LayerNorm 架构，更稳定的训练
4. 可学习的 dispersion 参数

参考:
- scVI: https://docs.scvi-tools.org
- DCA: https://github.com/theislab/dca
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, List, Literal, Tuple


class ResidualBlock(nn.Module):
    """
    残差块：Pre-LayerNorm + Linear + Activation + Dropout + Skip Connection

    Pre-LN 架构: x -> LN -> Linear -> Act -> Dropout -> + x
    """

    def __init__(
        self,
        in_dim: int,
        out_dim: int,
        dropout_rate: float = 0.1,
        activation: str = "GELU"
    ):
        super().__init__()

        self.use_projection = (in_dim != out_dim)

        # Pre-LayerNorm
        self.norm = nn.LayerNorm(in_dim)

        # Main path
        self.linear = nn.Linear(in_dim, out_dim)

        # Activation
        if activation == "GELU":
            self.act = nn.GELU()
        elif activation == "SiLU":
            self.act = nn.SiLU()
        elif activation == "ReLU":
            self.act = nn.ReLU()
        elif activation == "LeakyReLU":
            self.act = nn.LeakyReLU(0.2)
        else:
            self.act = nn.GELU()

        self.dropout = nn.Dropout(dropout_rate)

        # Skip connection projection (if dimensions don't match)
        if self.use_projection:
            self.proj = nn.Linear(in_dim, out_dim)
        else:
            self.proj = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Pre-norm
        h = self.norm(x)
        h = self.linear(h)
        h = self.act(h)
        h = self.dropout(h)

        # Skip connection
        if self.use_projection:
            x = self.proj(x)

        return x + h


class ResidualEncoder(nn.Module):
    """
    残差编码器

    结构: Input -> [ResBlock] * n_layers -> Bottleneck
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dims: List[int],
        latent_dim: int,
        dropout_rate: float = 0.1,
        activation: str = "GELU"
    ):
        super().__init__()

        # Input projection
        self.input_proj = nn.Linear(input_dim, hidden_dims[0])

        # Residual blocks
        blocks = []
        for i in range(len(hidden_dims)):
            in_d = hidden_dims[i]
            out_d = hidden_dims[i + 1] if i < len(hidden_dims) - 1 else hidden_dims[-1]
            blocks.append(ResidualBlock(in_d, out_d, dropout_rate, activation))

        self.blocks = nn.ModuleList(blocks)

        # Final norm before latent projection
        self.final_norm = nn.LayerNorm(hidden_dims[-1])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.input_proj(x)

        for block in self.blocks:
            h = block(h)

        h = self.final_norm(h)
        return h


class ResidualDecoder(nn.Module):
    """
    残差解码器

    结构: Latent -> [ResBlock] * n_layers -> Output heads

    注意: hidden_dims 传入的是 encoder 的 hidden_dims，decoder 会自动反转
    例如: encoder hidden_dims=[512, 256] -> decoder 内部使用 [256, 512]
    """

    def __init__(
        self,
        latent_dim: int,
        hidden_dims: List[int],
        output_dim: int,
        dropout_rate: float = 0.1,
        activation: str = "GELU"
    ):
        super().__init__()

        # Decoder: 从小到大（与 encoder 相反）
        # 例如 encoder: 1000 -> 512 -> 256 -> latent
        #      decoder: latent -> 256 -> 512 -> 1000
        reversed_dims = list(reversed(hidden_dims))

        # Latent to first hidden (最小的 hidden dim)
        self.latent_proj = nn.Linear(latent_dim, reversed_dims[0])

        # Residual blocks: 逐渐增大维度
        blocks = []
        for i in range(len(reversed_dims) - 1):
            in_d = reversed_dims[i]
            out_d = reversed_dims[i + 1]
            blocks.append(ResidualBlock(in_d, out_d, dropout_rate, activation))

        # 最后一个 block 保持维度
        if len(reversed_dims) > 0:
            blocks.append(ResidualBlock(reversed_dims[-1], reversed_dims[-1], dropout_rate, activation))

        self.blocks = nn.ModuleList(blocks)

        # Final norm before output
        self.final_norm = nn.LayerNorm(reversed_dims[-1])
        self.final_hidden_dim = reversed_dims[-1]

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        h = self.latent_proj(z)

        for block in self.blocks:
            h = block(h)

        h = self.final_norm(h)
        return h


class SCAE(nn.Module):
    """
    Single-Cell Autoencoder

    专为单细胞数据设计的自编码器：
    - 残差连接防止梯度消失
    - 支持多种输出分布: 'gaussian', 'nb', 'zinb'
    - 可学习的 gene-specific dispersion
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
        """
        super().__init__()

        self.input_dim = input_dim
        self.latent_dim = latent_dim
        self.distribution = distribution
        self.dispersion_type = dispersion

        # Encoder
        self.encoder = ResidualEncoder(
            input_dim=input_dim,
            hidden_dims=hidden_dims,
            latent_dim=latent_dim,
            dropout_rate=dropout_rate,
            activation=activation
        )

        # Latent projection
        self.fc_latent = nn.Linear(hidden_dims[-1], latent_dim)

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
            # 可学习的对数方差
            self.log_var_head = nn.Linear(hidden_dim, output_dim)
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

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """编码到潜在空间"""
        h = self.encoder(x)
        z = self.fc_latent(h)
        return z

    def decode(self, z: torch.Tensor) -> dict:
        """
        从潜在空间解码

        Returns:
            dict 包含:
                - 'mu': 预测均值
                - 'log_var': 对数方差 (gaussian)
                - 'theta': dispersion (nb, zinb)
                - 'pi': dropout logits (zinb)
        """
        h = self.decoder(z)

        outputs = {}

        # Mean
        outputs['mu'] = self.mean_head(h)

        # Distribution-specific outputs
        if self.distribution == 'gaussian':
            outputs['log_var'] = self.log_var_head(h)

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

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, dict]:
        """
        前向传播

        Args:
            x: 输入数据 (batch, genes)

        Returns:
            recon_x: 重建值 (即 mu)
            z: 潜在表示
            outputs: 完整输出字典（用于损失计算）
        """
        z = self.encode(x)
        outputs = self.decode(z)
        recon_x = outputs['mu']

        return recon_x, z, outputs

    def get_latent(self, x: torch.Tensor) -> torch.Tensor:
        """获取潜在表示（用于下游任务）"""
        return self.encode(x)
