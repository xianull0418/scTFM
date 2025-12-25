"""
scVI-style Single-Cell Autoencoder (SCAE)

核心设计（参考 scVI）:
1. Encoder 输入: log1p(normalized_counts)
2. Decoder 输出: softmax(rho) - 基因比例，和为 1
3. NB 均值: mu = library_size * rho
4. Loss: -log NB(raw_counts | mu, theta)

这种设计分离了:
- 生物学信号 (latent z -> rho)
- 技术噪声 (library_size)

参考:
- scVI: https://docs.scvi-tools.org
- Lopez et al., Nature Methods 2018
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, List, Literal, Tuple, Dict


class ResidualBlock(nn.Module):
    """
    残差块：Pre-LayerNorm + Linear + Activation + Dropout + Skip Connection
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

        self.norm = nn.LayerNorm(in_dim)
        self.linear = nn.Linear(in_dim, out_dim)

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

        if self.use_projection:
            self.proj = nn.Linear(in_dim, out_dim)
        else:
            self.proj = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.norm(x)
        h = self.linear(h)
        h = self.act(h)
        h = self.dropout(h)

        if self.use_projection:
            x = self.proj(x)

        return x + h


class Encoder(nn.Module):
    """scVI-style Encoder"""

    def __init__(
        self,
        n_input: int,
        n_hidden: int = 256,
        n_latent: int = 64,
        n_layers: int = 2,
        dropout_rate: float = 0.1,
        activation: str = "GELU",
        use_residual: bool = True,
    ):
        super().__init__()

        self.use_residual = use_residual

        # Input projection
        self.input_proj = nn.Sequential(
            nn.Linear(n_input, n_hidden),
            nn.LayerNorm(n_hidden),
            nn.GELU() if activation == "GELU" else nn.ReLU(),
            nn.Dropout(dropout_rate),
        )

        # Hidden layers
        if use_residual:
            self.layers = nn.ModuleList([
                ResidualBlock(n_hidden, n_hidden, dropout_rate, activation)
                for _ in range(n_layers)
            ])
        else:
            layers = []
            for _ in range(n_layers):
                layers.extend([
                    nn.Linear(n_hidden, n_hidden),
                    nn.LayerNorm(n_hidden),
                    nn.GELU() if activation == "GELU" else nn.ReLU(),
                    nn.Dropout(dropout_rate),
                ])
            self.layers = nn.Sequential(*layers)

        self.final_norm = nn.LayerNorm(n_hidden)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.input_proj(x)

        if self.use_residual:
            for layer in self.layers:
                h = layer(h)
        else:
            h = self.layers(h)

        h = self.final_norm(h)
        return h


class Decoder(nn.Module):
    """
    scVI-style Decoder

    输出 softmax normalized 的基因比例 rho
    """

    def __init__(
        self,
        n_latent: int,
        n_hidden: int = 256,
        n_output: int = 28231,
        n_layers: int = 2,
        dropout_rate: float = 0.1,
        activation: str = "GELU",
        use_residual: bool = True,
    ):
        super().__init__()

        self.use_residual = use_residual

        # Latent projection
        self.latent_proj = nn.Sequential(
            nn.Linear(n_latent, n_hidden),
            nn.LayerNorm(n_hidden),
            nn.GELU() if activation == "GELU" else nn.ReLU(),
            nn.Dropout(dropout_rate),
        )

        # Hidden layers
        if use_residual:
            self.layers = nn.ModuleList([
                ResidualBlock(n_hidden, n_hidden, dropout_rate, activation)
                for _ in range(n_layers)
            ])
        else:
            layers = []
            for _ in range(n_layers):
                layers.extend([
                    nn.Linear(n_hidden, n_hidden),
                    nn.LayerNorm(n_hidden),
                    nn.GELU() if activation == "GELU" else nn.ReLU(),
                    nn.Dropout(dropout_rate),
                ])
            self.layers = nn.Sequential(*layers)

        self.final_norm = nn.LayerNorm(n_hidden)

        # Output: rho (gene proportions, will be softmaxed)
        self.rho_decoder = nn.Linear(n_hidden, n_output)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        """
        Returns:
            rho: gene proportions after softmax, shape (batch, n_genes)
        """
        h = self.latent_proj(z)

        if self.use_residual:
            for layer in self.layers:
                h = layer(h)
        else:
            h = self.layers(h)

        h = self.final_norm(h)

        # Softmax 确保输出为正且和为 1
        rho = F.softmax(self.rho_decoder(h), dim=-1)

        return rho


class SCVIAE(nn.Module):
    """
    scVI-style Autoencoder

    核心设计:
    - Encoder: log1p(x) -> z
    - Decoder: z -> rho (softmax normalized proportions)
    - NB mean: mu = library_size * rho
    - Loss: -log NB(counts | mu, theta)

    这种设计的优势:
    1. Library size 分离 - latent 只编码生物信号
    2. Softmax 输出 - 可解释的基因比例
    3. NB 分布 - 正确建模 count 数据

    Args:
        n_input: 输入基因数
        n_hidden: 隐藏层维度
        n_latent: 潜在空间维度
        n_layers: encoder/decoder 层数
        dropout_rate: dropout 比例
        dispersion: 'gene' (推荐) 或 'gene-cell'
        gene_likelihood: 'nb' 或 'zinb'
    """

    def __init__(
        self,
        n_input: int = 28231,
        n_hidden: int = 256,
        n_latent: int = 64,
        n_layers: int = 2,
        dropout_rate: float = 0.1,
        activation: str = "GELU",
        use_residual: bool = True,
        dispersion: Literal['gene', 'gene-cell'] = 'gene',
        gene_likelihood: Literal['nb', 'zinb'] = 'nb',
        **kwargs
    ):
        super().__init__()

        self.n_input = n_input
        self.n_latent = n_latent
        self.dispersion = dispersion
        self.gene_likelihood = gene_likelihood

        # Encoder
        self.encoder = Encoder(
            n_input=n_input,
            n_hidden=n_hidden,
            n_latent=n_latent,
            n_layers=n_layers,
            dropout_rate=dropout_rate,
            activation=activation,
            use_residual=use_residual,
        )
        self.z_mean = nn.Linear(n_hidden, n_latent)

        # Decoder
        self.decoder = Decoder(
            n_latent=n_latent,
            n_hidden=n_hidden,
            n_output=n_input,
            n_layers=n_layers,
            dropout_rate=dropout_rate,
            activation=activation,
            use_residual=use_residual,
        )

        # Dispersion parameter (inverse dispersion of NB)
        if dispersion == 'gene':
            # Gene-specific dispersion (log scale for numerical stability)
            self.log_theta = nn.Parameter(torch.zeros(n_input))
        else:
            # Gene-cell specific dispersion
            self.theta_decoder = nn.Sequential(
                nn.Linear(n_hidden, n_input),
                nn.Softplus(),
            )

        # Dropout decoder for ZINB
        if gene_likelihood == 'zinb':
            # 需要 decoder hidden 输出
            self._decoder_hidden = None
            self.dropout_decoder = nn.Linear(n_hidden, n_input)

        self._init_weights()

    def _init_weights(self):
        """初始化权重"""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight, gain=0.6)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """
        Encode log1p normalized data to latent space

        Args:
            x: log1p(normalized_counts), shape (batch, n_genes)

        Returns:
            z: latent representation, shape (batch, n_latent)
        """
        h = self.encoder(x)
        z = self.z_mean(h)
        return z

    def decode(
        self,
        z: torch.Tensor,
        library_size: torch.Tensor
    ) -> Dict[str, torch.Tensor]:
        """
        Decode latent to NB distribution parameters

        Args:
            z: latent representation, shape (batch, n_latent)
            library_size: total UMI counts per cell, shape (batch,)

        Returns:
            dict with:
                - 'rho': gene proportions (softmax), shape (batch, n_genes)
                - 'mu': NB mean = library_size * rho, shape (batch, n_genes)
                - 'theta': NB dispersion, shape (batch, n_genes) or (n_genes,)
                - 'pi': dropout logits (ZINB only), shape (batch, n_genes)
        """
        # Get gene proportions
        rho = self.decoder(z)

        # Scale by library size to get NB mean
        # library_size: (batch,) -> (batch, 1)
        library_size = library_size.unsqueeze(-1)
        mu = library_size * rho

        outputs = {
            'rho': rho,
            'mu': mu,
        }

        # Dispersion
        if self.dispersion == 'gene':
            # Gene-specific: broadcast to batch
            theta = torch.exp(self.log_theta).unsqueeze(0).expand(z.size(0), -1)
        else:
            # This requires decoder hidden state, need to modify decoder
            theta = torch.exp(self.log_theta).unsqueeze(0).expand(z.size(0), -1)

        outputs['theta'] = theta

        # ZINB dropout
        if self.gene_likelihood == 'zinb':
            # 简化版: 使用单独的 dropout decoder
            h = self.encoder(self.z_mean.weight.new_zeros(z.size(0), self.n_input))
            pi = self.dropout_decoder(h)
            outputs['pi'] = pi

        return outputs

    def forward(
        self,
        x: torch.Tensor,
        library_size: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor, Dict[str, torch.Tensor]]:
        """
        Forward pass

        Args:
            x: log1p(normalized_counts), shape (batch, n_genes)
            library_size: total UMI counts, shape (batch,)
                         如果为 None，则从 x 估算

        Returns:
            mu: NB mean, shape (batch, n_genes)
            z: latent representation, shape (batch, n_latent)
            outputs: dict with all distribution parameters
        """
        # 如果没有提供 library_size，从 log1p 数据估算
        if library_size is None:
            # expm1 to get normalized counts, then sum
            library_size = torch.expm1(x).sum(dim=-1)
            # 防止 0
            library_size = library_size.clamp(min=1.0)

        # Encode
        z = self.encode(x)

        # Decode
        outputs = self.decode(z, library_size)
        mu = outputs['mu']

        return mu, z, outputs

    def get_latent(self, x: torch.Tensor) -> torch.Tensor:
        """获取 latent representation（用于下游任务）"""
        return self.encode(x)

    def get_normalized_expression(
        self,
        x: torch.Tensor,
        library_size: Optional[torch.Tensor] = None,
        scale: float = 1e4
    ) -> torch.Tensor:
        """
        获取去噪后的归一化表达

        Returns:
            rho * scale (类似 normalize_total 的输出)
        """
        if library_size is None:
            library_size = torch.expm1(x).sum(dim=-1).clamp(min=1.0)

        z = self.encode(x)
        outputs = self.decode(z, library_size)

        return outputs['rho'] * scale


class SCVIVAE(SCVIAE):
    """
    scVI-style Variational Autoencoder

    在 SCVIAE 基础上添加:
    - 变分推断 (mu, logvar)
    - KL 散度正则化
    - Reparameterization trick
    """

    def __init__(
        self,
        n_input: int = 28231,
        n_hidden: int = 256,
        n_latent: int = 64,
        n_layers: int = 2,
        dropout_rate: float = 0.1,
        activation: str = "GELU",
        use_residual: bool = True,
        dispersion: Literal['gene', 'gene-cell'] = 'gene',
        gene_likelihood: Literal['nb', 'zinb'] = 'nb',
        **kwargs
    ):
        super().__init__(
            n_input=n_input,
            n_hidden=n_hidden,
            n_latent=n_latent,
            n_layers=n_layers,
            dropout_rate=dropout_rate,
            activation=activation,
            use_residual=use_residual,
            dispersion=dispersion,
            gene_likelihood=gene_likelihood,
            **kwargs
        )

        # VAE: 添加 logvar 投影
        self.z_logvar = nn.Linear(n_hidden, n_latent)

        # 初始化 logvar 为较小的值
        nn.init.constant_(self.z_logvar.weight, 0.0)
        nn.init.constant_(self.z_logvar.bias, -2.0)

    def encode(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Encode to variational posterior q(z|x)

        Returns:
            q_mu: posterior mean
            q_logvar: posterior log variance
        """
        h = self.encoder(x)
        q_mu = self.z_mean(h)
        q_logvar = self.z_logvar(h)
        # 限制 logvar 范围
        q_logvar = q_logvar.clamp(min=-10, max=10)
        return q_mu, q_logvar

    def reparameterize(self, mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        """Reparameterization trick"""
        if self.training:
            std = torch.exp(0.5 * logvar)
            eps = torch.randn_like(std)
            return mu + eps * std
        else:
            return mu

    def forward(
        self,
        x: torch.Tensor,
        library_size: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, Dict[str, torch.Tensor]]:
        """
        Forward pass

        Returns:
            mu: NB mean
            z: sampled latent
            q_mu: posterior mean
            q_logvar: posterior log variance
            outputs: dict with all parameters
        """
        if library_size is None:
            library_size = torch.expm1(x).sum(dim=-1).clamp(min=1.0)

        # Encode
        q_mu, q_logvar = self.encode(x)

        # Sample
        z = self.reparameterize(q_mu, q_logvar)

        # Decode
        outputs = self.decode(z, library_size)
        outputs['q_mu'] = q_mu
        outputs['q_logvar'] = q_logvar

        mu = outputs['mu']

        return mu, z, q_mu, q_logvar, outputs

    def get_latent(self, x: torch.Tensor) -> torch.Tensor:
        """获取 posterior mean（确定性）"""
        q_mu, _ = self.encode(x)
        return q_mu

    @staticmethod
    def kl_divergence(mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        """KL(q(z|x) || p(z)) where p(z) = N(0, I)"""
        return -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), dim=-1)
