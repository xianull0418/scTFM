"""
Single-Cell Specific Loss Functions

实现适合单细胞 RNA-seq 数据的损失函数：
1. Negative Binomial (NB) Loss
2. Zero-Inflated Negative Binomial (ZINB) Loss
3. Gaussian Loss (适用于 log1p 归一化数据)

参考:
- scVI: https://github.com/scverse/scvi-tools
- DCA: https://github.com/theislab/dca
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal
from typing import Optional, Tuple
import math


def log_nb_positive(
    x: torch.Tensor,
    mu: torch.Tensor,
    theta: torch.Tensor,
    eps: float = 1e-8
) -> torch.Tensor:
    """
    Negative Binomial 分布的对数似然。

    使用 (mu, theta) 参数化：
    - mu: 均值 (mean)，必须 > 0
    - theta: 逆离散度 (inverse dispersion)，必须 > 0
      theta 越大，方差越接近均值（接近 Poisson）
      theta 越小，方差越大（过度离散）

    NB 分布：Var(X) = mu + mu^2 / theta

    Args:
        x: 观测值 (count data)，shape: (batch, genes)
        mu: NB 均值，shape: (batch, genes)
        theta: NB 逆离散度，shape: (batch, genes) 或 (genes,)
        eps: 数值稳定性常数

    Returns:
        对数似然，shape: (batch, genes)
    """
    # 确保数值稳定
    x = x.clamp(min=0)
    mu = mu.clamp(min=eps)
    theta = theta.clamp(min=eps)

    log_theta_mu_eps = torch.log(theta + mu + eps)

    # NB log-likelihood:
    # log P(x|mu,theta) = log Gamma(x + theta) - log Gamma(theta) - log Gamma(x + 1)
    #                   + theta * log(theta / (theta + mu))
    #                   + x * log(mu / (theta + mu))
    res = (
        theta * (torch.log(theta + eps) - log_theta_mu_eps)
        + x * (torch.log(mu + eps) - log_theta_mu_eps)
        + torch.lgamma(x + theta)
        - torch.lgamma(theta)
        - torch.lgamma(x + 1)
    )

    return res


def log_zinb_positive(
    x: torch.Tensor,
    mu: torch.Tensor,
    theta: torch.Tensor,
    pi: torch.Tensor,
    eps: float = 1e-8
) -> torch.Tensor:
    """
    Zero-Inflated Negative Binomial 分布的对数似然。

    ZINB = pi * delta_0 + (1 - pi) * NB(mu, theta)

    其中 pi 是零膨胀概率（dropout probability）

    Args:
        x: 观测值 (count data)，shape: (batch, genes)
        mu: NB 均值，shape: (batch, genes)
        theta: NB 逆离散度，shape: (batch, genes) 或 (genes,)
        pi: dropout logits（未经 sigmoid），shape: (batch, genes)
        eps: 数值稳定性常数

    Returns:
        对数似然，shape: (batch, genes)
    """
    # 确保数值稳定
    x = x.clamp(min=0)
    mu = mu.clamp(min=eps)
    theta = theta.clamp(min=eps)

    # softplus(-pi) = log(1 + exp(-pi)) = log(1 - sigmoid(pi)) = log(1 - dropout_prob)
    softplus_pi = F.softplus(-pi)

    log_theta_eps = torch.log(theta + eps)
    log_theta_mu_eps = torch.log(theta + mu + eps)

    # pi_theta_log = log(dropout_prob) + log(P(x=0|NB))
    #              = -log(1 + exp(-pi)) + theta * log(theta / (theta + mu))
    pi_theta_log = -pi + theta * (log_theta_eps - log_theta_mu_eps)

    # Case 1: x = 0
    # log P(x=0) = log(pi + (1-pi) * P_NB(0))
    #            = log(exp(log_pi) + exp(log(1-pi) + log_P_NB(0)))
    #            = logsumexp(log_pi, log(1-pi) + log_P_NB(0))
    case_zero = F.softplus(pi_theta_log) - softplus_pi

    # Case 2: x > 0
    # log P(x) = log(1 - pi) + log P_NB(x)
    case_non_zero = (
        -softplus_pi  # log(1 - pi)
        + pi_theta_log
        + x * (torch.log(mu + eps) - log_theta_mu_eps)
        + torch.lgamma(x + theta)
        - torch.lgamma(theta)
        - torch.lgamma(x + 1)
    )

    # 根据 x 是否为 0 选择对应的 case
    is_zero = (x < eps).float()
    res = is_zero * case_zero + (1 - is_zero) * case_non_zero

    return res


class NBLoss(nn.Module):
    """
    Negative Binomial Loss for single-cell data.

    适用于原始 count 数据或 size-factor 归一化后的数据。
    """

    def __init__(self, reduction: str = 'mean', eps: float = 1e-8):
        """
        Args:
            reduction: 'mean', 'sum', 或 'none'
            eps: 数值稳定性常数
        """
        super().__init__()
        self.reduction = reduction
        self.eps = eps

    def forward(
        self,
        x: torch.Tensor,
        mu: torch.Tensor,
        theta: torch.Tensor
    ) -> torch.Tensor:
        """
        计算 NB 负对数似然损失。

        Args:
            x: 观测值 (batch, genes)
            mu: 预测均值 (batch, genes)
            theta: 逆离散度 (batch, genes) 或 (genes,)

        Returns:
            NB 负对数似然损失
        """
        log_prob = log_nb_positive(x, mu, theta, self.eps)
        loss = -log_prob

        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        else:
            return loss


class ZINBLoss(nn.Module):
    """
    Zero-Inflated Negative Binomial Loss for single-cell data.

    适用于高度稀疏的 count 数据（90%+ 零值）。
    """

    def __init__(self, reduction: str = 'mean', eps: float = 1e-8):
        """
        Args:
            reduction: 'mean', 'sum', 或 'none'
            eps: 数值稳定性常数
        """
        super().__init__()
        self.reduction = reduction
        self.eps = eps

    def forward(
        self,
        x: torch.Tensor,
        mu: torch.Tensor,
        theta: torch.Tensor,
        pi: torch.Tensor
    ) -> torch.Tensor:
        """
        计算 ZINB 负对数似然损失。

        Args:
            x: 观测值 (batch, genes)
            mu: 预测均值 (batch, genes)
            theta: 逆离散度 (batch, genes) 或 (genes,)
            pi: dropout logits (batch, genes)

        Returns:
            ZINB 负对数似然损失
        """
        log_prob = log_zinb_positive(x, mu, theta, pi, self.eps)
        loss = -log_prob

        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        else:
            return loss


class GaussianLoss(nn.Module):
    """
    Gaussian Loss with learnable variance.

    适用于 log1p 归一化后的连续数据。
    比纯 MSE 更灵活，可以学习每个基因的噪声水平。
    """

    def __init__(
        self,
        reduction: str = 'mean',
        learn_var: bool = True,
        min_var: float = 1e-4,
        max_var: float = 10.0
    ):
        """
        Args:
            reduction: 'mean', 'sum', 或 'none'
            learn_var: 是否学习方差参数
            min_var: 最小方差（数值稳定性）
            max_var: 最大方差
        """
        super().__init__()
        self.reduction = reduction
        self.learn_var = learn_var
        self.min_var = min_var
        self.max_var = max_var

    def forward(
        self,
        x: torch.Tensor,
        mu: torch.Tensor,
        log_var: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        计算 Gaussian 负对数似然损失。

        Args:
            x: 观测值 (batch, genes)
            mu: 预测均值 (batch, genes)
            log_var: 对数方差 (batch, genes) 或 (genes,)，如果 learn_var=True

        Returns:
            Gaussian 负对数似然损失
        """
        if log_var is None or not self.learn_var:
            # 退化为 MSE
            loss = F.mse_loss(mu, x, reduction='none')
        else:
            # 限制方差范围
            var = torch.exp(log_var).clamp(min=self.min_var, max=self.max_var)

            # Gaussian NLL: 0.5 * (log(var) + (x - mu)^2 / var)
            loss = 0.5 * (log_var + (x - mu) ** 2 / var)

        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        else:
            return loss


class SCLoss(nn.Module):
    """
    Unified Single-Cell Loss Function.

    支持多种损失类型的统一接口：
    - 'mse': Mean Squared Error
    - 'gaussian': Gaussian NLL with learnable variance
    - 'nb': Negative Binomial
    - 'zinb': Zero-Inflated Negative Binomial
    """

    def __init__(
        self,
        loss_type: str = 'mse',
        reduction: str = 'mean',
        eps: float = 1e-8
    ):
        """
        Args:
            loss_type: 损失类型 ('mse', 'gaussian', 'nb', 'zinb')
            reduction: 'mean', 'sum', 或 'none'
            eps: 数值稳定性常数
        """
        super().__init__()
        self.loss_type = loss_type
        self.reduction = reduction
        self.eps = eps

        if loss_type == 'nb':
            self.loss_fn = NBLoss(reduction=reduction, eps=eps)
        elif loss_type == 'zinb':
            self.loss_fn = ZINBLoss(reduction=reduction, eps=eps)
        elif loss_type == 'gaussian':
            self.loss_fn = GaussianLoss(reduction=reduction)
        else:  # mse
            self.loss_fn = None  # 使用 F.mse_loss

    def forward(
        self,
        x: torch.Tensor,
        outputs: dict
    ) -> torch.Tensor:
        """
        统一的损失计算接口。

        Args:
            x: 观测值 (batch, genes)
            outputs: 模型输出字典，包含：
                - 'mu': 预测均值
                - 'theta': 逆离散度 (NB/ZINB)
                - 'pi': dropout logits (ZINB)
                - 'log_var': 对数方差 (Gaussian)

        Returns:
            损失值
        """
        mu = outputs['mu']

        if self.loss_type == 'mse':
            return F.mse_loss(mu, x, reduction=self.reduction)

        elif self.loss_type == 'gaussian':
            return self.loss_fn(x, mu, outputs.get('log_var'))

        elif self.loss_type == 'nb':
            theta = outputs['theta']
            return self.loss_fn(x, mu, theta)

        elif self.loss_type == 'zinb':
            theta = outputs['theta']
            pi = outputs['pi']
            return self.loss_fn(x, mu, theta, pi)

        else:
            raise ValueError(f"Unknown loss type: {self.loss_type}")
