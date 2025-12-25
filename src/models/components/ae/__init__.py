"""
Autoencoder Components for Single-Cell Data
"""

from .vanilla import VanillaAE
from .vae import VariationalAE
from .scae import SCAE, ResidualBlock, ResidualEncoder, ResidualDecoder
from .scvae import SCVAE
from .losses import NBLoss, ZINBLoss, GaussianLoss, SCLoss

__all__ = [
    "VanillaAE",
    "VariationalAE",
    "SCAE",
    "SCVAE",
    "ResidualBlock",
    "ResidualEncoder",
    "ResidualDecoder",
    "NBLoss",
    "ZINBLoss",
    "GaussianLoss",
    "SCLoss",
]
