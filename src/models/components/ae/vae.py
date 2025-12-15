import torch
import torch.nn as nn
import torch.nn.functional as F

class VariationalAE(nn.Module):
    """
    Variational Autoencoder (VAE) 实现。
    
    相比普通 AE，VAE 在 Latent Space 引入了 KL 散度约束，使其接近标准正态分布。
    这对后续使用 Rectified Flow 或 Diffusion Model 非常重要。
    """
    def __init__(
        self,
        input_dim: int,
        hidden_dims: list = [1024, 512],
        latent_dim: int = 64,
        dropout_rate: float = 0.1,
        use_batch_norm: bool = True,
        **kwargs,
    ):
        super().__init__()

        # Encoder
        encoder_layers = []
        curr_dim = input_dim
        
        for h_dim in hidden_dims:
            encoder_layers.append(nn.Linear(curr_dim, h_dim))
            if use_batch_norm:
                encoder_layers.append(nn.BatchNorm1d(h_dim))
            encoder_layers.append(nn.LeakyReLU())
            encoder_layers.append(nn.Dropout(dropout_rate))
            curr_dim = h_dim
            
        self.encoder_backbone = nn.Sequential(*encoder_layers)
        
        # VAE 特有的两个投影头：Mu (均值) 和 LogVar (对数方差)
        self.fc_mu = nn.Linear(curr_dim, latent_dim)
        self.fc_logvar = nn.Linear(curr_dim, latent_dim)

        # Decoder
        decoder_layers = []
        reversed_hidden = list(reversed(hidden_dims))
        
        # Latent -> First Hidden
        curr_dim = latent_dim
        
        for h_dim in reversed_hidden:
            decoder_layers.append(nn.Linear(curr_dim, h_dim))
            if use_batch_norm:
                decoder_layers.append(nn.BatchNorm1d(h_dim))
            decoder_layers.append(nn.LeakyReLU())
            decoder_layers.append(nn.Dropout(dropout_rate))
            curr_dim = h_dim
            
        self.decoder_net = nn.Sequential(*decoder_layers)
        self.output_layer = nn.Linear(curr_dim, input_dim)

    def reparameterize(self, mu, logvar):
        """
        重参数化技巧：z = mu + std * eps
        使得梯度可以通过随机节点反向传播
        """
        if self.training:
            std = torch.exp(0.5 * logvar)
            eps = torch.randn_like(std)
            return mu + eps * std
        else:
            # 推理阶段直接使用均值，保证确定性 (或者也可以采样，取决于需求)
            return mu

    def encode(self, x):
        """
        返回 mu 和 logvar
        """
        h = self.encoder_backbone(x)
        mu = self.fc_mu(h)
        logvar = self.fc_logvar(h)
        return mu, logvar

    def decode(self, z):
        h = self.decoder_net(z)
        recon = self.output_layer(h)
        return recon

    def forward(self, x):
        """
        返回: recon_x, z, mu, logvar
        """
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        recon_x = self.decode(z)
        
        return recon_x, z, mu, logvar

