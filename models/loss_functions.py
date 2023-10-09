import torch
import torch.nn as nn
import torch.nn.functional as F

class VAELoss(nn.Module):
    def __init__(self, beta=1.0):
        super().__init__()
        self.beta = beta  # Weight of the KL divergence term

    def forward(self, x_hat, x, mu, logvar):
        """
        Compute the VAE loss.
        vae_loss = MSE(x, x_hat) + beta * KL(N(\mu, \sigma), N(0, 1))
        where KL(N(\mu, \sigma), N(0, 1)) = \log \frac{1}{\sigma} + \frac{\sigma^2 + \mu^2}{2} - \frac{1}{2}
        :param x: <torch.tensor> ground truth tensor of shape (batch_size, state_size)
        :param x_hat: <torch.tensor> reconstructed tensor of shape (batch_size, state_size)
        :param mu: <torch.tensor> of shape (batch_size, state_size)
        :param logvar: <torch.tensor> of shape (batch_size, state_size)
        :return: <torch.tensor> scalar loss
        """
        mse_loss = F.mse_loss(x_hat.float(), x.float(), reduction='mean')
        kl_divergence = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), dim=1).mean()
        loss = mse_loss + self.beta * kl_divergence
        return loss
