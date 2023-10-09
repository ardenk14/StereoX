import torch
import torch.nn as nn


class StateEncoder(nn.Module):
    """
    Embeds the state into a latent space.
    State shape: (..., num_channels, 32, 32)
    latent shape: (..., latent_dim)
    Check the notebook for more details about the architecture.
    """

    def __init__(self, latent_dim, num_channels=3):
        super().__init__()
        self.latent_dim = latent_dim
        self.num_channels = num_channels
        self.encoder = nn.Sequential(
            nn.Conv2d(num_channels, 4, kernel_size=5, stride=1, padding=0, dilation=1),
            nn.ReLU(),
            nn.MaxPool2d(2, stride=2),
            nn.Conv2d(4, 4, kernel_size=5, stride=1, padding=0, dilation=1),
            nn.ReLU(),
            nn.MaxPool2d(2, stride=2),
            nn.Flatten(),
            nn.Linear(100, 100),
            nn.ReLU(),
            nn.Linear(100, latent_dim)
            )
        
    def forward(self, state):
        """
        :param state: <torch.Tensor> of shape (..., num_channels, 32, 32)
        :return latent_state: <torch.Tensor> of shape (..., latent_dim)
        """
        latent_state = None
        input_shape = state.shape
        state = state.reshape(-1, self.num_channels, 32, 32)
        latent_state = self.encoder(state)

        # convert to original multi-batch shape
        latent_state = latent_state.reshape(*input_shape[:-3], self.latent_dim)
        return latent_state


class StateVariationalEncoder(nn.Module):
    """
    Embeds the state into a latent space.
    State shape: (..., num_channels, 32, 32)
    latent shape: (..., latent_dim)
    Check the notebook for more details about the architecture.
    """

    def __init__(self, latent_dim, num_channels=3):
        super().__init__()
        self.latent_dim = latent_dim
        self.num_channels = num_channels
        self.encoder = nn.Sequential(
            nn.Conv2d(num_channels, 4, kernel_size=5, stride=1, padding=0, dilation=1),
            nn.ReLU(),
            nn.MaxPool2d(2, stride=2),
            nn.Conv2d(4, 4, kernel_size=5, stride=1, padding=0, dilation=1),
            nn.ReLU(),
            nn.MaxPool2d(2, stride=2),
            nn.Flatten(),
            nn.Linear(100, 100),
            nn.ReLU()
            )
        self.mean = nn.Linear(100, self.latent_dim)
        self.std = nn.Linear(100, self.latent_dim)

    def forward(self, state):
        """
        :param state: <torch.Tensor> of shape (..., num_channels, 32, 32)
        :return: 2 <torch.Tensor>
          :mu: <torch.Tensor> of shape (..., latent_dim)
          :log_var: <torch.Tensor> of shape (..., latent_dim)
        """
        mu = None
        log_var = None
        input_shape = state.shape
        state = state.reshape(-1, self.num_channels, 32, 32)
        encode = self.encoder(state)
        mu = self.mean(encode)
        log_var = self.std(encode)

        # convert to original multi-batch shape
        mu = mu.reshape(*input_shape[:-3], self.latent_dim)
        log_var = log_var.reshape(*input_shape[:-3], self.latent_dim)
        return mu, log_var

    def reparameterize(self, mu, logvar):
        """
        Reparametrization trick to sample from N(mu, std) from N(0,1)
        :param mu: <torch.Tensor> of shape (..., latent_dim)
        :param logvar: <torch.Tensor> of shape (..., latent_dim)
        :return: <torch.Tensor> of shape (..., latent_dim)
        """
        sampled_latent_state = None
        eps = torch.randn(logvar.shape)
        # compute the standard deviation from the log variance
        std = torch.sqrt(torch.exp(logvar))
        # use the reparameterization trick to sample from the latent space
        sampled_latent_state = mu + eps * std

        return sampled_latent_state


class StateDecoder(nn.Module):
    """
    Reconstructs the state from a latent space.
    Check the notebook for more details about the architecture.
    """

    def __init__(self, latent_dim, num_channels=3):
        super().__init__()
        self.latent_dim = latent_dim
        self.num_channels = num_channels
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 500),
            nn.ReLU(),
            nn.Linear(500, 500),
            nn.ReLU(),
            nn.Linear(500, num_channels * 32 * 32)
            )

    def forward(self, latent_state):
        """
        :param latent_state: <torch.Tensor> of shape (..., latent_dim)
        :return decoded_state: <torch.Tensor> of shape (..., num_channels, 32, 32)
        """
        decoded_state = None
        input_shape = latent_state.shape
        latent_state = latent_state.reshape(-1, self.latent_dim)

        decoded_state = self.decoder(latent_state)

        decoded_state = decoded_state.reshape(*input_shape[:-1], self.num_channels, 32, 32)

        return decoded_state


class StateVAE(nn.Module):
    """
    State AutoEncoder
    """

    def __init__(self, latent_dim, num_channels=3):
        super().__init__()
        self.latent_dim = latent_dim
        self.num_channels = num_channels
        self.encoder = StateVariationalEncoder(latent_dim, num_channels)
        self.decoder = StateDecoder(latent_dim, num_channels)

    def forward(self, state):
        """
        :param state: <torch.Tensor> of shape (..., num_channels, 32, 32)
        :return:
            reconstructed_state: <torch.Tensor> of shape (..., num_channels, 32, 32)
            mu: <torch.Tensor> of shape (..., latent_dim)
            log_var: <torch.Tensor> of shape (..., latent_dim)
            latent_state: <torch.Tensor> of shape (..., latent_dim)
        """
        reconstructed_state = None # decoded states from the latent_state
        mu, log_var = None, None # mean and log variance obtained from encoding state
        latent_state = None # sample from the latent space feeded to the decoder
        mu, log_var = self.encoder(state)
        latent_state = self.reparameterize(mu, log_var)
        reconstructed_state = self.decoder(latent_state)
        return reconstructed_state, mu, log_var, latent_state

    def encode(self, state):
        """
        :param state: <torch.Tensor> of shape (..., num_channels, 32, 32)
        :return: <torch.Tensor> of shape (..., latent_dim)
        """
        latent_state = None
        mu, log_var = self.encoder(state)
        latent_state = self.reparameterize(mu, log_var)
        return latent_state

    def decode(self, latent_state):
        """
        :param latent_state: <torch.Tensor> of shape (..., latent_dim)
        :return: <torch.Tensor> of shape (..., num_channels, 32, 32)
        """
        reconstructed_state = self.decoder(latent_state)
        return reconstructed_state

    def reparameterize(self, mu, logvar):
        return self.encoder.reparameterize(mu, logvar)
