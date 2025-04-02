import torch
from torch import nn
import torch.nn.functional as F
from base import BaseVAE
from types_ import *

class Encoder_LN(nn.Module):
    def __init__(self, latent_dim):
        super(Encoder_LN, self).__init__()
        self.relu = nn.ReLU()
        self.fc1 = nn.Linear(in_features=28 * 28, out_features=400)
        self.fc2 = nn.Linear(in_features=400, out_features=128)
        self.fc_mu = nn.Linear(in_features=128, out_features=latent_dim)  # Mu
        self.fc_log_var = nn.Linear(in_features=128, out_features=latent_dim)  # Log of variance

    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        # No activation function on mu and log_var
        mu = self.fc_mu(x)
        log_var = self.fc_log_var(x)
        return mu, log_var


class Decoder_LN(nn.Module):
    def __init__(self, latent_dim):
        super(Decoder_LN, self).__init__()
        self.relu = nn.ReLU()
        self.up1 = nn.Linear(in_features=latent_dim, out_features=128)
        self.up2 = nn.Linear(in_features=128, out_features=400)
        self.up3 = nn.Linear(in_features=400, out_features=28 * 28)
        self.sigmoid = nn.Sigmoid()

    def forward(self, z):
        # Decoder should take z (not mu, log_var)
        reconstructed = self.relu(self.up1(z))
        reconstructed = self.relu(self.up2(reconstructed))
        reconstructed = self.sigmoid(self.up3(reconstructed))
        return reconstructed


class VAE_LN(nn.Module):
    def __init__(self, latent_dim=2):
        super(VAE_LN, self).__init__()
        self.latent_dim = latent_dim
        self.encoder = Encoder_LN(latent_dim)
        self.decoder = Decoder_LN(latent_dim)

    def reparameterize(self, mu, log_var):
        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(std)
        z = mu + eps * std
        return z

    def forward(self, x):
        mu, log_var = self.encoder(x)
        z = self.reparameterize(mu, log_var)
        decoded = self.decoder(z)
        return decoded, mu, log_var


class VAE_CNN_Gen_Update(nn.Module):
    def __init__(self,
                 in_channels: int,
                 out_channels: int,
                 latent_dim: int,
                 hidden_dim : list = None,
                 img_size: int = 64,
                 **kwargs) -> None:
        super(VAE_CNN_Gen_Update, self).__init__()

        # Setup Modules
        modules = []
        if hidden_dim is None:
            hidden_dim = [32, 64, 128, 256, 512]

        encoded_img_size = img_size
        kernel_size = 3
        stride = 2
        padding = 1
        # Build Encoder
        for h_dim in hidden_dim:
            modules.append(
                nn.Sequential(
                    nn.Conv2d(in_channels=in_channels,
                             out_channels=h_dim,
                             kernel_size=kernel_size, stride=stride, padding=padding),
                    nn.BatchNorm2d(h_dim),
                    nn.LeakyReLU())
            )
            encoded_img_size = (encoded_img_size - 3 + 2*padding)//stride + 1 # Image size formula
            in_channels = h_dim

        self.encoded_img_size = encoded_img_size

        self.encoder = nn.Sequential(*modules)
        flatten_size = hidden_dim[-1] * encoded_img_size ** 2

        self.fc_mu = nn.Linear(in_features=flatten_size, out_features=latent_dim)
        self.fc_log_var = nn.Linear(in_features=flatten_size, out_features=latent_dim)

        # Decoder
        modules = []
        self.decoder_input = nn.Linear(latent_dim, flatten_size)
        self.hidden_dim = hidden_dim
        hidden_dim.reverse()
        for i in range(len(hidden_dim) -1):
            modules.append(
                nn.Sequential(
                    nn.ConvTranspose2d(in_channels=hidden_dim[i],
                                       out_channels=hidden_dim[i+1],
                                       kernel_size=3,
                                       stride=2,
                                       padding=1,
                                       output_padding=1),
                    nn.BatchNorm2d(hidden_dim[i+1]),
                    nn.LeakyReLU()
                )
            )

        self.decoder = nn.Sequential(*modules)

        self.final_layer = nn.Sequential(
            nn.ConvTranspose2d(in_channels=hidden_dim[-1],
                               out_channels=hidden_dim[-1],
                               kernel_size=3,
                               stride=2,
                               padding=1,
                               output_padding=1),
            nn.BatchNorm2d(hidden_dim[-1]),
            nn.LeakyReLU(),
            nn.Conv2d(in_channels=hidden_dim[-1],
                      out_channels=out_channels,
                      kernel_size=3,
                      padding=1,),
            nn.Sigmoid()
        )

    def encode(self, input: Tensor) -> List[Tensor]:
        """
        Encodes the input by passing through the encoder network and return the latent codes.
        :param input: (Tensor) Input tensor to encoder [N x C x H x W]
        :return: (Tensor) List of latent codes (mu, log_var)
        """
        result = self.encoder(input)
        result = torch.flatten(result, start_dim=1)
        mu = self.fc_mu(result)
        log_var = self.fc_log_var(result)
        return [mu, log_var]

    def decode(self, z: Tensor) -> Tensor:
        """
        Maps the given latent codes onto the image space.
        :param z: (Tensor) [N x LD]
        :return: (Tensor) [N x C x H x W]
        """
        result = self.decoder_input(z)
        result =result.view(-1, self.hidden_dim[0], self.encoded_img_size, self.encoded_img_size)
        result = self.decoder(result)
        result = self.final_layer(result)
        return result

    def reparameterize(self, mu : Tensor, log_var: Tensor) -> Tensor:
        """
        Reparameterization trick to sample from N(mu, log_var) from N(0,1).
        :param mu: (Tensor) Mean of the latent Gaussian [N x LD]
        :param log_var: (Tensor) Variance of the latent Gaussian [N x LD]
        :return: (Tensor) [B x LD]
        """
        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(std)
        return eps * std + mu

    def forward(self, input: Tensor) -> Tensor:
        mu, log_var = self.encode(input)
        z = self.reparameterize(mu, log_var)
        decoded = self.decode(z)
        return [decoded, mu, log_var]

    def loss_function(self,
                      *args,
                      **kwargs) -> dict:
        """
        Compute the VAE Loss fucntion.
        KL(N(\mu, \sigma), N(0, 1)) = \log \frac{1}{\sigma} + \frac{\sigma^2 + \mu^2}{2} - \frac{1}{2}
        :param args:
        :param kwargs:
        :return:
        """
        recons = args[0]
        input = args[1]
        mu = args[2]
        log_var = args[3]

        kld_weight = kwargs['M_N'] # Account for the minibatch samples from the dataset
        recon_loss = F.mse_loss(recons, input)

        kld_loss = torch.mean(-0.5 * torch.sum(1 + log_var - mu ** 2 - torch.exp(log_var), dim=1), dim=0)
        loss = recon_loss + kld_weight * kld_loss
        return {'loss': loss, 'recon_loss': recon_loss.detach(), 'kld_loss': -kld_loss.detach()}

    def sample(self,
               num_samples: int,
               current_device: int,
               **kwargs) -> Tensor:
        """
        Samples from the latent space and return the corresponding image space map.
        :param num_samples:
        :param current_device:
        :param kwargs:
        :return:
        """
        z = torch.randn(num_samples,
                        self.latent_dim)

        z = z.to(current_device)

        samples = self.decode(z)
        return samples

    def generate(self,
                 x: Tensor,
                 **kwargs) -> Tensor:
        """
        Given an input image x, returns the reconstructed image.
        :param x:
        :param kwargs:
        :return:
        """

        return self.forward(x)[0]


class VAE_CNN(BaseVAE):
    """
    Convolutional Variational AutoEncoder
    """
    def __init__(self,
                 in_channels: int = 3,
                 out_channels: int = 1,
                 latent_dim: int = 20,
                 hidden_dim: List = None,
                 img_size: int = 32) -> None:
        super(VAE_CNN, self).__init__()

        self.latent_dim = latent_dim
        self.img_size = img_size

        # Set default hidden dimensions if not provided
        if hidden_dim is None:
            hidden_dim = [32, 64, 128, 256, 512]

        self.hidden_dim = hidden_dim.copy()

        # Build Encoder
        modules = []

        current_img_size = img_size
        for h_dim in self.hidden_dim:
            modules.append(
                nn.Sequential(
                    nn.Conv2d(in_channels, h_dim, kernel_size=3, stride=2, padding=1), # Stride = 2 decrease size 2 times
                    nn.BatchNorm2d(h_dim),
                    nn.LeakyReLU(),
                )
            )
            current_img_size = (current_img_size - 3 + 2 * 1)//2 + 1 # Update img size based on architecture
            in_channels = h_dim

        self.encoder = nn.Sequential(*modules)
        self.encoded_img_size = current_img_size
        flattened_size = self.hidden_dim[-1] * current_img_size ** 2

        # Latent space parameters
        self.fc_mu = nn.Linear(flattened_size, latent_dim)
        self.fc_log_var = nn.Linear(flattened_size, latent_dim)

        # Build Decoder
        modules = []
        self.decoder_input = nn.Linear(latent_dim, flattened_size)

        # Reverse hidden dimensions for decoder
        hidden_dim_reversed = list(reversed(self.hidden_dim))

        for i in range(len(hidden_dim_reversed) - 1):
            modules.append(
                nn.Sequential(
                    nn.ConvTranspose2d(
                        hidden_dim_reversed[i],
                        hidden_dim_reversed[i + 1],
                        kernel_size=3,
                        stride=2,
                        padding=1,
                        output_padding=1,
                    ),
                    nn.BatchNorm2d(hidden_dim_reversed[i + 1]),
                    nn.LeakyReLU(),
                )
            )

        self.decoder = nn.Sequential(*modules)

        self.final_layer = nn.Sequential(
            nn.ConvTranspose2d(
                hidden_dim_reversed[-1],
                hidden_dim_reversed[-1],
                kernel_size=3,
                stride=2,
                padding=1,
                output_padding=1,
            ),
            nn.BatchNorm2d(hidden_dim_reversed[-1]),
            nn.LeakyReLU(),
            nn.Conv2d(
                hidden_dim_reversed[-1],
                out_channels,
                kernel_size=3,
                padding=1,
            ),
            nn.Sigmoid()
        )

    def encode(self, input: Tensor) -> List[Tensor]:
        result = self.encoder(input)
        result = torch.flatten(result, 1)
        mu = self.fc_mu(result)
        log_var = self.fc_log_var(result)
        return [mu, log_var]

    def decode(self, z: Tensor) -> Tensor:
        result = self.decoder_input(z)
        result = result.view(-1, self.hidden_dim[-1], self.encoded_img_size, self.encoded_img_size)
        result = self.decoder(result)
        return self.final_layer(result)

    def sample(self,
               num_samples: int,
               current_device: int, **kwargs) -> Tensor:
        """
        Samples from the latent space and return the corresponding
        image space map.
        :param num_samples: (Int) Number of samples
        :param current_device: (Int) Device to run the model
        :return: (Tensor)
        """
        z = torch.randn(num_samples,
                        self.latent_dim)

        z = z.to(current_device)

        samples = self.decode(z)
        return samples

    def generate(self, x: Tensor, **kwargs) -> Tensor:
        """
        Given an input image x, returns the reconstructed image
        :param x: (Tensor) [B x C x H x W]
        :return: (Tensor) [B x C x H x W]
        """

        return self.forward(x)[0]

    def reparameterize(self, mu: Tensor, log_var: Tensor) -> Tensor:
        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(std)
        return eps.mul(std).add_(mu)

    def forward(self, x: Tensor) -> Tensor:
        mu, log_var = self.encode(x)
        z = self.reparameterize(mu, log_var)
        decoded = self.decode(z)
        return [decoded, mu, log_var]

    def loss_function(self, *args) -> Tensor:
        recons, input, mu, log_var = args[:4]
        kld_weight = 0.5

        recon_loss = F.mse_loss(recons, input, reduction='sum')
        kld_loss = -0.5*torch.sum(1 + log_var - mu.pow(2) - log_var.exp(), dim=1).mean()
        kld_loss = kld_weight * kld_loss

        return recon_loss + kld_loss