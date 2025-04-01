import torch
from torch import nn
import torch.nn.functional as F

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


class Encoder_CNN(nn.Module):
    def __init__(self, in_channels, latent_dim):
        super(Encoder_CNN, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, 32, 3, stride=2, padding=1)
        self.conv2 = nn.Conv2d(32, 64, 3, stride=2, padding=1)
        self.conv3 = nn.Conv2d(64, 128, 3, stride=2, padding=1)
        self.fc_size = 128 * 4 * 4
        self.fc_mu = nn.Linear(in_features=self.fc_size, out_features=latent_dim)
        self.fc_log_var = nn.Linear(in_features=self.fc_size, out_features=latent_dim)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = x.view(-1, self.fc_size)
        mu = self.fc_mu(x)
        log_var = self.fc_log_var(x)
        return mu, log_var

class Decoder_CNN(nn.Module):
    def __init__(self, in_channels, latent_dim):
        super(Decoder_CNN, self).__init__()
        self.fc = nn.Linear(in_features=latent_dim, out_features=128*4*4)

        self.de_conv1 = nn.ConvTranspose2d(128, 64, 3, stride=2, padding=1)
        self.de_conv2 = nn.ConvTranspose2d(64, 32, 3, stride=2, padding=1, output_padding=1)
        self.de_conv3 = nn.ConvTranspose2d(32, in_channels, 3, stride=2, padding=1, output_padding=1)

    def forward(self, z):
        z = F.relu(self.fc(z))
        z = z.view(-1, 128, 4, 4)
        z = F.relu(self.de_conv1(z))
        z = F.relu(self.de_conv2(z))
        z = F.sigmoid(self.de_conv3(z))
        return z

class VAE_CNN(nn.Module):
    def __init__(self, in_channels = 1, latent_dim=2):
        super(VAE_CNN, self).__init__()
        self.latent_dim = latent_dim
        self.encoder = Encoder_CNN(in_channels, latent_dim)
        self.decoder = Decoder_CNN(in_channels, latent_dim)

    def reparameterize(self, mu, log_var):
        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(std)
        z = mu + eps * std
        return z

    def forward(self, x):
        mu, log_var = self.encoder(x)
        z = self.reparameterize(mu, log_var)

        decoded = self.decoder(z)
        return decoded, mu, log_var # Return mu and log_var for loss function