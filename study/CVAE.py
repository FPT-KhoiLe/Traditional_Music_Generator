from torch import nn
from torch.utils.data import Dataset
from torchvision import datasets, transforms
import torch.nn.functional as F
from torchvision import transforms
import torch

class ColorizationDataset(Dataset):
    def __init__(self, data):
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        color_img, label = self.data[idx]
        gray_img = transforms.Grayscale()(color_img)
        return color_img, gray_img

class PriorNet(nn.Module):
    def __init__(self, latent_dim):
        super(PriorNet, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=32, kernel_size=5, stride=1, padding=2)
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=5, stride=1, padding=2)
        self.conv3 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=5, stride=1, padding=2)
        self.conv4 = nn.Conv2d(in_channels=128, out_channels=256, kernel_size=5, stride=1, padding=2)
        self.conv5 = nn.Conv2d(in_channels=256, out_channels=512, kernel_size=5, stride=1, padding=2)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.fc = nn.Linear(in_features=512 * 4 * 4, out_features=512)
        self.fc_mu = nn.Linear(in_features=512, out_features=latent_dim)
        self.fc_log_var = nn.Linear(in_features=512, out_features=latent_dim)
    def forward(self, c):
        c = self.pool(F.relu(self.conv1(c)))
        c = self.pool(F.relu(self.conv2(c)))
        c = self.pool(F.relu(self.conv3(c)))
        c = self.pool(F.relu(self.conv4(c)))
        c = self.pool(F.relu(self.conv5(c)))

        c = c.view(-1, 512 * 4 * 4)
        c = F.relu(self.fc(c))

        mu_p = self.fc_mu(c)
        log_var_p = self.fc_log_var(c)
        return mu_p, log_var_p

class Encoder(nn.Module):
    def __init__(self, latent_dim):
        super(Encoder, self).__init__()
        self.latent_dim = latent_dim
        self.conv1 = nn.Conv2d(in_channels=4, out_channels=32, kernel_size=5, stride=1, padding=2)
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=5, stride=1, padding=2)
        self.conv3 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=5, stride=1, padding=2)
        self.conv4 = nn.Conv2d(in_channels=128, out_channels=256, kernel_size=5, stride=1, padding=2)
        self.conv5 = nn.Conv2d(in_channels=256, out_channels=512, kernel_size=5, stride=1, padding=2)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.fc = nn.Linear(in_features=512 * 4 * 4, out_features=512)
        self.fc_mu = nn.Linear(in_features=512, out_features=latent_dim)
        self.fc_log_var = nn.Linear(in_features=512, out_features=latent_dim)

    def forward(self, x, c):
        # cat([x,c], dim=1)
        input = torch.cat((x, c), dim=1)

        # Pool(Relu(Conv1(x)))
        input = self.pool(F.relu(self.conv1(input)))
        # Pool(Relu(Conv2(x)))
        input = self.pool(F.relu(self.conv2(input)))
        # Pool(Relu(Conv3(x)))
        input = self.pool(F.relu(self.conv3(input)))
        # Pool(Relu(Conv4(x))
        input = self.pool(F.relu(self.conv4(input)))
        # Pool(Relu(Conv5(x))
        input = self.pool(F.relu(self.conv5(input))) # => (512 , 4 ,4)

        # view (-1, out_channels * current_size**2)
        input = input.view(-1, 512 * 4 * 4)
        input = F.relu(self.fc(input))

        mu_q = self.fc_mu(input)
        log_var_q = self.fc_log_var(input)

        return mu_q, log_var_q

class Decoder(nn.Module):
    def __init__(self, latent_dim):
        super(Decoder, self).__init__()
        self.fc = nn.Linear(latent_dim, 512 * 4 * 4)
        self.upsample1 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)
        self.conv1 = nn.Conv2d(512, 256, kernel_size=5, stride=1, padding=2)
        self.upsample2 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)
        self.conv2 = nn.Conv2d(256, 128, kernel_size=5, stride=1, padding=2)
        self.upsample3 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)
        self.conv3 = nn.Conv2d(128, 64, kernel_size=5, stride=1, padding=2)
        self.upsample4 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)
        self.conv4 = nn.Conv2d(64, 32, kernel_size=5, stride=1, padding=2)
        self.upsample5 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)
        self.conv5 = nn.Conv2d(32, 3, kernel_size=5, stride=1, padding=2)

    def forward(self, z):
        # Use fc to reconstruct x size from latent dim
        x = F.relu(self.fc(z))
        x = x.view(-1, 512, 4, 4)
        x = F.relu(self.conv1(self.upsample1(x)))
        x = F.relu(self.conv2(self.upsample2(x)))
        x = F.relu(self.conv3(self.upsample3(x)))
        x = F.relu(self.conv4(self.upsample4(x)))
        x = F.relu(self.conv5(self.upsample5(x)))
        x = F.sigmoid(x)
        return x


class CVAE(nn.Module):
    def __init__(self, latent_dim):
        super(CVAE, self).__init__()
        self.encoder = Encoder(latent_dim=latent_dim)
        self.prior = PriorNet(latent_dim=latent_dim)
        self.decoder = Decoder(latent_dim=latent_dim)

    def reparameterize(self, mu, log_var):
        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(std)
        z = mu + eps * std
        return z

    def forward(self, x, c):
        mu_q, log_var_q = self.encoder(x, c)
        mu_p, log_var_p = self.prior(c)
        z = self.reparameterize(mu_q, log_var_q)
        output = self.decoder(z)
        return output, mu_p, log_var_p, mu_q, log_var_q

class CVAE_Generator(nn.Module):
    def __init__(self, encoder, decoder):
        super(CVAE_Generator, self).__init__()
        self.encoder = encoder
        self.decoder = decoder

    def reparameterize(self, mu, log_var):
        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(std)
        z = mu + eps * std
        return z

    def forward(self, x):
        mu, log_var = self.encoder(x)
        z = self.reparameterize(mu, log_var)
        output = self.decoder(z)
        return output