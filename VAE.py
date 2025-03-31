import torch
import torch.nn.functional as F
from torch import nn
from torchvision import datasets, transforms
import matplotlib.pyplot as plt

class Encoder(nn.Module):
    def __init__(self, latent_dim):
        super(Encoder, self).__init__()

        # Build Model With LN
        self.relu = nn.ReLU()
        self.fc1 = nn.Linear(in_features=28*28, out_features=400)
        self.fc2 = nn.Linear(in_features=400, out_features=128)
        self.fc_mu = nn.Linear(in_features=128, out_features=latent_dim) # Mu
        self.fc_log_var = nn.Linear(in_features=128, out_features=latent_dim) # Log of variance

    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        mu = self.relu(self.fc_mu(x))
        log_var = self.relu(self.fc_log_var(x))
        return mu, log_var

class Decoder(nn.Module):
    def __init__(self, latent_dim):
        super(Decoder, self).__init__()
        self.relu = nn.ReLU()
        self.up1 = nn.Linear(in_features=latent_dim, out_features=128)
        self.up2 = nn.Linear(in_features=128, out_features=400)
        self.up3 = nn.Linear(in_features=400, out_features=28*28)
        self.sigmoid = nn.Sigmoid()

    def forward(self, mu, log_var):
        sigma = torch.exp(0.5 * log_var)
        epsilon = torch.randn_like(sigma)
        z = mu + sigma * epsilon

        reconstructed = self.relu(self.up1(z))
        reconstructed = self.relu(self.up2(reconstructed))
        reconstructed = self.sigmoid(self.up3(reconstructed))
        return reconstructed

class VAE(nn.Module):
    def __init__(self, latent_dim = 2):
        """
        :param latent_dim: Latent dim <= 128
        """
        super(VAE, self).__init__()
        self.encoder = Encoder(latent_dim)
        self.decoder = Decoder(latent_dim)

    def forward(self, x):
        mu, log_var = self.encoder(x)
        decoded = self.decoder(mu, log_var)
        return decoded, mu, log_var

def visualize():
    data = datasets.MNIST(root='./data', train=True, transform=transforms.ToTensor(), download=True)
    dataloader = torch.utils.data.DataLoader(data, batch_size=64, shuffle=False)

    # dataiter = iter(dataloader)
    # x, label = next(dataiter)
    # image = x[0].cpu().detach().numpy()
    # image = image.transpose((1, 2, 0))
    # plt.imshow(image, cmap='gray')
    # plt.axis('off')
    # plt.show()

    model = VAE(latent_dim=128).cuda()
    rec_loss_func = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    num_epochs = 1
    outputs = []
    print(model)
    for epoch in range(num_epochs):
        for x, y in dataloader:
            x = x.view(-1, 28*28) # For Linear neuron network
            x = x.cuda()
            x_rec, mu, log_var = model(x)
            rec_loss = rec_loss_func(x_rec, x)
            kl_div_loss = torch.sum(1 + log_var - mu.pow(2) - log_var.exp())
            loss = rec_loss + kl_div_loss
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        outputs.append([epoch, x, x_rec])

    for k in range(0, num_epochs, 4):
        plt.figure(figsize=(10, 10))
        plt.gray()
        imgs = outputs[k][1].cpu().detach().numpy()
        recons = outputs[k][2].cpu().detach().numpy()
        print(imgs)
        for i, item in enumerate(imgs):
            if i >= 9: break
            plt.subplot(2, 9, i + 1)
            item = item.reshape(-1, 28, 28)
            plt.imshow(item[0], cmap='gray')
            plt.axis('off')
        # plt.show()
        for i, item in enumerate(recons):
            if i >= 9: break
            plt.subplot(2, 9, 9 + i + 1)
            item = item.reshape(-1, 28, 28)
            plt.imshow(item[0], cmap='gray')
            plt.axis('off')
        plt.show()

if __name__ == '__main__':
    visualize()