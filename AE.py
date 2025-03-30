import torch
from matplotlib import pyplot as plt
from torch import nn
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms, datasets

import os
from PIL import Image

class AE(nn.Module):
    def __init__(self):
        super(AE, self).__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(1, 16, 3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(16, 32, 3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 64, 7, 2),
        )
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(64, 32, 7),
            nn.ReLU(),
            nn.ConvTranspose2d(32, 16, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(16, 1, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.Sigmoid(),
        )

    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded


def main():
    mnist_data = datasets.MNIST(root='./data', train=True, transform=transforms.ToTensor(), download=True)
    train_loader = DataLoader(dataset=mnist_data, batch_size=64, shuffle=True)

    model = AE().cuda()
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=0.0001)
    outputs = []
    num_epochs = 15
    # dataiter = iter(train_loader)
    # img, label = dataiter._next_data()
    # print(img.shape, label.shape)
    for epoch in range(num_epochs):
        # Train loop
        for x, y in train_loader:
            x = x.cuda()
            x_hat = model(x)
            loss = criterion(x_hat, x)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        print("Epoch [{}/{}], Loss: {:.4f}".format(epoch + 1, num_epochs, loss.item()))
        outputs.append((epoch, x, x_hat))

    for k in range(0, num_epochs, 4):
        plt.figure(figsize=(10, 10))
        plt.gray()
        imgs = outputs[k][1].cpu().detach().numpy()
        recons = outputs[k][2].cpu().detach().numpy()
        for i, item in enumerate(imgs):
            if i >= 9: break
            plt.subplot(2, 9, i + 1)
            # item = item.reshape(-1, 28, 28)
            plt.imshow(item[0], cmap='gray')
            plt.axis('off')

        for i, item in enumerate(recons):
            if i >= 9: break
            plt.subplot(2, 9, 9+ i + 1)
            # item = item.reshape(-1, 28, 28)
            plt.imshow(item[0], cmap='gray')
            plt.axis('off')
        plt.show()
if __name__ == '__main__':
    main()