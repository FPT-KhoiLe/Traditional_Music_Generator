import torch
from matplotlib import pyplot as plt
from torch import nn
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms, datasets
import torch.nn.functional as F
import os
from PIL import Image

class AE_Visualize(nn.Module):
    def __init__(self):
        super().__init__()
        # Encoder
        self.enc_conv1 = nn.Conv2d(1, 64, 3, padding=1)
        self.enc_conv2 = nn.Conv2d(64, 128, 3, padding=1)
        self.enc_conv3 = nn.Conv2d(128, 256, 3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)

        # Decoder
        self.up1 = nn.ConvTranspose2d(256, 128, 2, stride=2)
        self.dec_conv1 = nn.Conv2d(256, 128, kernel_size=3, padding=1) # + 128 from skip connection
        self.up2 = nn.ConvTranspose2d(128, 64, 2, stride=2)
        self.dec_conv2 = nn.Conv2d(128, 64, kernel_size=3, padding=1) # + 64 from skip
        self.final_conv = nn.Conv2d(64, 1, kernel_size=3, padding=1)
    def forward(self, x):
        # Encoder
        print("Input Shape: ",x.shape)
        x_cpu = x.cpu().detach().numpy()
        plt.imshow(x_cpu[0][0], cmap='gray')
        plt.axis('off')
        plt.show()

        x1 = F.relu(self.enc_conv1(x)) # Save for skip connection
        print("Conv 1: ",x1.shape)

        x_1 = x1.cpu().detach().numpy()
        for i in range(10):
            plt.subplot(1, 10, i+1)
            plt.imshow(x_1[0][i], cmap='gray')
            plt.axis('off')
        plt.show()

        x1p = self.pool(x1)
        print("Pool 1: ",x1p.shape)
        x2 = F.relu(self.enc_conv2(x1p))  # Save for skip connection
        print("Conv 2: ",x2.shape)

        x_2 = x2.cpu().detach().numpy()
        for i in range(10):
            plt.subplot(1, 10, i+1)
            plt.imshow(x_2[0][i], cmap='gray')
            plt.axis('off')
        plt.show()

        x2p = self.pool(x2)
        print("Pool 2: ",x2p.shape)
        encoded = F.relu(self.enc_conv3(x2p))
        print("Conv 3: ",encoded.shape)

        encoded_np = encoded.cpu().detach().numpy()
        for i in range(10):
            plt.subplot(1, 10, i+1)
            plt.imshow(encoded_np[0][i], cmap='gray')
            plt.axis('off')
        plt.show()

        # Decoder
        u1 = self.up1(encoded)
        print("Up 1: ",u1.shape)
        u1_np = u1.cpu().detach().numpy()
        for i in range(10):
            plt.subplot(1, 10, i+1)
            plt.imshow(u1_np[0][i], cmap='gray')
            plt.axis('off')
        plt.show()

        d1 = F.relu(self.dec_conv1(torch.cat((u1, x2), 1))) # Connect with x2
        print("Connect 1: ", d1.shape)
        d1_np = d1.cpu().detach().numpy()
        for i in range(10):
            plt.subplot(1, 10, i+1)
            plt.imshow(d1_np[0][i], cmap='gray')
            plt.axis('off')
        plt.show()

        u2 = self.up2(u1)
        print("Up 2: ", u2.shape)
        u2_np = u2.cpu().detach().numpy()
        for i in range(10):
            plt.subplot(1, 10, i+1)
            plt.imshow(u2_np[0][i], cmap='gray')
            plt.axis('off')
        plt.show()

        d2 = F.relu(self.dec_conv2(torch.cat((u2, x1), 1)))
        print("Connect 2: ",d2.shape)
        d2_np = d2.cpu().detach().numpy()
        for i in range(10):
            plt.subplot(1, 10, i+1)
            plt.imshow(d2_np[0][i], cmap='gray')
            plt.axis('off')
        plt.show()

        decoded = F.sigmoid(self.final_conv(d2))
        print("Final Conv: ", decoded.shape)
        decoded_np = decoded.cpu().detach().numpy()

        plt.imshow(decoded_np[0][0], cmap='gray')
        plt.axis('off')
        plt.show()
        return
        return decoded

class AE(nn.Module):
    def __init__(self):
        super().__init__()
        # Encoder
        self.enc_conv1 = nn.Conv2d(1, 64, 3, padding=1)
        self.enc_conv2 = nn.Conv2d(64, 128, 3, padding=1)
        self.enc_conv3 = nn.Conv2d(128, 256, 3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)

        # Decoder
        self.up1 = nn.ConvTranspose2d(256, 128, 2, stride=2)
        self.dec_conv1 = nn.Conv2d(256, 128, kernel_size=3, padding=1) # + 128 from skip connection
        self.up2 = nn.ConvTranspose2d(128, 64, 2, stride=2)
        self.dec_conv2 = nn.Conv2d(128, 64, kernel_size=3, padding=1) # + 64 from skip
        self.final_conv = nn.Conv2d(64, 1, kernel_size=3, padding=1)

    def forward(self, x):
        # Encoder
        x1 = F.relu(self.enc_conv1(x)) # Save for skip connection
        x1p = self.pool(x1)
        x2 = F.relu(self.enc_conv2(x1p))  # Save for skip connection
        x2p = self.pool(x2)
        encoded = F.relu(self.enc_conv3(x2p))

        # Decoder
        u1 = self.up1(encoded)
        d1 = F.relu(self.dec_conv1(torch.cat((u1, x2), 1))) # Connect with x2
        u2 = self.up2(d1)
        d2 = F.relu(self.dec_conv2(torch.cat((u2, x1), 1)))
        decoded = F.sigmoid(self.final_conv(d2))
        return decoded

def visualize():
    mnist_data = datasets.MNIST(root='./data', train=True, transform=transforms.ToTensor(), download=True)
    train_loader = DataLoader(dataset=mnist_data, batch_size=64, shuffle=True)

    model = AE_Visualize().cuda()
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=0.0001)
    outputs = []
    num_epochs = 1
    # dataiter = iter(train_loader)
    # img, label = dataiter._next_data()
    # print(img.shape, label.shape)
    try:
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
    except AttributeError:
        print("Visualization finished.")
