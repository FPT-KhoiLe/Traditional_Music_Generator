from sympy.solvers.diophantine.diophantine import reconstruct

from VAE import VAE_LN, VAE_CNN
from AE import AE
from torchvision import datasets, transforms
import matplotlib.pyplot as plt
import torch
from torch import nn


def ae_cnn():
    mnist_data = datasets.MNIST(root='./data', train=True, transform=transforms.ToTensor(), download=True)
    train_loader = torch.utils.data.DataLoader(dataset=mnist_data, batch_size=64, shuffle=True)

    model = AE().cuda()
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
                plt.subplot(2, 9, 9 + i + 1)
                # item = item.reshape(-1, 28, 28)
                plt.imshow(item[0], cmap='gray')
                plt.axis('off')
            plt.show()
    except AttributeError:
        print("Visualization finished.")

def vae_ln():
    data = datasets.MNIST(root='./data', train=True, transform=transforms.ToTensor(), download=True)
    dataloader = torch.utils.data.DataLoader(data, batch_size=64, shuffle=True)

    model = VAE_LN(latent_dim=2).cuda()  # Smaller latent dim to start
    rec_loss_func = nn.MSELoss(reduction='sum')  # Binary cross entropy for MNIST
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    num_epochs = 1
    outputs = []

    for epoch in range(num_epochs):
        train_loss = 0
        for i, (x, _) in enumerate(dataloader):
            x = x.view(-1, 28 * 28)
            x = x.cuda()

            # Forward pass
            x_rec, mu, log_var = model(x)

            # Compute losses
            rec_loss = rec_loss_func(x_rec, x)

            # KL divergence: -0.5 * sum(1 + log_var - mu^2 - exp(log_var))
            kl_div_loss = -0.5 * torch.sum(1 + log_var - mu.pow(2) - log_var.exp())

            # Total loss
            loss = rec_loss + kl_div_loss

            # Backward and optimize
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            train_loss += loss.item()

        # Print epoch stats
        avg_loss = train_loss / len(data)
        print(f"Epoch: {epoch + 1}/{num_epochs}. Loss: {avg_loss:.6f}")

        # Save outputs for visualization
        if epoch % 2 == 0 or epoch == num_epochs - 1:
            outputs.append([epoch, x, x_rec])

    # Visualization
    for k in range(0, len(outputs), 4):
        plt.figure(figsize=(10, 4))
        plt.gray()
        imgs = outputs[k][1].cpu().detach().numpy()
        recons = outputs[k][2].cpu().detach().numpy()

        for i in range(5):
            # Original images
            plt.subplot(2, 5, i + 1)
            plt.imshow(imgs[i].reshape(28, 28), cmap='gray')
            plt.axis('off')

            # Reconstructed images
            plt.subplot(2, 5, 5 + i + 1)
            plt.imshow(recons[i].reshape(28, 28), cmap='gray')
            plt.axis('off')

        plt.suptitle(f"Epoch {outputs[k][0] + 1}")
        plt.tight_layout()
        plt.show()

def vae_cnn():
    data = datasets.MNIST(root='./data', train=True, transform=transforms.ToTensor(), download=True)
    train_loader = torch.utils.data.DataLoader(dataset=data, batch_size=64, shuffle=True)

    model = VAE_CNN(latent_dim=20, in_channels=1).cuda()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=0.0001)
    outputs = []
    num_epochs = 5
# try:
    for epoch in range(num_epochs):
        for x, y in train_loader:
            x = x.cuda()
            x_rec, mu, log_var = model(x)
            reconstruction_loss = nn.MSELoss(reduction="sum")(x_rec, x)
            kl_div_loss = 0.5 * torch.sum(1 + log_var - mu.pow(2) - log_var.exp())
            loss = reconstruction_loss + kl_div_loss
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        outputs.append((epoch, x, x_rec))
        print(f"Epoch: {epoch + 1}/{num_epochs}. Loss: {loss:.6f}")

    # Visualization
    for k in range(0, len(outputs), 4):
        plt.figure(figsize=(10, 4))
        plt.gray()
        imgs = outputs[k][1].cpu().detach().numpy()
        recons = outputs[k][2].cpu().detach().numpy()

        for i in range(5):
            # Original images
            plt.subplot(2, 5, i + 1)
            plt.imshow(imgs[i].reshape(28, 28), cmap='gray')
            plt.axis('off')

            # Reconstructed images
            plt.subplot(2, 5, 5 + i + 1)
            plt.imshow(recons[i].reshape(28, 28), cmap='gray')
            plt.axis('off')

        plt.suptitle(f"Epoch {outputs[k][0] + 1}")
        plt.tight_layout()
        plt.show()

# except AttributeError:
#     pass
if __name__ == '__main__':
    # vae_ln()
    vae_cnn()
    # ae_cnn()