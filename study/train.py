import sys

import numpy as np
from torch.utils.data import Dataset
from tqdm import tqdm
from VAE import VAE_LN,VAE_CNN, VAE_CNN_Gen_Update
from AE import AE
from CVAE import CVAE, ColorizationDataset
from torchvision import datasets, transforms
import matplotlib.pyplot as plt
import torch
from torch import nn
import os
from PIL import Image
from torch.utils.tensorboard import SummaryWriter
import torchvision

class CustomDataset(Dataset):
    def __init__(self, root, transform=None):
        self.transform = transform
        self.samples = []

        # Tạo danh sách lớp và ánh xạ sang index
        self.classes = sorted(entry.name for entry in os.scandir(root) if entry.is_dir())
        self.class_to_idx = {cls_name: i for i, cls_name in enumerate(self.classes)}

        # Duyệt cây thư mục và lưu đường dẫn ảnh + nhãn
        for class_name in self.classes:
            class_dir = os.path.join(root, class_name)
            for file_name in os.scandir(class_dir):
                if file_name.is_file() and file_name.name.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp')):
                    self.samples.append((
                        os.path.join(class_dir, file_name.name),
                        self.class_to_idx[class_name]
                    ))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, label = self.samples[idx]

        # Load ảnh và convert về RGB để xử lý ảnh grayscale/transparency
        img = Image.open(img_path).convert('RGB')

        if self.transform:
            img = self.transform(img)

        return img, label


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
    num_epochs = 10
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
    img_size = 128
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Resize((img_size, img_size)),
    ])

    data = CustomDataset(root="./data/Intel_image_classification/seg_train/seg_train/",
                         transform=transform)

    train_loader = torch.utils.data.DataLoader(dataset=data, batch_size=64, shuffle=True)

    model = VAE_CNN(in_channels=3, out_channels=3, latent_dim=5000, img_size=img_size).cuda()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=0.0001)
    scaler = torch.cuda.amp.GradScaler()  # Tự động mixed precision
    writer = SummaryWriter("runs/vae")  # Khởi tạo TensorBoard

    # Early stopping
    best_loss = float('inf')
    patience = 10
    patience_counter = 0
    best_model_path = "VAE_BestResult/best_vae_model.pth"

    num_epochs = 100  # Đặt lớn để early stopping có thể hoạt động
    for epoch in range(num_epochs):
        epoch_loss = 0.0
        model.train()

        for x, _ in train_loader:  # Giả sử dataset không có label
            x = x.cuda()

            # Forward pass với autograd
            with torch.cuda.amp.autocast():
                x_rec, mu, log_var = model(x)
                loss = model.loss_function(x_rec, x, mu, log_var)

            # Backpropagation tối ưu hóa
            optimizer.zero_grad()
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            epoch_loss += loss.item()

        # Tính loss trung bình epoch
        avg_loss = epoch_loss / len(train_loader)

        # Ghi log lên TensorBoard
        writer.add_scalar('Loss/train', avg_loss, epoch)

        # Early stopping logic
        if avg_loss < best_loss:
            best_loss = avg_loss
            patience_counter = 0
            # Lưu model tốt nhất
            torch.save(model.state_dict(), best_model_path)

            plt.figure(figsize=(10, 4))
            imgs = x.cpu().detach().numpy()
            recons = x_rec.cpu().detach().numpy()
            for i in range(5):
                plt.subplot(2, 5, i + 1)
                print(np.transpose(imgs[i], (1, 2, 0)).dtype)
                print(np.transpose(recons[i], (1, 2, 0)).dtype)
                plt.imshow(np.transpose(imgs[i], (1, 2, 0)).astype(np.float32))
                plt.axis('off')
                plt.subplot(2, 5, 5 + i + 1)
                plt.imshow(np.transpose(recons[i], (1, 2, 0)).astype(np.float32))
                plt.axis('off')
            plt.show()
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"\nEarly stopping triggered at epoch {epoch + 1}!")
                break

        print(f"Epoch: {epoch + 1}/{num_epochs} | Loss: {avg_loss:.6f} | Best Loss: {best_loss:.6f}")

    # Load lại model tốt nhất sau training
    model.load_state_dict(torch.load(best_model_path))

    # Visualization code ở đây (giữ nguyên như của bạn)


    writer.close()  # Đóng TensorBoard
    sys.exit()

def cvae():
    transform = transforms.Compose([transforms.ToTensor(),
                                    transforms.Resize((128, 128))])
    origin_train_data = CustomDataset(root="./data/Intel_image_classification/seg_train/seg_train/",
                                      transform=transform)
    batch_size = 120
    data = ColorizationDataset(data=origin_train_data)
    dataloader = torch.utils.data.DataLoader(dataset=data,
                                             batch_size=batch_size,
                                             shuffle=True,
                                             pin_memory=True,
                                             num_workers=2,
                                             persistent_workers=True)

    dataiter = iter(dataloader)
    color_img, gray_img = next(dataiter)
    print(torch.min(color_img), torch.max(color_img))
    # plt.figure(figsize=(10, 4))
    # plt.subplot(1, 3, 1)
    # plt.imshow(color_img[0].permute(1, 2, 0))
    # plt.subplot(1, 3, 2)
    # plt.imshow(gray_img[0].permute(1, 2, 0), cmap='gray')
    # plt.axis('off')
    #
    # input_data = torch.cat((color_img, gray_img), dim=1)
    # print(input_data.shape)
    # plt.subplot(1, 3, 3)
    # plt.imshow(input_data[0].permute(1, 2, 0))
    # plt.show()

    model = CVAE(latent_dim=200).cuda()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=0.0001)
    outputs = []
    num_epochs = 1
    try:
        for epoch in range(num_epochs):
            epoch_bar = tqdm(dataloader, desc=f"Epoch [{epoch + 1}/{num_epochs}]")
            for x, c in epoch_bar:
                x = x.cuda()
                c = c.cuda()
                x_rec, mu_q, log_var_q, mu_p, log_var_p = model(x, c)
                var_q = torch.exp(log_var_q)
                var_p = torch.exp(log_var_p)
                rec_loss = nn.MSELoss(reduction="sum")(x_rec, x) / batch_size
                kl_div_loss = (0.5 * torch.sum(log_var_p - log_var_q + (var_q + (mu_q - mu_p).pow(2)) / var_p - 1))
                loss = rec_loss + kl_div_loss
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                epoch_bar.set_postfix(loss=loss.item())
            outputs.append((epoch, c, x_rec))
            print(f"Epoch: {epoch + 1}/{num_epochs}. Loss: {loss:.6f}")

    except:
        print("Got Error!")

    # Visualization
    for k in range(0, len(outputs), 4):
        plt.figure(figsize=(10, 4))
        plt.gray()
        imgs = outputs[k][1].cpu().detach().numpy()
        recons = outputs[k][2].cpu().detach().numpy()

        for i in range(5):
            # Original images
            plt.subplot(2, 5, i + 1)
            plt.imshow(imgs[i].squeeze(0), cmap='gray')
            plt.axis('off')

            # Reconstructed images
            plt.subplot(2, 5, 5 + i + 1)
            plt.imshow(recons[i].transpose(1,2,0))
            plt.axis('off')

        plt.suptitle(f"Epoch {outputs[k][0] + 1}")
        plt.tight_layout()
        plt.show()


if __name__ == '__main__':
    # vae_ln()
    vae_cnn()
    # ae_cnn()
    # cvae()