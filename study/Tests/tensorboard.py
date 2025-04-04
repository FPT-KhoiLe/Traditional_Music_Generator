import torch
from torch import nn
import torchvision
from torchvision import transforms, datasets
import matplotlib.pyplot as plt
import sys
from torch.utils.tensorboard import SummaryWriter
writer = SummaryWriter("runs/mnist")

input_size = 28*28
hidden_size = 500
num_classes = 10
num_epochs = 1
batch_size = 64
learning_rate = 0.001

# Train dataset
train_dataset = datasets.MNIST(
    root="../data",
    train=True,
    transform=transforms.ToTensor(),
    download=True
)

test_dataset = datasets.MNIST(
    root="../data",
    train=False,
    transform=transforms.ToTensor(),
    download=True
)

train_loader = torch.utils.data.DataLoader(
    dataset=train_dataset,
    batch_size=batch_size,
    shuffle=True,
    num_workers=2,
    pin_memory=True
)

test_loader = torch.utils.data.DataLoader(
    dataset=test_dataset,
    batch_size=batch_size,
    shuffle=False,
    num_workers=2,
    pin_memory=True
)

examples = iter(train_loader)
images, labels = next(examples)

for i in range(6):
    plt.subplot(2, 6, i + 1)
    plt.imshow(images[i].numpy().squeeze(), cmap="gray")
# plt.show()
img_grid = torchvision.utils.make_grid(images)
writer.add_image('mnist_images', img_grid)
writer.close()
sys.exit()
# Define Model : Fully connected neural network with one hidden layer
class Net(nn.Module):
    def __init__(self, input_size, hidden_size, num_classes):
        super(Net, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, num_classes)
        )

    def forward(self, x):
        return self.net(x)

model = Net(input_size, hidden_size, num_classes).cuda()
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

n_total_steps = len(train_loader)
for epoch in range(num_epochs):
    for images, labels in train_loader:
        images, labels = images.cuda(), labels.cuda()
        outputs = model(images)
        loss = criterion(outputs, labels)
        optimizer.zero_grad()
        loss.backward()

        print(f"Epoch {epoch+1}/{num_epochs}, Loss: {loss.item():.4f}")