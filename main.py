import logging

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.utils
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader

DATA_PATH = "./data"
CLASSES = (
    "plane",
    "car",
    "bird",
    "cat",
    "deer",
    "dog",
    "frog",
    "horse",
    "ship",
    "truck",
)
MODEL_PATH = "./cifar_net.pth"
EPOCH = 10
LEARNING_RATE = 0.001

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
logger.info(f"Running on {device.type}")

transform = transforms.Compose(
    [transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]
)

trainset = torchvision.datasets.CIFAR10(
    root=DATA_PATH, train=True, download=True, transform=transform
)
trainloader = DataLoader(trainset, batch_size=4, shuffle=True, num_workers=2)

testset = torchvision.datasets.CIFAR10(
    root=DATA_PATH, train=False, download=True, transform=transform
)
testloader = DataLoader(testset, batch_size=4, shuffle=False, num_workers=2)


def imshow(img):
    img = img / 2 + 0.5
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.show()


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, 3, padding=1)  # input: 3x32x32, output: 32x32x32
        self.pool = nn.MaxPool2d(2, 2)  # output: 32x16x16
        self.conv2 = nn.Conv2d(32, 64, 3, padding=1)  # output: 64x16x16
        self.conv3 = nn.Conv2d(64, 64, 3, padding=1)  # output: 64x8x8
        self.fc1 = nn.Linear(64 * 4 * 4, 64)  # input: 64x4x4 -> 64*4*4, output: 64
        self.fc2 = nn.Linear(64, 10)  # output: 10 (number of classes)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = self.pool(F.relu(self.conv3(x)))
        x = x.view(-1, 64 * 4 * 4)  # Flatten the image
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x


def train(device, trainloader, net, criterion, optimizer):
    for epoch in range(EPOCH):  # loop over the dataset multiple times
        running_loss = 0.0
        for batch, (inputs, labels) in enumerate(trainloader):
            inputs, labels = inputs.to(device), labels.to(device)
            # Zero the parameter gradients
            optimizer.zero_grad()
            # Forward pass
            outputs = net(inputs)
            loss = criterion(outputs, labels)

            # Backward pass and optimize
            loss.backward()
            optimizer.step()

            # Print statistics
            running_loss += loss.item()
            if batch % 2000 == 1999:  # Print every 2000 mini-batches
                logger.info(
                    f"[Epoch {epoch + 1}, Mini-batch {batch + 1}] loss: {running_loss / 2000:.3f}"
                )
                running_loss = 0.0
    logger.info("Finished Training")
    torch.save(net.state_dict(), MODEL_PATH)


def testing(device, testloader, net):
    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in testloader:
            images, labels = images.to(device), labels.to(device)
            outputs = net(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    logger.info(
        f"Accuracy of the network on the 10000 test images: {100 * correct / total:.2f}%"
    )

    # Display the accuracy per class
    class_correct = list(0.0 for i in range(10))
    class_total = list(0.0 for i in range(10))
    with torch.no_grad():
        for data in testloader:
            images, labels = data
            outputs = net(images.to(device))
            _, predicted = torch.max(outputs, 1)
            c = (predicted == labels.to(device)).squeeze()
            for i in range(4):
                label = labels[i]
                class_correct[label] += c[i].item()
                class_total[label] += 1

    for i in range(10):
        logger.info(
            f"Accuracy of {CLASSES[i]}: {100 * class_correct[i] / class_total[i]:.2f}%"
        )


if __name__ == "__main__":
    dataiter = iter(trainloader)
    images, labels = next(dataiter)

    # logger.info(" ".join(f"{CLASSES[labels[j]]}" for j in range(4)))
    # imshow(torchvision.utils.make_grid(images))

    # Instantiate the model and move to device
    net = Net().to(device)

    # Define Loss function and Optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(net.parameters(), lr=LEARNING_RATE)

    # Training the Model
    train(device, trainloader, net, criterion, optimizer)

    # Testing the Model
    testing(device, testloader, net)
