import torch
from torch import nn
import torch.nn.functional as F
from torch import optim
from torchvision import datasets, transforms


class MyNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        self.hidden1 = nn.Linear(INPUT_SHAPE, HIDDEN_SIZE1)
        self.hidden2 = nn.Linear(HIDDEN_SIZE1, HIDDEN_SIZE2)
        self.hidden3 = nn.Linear(HIDDEN_SIZE2, 10)

    def forward(self, x):
        x = x.view(-1, INPUT_SHAPE)
        x = F.relu(self.hidden1(x))
        x = F.relu(self.hidden2(x))
        x = F.log_softmax(self.hidden3(x), dim=1)
        return x


def train(trainloader, testloader, epochs, device):
    for _ in range(epochs):

        train_loss = 0
        valid_loss = 0

        model.train()
        for xb, yb in trainloader:
            xb, yb = xb.to(device), yb.to(device)
            logits = model(xb)
            loss = criterion(logits, yb)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            train_loss += loss

        model.eval()
        with torch.no_grad():
            for xb, yb in testloader:
                xb, yb = xb.to(device), yb.to(device)
                logits = model(xb)
                loss = criterion(logits, yb)
                valid_loss += loss

        avg_train_loss = train_loss / len(trainloader)
        avg_valid_loss = valid_loss / len(testloader)
        print(f'Train loss: {avg_train_loss}; Valid. loss: {avg_valid_loss}')


if __name__ == '__main__':
    INPUT_SHAPE = 28*28
    HIDDEN_SIZE1 = 128
    HIDDEN_SIZE2 = 64
    BATCH_SIZE = 32
    LEARNING_RATE = 0.05

    device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")

    # Create a set of tranformations
    transform = transforms.Compose([transforms.ToTensor(),
                                    transforms.Normalize((0.5,), (0.5,))])
    # Create training and test datasets
    trainset = datasets.FashionMNIST('~/.pytorch/F_MNIST_data/', download=True,
                                     train=True, transform=transform)
    testset = datasets.FashionMNIST('~/.pytorch/F_MNIST_data/', download=True,
                                    train=False, transform=transform)
    # Create dataloaders
    trainloader = torch.utils.data.DataLoader(trainset,
                                              batch_size=BATCH_SIZE,
                                              shuffle=True)
    testloader = torch.utils.data.DataLoader(testset,
                                             batch_size=BATCH_SIZE,
                                             shuffle=False)
    # Instantiate model
    model = MyNetwork()
    model.to(device)
    # Define loss function
    criterion = nn.NLLLoss()
    # Choose optimizer
    optimizer = optim.SGD(model.parameters(), lr=LEARNING_RATE)
    # Train the model
    train(trainloader, testloader, 3, device)
