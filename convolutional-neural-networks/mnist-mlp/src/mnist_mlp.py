import numpy as np
import torch
from torchvision import datasets
import torchvision.transforms as transforms
from torch.utils.data.sampler import SubsetRandomSampler
import torch.nn as nn
import torch.nn.functional as F

# location of the MNIST images
ROOT = '~/.pytorch'
# number of subprocesses to use for data loading
NUM_WORKERS = 0
# how many samples per batch to load
BATCH_SIZE = 32
# percentage of training set to use as validation
VALID_SIZE = 0.2
# Size of the hidden layers
HIDDEN_SIZE = 512
# Size of the output layer
OUTPUT_SIZE = 10

# Set device
CUDA_DEVICE = 'cuda:0'
if torch.cuda.is_available():
    device = torch.device(CUDA_DEVICE)
else:
    torch.device('cpu')

# convert data to torch.FloatTensor
transform = transforms.ToTensor()

# choose the training and test datasets
train_data = datasets.MNIST(root=ROOT, train=True,
                            download=True, transform=transform)
test_data = datasets.MNIST(root=ROOT, train=False,
                           download=True, transform=transform)

# obtain training indices that will be used for validation
num_train = len(train_data)
indices = list(range(num_train))
np.random.shuffle(indices)
split = int(np.floor(VALID_SIZE * num_train))
train_idx, valid_idx = indices[split:], indices[:split]

# define samplers for obtaining training and validation batches
train_sampler = SubsetRandomSampler(train_idx)
valid_sampler = SubsetRandomSampler(valid_idx)

# prepare data loaders
train_loader = torch.utils.data.DataLoader(train_data, batch_size=BATCH_SIZE,
                                           sampler=train_sampler,
                                           num_workers=NUM_WORKERS)
valid_loader = torch.utils.data.DataLoader(train_data, batch_size=BATCH_SIZE,
                                           sampler=valid_sampler,
                                           num_workers=NUM_WORKERS)
test_loader = torch.utils.data.DataLoader(test_data, batch_size=BATCH_SIZE,
                                          num_workers=NUM_WORKERS)

images, labels = next(iter(train_loader))

# Define network architecture


class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(784, HIDDEN_SIZE)
        self.fc2 = nn.Linear(HIDDEN_SIZE, HIDDEN_SIZE)
        self.fc3 = nn.Linear(HIDDEN_SIZE, OUTPUT_SIZE)

    def forward(self, x):
        x = x.view(-1, 784)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


model = Net()
model = model.to(device)
criterion = nn.CrossEntropyLoss(reduction='sum')
optimizer = torch.optim.RMSprop(model.parameters())


def train_valid_loop(dataloader, model, criterion, optimizer, every=10, what='train'):
    running_loss = 0
    running_acc = 0
    batch_num = 0

    for images, labels in dataloader:
        batch_num += 1
        images, labels = images.to(device), labels.to(device)
        logits = model(images)
        loss = criterion(logits, labels)
        if what == 'train':
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        running_loss += loss
        _, preds = torch.max(logits, dim=1)
        running_acc += (preds == labels).type(torch.IntTensor).sum()

        if batch_num % every == 0:
            print(('Batch N.: {:3}, ' + what.title() + 'Loss: {:.3f}, ' +
                   what.title() + 'Acc.: {:.3f}').format(
                       batch_num,
                       running_loss.item()/(batch_num*dataloader.batch_size),
                       running_acc.item()/(batch_num*dataloader.batch_size)
            ))


def fit(epochs, model, criterion, optimizer, every):
    for epoch in range(epochs):
        print(f'--- Epoch: {epoch} ---')
        # Train loop
        model.train()
        print('\n--- Train loop ----\n')
        train_valid_loop(train_loader, model, criterion,
                         optimizer, every, 'train')

        # Validation loop
        model.eval()
        print('\n--- Validation loop ---\n')
        with torch.no_grad():
            train_valid_loop(valid_loader, model, criterion,
                             optimizer, every, what='valid')


fit(5, model, criterion, optimizer, every=100)
