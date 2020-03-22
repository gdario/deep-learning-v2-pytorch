from collections import OrderedDict
import torch
from torch import nn
from torch import optim
from torchvision import datasets, transforms, models

if torch.cuda.is_available():
    device = torch.device('cuda:1')
else:
    device = torch.device('cpu')

DATA_PATH = 'Cat_Dog_data/'
BATCH_SIZE = 64
EPOCHS = 2
EVERY = 1

train_transforms = transforms.Compose([
    transforms.RandomRotation(30),
    transforms.RandomResizedCrop(224),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

test_transforms = transforms.Compose([
    transforms.Resize(255),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

train_data = datasets.ImageFolder(DATA_PATH + 'train/', train_transforms)
test_data = datasets.ImageFolder(DATA_PATH + 'test/', test_transforms)

trainloader = torch.utils.data.DataLoader(train_data, BATCH_SIZE, True)
testloader = torch.utils.data.DataLoader(test_data, BATCH_SIZE)

model = models.densenet121(pretrained=True)

for param in model.parameters():
    param.requires_grad = False

classifier = nn.Sequential(
    OrderedDict([
        ('fc1', nn.Linear(1024, 512)),
        ('relu', nn.ReLU()),
        ('fc2', nn.Linear(512, 2)),
        ('output', nn.LogSoftmax(dim=1))
    ])
)

model.classifier = classifier

# Train the model
model = model.to(device)
criterion = nn.NLLLoss()
optimizer = optim.Adam(model.classifier.parameters())

for epoch in range(EPOCHS):

    valid_loss = 0
    valid_acc = 0

    valid_loss_history = []
    valid_acc_history = []

    # Training loop
    model.train()
    for inputs, labels in trainloader:
        inputs, labels = inputs.to(device), labels.to(device)
        optimizer.zero_grad()
        log_ps = model(inputs)
        loss = criterion(log_ps, labels)
        loss.backward()
        optimizer.step()

    # Validation loop
    with torch.no_grad():
        model.eval()
        valid_batch = 0
        for inputs, labels in testloader:
            valid_batch += 1
            inputs, labels = inputs.to(device), labels.to(device)
            log_ps = model(inputs)
            loss = criterion(log_ps, labels)
            valid_loss += loss
            _, preds = torch.topk(log_ps, 1)
            valid_acc += (preds.flatten() == labels.flatten()).type(
                torch.FloatTensor).mean()

            if valid_batch % EVERY == 0:
                vll = valid_loss.item()/valid_batch
                vac = valid_acc.item()/valid_batch
                print(
                    ('Epoch: {:2}, Batch: {:3}, valid loss: {:.3f}, '
                     'valid acc: {:.3f}').format(epoch, valid_batch, vll, vac))
                valid_loss_history.append(vll)
                valid_acc_history.append(vac)
