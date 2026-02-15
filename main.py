# imports
import torch
import medmnist
from medmnist import INFO
from torch.utils.data import DataLoader
from model import SimpleCNN
from torch import nn
import torch.optim as optim
from torchvision import transforms


# dataset import 
data_flag = 'chestmnist'
info = INFO[data_flag]
DataClass = getattr(medmnist, info['python_class'])

transform = transforms.Compose([
    transforms.ToTensor()
])

train_dataset = DataClass(
    split='train',
    transform=transform,
    download=True
)

test_dataset = DataClass(
    split='test',
    transform=transform,
    download=True
)


# data loaders 
train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)

print("Train size:", len(train_dataset))


# baseline model
model = SimpleCNN()
optimizer = optim.Adam(model.parameters(), lr=0.001)
loss_fn = nn.CrossEntropyLoss()

for epoch in range(3):
    model.train()
    total_loss = 0

    for x, y in train_loader:
        y = (y.sum(dim=1) > 0).long()

        optimizer.zero_grad()
        logits = model(x.float())
        loss = loss_fn(logits, y)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    print(f"Epoch {epoch+1}, Loss: {total_loss:.4f}")