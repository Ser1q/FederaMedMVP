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
    

# simulating 3 clinics
from torch.utils.data import random_split

total_size = len(train_dataset)
part_size = total_size // 3

hospital_a, hospital_b, hospital_c = random_split(
    train_dataset,
    [part_size, part_size, total_size - 2 * part_size]
)


# splitting dataset for individual hospitals
loader_a = DataLoader(hospital_a, batch_size=64, shuffle=True)
loader_b = DataLoader(hospital_b, batch_size=64, shuffle=True)
loader_c = DataLoader(hospital_c, batch_size=64, shuffle=True)


# evaluating function
def evaluate(model, loader):
    model.eval()
    correct = 0
    total = 0

    with torch.no_grad():
        for x, y in loader:
            y = (y.sum(dim=1) > 0).long()
            logits = model(x.float())
            preds = torch.argmax(logits, dim=1)

            correct += (preds == y).sum().item()
            total += y.size(0)

    return correct / total


# copying global model into hospitals
import copy

def train_local(model, loader, epochs=1):
    model.train()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    loss_fn = nn.CrossEntropyLoss()

    for _ in range(epochs):
        for x, y in loader:
            y = (y.sum(dim=1) > 0).long()
            optimizer.zero_grad()
            logits = model(x.float())
            loss = loss_fn(logits, y)
            loss.backward()
            optimizer.step()

    return model.state_dict()


# FedAvg
def average_weights(weights_list):
    avg_weights = copy.deepcopy(weights_list[0])

    for key in avg_weights.keys():
        for i in range(1, len(weights_list)):
            avg_weights[key] += weights_list[i][key]
        avg_weights[key] = avg_weights[key] / len(weights_list)

    return avg_weights


# federated loop
global_model = SimpleCNN()

for round in range(5):

    weights_a = train_local(copy.deepcopy(global_model), loader_a)
    weights_b = train_local(copy.deepcopy(global_model), loader_b)
    weights_c = train_local(copy.deepcopy(global_model), loader_c)

    new_weights = average_weights([weights_a, weights_b, weights_c])

    global_model.load_state_dict(new_weights)

    acc = evaluate(global_model, test_loader)

    print(f"Round {round+1}, Accuracy: {acc:.4f}")