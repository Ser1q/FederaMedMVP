import os
import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import numpy as np
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models
from PIL import Image
import pydicom
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score


# Device (MPS for Mac M4)

device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
print("Using device:", device)


# Dataset Class
class RSNADataset(Dataset):
    def __init__(self, df, image_dir, transform=None):
        self.df = df
        self.image_dir = image_dir
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        patient_id = self.df.iloc[idx]["patientId"]
        label = self.df.iloc[idx]["Target"]

        image_path = os.path.join(self.image_dir, patient_id + ".dcm")

        dicom = pydicom.dcmread(image_path)
        image = dicom.pixel_array

        image = Image.fromarray(image).convert("RGB")

        if self.transform:
            image = self.transform(image)

        return image, torch.tensor(label, dtype=torch.long)



# Paths
IMAGE_DIR = "./rsna-pneumonia-dataset/stage_2_train_images"
CSV_PATH = "./rsna-pneumonia-dataset/stage_2_train_labels.csv"


# Load and preprocess labels
df = pd.read_csv(CSV_PATH)

# Group by patientId (если несколько bbox)
df = df.groupby("patientId")["Target"].max().reset_index()

# check
available_files = set(
    [f.replace(".dcm", "") for f in os.listdir(IMAGE_DIR)]
)

df = df[df["patientId"].isin(available_files)].reset_index(drop=True)

print("After filtering missing files:", len(df))

# Train / Test
train_df, test_df = train_test_split(
    df,
    test_size=0.2,
    stratify=df["Target"],
    random_state=42
)

print("Train size:", len(train_df))
print("Test size:", len(test_df))


# Transforms
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])


# Dataset & Loader
train_dataset = RSNADataset(train_df, IMAGE_DIR, transform)
test_dataset = RSNADataset(test_df, IMAGE_DIR, transform)

train_loader = DataLoader(
    train_dataset,
    batch_size=16,
    shuffle=True,
    num_workers=0
)

test_loader = DataLoader(
    test_dataset,
    batch_size=16,
    shuffle=False,
    num_workers=0
)


# Model (ResNet18)
model = models.resnet18(weights="IMAGENET1K_V1")
model.fc = nn.Linear(model.fc.in_features, 2)
model = model.to(device)


# Optimizer & Loss
optimizer = optim.Adam(model.parameters(), lr=1e-4)
criterion = nn.CrossEntropyLoss()


# Training Loop
def train_one_epoch(model, loader):
    model.train()
    total_loss = 0

    for x, y in loader:
        x = x.to(device)
        y = y.to(device)

        optimizer.zero_grad()
        outputs = model(x)
        loss = criterion(outputs, y)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    return total_loss / len(loader)



# Evaluation
def evaluate(model, loader):
    model.eval()
    correct = 0
    total = 0

    all_probs = []
    all_labels = []

    with torch.no_grad():
        for x, y in loader:
            x = x.to(device)
            y = y.to(device)

            outputs = model(x)

            probs = torch.softmax(outputs, dim=1)[:, 1]  # probability of class 1
            preds = torch.argmax(outputs, dim=1)

            correct += (preds == y).sum().item()
            total += y.size(0)

            all_probs.extend(probs.cpu().numpy())
            all_labels.extend(y.cpu().numpy())

    accuracy = correct / total
    roc_auc = roc_auc_score(all_labels, all_probs)

    return accuracy, roc_auc



# Main Training
EPOCHS = 5

for epoch in range(EPOCHS):
    loss = train_one_epoch(model, train_loader)
    acc, auc = evaluate(model, test_loader)

    print(f"Epoch {epoch+1}/{EPOCHS} | "
        f"Loss: {loss:.4f} | "
        f"Acc: {acc:.4f} | "
        f"ROC-AUC: {auc:.4f}")


# Save
torch.save(model.state_dict(), "rsna_resnet18_centralized.pth")
print("Model saved.")