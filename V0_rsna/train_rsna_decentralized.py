import copy
import os
import random

import numpy as np
import pandas as pd
import pydicom
import torch
import torch.nn as nn
import torch.optim as optim
from PIL import Image
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset, Subset
from torchvision import models, transforms


CONFIG = {
    "num_clients": 3,
    "rounds": 5,
    "local_epochs": 1,
    "batch_size": 16,
    "lr": 1e-4,
    "test_size": 0.2,
    "seed": 42,
    "image_dir": os.path.join(
        os.path.dirname(__file__), "rsna-pneumonia-dataset", "stage_2_train_images"
    ),
    "csv_path": os.path.join(
        os.path.dirname(__file__), "rsna-pneumonia-dataset", "stage_2_train_labels.csv"
    ),
    "output": os.path.join(os.path.dirname(__file__), "rsna_resnet18_decentralized.pth"),
}


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


class RSNADataset(Dataset):
    def __init__(self, df: pd.DataFrame, image_dir: str, transform=None):
        self.df = df.reset_index(drop=True)
        self.image_dir = image_dir
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        patient_id = self.df.iloc[idx]["patientId"]
        label = int(self.df.iloc[idx]["Target"])

        image_path = os.path.join(self.image_dir, patient_id + ".dcm")
        dicom = pydicom.dcmread(image_path)
        image = Image.fromarray(dicom.pixel_array).convert("RGB")

        if self.transform:
            image = self.transform(image)

        return image, torch.tensor(label, dtype=torch.long)


def build_model() -> nn.Module:
    model = models.resnet18(weights="IMAGENET1K_V1")
    model.fc = nn.Linear(model.fc.in_features, 2)
    return model


def split_indices_stratified(labels: np.ndarray, num_clients: int, seed: int):
    rng = np.random.default_rng(seed)
    labels = np.asarray(labels)

    client_indices = [[] for _ in range(num_clients)]
    for cls in np.unique(labels):
        cls_idx = np.where(labels == cls)[0]
        rng.shuffle(cls_idx)
        chunks = np.array_split(cls_idx, num_clients)
        for i, chunk in enumerate(chunks):
            client_indices[i].extend(chunk.tolist())

    for idx in client_indices:
        rng.shuffle(idx)

    return client_indices


def train_local(model, loader, device, lr: float, local_epochs: int):
    model = model.to(device)
    model.train()

    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()

    total_loss = 0.0
    steps = 0

    for _ in range(local_epochs):
        for x, y in loader:
            x = x.to(device)
            y = y.to(device)

            optimizer.zero_grad()
            logits = model(x)
            loss = criterion(logits, y)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            steps += 1

    avg_loss = total_loss / max(steps, 1)
    return model.state_dict(), avg_loss


def aggregate_weighted(state_dicts, sample_counts):
    total = float(sum(sample_counts))
    aggregated = copy.deepcopy(state_dicts[0])

    for key in aggregated.keys():
        aggregated[key] = state_dicts[0][key] * (sample_counts[0] / total)
        for i in range(1, len(state_dicts)):
            aggregated[key] += state_dicts[i][key] * (sample_counts[i] / total)

    return aggregated


def evaluate(model, loader, device):
    model = model.to(device)
    model.eval()

    correct = 0
    total = 0
    all_probs = []
    all_labels = []

    with torch.no_grad():
        for x, y in loader:
            x = x.to(device)
            y = y.to(device)

            logits = model(x)
            probs = torch.softmax(logits, dim=1)[:, 1]
            preds = torch.argmax(logits, dim=1)

            correct += (preds == y).sum().item()
            total += y.size(0)
            all_probs.extend(probs.cpu().numpy())
            all_labels.extend(y.cpu().numpy())

    accuracy = correct / max(total, 1)
    try:
        auc = roc_auc_score(all_labels, all_probs)
    except ValueError:
        auc = float("nan")

    return accuracy, auc


def run_training(config=None):
    cfg = dict(CONFIG)
    if config:
        cfg.update(config)

    seed_everything(cfg["seed"])

    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    print(f"Using device: {device}")

    df = pd.read_csv(cfg["csv_path"])
    df = df.groupby("patientId")["Target"].max().reset_index()

    available_files = {f.replace(".dcm", "") for f in os.listdir(cfg["image_dir"]) if f.endswith(".dcm")}
    df = df[df["patientId"].isin(available_files)].reset_index(drop=True)
    print(f"After filtering missing files: {len(df)}")

    train_df, test_df = train_test_split(
        df,
        test_size=cfg["test_size"],
        stratify=df["Target"],
        random_state=cfg["seed"],
    )
    print(f"Train size: {len(train_df)}")
    print(f"Test size: {len(test_df)}")

    transform = transforms.Compose(
        [
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )

    train_dataset = RSNADataset(train_df, cfg["image_dir"], transform)
    test_dataset = RSNADataset(test_df, cfg["image_dir"], transform)
    test_loader = DataLoader(test_dataset, batch_size=cfg["batch_size"], shuffle=False, num_workers=0)

    client_idx = split_indices_stratified(train_df["Target"].values, cfg["num_clients"], cfg["seed"])
    client_loaders = [
        DataLoader(Subset(train_dataset, idx), batch_size=cfg["batch_size"], shuffle=True, num_workers=0)
        for idx in client_idx
    ]

    for i, idx in enumerate(client_idx):
        positives = int(train_df.iloc[idx]["Target"].sum())
        print(f"Client {i+1}: samples={len(idx)}, positives={positives}")

    global_model = build_model().to(device)

    for rnd in range(cfg["rounds"]):
        client_states = []
        client_losses = []
        sample_counts = []

        for loader in client_loaders:
            local_model = build_model().to(device)
            local_model.load_state_dict(global_model.state_dict())

            state, local_loss = train_local(
                local_model,
                loader,
                device,
                lr=cfg["lr"],
                local_epochs=cfg["local_epochs"],
            )

            client_states.append(state)
            client_losses.append(local_loss)
            sample_counts.append(len(loader.dataset))

        new_state = aggregate_weighted(client_states, sample_counts)
        global_model.load_state_dict(new_state)

        acc, auc = evaluate(global_model, test_loader, device)
        mean_local_loss = float(np.mean(client_losses))
        print(
            f"Round {rnd + 1}/{cfg['rounds']} | "
            f"Local Loss: {mean_local_loss:.4f} | "
            f"Acc: {acc:.4f} | "
            f"ROC-AUC: {auc:.4f}"
        )

    torch.save(global_model.state_dict(), cfg["output"])
    print(f"Model saved to: {cfg['output']}")


def main():
    run_training()


if __name__ == "__main__":
    main()
    
# Results:
# Train size: 20011
# Test size: 5003
# Client 1: samples=6671, positives=1537
# Client 2: samples=6670, positives=1536
# Client 3: samples=6670, positives=1536

# Round 1/5 | Local Loss: 0.4282 | Acc: 0.8227 | ROC-AUC: 0.8563
# Round 2/5 | Local Loss: 0.3912 | Acc: 0.8175 | ROC-AUC: 0.8641
# Round 3/5 | Local Loss: 0.3730 | Acc: 0.8357 | ROC-AUC: 0.8683
# Round 4/5 | Local Loss: 0.3485 | Acc: 0.8415 | ROC-AUC: 0.8758
# Round 5/5 | Local Loss: 0.3173 | Acc: 0.8409 | ROC-AUC: 0.8737