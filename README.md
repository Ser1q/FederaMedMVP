# FederaMed MVP: Dataset and Training Summary

This repository currently contains two main experiment tracks:

- `V0_rsna`: centralized baseline training on RSNA pneumonia chest X-rays
- `V0_MNIST`: prototype federated learning (FedAvg) on ChestMNIST-like data (`medmnist`)

## Datasets

### 1) RSNA Pneumonia Detection Dataset (`V0_rsna/rsna-pneumonia-dataset`)

- Modality: chest X-ray DICOM images (`.dcm`)
- Training images: `stage_2_train_images`
- Labels: `stage_2_train_labels.csv`
- Metadata/sample files: `stage_2_detailed_class_info.csv`, `stage_2_sample_submission.csv`

How labels are prepared in code:

- Rows are grouped by `patientId`
- Final patient label is `Target=max`, which merges multiple bounding box annotations into one binary label per patient
- Patients with missing image files are removed before training

### 2) ChestMNIST (via `medmnist`) in `V0_MNIST/mnist_test.py`

- Loaded with `data_flag = 'chestmnist'`
- Train/test splits are downloaded through `medmnist`
- Multi-label targets are converted to binary with:
	- `y = (y.sum(dim=1) > 0).long()`
	- This means at least one positive condition becomes class `1`, otherwise class `0`

## Model

- Backbone: `ResNet18` (`torchvision.models.resnet18`) initialized with ImageNet weights (`IMAGENET1K_V1`)
- Classifier head: final fully connected layer replaced with a 2-class output (`pneumonia` vs `non-pneumonia`)
- Device: Apple MPS when available, otherwise CPU

## Data Pipeline

- Images: DICOM files from `V0_rsna/rsna-pneumonia-dataset/stage_2_train_images`
- Labels: `V0_rsna/rsna-pneumonia-dataset/stage_2_train_labels.csv`
- Label preprocessing: grouped by `patientId` and `Target=max` to merge multiple annotations per patient
- Missing-file filtering: only patients with corresponding `.dcm` files are kept
- Train/test split: 80/20 with stratification (`random_state=42`)

## Input Preprocessing

- DICOM pixel array loaded with `pydicom`
- Converted to RGB PIL image
- Transforms:
	- Resize to `224x224`
	- Normalize with ImageNet mean/std

## Training Setup

- Loss: `CrossEntropyLoss`
- Optimizer: `Adam` with learning rate `1e-4`
- Batch size: `16`
- Epochs: `5`
- Evaluation metrics: Accuracy and ROC-AUC (class-1 probability)

## Reported Results

### Centralized RSNA (`V0_rsna/train_rsna_centralized.py`)

From the script output:

- Epoch 1: Loss `0.4001`, Acc `0.8337`, ROC-AUC `0.8655`
- Epoch 2: Loss `0.3567`, Acc `0.8223`, ROC-AUC `0.8673`
- Epoch 3: Loss `0.3208`, Acc `0.8319`, ROC-AUC `0.8648`
- Epoch 4: Loss `0.2688`, Acc `0.8181`, ROC-AUC `0.8458`
- Epoch 5: Loss `0.1794`, Acc `0.8245`, ROC-AUC `0.8463`

### Decentralized RSNA FedAvg (`V0_rsna/train_rsna_decentralized.py`)

From the script output:

- Client 1: samples `6671`, positives `1537`
- Client 2: samples `6670`, positives `1536`
- Client 3: samples `6670`, positives `1536`

- Round 1: Local Loss `0.4282`, Acc `0.8227`, ROC-AUC `0.8563`
- Round 2: Local Loss `0.3912`, Acc `0.8175`, ROC-AUC `0.8641`
- Round 3: Local Loss `0.3730`, Acc `0.8357`, ROC-AUC `0.8683`
- Round 4: Local Loss `0.3485`, Acc `0.8415`, ROC-AUC `0.8758`
- Round 5: Local Loss `0.3173`, Acc `0.8409`, ROC-AUC `0.8737`

## Centralized vs Decentralized Comparison (RSNA)

| Metric | Centralized (Epoch 5) | Decentralized (Round 5) | Delta (Decentralized - Centralized) |
|---|---:|---:|---:|
| Accuracy | 0.8245 | 0.8409 | +0.0164 |
| ROC-AUC | 0.8463 | 0.8737 | +0.0274 |

Best observed values during training:

| Metric | Best Centralized | Best Decentralized | Delta |
|---|---:|---:|---:|
| Accuracy | 0.8337 | 0.8415 | +0.0078 |
| ROC-AUC | 0.8673 | 0.8758 | +0.0085 |

Interpretation:

- In this setup, decentralized FedAvg achieved higher final and best test performance than the centralized baseline.
- The gain is more pronounced on ROC-AUC, indicating better ranking/separation of positive vs negative cases.

## Saved Weights

- Output checkpoint: `V0_rsna/rsna_resnet18_centralized.pth`

This model serves as the centralized baseline before applying federated approaches to medical imaging data.

## Federated Learning Architecture Applied

The federated learning pipeline is implemented in `V0_MNIST/mnist_test.py` as a proof-of-concept using FedAvg.

### Components

- Global model: `SimpleCNN`
- Clients: 3 simulated hospitals (`hospital_a`, `hospital_b`, `hospital_c`)
- Client data partitioning: training data is split into 3 non-overlapping subsets via `random_split`
- Local training: each client trains a local copy of the global model with Adam + CrossEntropy
- Aggregation: server performs parameter-wise averaging (`average_weights`) across client model weights
- Communication rounds: 5 federated rounds
- Evaluation: global model accuracy on the shared test set after each round

### FedAvg Flow in This Repo

1. Initialize a global model on the server.
2. Broadcast global weights to each client (copy of the model).
3. Train each client locally for 1 epoch.
4. Send local model weights back to the server.
5. Average weights to form updated global weights.
6. Evaluate global model and repeat for the next round.

## Current Status

- RSNA track: centralized baseline and decentralized FedAvg example are both implemented
- FL track: implemented on both ChestMNIST prototype and RSNA partitioned training
- Next step: run repeated seeds and non-IID client splits to measure robustness of the observed gains

Update: an RSNA decentralized/federated example is now available in `V0_rsna/train_rsna_decentralized.py`.

## Run RSNA Decentralized Example

From the project root:

```bash
python3 V0_rsna/train_rsna_decentralized.py
```

Optional parameters:

- `--num_clients` (default: `3`)
- `--rounds` (default: `5`)
- `--local_epochs` (default: `1`)
- `--batch_size` (default: `16`)
- `--lr` (default: `1e-4`)

This script uses weighted FedAvg aggregation across clients and saves the final model to `V0_rsna/rsna_resnet18_decentralized.pth`.
