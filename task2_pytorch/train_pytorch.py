
import os
import copy
import random
import numpy as np
import pandas as pd

import torch
from torch import nn

from torch.utils import data

Dataset = data.Dataset
DataLoader = data.DataLoader
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score


# -----------------------------------------
# 1. Reproducibility and folders
# -----------------------------------------

SEED = 42

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

os.makedirs("datasets", exist_ok=True)
os.makedirs("task2_pytorch", exist_ok=True)
os.makedirs("results", exist_ok=True)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print("Using device:", DEVICE)


# -----------------------------------------
# 2. Load Breast Cancer Dataset
# -----------------------------------------

data = load_breast_cancer()

df = pd.DataFrame(
    data.data,
    columns=data.feature_names
)

df["target"] = data.target

CSV_PATH = "datasets/breast_cancer.csv"

df.to_csv(CSV_PATH, index=False)

print("\nDataset created successfully!")
print("Dataset shape:", df.shape)
print("Features:", len(data.feature_names))


# -----------------------------------------
# 3. Train Validation Split
# -----------------------------------------

X = df.drop("target", axis=1)
y = df["target"]

train_idx, val_idx = train_test_split(
    np.arange(len(df)),
    test_size=0.2,
    random_state=SEED,
    stratify=y
)

# Fit scaler only on training data
scaler = StandardScaler()

scaler.fit(X.iloc[train_idx])

print("\nTraining samples:", len(train_idx))
print("Validation samples:", len(val_idx))


# -----------------------------------------
# 4. Custom CsvDataset
# -----------------------------------------

class CsvDataset(Dataset):

    def __init__(self, csv_path, indices, scaler):

        self.data = pd.read_csv(csv_path)

        self.X = self.data.drop(
            "target", axis=1
        ).values.astype(np.float32)

        self.y = self.data["target"].values.astype(
            np.float32
        )

        self.X = scaler.transform(self.X).astype(
            np.float32
        )

        self.X = self.X[indices]
        self.y = self.y[indices]

    def __len__(self):
        return len(self.X)

    def __getitem__(self, index):

        features = torch.tensor(
            self.X[index],
            dtype=torch.float32
        )

        label = torch.tensor(
            self.y[index],
            dtype=torch.float32
        )

        return features, label


train_dataset = CsvDataset(
    CSV_PATH, train_idx, scaler
)

val_dataset = CsvDataset(
    CSV_PATH, val_idx, scaler
)


# -----------------------------------------
# 5. DataLoaders
# -----------------------------------------

train_loader = DataLoader(
    train_dataset,
    batch_size=32,
    shuffle=True
)

val_loader = DataLoader(
    val_dataset,
    batch_size=32,
    shuffle=False
)

print("\nDataLoaders created successfully!")


# -----------------------------------------
# 6. Build MLP Neural Network
# -----------------------------------------

class MLP(nn.Module):

    def __init__(self, input_size):

        super().__init__()

        self.network = nn.Sequential(

            nn.Linear(input_size, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(0.3),

            nn.Linear(64, 32),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.Dropout(0.2),

            nn.Linear(32, 1)
        )

    def forward(self, x):
        return self.network(x)


model = MLP(input_size=X.shape[1]).to(DEVICE)

# Count trainable parameters
params = sum(
    p.numel()
    for p in model.parameters()
    if p.requires_grad
)

print("\nTrainable Parameters:", params)


# -----------------------------------------
# 7. Loss and Optimizer
# -----------------------------------------

criterion = nn.BCEWithLogitsLoss()

optimizer = torch.optim.Adam(
    model.parameters(),
    lr=0.001
)

EPOCHS = 40
PATIENCE = 7

best_val_accuracy = 0.0
best_state = None
best_epoch = 0
epochs_without_improvement = 0

train_losses = []


# -----------------------------------------
# 8. Training and Validation Functions
# -----------------------------------------

def evaluate(model, loader):

    model.eval()

    correct = 0
    total = 0

    with torch.no_grad():

        for features, labels in loader:

            features = features.to(DEVICE)
            labels = labels.to(DEVICE)

            outputs = model(features)

            probabilities = torch.sigmoid(outputs)

            predictions = (probabilities >= 0.5).float()

            correct += (predictions == labels.unsqueeze(1)).sum().item()

            total += labels.size(0)

    return correct / total


# -----------------------------------------
# 9. Training Loop + Early Stopping
# -----------------------------------------

print("\nStarting Training...\n")

for epoch in range(EPOCHS):

    model.train()

    running_loss = 0.0

    for features, labels in train_loader:

        features = features.to(DEVICE)
        labels = labels.to(DEVICE).unsqueeze(1)

        # Zero gradients
        optimizer.zero_grad()

        # Forward pass
        outputs = model(features)

        # Calculate loss
        loss = criterion(outputs, labels)

        # Backpropagation
        loss.backward()

        # Update weights
        optimizer.step()

        running_loss += loss.item() * labels.size(0)

    epoch_loss = running_loss / len(train_dataset)

    train_losses.append(epoch_loss)

    # Validation accuracy
    val_accuracy = evaluate(model, val_loader)

    print(
        f"Epoch [{epoch + 1:02d}/{EPOCHS}] "
        f"Loss: {epoch_loss:.4f} "
        f"Val Accuracy: {val_accuracy:.4f}"
    )

    # Early stopping based on validation accuracy
    if val_accuracy > best_val_accuracy:

        best_val_accuracy = val_accuracy

        best_state = copy.deepcopy(model.state_dict())

        best_epoch = epoch + 1

        epochs_without_improvement = 0

    else:

        epochs_without_improvement += 1

    if epochs_without_improvement >= PATIENCE:

        print("\nEarly stopping triggered!")

        break


# -----------------------------------------
# 10. Save Best Model
# -----------------------------------------

MODEL_PATH = "task2_pytorch/saved_model.pth"

model.load_state_dict(best_state)

torch.save(
    model.state_dict(),
    MODEL_PATH
)

print("\nBest model saved successfully!")

print("Best validation accuracy:", round(best_val_accuracy, 4))
print("Best epoch:", best_epoch)


# -----------------------------------------
# 11. Reload Model and Verify
# -----------------------------------------

loaded_model = MLP(input_size=X.shape[1]).to(DEVICE)

loaded_model.load_state_dict(
    torch.load(
        MODEL_PATH,
        map_location=DEVICE,
        weights_only=True
    )
)

loaded_model.eval()

reloaded_accuracy = evaluate(
    loaded_model,
    val_loader
)

print("\nReloaded Model Accuracy:", round(reloaded_accuracy, 4))

print(
    "Accuracy identical:",
    abs(reloaded_accuracy - best_val_accuracy) < 1e-9
)


# -----------------------------------------
# 12. Compare with Logistic Regression
# -----------------------------------------

X_train = scaler.transform(X.iloc[train_idx])
X_val = scaler.transform(X.iloc[val_idx])

y_train = y.iloc[train_idx]
y_val = y.iloc[val_idx]

logistic_model = LogisticRegression(
    max_iter=2000,
    random_state=SEED
)

logistic_model.fit(X_train, y_train)

logistic_predictions = logistic_model.predict(X_val)

logistic_accuracy = accuracy_score(
    y_val,
    logistic_predictions
)


# -----------------------------------------
# 13. Final Comparison
# -----------------------------------------

results = pd.DataFrame({

    "Model": [
        "PyTorch MLP",
        "Logistic Regression"
    ],

    "Validation Accuracy": [
        best_val_accuracy,
        logistic_accuracy
    ]

})

print("\nFINAL MODEL COMPARISON")
print(results.to_string(index=False))

results.to_csv(
    "results/task2_comparison.csv",
    index=False
)

print("\nTask 2 completed successfully!")
