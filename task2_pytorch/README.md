
# Task 2 - PyTorch Training Pipeline

## Objective
Build a neural network classification model
using the Breast Cancer Wisconsin dataset.

## Dataset
- Samples: 569
- Features: 30
- Target: Benign / Malignant

## Preprocessing
- Train/validation split: 80/20
- Feature scaling: StandardScaler
- Custom CsvDataset class
- DataLoader batch size: 32

## Neural Network
- Input: 30
- Hidden Layer 1: 64 neurons
- Hidden Layer 2: 32 neurons
- Output: 1 neuron
- Activation: ReLU
- Regularization: Dropout
- Batch normalization: BatchNorm1d

## Training
- Optimizer: Adam
- Loss: BCEWithLogitsLoss
- Maximum epochs: 40
- Early stopping patience: 7

## Model Saving
Saved using PyTorch state_dict.

## Model Comparison
Compared PyTorch MLP with
Scikit-learn Logistic Regression.

## Results
Refer to results/task2_comparison.csv
for actual validation accuracy.

## Files
- train_pytorch.py
- saved_model.pth
- README.md
