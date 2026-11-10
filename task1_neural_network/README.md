
# Task 1 - Neural Network Fundamentals

## Objective
Build a 2-4-1 neural network using NumPy
to solve the XOR classification problem.

## Dataset
X = [[0,0], [0,1], [1,0], [1,1]]
y = [0, 1, 1, 0]

## Architecture
- Input layer: 2 neurons
- Hidden layer: 4 neurons
- Hidden activation: ReLU
- Output layer: 1 neuron
- Output activation: Sigmoid

## Training Configuration
- Epochs: 5000
- Learning rate: 0.5
- Loss: Binary Cross Entropy
- Optimizer: Gradient Descent

## Implemented Functions
- ReLU
- Sigmoid
- Softmax
- Mean Squared Error
- Binary Cross Entropy

## PyTorch Autograd
Manual weight gradient: -32
Manual bias gradient: -16

Verified using PyTorch autograd.

## Regularization Concepts
- Dropout
- Batch Normalization
- Early Stopping

## Files
- neural_network.py
- gradient_autograd.py
- learning_rate_comparison.py
- loss_curve.png

## Result
Successfully trained the XOR neural network.
Predicted classes: [0, 1, 1, 0]
