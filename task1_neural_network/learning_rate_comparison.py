
import numpy as np
import matplotlib.pyplot as plt

# XOR Dataset
X = np.array([
    [0, 0],
    [0, 1],
    [1, 0],
    [1, 1]
], dtype=float)

y = np.array([[0], [1], [1], [0]], dtype=float)


# Sigmoid function
def sigmoid(x):
    return 1 / (1 + np.exp(-np.clip(x, -500, 500)))


# ReLU function
def relu(x):
    return np.maximum(0, x)


# Binary Cross Entropy
def cross_entropy(y_true, y_pred):
    y_pred = np.clip(y_pred, 1e-7, 1 - 1e-7)
    return -np.mean(
        y_true * np.log(y_pred)
        + (1 - y_true) * np.log(1 - y_pred)
    )


# Training function
def train_network(lr, epochs=5000):

    np.random.seed(42)

    W1 = np.random.randn(2, 4)
    b1 = np.zeros((1, 4))

    W2 = np.random.randn(4, 1)
    b2 = np.zeros((1, 1))

    losses = []

    for epoch in range(epochs):

        # Forward pass
        z1 = X @ W1 + b1
        a1 = relu(z1)

        z2 = a1 @ W2 + b2
        a2 = sigmoid(z2)

        loss = cross_entropy(y, a2)

        if not np.isfinite(loss) or loss > 100:
            print(f"LR {lr}: Training became unstable at epoch {epoch}")
            break

        losses.append(loss)

        # Backpropagation
        dz2 = (a2 - y) / len(X)

        dW2 = a1.T @ dz2
        db2 = np.sum(dz2, axis=0, keepdims=True)

        da1 = dz2 @ W2.T
        dz1 = da1 * (z1 > 0)

        dW1 = X.T @ dz1
        db1 = np.sum(dz1, axis=0, keepdims=True)

        # Update weights
        W1 -= lr * dW1
        b1 -= lr * db1

        W2 -= lr * dW2
        b2 -= lr * db2

        if not all(np.all(np.isfinite(v)) for v in [W1, b1, W2, b2]):
            print(f"LR {lr}: Weights diverged at epoch {epoch}")
            break

    return losses


# Compare learning rates
learning_rates = [0.01, 0.5, 5]

plt.figure(figsize=(10, 6))

for lr in learning_rates:

    losses = train_network(lr)

    if losses:
        plt.plot(losses, label=f"Learning Rate = {lr}")

        print(
            f"Learning Rate: {lr}, "
            f"Final Loss: {losses[-1]:.6f}, "
            f"Epochs: {len(losses)}"
        )

plt.title("XOR Neural Network - Learning Rate Comparison")
plt.xlabel("Epochs")
plt.ylabel("Binary Cross Entropy Loss")

plt.legend()
plt.grid(True)

plt.savefig(
    "task1_neural_network/loss_curve.png",
    dpi=300,
    bbox_inches="tight"
)

plt.show()

print("\nLoss curve saved successfully!")
