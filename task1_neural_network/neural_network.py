
import numpy as np

# 1. ReLU Activation Function
def relu(x):
    return np.maximum(0, x)


# 2. Sigmoid Activation Function
def sigmoid(x):
    return 1 / (1 + np.exp(-np.clip(x, -500, 500)))


# 3. Softmax Activation Function
def softmax(x):
    exp_values = np.exp(x - np.max(x))
    return exp_values / np.sum(exp_values)


# 4. Mean Squared Error
def mse(y_true, y_pred):
    return np.mean((y_true - y_pred) ** 2)


# 5. Binary Cross Entropy
def cross_entropy(y_true, y_pred):
    epsilon = 1e-7
    y_pred = np.clip(y_pred, epsilon, 1 - epsilon)

    loss = -np.mean(
        y_true * np.log(y_pred)
        + (1 - y_true) * np.log(1 - y_pred)
    )

    return loss


# Test all functions
if __name__ == "__main__":

    print("ReLU:", relu(np.array([-2, 0, 3])))

    print("Sigmoid:", sigmoid(np.array([0, 1, -1])))

    print("Softmax:", np.round(
        softmax(np.array([2.0, 1.0, 0.1])), 3
    ))

    print("MSE:", mse(
        np.array([1, 0]),
        np.array([0.8, 0.2])
    ))

    print("Cross Entropy:", cross_entropy(
        np.array([1, 0]),
        np.array([0.8, 0.2])
    ))



import numpy as np

# --------------------------------
# 1. Activation Functions
# --------------------------------

def relu(x):
    return np.maximum(0, x)


def sigmoid(x):
    return 1 / (1 + np.exp(-np.clip(x, -500, 500)))


def softmax(x):
    exp_values = np.exp(x - np.max(x))
    return exp_values / np.sum(exp_values)


# --------------------------------
# 2. Loss Functions
# --------------------------------

def mse(y_true, y_pred):
    return np.mean((y_true - y_pred) ** 2)


def cross_entropy(y_true, y_pred):
    epsilon = 1e-7
    y_pred = np.clip(y_pred, epsilon, 1 - epsilon)

    return -np.mean(
        y_true * np.log(y_pred)
        + (1 - y_true) * np.log(1 - y_pred)
    )


# --------------------------------
# 3. XOR Dataset
# --------------------------------

X = np.array([
    [0, 0],
    [0, 1],
    [1, 0],
    [1, 1]
], dtype=float)

y = np.array([
    [0],
    [1],
    [1],
    [0]
], dtype=float)


# --------------------------------
# 4. Initialize Neural Network
# Architecture: 2 - 4 - 1
# --------------------------------

np.random.seed(42)

W1 = np.random.randn(2, 4)
b1 = np.zeros((1, 4))

W2 = np.random.randn(4, 1)
b2 = np.zeros((1, 1))


# --------------------------------
# 5. Training Configuration
# --------------------------------

learning_rate = 0.5
epochs = 5000
m = len(X)

loss_history = []


# --------------------------------
# 6. Training Loop
# --------------------------------

for epoch in range(epochs):

    # Forward Pass
    z1 = np.dot(X, W1) + b1
    a1 = relu(z1)

    z2 = np.dot(a1, W2) + b2
    a2 = sigmoid(z2)

    # Calculate Binary Cross Entropy
    loss = cross_entropy(y, a2)
    loss_history.append(loss)

    # Backpropagation
    dz2 = (a2 - y) / m

    dW2 = np.dot(a1.T, dz2)
    db2 = np.sum(dz2, axis=0, keepdims=True)

    da1 = np.dot(dz2, W2.T)

    dz1 = da1 * (z1 > 0)

    dW1 = np.dot(X.T, dz1)
    db1 = np.sum(dz1, axis=0, keepdims=True)

    # Gradient Descent Weight Update
    W2 -= learning_rate * dW2
    b2 -= learning_rate * db2

    W1 -= learning_rate * dW1
    b1 -= learning_rate * db1


# --------------------------------
# 7. Final Predictions
# --------------------------------

z1 = np.dot(X, W1) + b1
a1 = relu(z1)

z2 = np.dot(a1, W2) + b2
predictions = sigmoid(z2)

print("\nTraining Completed!")
print("Epochs:", epochs)
print("Learning Rate:", learning_rate)

print("\nFinal Loss:", round(loss_history[-1], 6))

print("\nXOR Predictions:")
print(np.round(predictions.flatten(), 2))

print("\nPredicted Classes:")
print((predictions >= 0.5).astype(int).flatten())

print("\nActual Outputs:")
print(y.flatten())
