
import importlib

torch = importlib.import_module("torch")

# Step 1: Define variables
w = torch.tensor(1.0, requires_grad=True)
x = torch.tensor(2.0)
b = torch.tensor(0.0, requires_grad=True)
t = torch.tensor(10.0)

# Step 2: Forward calculation
prediction = w * x + b

# Step 3: Calculate squared error loss
loss = (prediction - t) ** 2

print("Prediction:", prediction.item())
print("Loss:", loss.item())

# Step 4: Backpropagation
loss.backward()

# Step 5: Print gradients
print("w.grad:", w.grad.item())
print("b.grad:", b.grad.item())
