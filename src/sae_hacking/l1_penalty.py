#!/usr/bin/env python3

import matplotlib.pyplot as plt
import torch

# Set random seed for reproducibility
torch.manual_seed(42)

# Define the variables to optimize
x = torch.tensor(0.0, requires_grad=True)
y = torch.tensor(0.0, requires_grad=True)

# Define optimization parameters
learning_rate = 0.01
num_iterations = 2000

# Use Adam optimizer
optimizer = torch.optim.Adam([x, y], lr=learning_rate)

# Lists to store history for visualization
loss_history = []
x_history = []
y_history = []


def penalty_fn(x, y):
    return 4 * (torch.abs(x) + torch.abs(y))


# Optimization loop
for i in range(num_iterations):
    # Zero gradients
    optimizer.zero_grad()

    # Compute objective function
    objective = (x - 3) ** 2 + (y - 2) ** 2

    # Compute penalty term (L1 norm)
    penalty = penalty_fn(x, y)

    # Total loss
    loss = objective + penalty

    # Compute gradients
    loss.backward()

    # Update parameters
    optimizer.step()

    # Store history
    loss_history.append(loss.item())
    x_history.append(x.item())
    y_history.append(y.item())

    # Print progress occasionally
    if (i + 1) % 200 == 0:
        print(
            f"Iteration {i + 1}: x = {x.item():.4f}, y = {y.item():.4f}, Loss = {loss.item():.4f}"
        )

# Print final result
print("\nFinal solution:")
print(f"x = {x.item():.6f}")
print(f"y = {y.item():.6f}")
print(f"Objective value = {((x - 3) ** 2 + (y - 2) ** 2).item():.6f}")
print(f"Penalty value = {penalty_fn(x, y).item():.6f}")
print(f"Total loss = {loss.item():.6f}")

# Visualize optimization progress
plt.figure(figsize=(12, 5))

# Plot loss over iterations
plt.subplot(1, 2, 1)
plt.plot(loss_history)
plt.title("Loss vs Iterations")
plt.xlabel("Iteration")
plt.ylabel("Loss")

# Plot optimization trajectory in x-y plane
plt.subplot(1, 2, 2)
plt.plot(x_history, y_history, "r-", alpha=0.5)
plt.plot(x_history[-1], y_history[-1], "ro", markersize=8)
plt.scatter([3], [2], color="green", s=100, label="Objective minimum (without penalty)")
plt.title("Optimization Path in x-y Plane")
plt.xlabel("x")
plt.ylabel("y")
plt.grid(True)
plt.legend()

plt.tight_layout()
plt.savefig("/tmp/plot.png")
