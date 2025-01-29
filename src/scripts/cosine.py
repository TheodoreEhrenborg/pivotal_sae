import matplotlib.pyplot as plt
import numpy as np


def random_unit_vector(d):
    # Generate random vector from standard normal distribution
    v = np.random.normal(0, 1, d)
    # Normalize to unit length
    return v / np.linalg.norm(v)


def cosine_similarity(v1, v2):
    return np.dot(v1, v2)


# Parameters
d = 100  # dimensions
n_trials = 1000  # number of trials

# Generate similarities
similarities = []
for _ in range(n_trials):
    v1 = random_unit_vector(d)
    v2 = random_unit_vector(d)
    sim = cosine_similarity(v1, v2)
    similarities.append(sim)

# Create histogram
plt.figure(figsize=(10, 6))
plt.hist(similarities, bins=50, density=True, alpha=0.7, color="blue")
plt.xlabel("Cosine Similarity")
plt.ylabel("Density")
plt.title(
    f"Distribution of Cosine Similarities between Random Unit Vectors\nin {d} Dimensions ({n_trials} trials)"
)

# Add theoretical normal distribution
x = np.linspace(-0.4, 0.4, 100)
plt.plot(
    x,
    1 / (np.sqrt(2 * np.pi * (1 / d))) * np.exp(-(x**2) / (2 * (1 / d))),
    "r--",
    lw=2,
    label="Theoretical N(0, 1/d)",
)
plt.legend()

# Add grid
plt.grid(True, alpha=0.3)

# Save the plot
plt.savefig("/tmp/graph.png")
