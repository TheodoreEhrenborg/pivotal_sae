import numpy as np
from scipy.stats import binom

# Parameters
n = 30  # number of coins
p = 0.03  # probability of heads
min_heads = 4

# Exact calculation
exact_prob = 1 - sum(binom.pmf(k, n, p) for k in range(min_heads))
print(f"Exact probability: {exact_prob:.4f}")

# Simulation
num_trials = 100000
successes = 0

for _ in range(num_trials):
    flips = np.random.random(n) < p
    if sum(flips) >= min_heads:
        successes += 1

simulated_prob = successes / num_trials
print(f"Simulated probability: {simulated_prob:.4f}")
