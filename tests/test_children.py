#!/usr/bin/env python3

import torch

from sae_hacking.common.toy_dataset import compute_result, compute_result2


def test_compute_results_match():
    # Set random seed for reproducibility
    torch.manual_seed(0)

    # Test parameters
    batch_size = 3
    n_features = 4
    n_children = 2
    model_dim = 5
    device = torch.device("cpu")

    # Generate random test inputs
    activations = torch.randint(
        0, 2, (batch_size, n_features), dtype=torch.bool, device=device
    )
    perturbation_choices = torch.randint(
        0, n_children, (batch_size, n_features), device=device
    )
    features = torch.randn(n_features, model_dim, device=device)
    perturbations = torch.randn(n_features, n_children, model_dim, device=device)

    # Compute results with both versions
    result1 = compute_result(
        activations, perturbation_choices, features, perturbations, device
    )
    result2 = compute_result2(
        activations, perturbation_choices, features, perturbations, device
    )

    # Assert results are equal within numerical tolerance
    assert torch.allclose(result1, result2, rtol=1e-5, atol=1e-5)
