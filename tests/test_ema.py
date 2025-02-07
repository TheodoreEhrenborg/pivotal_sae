import torch

from sae_hacking.common.sae import update_parent_child_ratio, update_parent_child_ratio2


def test_update_parent_child_ratio():
    # Set random seed for reproducibility
    torch.manual_seed(42)

    # Test parameters
    batch_size = 9
    sae_dim = 5

    # Create random test data
    parent_activations = torch.rand(batch_size, sae_dim)
    child1_activations = torch.rand(batch_size, sae_dim)
    child2_activations = torch.rand(batch_size, sae_dim)

    # Set some values to zero to test edge cases
    parent_activations[parent_activations < 0.3] = 0
    child1_activations[child1_activations < 0.3] = 0
    child2_activations[child2_activations < 0.3] = 0

    # Create two copies of initial ratios
    child1_parent_ratios_original = torch.ones(sae_dim)
    child2_parent_ratios_original = torch.ones(sae_dim)
    child1_parent_ratios_vectorized = child1_parent_ratios_original.clone()
    child2_parent_ratios_vectorized = child2_parent_ratios_original.clone()

    # Run both implementations
    update_parent_child_ratio(
        parent_activations,
        child1_activations,
        child2_activations,
        child1_parent_ratios_original,
        child2_parent_ratios_original,
    )

    update_parent_child_ratio2(
        parent_activations,
        child1_activations,
        child2_activations,
        child1_parent_ratios_vectorized,
        child2_parent_ratios_vectorized,
    )

    # Compare results
    print("Test data:")
    print("Parent activations:\n", parent_activations)
    print("Child1 activations:\n", child1_activations)
    print("Child2 activations:\n", child2_activations)

    print("\nResults:")
    print("Child1 ratios (original):", child1_parent_ratios_original)
    print("Child1 ratios (vectorized):", child1_parent_ratios_vectorized)
    print("Child2 ratios (original):", child2_parent_ratios_original)
    print("Child2 ratios (vectorized):", child2_parent_ratios_vectorized)

    # Check if results are close (using a small tolerance due to potential floating-point differences)
    assert torch.allclose(
        child1_parent_ratios_original, child1_parent_ratios_vectorized, rtol=1e-5
    ), "Child1 ratios don't match!"
    assert torch.allclose(
        child2_parent_ratios_original, child2_parent_ratios_vectorized, rtol=1e-5
    ), "Child2 ratios don't match!"

    print("\nAll tests passed! Both implementations give the same results.")
