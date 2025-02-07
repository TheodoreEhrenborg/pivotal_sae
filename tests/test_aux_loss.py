#!/usr/bin/env python3
import torch

from sae_hacking.common.sae import auxiliary_loss, auxiliary_loss_reference


def test_auxiliary_loss_implementations():
    # Set random seed for reproducibility
    torch.manual_seed(42)

    # Create sample inputs
    batch_size = 10
    n_features = 15
    model_dim = 20

    sae_activations = torch.rand(batch_size, n_features).to(dtype=torch.float32)
    model_activations = torch.rand(batch_size, model_dim).to(dtype=torch.float32)
    winners_mask = torch.randint(0, 2, (batch_size, n_features)).to(dtype=torch.bool)
    final_activations_child1 = torch.rand(batch_size, n_features).to(
        dtype=torch.float32
    )
    final_activations_child2 = torch.rand(batch_size, n_features).to(
        dtype=torch.float32
    )
    decoder_weight = torch.rand(model_dim, n_features).to(dtype=torch.float32)
    decoder_child1_weight = torch.rand(model_dim, n_features).to(dtype=torch.float32)
    decoder_child2_weight = torch.rand(model_dim, n_features).to(dtype=torch.float32)

    # Compute losses using both implementations
    loss_reference = auxiliary_loss_reference(
        sae_activations,
        model_activations,
        winners_mask,
        final_activations_child1,
        final_activations_child2,
        decoder_weight,
        decoder_child1_weight,
        decoder_child2_weight,
    )

    loss_vectorized = auxiliary_loss(
        sae_activations,
        model_activations,
        winners_mask,
        final_activations_child1,
        final_activations_child2,
        decoder_weight,
        decoder_child1_weight,
        decoder_child2_weight,
    )

    # Assert they're approximately equal
    torch.testing.assert_close(loss_reference, loss_vectorized, rtol=1e-4, atol=1e-4)
