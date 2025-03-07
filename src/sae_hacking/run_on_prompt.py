#!/usr/bin/env python3


import torch
from beartype import beartype
from sae_lens import SAE, HookedSAETransformer


@beartype
def get_feature_activation_per_token(
    model: HookedSAETransformer, sae: SAE, feature_idx: int, prompt: str
) -> torch.Tensor:
    """
    Returns an array showing how much a specific SAE feature activated on each token of the prompt.

    Args:
        model: The transformer model with SAE hooks
        sae: The SAE to analyze
        feature_idx: Index of the specific feature to track
        prompt: The input text prompt

    Returns:
        Tensor of shape [num_tokens] containing activation values for the specified feature
        across all tokens in the prompt
    """
    # Ensure the SAE uses its error term for accurate activation measurement
    sae.use_error_term = True

    # Reset the model and SAEs to ensure clean state
    model.reset_hooks()
    model.reset_saes()

    # Run the model with the SAE to get activations
    _, cache = model.run_with_cache_with_saes(prompt, saes=[sae])

    # Get the SAE activations from the cache
    # Shape: [batch_size, sequence_length, n_features]
    sae_acts = cache[f"{sae.cfg.hook_name}.hook_sae_acts_post"]

    # Extract activations for the specified feature across all tokens
    # Assuming batch_size is 1, we take the first batch with sae_acts[0]
    feature_acts = sae_acts[0, :, feature_idx]

    return feature_acts
