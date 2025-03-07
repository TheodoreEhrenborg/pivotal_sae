#!/usr/bin/env python3

from argparse import ArgumentParser, Namespace

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


@beartype
def make_parser() -> ArgumentParser:
    parser = ArgumentParser()
    parser.add_argument("--model", default="google/gemma-2-2b")
    parser.add_argument("--sae-release", required=True)
    parser.add_argument("--sae-id", default="layer_20/width_65k/canonical")
    parser.add_argument(
        "--reader-sae-release", default="gemma-scope-2b-pt-mlp-canonical"
    )
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--feature-idx", required=True, type=int)
    parser.add_argument("--prompt", required=True)
    return parser


@torch.inference_mode()
@beartype
def main(args: Namespace) -> None:
    model = HookedSAETransformer.from_pretrained(args.model, device=args.device)

    sae, _, _ = SAE.from_pretrained(
        release=args.sae_release, sae_id=args.sae_id, device=args.device
    )
    activations_S = get_feature_activation_per_token(
        model, sae, args.feature_idx, args.prompt
    )


if __name__ == "__main__":
    main(make_parser().parse_args())
