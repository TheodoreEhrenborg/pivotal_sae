#!/usr/bin/env python3
import time
from argparse import ArgumentParser, Namespace
from pathlib import Path
from typing import Dict

import torch
from beartype import beartype
from coolname import generate_slug
from datasets import load_dataset
from jaxtyping import Float, jaxtyped
from sae_lens import SAE, HookedSAETransformer
from tqdm import tqdm
from transformers import AutoTokenizer

from sae_hacking.safetensor_utils import save_v2
from sae_hacking.timeprint import timeprint

# Gemma-scope based on https://colab.research.google.com/drive/17dQFYUYnuKnP6OwQPH9v_GSYUW5aj-Rp
# Neuronpedia API based on https://colab.research.google.com/github/jbloomAus/SAELens/blob/main/tutorials/tutorial_2_0.ipynb


@beartype
def make_parser() -> ArgumentParser:
    parser = ArgumentParser()
    parser.add_argument("--model", default="google/gemma-2-2b")
    parser.add_argument(
        "--ablator-sae-release", default="gemma-scope-2b-pt-res-canonical"
    )
    parser.add_argument("--ablator-sae-id", default="layer_20/width_65k/canonical")
    parser.add_argument(
        "--reader-sae-release", default="gemma-scope-2b-pt-mlp-canonical"
    )
    parser.add_argument("--reader-sae-id", default="layer_21/width_65k/canonical")
    parser.add_argument("--max-tokens-in-prompt", type=int, default=125)
    parser.add_argument("--abridge-ablations-to", type=int, default=1000)
    parser.add_argument("--n-prompts", type=int, default=1)
    parser.add_argument("--keep-frequent-features", action="store_true")
    parser.add_argument("--save-frequency", type=int, default=240)
    parser.add_argument(
        "--exclude-latent-threshold",
        type=float,
        default=0.1,
        help="Latents more frequent than this are excluded from ablation",
    )
    return parser


@beartype
def generate_prompts(
    model: str, n_prompts: int, max_tokens_in_prompt: int
) -> list[str]:
    dataset = load_dataset("NeelNanda/pile-10k", split="train")
    tokenizer = AutoTokenizer.from_pretrained(model)
    prompts = [dataset[i]["text"] for i in range(n_prompts)]

    processed_prompts = []
    for prompt in prompts:
        tokenized_prompt_1S = tokenizer(prompt)["input_ids"]
        # Skip the BOS that the tokenizer adds
        processed_prompt = tokenizer.decode(
            tokenized_prompt_1S[1 : max_tokens_in_prompt + 1]
        )
        processed_prompts.append(processed_prompt)

    return processed_prompts


@beartype
def find_frequently_activating_features(
    model: HookedSAETransformer,
    ablator_sae: SAE,
    prompts: list[str],
    exclude_latent_threshold: float,
) -> list[int]:
    """
    For each prompt, checks which ablator SAE features activate on which tokens.
    Returns a list of features that activate on at least the specified percentage of tokens.

    Args:
        model: The transformer model with SAE hooks
        ablator_sae: The SAE to analyze for feature activations
        prompts: List of text prompts to process
        min_activation_percentage: Minimum percentage of tokens a feature must activate on

    Returns:
        List of feature indices that activate on at least min_activation_percentage of tokens
    """
    ablator_sae.use_error_term = True

    # Count of total tokens across all prompts
    total_token_count = 0

    # Dictionary to track activation counts for each feature
    feature_activation_counts: Dict[int, int] = {}

    # Process each prompt
    for prompt in prompts:
        timeprint(f"Processing prompt: {prompt}")

        # Run the model with ablator SAE to get its activations
        model.reset_hooks()
        model.reset_saes()
        _, ablator_cache = model.run_with_cache_with_saes(prompt, saes=[ablator_sae])
        ablator_acts_1Se = ablator_cache[
            f"{ablator_sae.cfg.hook_name}.hook_sae_acts_post"
        ]

        # Count the number of tokens in this prompt
        num_tokens = ablator_acts_1Se.shape[1]
        total_token_count += num_tokens

        # For each feature, count on how many tokens it activates
        for feature_idx in range(ablator_acts_1Se.shape[2]):
            # Get activations for this feature across all token positions
            feature_acts_e = ablator_acts_1Se[0, :, feature_idx]

            # Count token positions where activation exceeds threshold
            activations = (feature_acts_e > 0).sum().item()

            # Update the count for this feature
            if feature_idx in feature_activation_counts:
                feature_activation_counts[feature_idx] += activations
            else:
                feature_activation_counts[feature_idx] = activations

    # Calculate which features activate on at least min_activation_percentage of tokens
    frequently_activating_features = []
    for feature_idx, activation_count in feature_activation_counts.items():
        activation_percentage = activation_count / total_token_count
        if activation_percentage >= exclude_latent_threshold:
            frequently_activating_features.append(feature_idx)

    return sorted(frequently_activating_features)


@jaxtyped(typechecker=beartype)
def compute_ablation_matrix(
    model: HookedSAETransformer,
    ablator_sae: SAE,
    reader_sae: SAE,
    prompt: str,
    frequent_features: list[int],
    ablation_results_eE: Float[
        torch.Tensor, "num_ablator_features num_reader_features"
    ],
    abridge_ablations_to: int,
    cooccurrences_ee: Float[torch.Tensor, "num_ablator_features num_ablator_features"],
    how_often_activated_e: Float[torch.Tensor, " num_ablator_features"],
) -> None:
    """
    - e: number of features in ablator SAE
    - E: number of features in reader SAE
    """
    timeprint(prompt)
    ablator_sae.use_error_term = True
    reader_sae.use_error_term = True

    # First, run the model with ablator SAE to get its activations
    model.reset_hooks()
    model.reset_saes()
    _, ablator_cache = model.run_with_cache_with_saes(prompt, saes=[ablator_sae])
    ablator_acts_1Se = ablator_cache[f"{ablator_sae.cfg.hook_name}.hook_sae_acts_post"]

    timeprint("Starting to update co-occurrence matrix")
    cooccurrences_ee += gather_co_occurrences2(ablator_acts_1Se)
    timeprint("Done updating co-occurrence matrix")

    # Find the features with highest activation summed across all positions
    summed_acts_e = ablator_acts_1Se[0].sum(dim=0)
    tentative_top_features_k = torch.topk(
        summed_acts_e, k=abridge_ablations_to + len(frequent_features)
    ).indices

    top_features_K = [
        i for i in tentative_top_features_k if i.item() not in frequent_features
    ][:abridge_ablations_to]
    assert len(top_features_K) == abridge_ablations_to

    how_often_activated_e[top_features_K] += 1

    # Get baseline activations for reader SAE
    model.reset_hooks()
    model.reset_saes()
    _, baseline_cache = model.run_with_cache_with_saes(prompt, saes=[reader_sae])
    baseline_acts_1SE = baseline_cache[f"{reader_sae.cfg.hook_name}.hook_sae_acts_post"]
    baseline_acts_E = baseline_acts_1SE[0, -1, :]

    # Add the ablator SAE to the model
    model.add_sae(ablator_sae)
    hook_point = ablator_sae.cfg.hook_name + ".hook_sae_acts_post"

    # For each top feature in the ablator SAE
    for ablator_idx in top_features_K:
        # Set up ablation hook for this feature
        def ablation_hook(acts_BSe, hook):
            acts_BSe[:, :, ablator_idx] = 0
            return acts_BSe

        model.add_hook(hook_point, ablation_hook, "fwd")

        # Run with this feature ablated
        _, ablated_cache = model.run_with_cache_with_saes(prompt, saes=[reader_sae])
        ablated_acts_1SE = ablated_cache[
            f"{reader_sae.cfg.hook_name}.hook_sae_acts_post"
        ]
        ablated_acts_E = ablated_acts_1SE[0, -1, :]

        result = baseline_acts_E - ablated_acts_E

        ablator_idx_int = ablator_idx.item()
        ablation_results_eE[ablator_idx_int] += result.cpu()

        # Reset hooks for next iteration
        model.reset_hooks()


@torch.inference_mode()
@beartype
def main(args: Namespace) -> None:
    output_dir = f"/results/{time.strftime('%Y%m%d-%H%M%S')}{generate_slug()}"
    Path(output_dir).mkdir()
    timeprint(f"Writing to {output_dir}")
    device = "cuda"
    model = HookedSAETransformer.from_pretrained(args.model, device=device)

    ablator_sae, ablator_sae_config, _ = SAE.from_pretrained(
        release=args.ablator_sae_release, sae_id=args.ablator_sae_id, device=device
    )
    e = ablator_sae_config["d_sae"]
    reader_sae, reader_sae_config, _ = SAE.from_pretrained(
        release=args.reader_sae_release, sae_id=args.reader_sae_id, device=device
    )
    E = reader_sae_config["d_sae"]
    prompts = generate_prompts(args.model, args.n_prompts, args.max_tokens_in_prompt)

    frequent_features = (
        []
        if args.keep_frequent_features
        else find_frequently_activating_features(
            model,
            ablator_sae,
            prompts,
            exclude_latent_threshold=args.exclude_latent_threshold,
        )
    )

    ablation_results_eE = torch.zeros(e, E)
    cooccurrences_ee = torch.zeros(e, e).to_sparse()
    how_often_activated_e = torch.zeros(e)
    for i, prompt in enumerate(tqdm(prompts)):
        timeprint("Computing ablation matrix...")
        compute_ablation_matrix(
            model,
            ablator_sae,
            reader_sae,
            prompt,
            frequent_features,
            ablation_results_eE,
            args.abridge_ablations_to,
            cooccurrences_ee,
            how_often_activated_e,
        )
        timeprint("Done computing ablation matrix")
        if i % args.save_frequency == 0:
            save_v2(
                ablation_results_eE,
                f"{output_dir}/{time.strftime('%Y%m%d-%H%M%S')}intermediate.safetensors.zst",
                cooccurrences_ee.to_dense(),
                how_often_activated_e,
            )


@beartype
def gather_co_occurrences2(ablator_acts_1Se) -> torch.Tensor:
    # Convert to binary activation (1 where features are active, 0 otherwise)
    active_binary_Se = (ablator_acts_1Se[0] > 0).float().to_sparse()

    # Compute co-occurrences using matrix multiplication
    these_cooccurrences_ee = active_binary_Se.T @ active_binary_Se

    return these_cooccurrences_ee.cpu()


@beartype
def gather_co_occurrences(ablator_acts_1Se) -> torch.Tensor:
    e = ablator_acts_1Se.shape[2]
    these_cooccurrences_ee = torch.zeros(e, e)
    # Update co-occurrence matrix for ablator features
    # For each position in the sequence
    for pos in range(ablator_acts_1Se.shape[1]):
        # Get feature activations at this position
        pos_acts_e = ablator_acts_1Se[0, pos]
        # Find which features are active (> 0)
        active_features = torch.where(pos_acts_e > 0)[0]

        # Update co-occurrence counts for each pair of active features
        for i in range(len(active_features)):
            for j in range(len(active_features)):
                feat1, feat2 = active_features[i].item(), active_features[j].item()
                these_cooccurrences_ee[feat1, feat2] += 1
    return these_cooccurrences_ee


if __name__ == "__main__":
    main(make_parser().parse_args())
