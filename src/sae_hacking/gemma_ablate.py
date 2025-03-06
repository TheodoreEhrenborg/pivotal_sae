#!/usr/bin/env python3
import asyncio
import time
from argparse import ArgumentParser, Namespace
from pathlib import Path
from typing import Dict

import aiohttp
import torch
from beartype import beartype
from coolname import generate_slug
from datasets import load_dataset
from sae_lens import SAE, HookedSAETransformer
from tqdm import tqdm
from transformers import AutoTokenizer

from sae_hacking.graph_network import graph_ablation_matrix
from sae_hacking.safetensor_utils import save_dict_with_tensors

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
    parser.add_argument("--n-edges", type=int, default=10000)
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
        print(f"Processing prompt: {prompt}")

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


@beartype
def compute_ablation_matrix(
    model: HookedSAETransformer,
    ablator_sae: SAE,
    reader_sae: SAE,
    prompt: str,
    frequent_features: list[int],
    ablation_results_mut: dict,
    abridge_ablations_to: int,
) -> None:
    """
    - e: number of features in ablator SAE
    - E: number of features in reader SAE
    """
    print(prompt)
    ablator_sae.use_error_term = True
    reader_sae.use_error_term = True

    # First, run the model with ablator SAE to get its activations
    model.reset_hooks()
    model.reset_saes()
    _, ablator_cache = model.run_with_cache_with_saes(prompt, saes=[ablator_sae])
    ablator_acts_1Se = ablator_cache[f"{ablator_sae.cfg.hook_name}.hook_sae_acts_post"]

    # Find the features with highest activation summed across all positions
    summed_acts_e = ablator_acts_1Se[0].sum(dim=0)
    tentative_top_features_k = torch.topk(
        summed_acts_e, k=abridge_ablations_to + len(frequent_features)
    ).indices

    top_features_K = [
        i for i in tentative_top_features_k if i.item() not in frequent_features
    ][:abridge_ablations_to]
    assert len(top_features_K) == abridge_ablations_to

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
    for ablator_idx in tqdm(top_features_K):
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
        if ablator_idx_int in ablation_results_mut:
            ablation_results_mut[ablator_idx_int] += result.cpu()
        else:
            ablation_results_mut[ablator_idx_int] = result.cpu()

        # Reset hooks for next iteration
        model.reset_hooks()


@beartype
def analyze_ablation_matrix(
    ablation_matrix_eE: torch.Tensor, ablator_sae: SAE, reader_sae: SAE, top_k: int = 5
) -> None:
    """
    Analyzes the ablation matrix and prints the strongest interactions.
    Uses sequential topk operations on the original matrix to find largest magnitude values.
    """
    # Find top k positive values
    _, top_indices = torch.topk(ablation_matrix_eE.view(-1), top_k)

    # Find top k negative values (by finding bottom k)
    _, bottom_indices = torch.topk(ablation_matrix_eE.view(-1), top_k, largest=False)

    # Combine indices and get their values
    all_indices = torch.cat([top_indices, bottom_indices])
    all_values = ablation_matrix_eE.view(-1)[all_indices]

    # Find the top k by absolute value
    _, final_idx = torch.topk(all_values.abs(), top_k)
    flat_indices_K = all_indices[final_idx]

    # Convert flat indices to 2D coordinates
    num_reader_features = ablation_matrix_eE.size(1)
    ablator_indices_K = flat_indices_K.div(num_reader_features, rounding_mode="floor")
    reader_indices_K = flat_indices_K % num_reader_features

    # Get descriptions for the features
    ablator_descriptions = asyncio.run(
        get_all_descriptions(ablator_indices_K.tolist(), ablator_sae.cfg.neuronpedia_id)
    )
    reader_descriptions = asyncio.run(
        get_all_descriptions(reader_indices_K.tolist(), reader_sae.cfg.neuronpedia_id)
    )

    print("\nStrongest feature interactions:")
    for ablator_idx, reader_idx, ablator_desc, reader_desc in zip(
        ablator_indices_K,
        reader_indices_K,
        ablator_descriptions,
        reader_descriptions,
        strict=True,
    ):
        effect = ablation_matrix_eE[ablator_idx, reader_idx].item()
        direction = "increases" if effect < 0 else "decreases"
        print(
            f"\nAblating feature {ablator_idx} {direction} feature {reader_idx} by {abs(effect):.2f}"
        )
        print(f"Ablator feature description: {ablator_desc}")
        print(f"Reader feature description: {reader_desc}")


@beartype
async def get_description_async(
    idx: int, session: aiohttp.ClientSession, neuronpedia_id: str
) -> str:
    url = f"https://www.neuronpedia.org/api/feature/{neuronpedia_id}/{idx}"
    async with session.get(url) as response:
        data = await response.json()
        try:
            if data["explanations"]:
                return data["explanations"][0]["description"]
            else:
                return "No explanation found"
        except:
            # Sometimes we get rate-limited
            print(data)
            raise


@beartype
async def get_all_descriptions(indices: list[int], neuronpedia_id: str) -> list[str]:
    async with aiohttp.ClientSession() as session:
        tasks = [get_description_async(idx, session, neuronpedia_id) for idx in indices]
        return await asyncio.gather(*tasks)


@torch.inference_mode()
@beartype
def main(args: Namespace) -> None:
    output_dir = f"/results/{time.strftime('%Y%m%d-%H%M%S')}{generate_slug()}"
    Path(output_dir).mkdir()
    print(f"Writing to {output_dir}")
    device = "cuda"
    model = HookedSAETransformer.from_pretrained(args.model, device=device)

    ablator_sae, _, _ = SAE.from_pretrained(
        release=args.ablator_sae_release, sae_id=args.ablator_sae_id, device=device
    )
    reader_sae, _, _ = SAE.from_pretrained(
        release=args.reader_sae_release, sae_id=args.reader_sae_id, device=device
    )
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

    ablation_results_mut = {}
    for i, prompt in enumerate(prompts):
        print("Computing ablation matrix...")
        compute_ablation_matrix(
            model,
            ablator_sae,
            reader_sae,
            prompt,
            frequent_features,
            ablation_results_mut,
            args.abridge_ablations_to,
        )
        if i % args.save_frequency == 0:
            save_dict_with_tensors(
                ablation_results_mut,
                f"{output_dir}/{time.strftime('%Y%m%d-%H%M%S')}intermediate.safetensors.zst",
            )

    # print("Analyzing results...")
    # analyze_ablation_matrix(ablation_matrix_eE, ablator_sae, reader_sae)
    print("Graphing results...")
    ablation_results = ablation_results_mut
    graph_ablation_matrix(
        ablation_results,
        ablator_sae.cfg.neuronpedia_id,
        reader_sae.cfg.neuronpedia_id,
        output_dir,
        args.n_edges,
    )


if __name__ == "__main__":
    main(make_parser().parse_args())
