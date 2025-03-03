#!/usr/bin/env python3
import asyncio
import json
import os
import time
from argparse import ArgumentParser, Namespace
from pathlib import Path
from typing import Dict

import aiohttp
import networkx as nx
import requests
import torch
from beartype import beartype
from coolname import generate_slug
from datasets import load_dataset
from pyvis.network import Network
from sae_lens import SAE, HookedSAETransformer
from tqdm import tqdm
from transformers import AutoTokenizer

# Gemma-scope based on https://colab.research.google.com/drive/17dQFYUYnuKnP6OwQPH9v_GSYUW5aj-Rp
# Neuronpedia API based on https://colab.research.google.com/github/jbloomAus/SAELens/blob/main/tutorials/tutorial_2_0.ipynb


@beartype
def make_parser() -> ArgumentParser:
    parser = ArgumentParser()
    parser.add_argument("--model", default="google/gemma-2-2b")
    parser.add_argument(
        "--ablater-sae-release", default="gemma-scope-2b-pt-res-canonical"
    )
    parser.add_argument("--ablater-sae-id", default="layer_20/width_65k/canonical")
    parser.add_argument(
        "--reader-sae-release", default="gemma-scope-2b-pt-mlp-canonical"
    )
    parser.add_argument("--reader-sae-id", default="layer_21/width_65k/canonical")
    parser.add_argument("--max-tokens-in-prompt", type=int, default=125)
    parser.add_argument("--abridge-ablations-to", type=int, default=1000)
    parser.add_argument("--n-edges", type=int, default=10000)
    parser.add_argument("--n-prompts", type=int, default=1)
    parser.add_argument("--json-save-frequency", type=int, default=240)
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
    ablater_sae: SAE,
    prompts: list[str],
    min_activation_percentage: float,
) -> list[int]:
    """
    For each prompt, checks which ablater SAE features activate on which tokens.
    Returns a list of features that activate on at least the specified percentage of tokens.

    Args:
        model: The transformer model with SAE hooks
        ablater_sae: The SAE to analyze for feature activations
        prompts: List of text prompts to process
        min_activation_percentage: Minimum percentage of tokens a feature must activate on

    Returns:
        List of feature indices that activate on at least min_activation_percentage of tokens
    """
    ablater_sae.use_error_term = True

    # Count of total tokens across all prompts
    total_token_count = 0

    # Dictionary to track activation counts for each feature
    feature_activation_counts: Dict[int, int] = {}

    # Process each prompt
    for prompt in prompts:
        print(f"Processing prompt: {prompt}")

        # Run the model with ablater SAE to get its activations
        model.reset_hooks()
        model.reset_saes()
        _, ablater_cache = model.run_with_cache_with_saes(prompt, saes=[ablater_sae])
        ablater_acts_1Se = ablater_cache[
            f"{ablater_sae.cfg.hook_name}.hook_sae_acts_post"
        ]

        # Count the number of tokens in this prompt
        num_tokens = ablater_acts_1Se.shape[1]
        total_token_count += num_tokens

        # For each feature, count on how many tokens it activates
        for feature_idx in range(ablater_acts_1Se.shape[2]):
            # Get activations for this feature across all token positions
            feature_acts_e = ablater_acts_1Se[0, :, feature_idx]

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
        if activation_percentage >= min_activation_percentage:
            frequently_activating_features.append(feature_idx)

    return sorted(frequently_activating_features)


@beartype
def compute_ablation_matrix(
    model: HookedSAETransformer,
    ablater_sae: SAE,
    reader_sae: SAE,
    prompt: str,
    ablation_results_mut: dict,
    abridge_ablations_to: int,
) -> None:
    """
    - e: number of features in ablater SAE
    - E: number of features in reader SAE
    """
    print(prompt)
    ablater_sae.use_error_term = True
    reader_sae.use_error_term = True

    # First, run the model with ablater SAE to get its activations
    model.reset_hooks()
    model.reset_saes()
    _, ablater_cache = model.run_with_cache_with_saes(prompt, saes=[ablater_sae])
    ablater_acts_1Se = ablater_cache[f"{ablater_sae.cfg.hook_name}.hook_sae_acts_post"]

    # Find the features with highest activation summed across all positions
    summed_acts_e = ablater_acts_1Se[0].sum(dim=0)
    top_features_K = torch.topk(summed_acts_e, k=abridge_ablations_to).indices

    # Get baseline activations for reader SAE
    model.reset_hooks()
    model.reset_saes()
    _, baseline_cache = model.run_with_cache_with_saes(prompt, saes=[reader_sae])
    baseline_acts_1SE = baseline_cache[f"{reader_sae.cfg.hook_name}.hook_sae_acts_post"]
    baseline_acts_E = baseline_acts_1SE[0, -1, :]

    # Add the ablater SAE to the model
    model.add_sae(ablater_sae)
    hook_point = ablater_sae.cfg.hook_name + ".hook_sae_acts_post"

    # For each top feature in the ablater SAE
    for ablater_idx in tqdm(top_features_K):
        # Set up ablation hook for this feature

        def ablation_hook(acts_BSe, hook):
            acts_BSe[:, :, ablater_idx] = 0
            return acts_BSe

        model.add_hook(hook_point, ablation_hook, "fwd")

        # Run with this feature ablated
        _, ablated_cache = model.run_with_cache_with_saes(prompt, saes=[reader_sae])
        ablated_acts_1SE = ablated_cache[
            f"{reader_sae.cfg.hook_name}.hook_sae_acts_post"
        ]
        ablated_acts_E = ablated_acts_1SE[0, -1, :]

        result = baseline_acts_E - ablated_acts_E

        ablater_idx_int = ablater_idx.item()
        if ablater_idx_int in ablation_results_mut:
            ablation_results_mut[ablater_idx_int] += result
        else:
            ablation_results_mut[ablater_idx_int] = result

        # Reset hooks for next iteration
        model.reset_hooks()


@beartype
def graph_ablation_matrix(
    ablation_results: dict,
    ablater_sae_id: str,
    reader_sae_id: str,
    output_dir: str,
    n_edges: int,
) -> None:
    """
    Creates and saves a network plot of the interactions.
    """

    # Get the total number of reader neurons from any entry in the dictionary
    n_reader = next(iter(ablation_results.values())).shape[0]

    # Collect all values and their indices
    all_values = []
    all_indices = []
    for ablater_idx, tensor in ablation_results.items():
        values = tensor.view(-1)
        indices = torch.arange(len(values)).cuda()
        all_values.append(values)
        all_indices.append(
            indices + ablater_idx * n_reader
        )  # Offset indices by ablater position

    # Stack all values and indices
    all_values = torch.cat(all_values)
    all_indices = torch.cat(all_indices)

    # Get top k values
    _, top_k_indices = torch.topk(all_values, n_edges)
    flat_indices = all_indices[top_k_indices]

    # Create graph
    G = nx.Graph()

    # Get all descriptions first
    ablater_indices = flat_indices.div(n_reader, rounding_mode="floor")
    reader_indices = flat_indices % n_reader

    print("Loading auto-interp explanations")
    ablater_descriptions = NeuronExplanationLoader(ablater_sae_id)
    reader_descriptions = NeuronExplanationLoader(reader_sae_id)

    # Add nodes with attributes
    print("Adding nodes to graph")
    for i in ablater_indices:
        name = f"A{i.item()}"
        G.add_node(
            name,
            title=f"{name} {ablater_descriptions.get_explanation(i.item())}",
            group="ablater",
        )
    for i in reader_indices:
        name = f"R{i.item()}"
        G.add_node(
            name,
            title=f"{name} {reader_descriptions.get_explanation(i.item())}",
            group="reader",
        )

    # Add edges with weights
    print("Adding edges to graph")
    for ablater_idx, reader_idx in zip(ablater_indices, reader_indices):
        weight = ablation_results[ablater_idx.item()][reader_idx.item()].item()
        G.add_edge(
            f"A{ablater_idx.item()}",
            f"R{reader_idx.item()}",
            weight=weight,
            abs_weight=abs(weight),
        )

    nt = Network("500px", "1000px", select_menu=True, cdn_resources="remote")
    print("Calculating layout")
    nt.from_nx(G)
    nt.show_buttons(filter_=["physics"])
    nt.save_graph(f"{output_dir}/ablation_network.html")


@beartype
class NeuronExplanationLoader:
    def __init__(self, combined_id: str):
        """
        Initialize the loader with a combined model/sae id string.
        Downloads and caches data if not already present.

        Args:
            combined_id (str): Combined identifier (e.g., "gemma-2-2b/20-gemmascope-res-65k")
        """
        self.model_id, self.sae_id = self._parse_combined_id(combined_id)
        self.cache_path = f"/tmp/neuron_explanations_{self.model_id}_{self.sae_id}.json"
        self.explanations = self._preprocess(self._load_or_download_data())

    def _preprocess(self, data: list) -> dict:
        return {int(item["index"]): item["description"] for item in data}

    def _parse_combined_id(self, combined_id: str) -> tuple[str, str]:
        """
        Parse the combined ID into model_id and sae_id.

        Args:
            combined_id (str): The combined identifier string

        Returns:
            Tuple of (model_id, sae_id)
        """
        model_id, sae_id = combined_id.split("/")
        return model_id, sae_id

    def _load_or_download_data(self) -> list[dict]:
        """
        Load data from cache if it exists, otherwise download and cache it.

        Returns:
            Dict containing the explanations data
        """
        if os.path.exists(self.cache_path):
            with open(self.cache_path, "r") as f:
                return json.load(f)

        print("Downloading data from neuronpedia")
        url = "https://www.neuronpedia.org/api/explanation/export"
        params = {"modelId": self.model_id, "saeId": self.sae_id}

        response = requests.get(url, params=params)
        response.raise_for_status()

        data = response.json()

        with open(self.cache_path, "w") as f:
            json.dump(data, f)

        return data

    def get_explanation(self, index: int) -> str:
        """
        Get the explanation for a specific neuron index.

        Args:
            index (str): The neuron index to look up

        Returns:
            Dict containing the explanation data for the specified index

        """
        return self.explanations.get(index, f"No explanation found for index {index}")


@beartype
def analyze_ablation_matrix(
    ablation_matrix_eE: torch.Tensor, ablater_sae: SAE, reader_sae: SAE, top_k: int = 5
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
    ablater_indices_K = flat_indices_K.div(num_reader_features, rounding_mode="floor")
    reader_indices_K = flat_indices_K % num_reader_features

    # Get descriptions for the features
    ablater_descriptions = asyncio.run(
        get_all_descriptions(ablater_indices_K.tolist(), ablater_sae.cfg.neuronpedia_id)
    )
    reader_descriptions = asyncio.run(
        get_all_descriptions(reader_indices_K.tolist(), reader_sae.cfg.neuronpedia_id)
    )

    print("\nStrongest feature interactions:")
    for ablater_idx, reader_idx, ablater_desc, reader_desc in zip(
        ablater_indices_K,
        reader_indices_K,
        ablater_descriptions,
        reader_descriptions,
        strict=True,
    ):
        effect = ablation_matrix_eE[ablater_idx, reader_idx].item()
        direction = "increases" if effect < 0 else "decreases"
        print(
            f"\nAblating feature {ablater_idx} {direction} feature {reader_idx} by {abs(effect):.2f}"
        )
        print(f"Ablater feature description: {ablater_desc}")
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


@beartype
def save_dict_with_tensors_to_json(tensor_dict: dict, save_path: str) -> None:
    json_dict = {}

    for key, value in tensor_dict.items():
        if isinstance(value, torch.Tensor):
            assert value.dim() == 1, f"Tensor for key '{key}' is not 1D"
            json_dict[key] = value.tolist()
        else:
            json_dict[key] = value

    with open(save_path, "w") as f:
        json.dump(json_dict, f)


@beartype
def main(args: Namespace) -> None:
    output_dir = f"/results/{time.strftime('%Y%m%d-%H%M%S')}{generate_slug()}"
    Path(output_dir).mkdir()
    print(f"Writing to {output_dir}")
    device = "cuda"
    model = HookedSAETransformer.from_pretrained(args.model, device=device)

    ablater_sae, _, _ = SAE.from_pretrained(
        release=args.ablater_sae_release, sae_id=args.ablater_sae_id, device=device
    )
    reader_sae, _, _ = SAE.from_pretrained(
        release=args.reader_sae_release, sae_id=args.reader_sae_id, device=device
    )
    prompts = generate_prompts(args.model, args.n_prompts, args.max_tokens_in_prompt)

    ablation_results_mut = {}
    for i, prompt in enumerate(prompts):
        print("Computing ablation matrix...")
        compute_ablation_matrix(
            model,
            ablater_sae,
            reader_sae,
            prompt,
            ablation_results_mut,
            abridge_ablations_to=args.abridge_ablations_to,
        )
        if i % args.json_save_frequency == 0:
            save_dict_with_tensors_to_json(
                ablation_results_mut,
                f"{output_dir}/{time.strftime('%Y%m%d-%H%M%S')}intermediate.json",
            )

    # print("Analyzing results...")
    # analyze_ablation_matrix(ablation_matrix_eE, ablater_sae, reader_sae)
    print("Graphing results...")
    ablation_results = ablation_results_mut
    graph_ablation_matrix(
        ablation_results,
        ablater_sae.cfg.neuronpedia_id,
        reader_sae.cfg.neuronpedia_id,
        output_dir,
        args.n_edges,
    )


if __name__ == "__main__":
    main(make_parser().parse_args())
