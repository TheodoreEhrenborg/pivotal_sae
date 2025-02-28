#!/usr/bin/env python3
import asyncio
import json
import os
import time
from argparse import ArgumentParser, Namespace

import aiohttp
import networkx as nx
import requests
import torch
from beartype import beartype
from datasets import load_dataset
from pyvis.network import Network
from sae_lens import SAE, HookedSAETransformer
from tqdm import tqdm

# Gemma-scope based on https://colab.research.google.com/drive/17dQFYUYnuKnP6OwQPH9v_GSYUW5aj-Rp
# Neuronpedia API based on https://colab.research.google.com/github/jbloomAus/SAELens/blob/main/tutorials/tutorial_2_0.ipynb


# Add this function to get the first prompt from pile-10k
def get_pile_prompt():
    dataset = load_dataset("NeelNanda/pile-10k", split="train")
    return dataset[0]["text"]


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
    parser.add_argument("--abridge-prompt-to", type=int, default=750)
    parser.add_argument("--abridge-ablations-to", type=int, default=1000)
    parser.add_argument("--n-edges", type=int, default=10000)
    return parser


@beartype
def compute_ablation_matrix(
    model: HookedSAETransformer,
    ablater_sae: SAE,
    reader_sae: SAE,
    prompt: str,
    abridge_ablations_to: int,
) -> torch.Tensor:
    """
    Computes a matrix where each element (i,j) represents the effect of ablating
    feature i in the ablater SAE on feature j in the reader SAE.
    - e: number of features in ablater SAE
    - E: number of features in reader SAE
    """
    ablater_sae.use_error_term = True
    reader_sae.use_error_term = True

    # First, run the model with ablater SAE to get its activations
    model.reset_hooks()
    model.reset_saes()
    model.add_sae(ablater_sae)
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

    # Initialize the ablation matrix
    e = ablater_sae.cfg.d_sae
    E = reader_sae.cfg.d_sae
    ablation_matrix_eE = torch.zeros((e, E), device="cpu")  # [e, E]

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

        # Compute differences
        ablation_matrix_eE[ablater_idx, :] = baseline_acts_E - ablated_acts_E

        # Reset hooks for next iteration
        model.reset_hooks()

    return ablation_matrix_eE


@beartype
def graph_ablation_matrix(
    ablation_matrix_eE: torch.Tensor,
    ablater_sae_id: str,
    reader_sae_id: str,
    output_dir: str,
    n_edges: int,
) -> None:
    """
    Creates and saves a network plot of the interactions.
    """

    # Get the size of the matrix
    _, n_reader = ablation_matrix_eE.shape

    _, flat_indices = torch.topk(ablation_matrix_eE.view(-1), n_edges)

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
        weight = ablation_matrix_eE[ablater_idx, reader_idx].item()
        G.add_edge(
            f"A{ablater_idx.item()}",
            f"R{reader_idx.item()}",
            weight=weight,
            abs_weight=abs(weight),
        )

    nt = Network("500px", "1000px")
    print("Calculating layout")
    nt.from_nx(G)
    nt.save_graph(f"{output_dir}/{time.strftime('%Y%m%d-%H%M%S')}ablation_network.html")


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
    """
    # Find the strongest effects (both positive and negative)
    abs_matrix_AE = torch.abs(ablation_matrix_eE)
    vals_K, flat_indices_K = torch.topk(abs_matrix_AE.view(-1), top_k)

    # Convert flat indices back to 2D
    ablater_indices_K = flat_indices_K.div(
        ablation_matrix_eE.size(1), rounding_mode="floor"
    )
    reader_indices_K = flat_indices_K % ablation_matrix_eE.size(1)

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
def main(args: Namespace) -> None:
    device = "cuda"
    model = HookedSAETransformer.from_pretrained(args.model, device=device)

    ablater_sae, _, _ = SAE.from_pretrained(
        release=args.ablater_sae_release, sae_id=args.ablater_sae_id, device=device
    )
    reader_sae, _, _ = SAE.from_pretrained(
        release=args.reader_sae_release, sae_id=args.reader_sae_id, device=device
    )

    prompt = get_pile_prompt()
    if args.abridge_prompt_to:
        prompt = prompt[: args.abridge_prompt_to]

    print("Computing ablation matrix...")
    ablation_matrix_eE = compute_ablation_matrix(
        model,
        ablater_sae,
        reader_sae,
        prompt,
        abridge_ablations_to=args.abridge_ablations_to,
    )

    print("Analyzing results...")
    analyze_ablation_matrix(ablation_matrix_eE, ablater_sae, reader_sae)
    print("Graphing results...")
    graph_ablation_matrix(
        ablation_matrix_eE,
        ablater_sae.cfg.neuronpedia_id,
        reader_sae.cfg.neuronpedia_id,
        "/tmp",
        args.n_edges,
    )


if __name__ == "__main__":
    main(make_parser().parse_args())
