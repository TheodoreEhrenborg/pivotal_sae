#!/usr/bin/env python3

import json
import os

import networkx as nx
import requests
import torch
from beartype import beartype
from pyvis.network import Network
from tqdm import tqdm


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
    Shows top positive edges in blue and top negative edges in red.
    """
    # Get the total number of reader neurons from any entry in the dictionary
    print("Entered graph_ablation_matrix")
    n_reader = next(iter(ablation_results.values())).shape[0]

    # Collect all values and their indices
    all_values = []
    all_indices = []
    for ablater_idx, tensor in tqdm(ablation_results.items()):
        values = tensor.view(-1)
        indices = torch.arange(len(values))
        all_values.append(values)
        all_indices.append(
            indices + ablater_idx * n_reader
        )  # Offset indices by ablater position

    # Stack all values and indices
    all_values = torch.cat(all_values)
    all_indices = torch.cat(all_indices)

    # Split edges by positive and negative values
    n_pos_edges = n_edges // 2
    n_neg_edges = n_edges - n_pos_edges

    # Get top positive edges
    positive_mask = all_values > 0
    positive_values = all_values[positive_mask]
    positive_indices = all_indices[positive_mask]

    # Get top negative edges
    negative_mask = all_values < 0
    negative_values = all_values[negative_mask]
    negative_indices = all_indices[negative_mask]

    # Get top k positive and negative values
    if len(positive_values) > 0:
        _, top_pos_indices = torch.topk(
            positive_values, min(n_pos_edges, len(positive_values))
        )
        top_pos_flat_indices = positive_indices[top_pos_indices]
    else:
        top_pos_flat_indices = torch.tensor([], dtype=torch.long)

    if len(negative_values) > 0:
        _, top_neg_indices = torch.topk(
            negative_values.abs(), min(n_neg_edges, len(negative_values))
        )
        top_neg_flat_indices = negative_indices[top_neg_indices]
    else:
        top_neg_flat_indices = torch.tensor([], dtype=torch.long)

    # Create graph
    G = nx.Graph()

    # Process all selected indices
    all_flat_indices = torch.cat([top_pos_flat_indices, top_neg_flat_indices])
    ablater_indices = all_flat_indices.div(n_reader, rounding_mode="floor")
    reader_indices = all_flat_indices % n_reader

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

    # Add edges with weights and colors
    print("Adding edges to graph")
    for ablater_idx, reader_idx in zip(ablater_indices, reader_indices, strict=True):
        weight = ablation_results[ablater_idx.item()][reader_idx.item()].item()
        edge_color = "blue" if weight > 0 else "red"
        G.add_edge(
            f"A{ablater_idx.item()}",
            f"R{reader_idx.item()}",
            weight=abs(weight),
            color=edge_color,
            title=f"Weight: {weight:.4f}",
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
