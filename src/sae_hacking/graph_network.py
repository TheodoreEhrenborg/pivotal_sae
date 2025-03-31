#!/usr/bin/env python3
from datetime import datetime

import networkx as nx
import torch
from beartype import beartype
from pyvis.network import Network
from tqdm import tqdm

from sae_hacking.neuronpedia_utils import NeuronExplanationLoader
from sae_hacking.timeprint import timeprint


@beartype
def graph_ablation_matrix(
    ablation_results: dict,
    ablator_sae_id: str,
    reader_sae_id: str,
    output_dir: str,
    n_edges: int,
) -> None:
    """
    Creates and saves a network plot of the interactions.
    Shows top positive edges in blue and top negative edges in red.
    """
    # Get the total number of reader neurons from any entry in the dictionary
    timeprint("Entered graph_ablation_matrix")
    n_reader = next(iter(ablation_results.values())).shape[0]

    # Collect all values and their indices
    all_values_list = []
    for ablator_idx, tensor in tqdm(ablation_results.items()):
        values = tensor.view(-1)
        assert len(values) == n_reader
        all_values_list.append(values)

    # We're going to do an in-place version of torch.cat, where the result tensor
    # is on the GPU. Weirdly this is at least 1000x faster than using the CPU,
    # and using torch.cat with inputs on the GPU is also slower
    timeprint("Finished loop")
    all_values = torch.zeros(len(all_values_list) * n_reader, device="cuda")
    timeprint("Made blank tensor")
    for i, v in enumerate(tqdm(all_values_list)):
        all_values[i * n_reader : (i + 1) * n_reader] = v
    timeprint("Have constructed all_values")
    all_values = all_values.to("cpu")
    timeprint("Have moved all_values to cpu")

    ablator_idxs = torch.tensor([ablator_idx for ablator_idx in ablation_results])
    all_indices = (
        (ablator_idxs * n_reader).unsqueeze(1) + torch.arange(n_reader)
    ).view(-1)

    timeprint("Have constructed all_indices")
    # Split edges by positive and negative values
    n_pos_edges = n_edges // 2
    n_neg_edges = n_edges - n_pos_edges

    timeprint("Starting to construct masks")
    # Get top positive edges
    positive_mask = all_values > 0
    positive_values = all_values[positive_mask]
    positive_indices = all_indices[positive_mask]

    # Get top negative edges
    negative_mask = all_values < 0
    negative_values = all_values[negative_mask]
    negative_indices = all_indices[negative_mask]
    timeprint("Done constructing masks")

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
    timeprint("Done collecting indices")

    # Create graph
    G = nx.Graph()

    # Process all selected indices
    all_flat_indices = torch.cat([top_pos_flat_indices, top_neg_flat_indices])
    ablator_indices = all_flat_indices.div(n_reader, rounding_mode="floor")
    reader_indices = all_flat_indices % n_reader

    timeprint("Loading auto-interp explanations")
    ablator_descriptions = NeuronExplanationLoader(ablator_sae_id)
    reader_descriptions = NeuronExplanationLoader(reader_sae_id)

    # Add nodes with attributes
    timeprint("Adding nodes to graph")
    for i in ablator_indices:
        name = f"A{i.item()}"
        G.add_node(
            name,
            title=f"{name} {ablator_descriptions.get_explanation(i.item())}",
            group="ablator",
        )
    for i in reader_indices:
        name = f"R{i.item()}"
        G.add_node(
            name,
            title=f"{name} {reader_descriptions.get_explanation(i.item())}",
            group="reader",
        )

    # Add edges with weights and colors
    timeprint("Adding edges to graph")
    for ablator_idx, reader_idx in zip(ablator_indices, reader_indices, strict=True):
        weight = ablation_results[ablator_idx.item()][reader_idx.item()].item()
        edge_color = "blue" if weight > 0 else "red"
        G.add_edge(
            f"{ablator_sae_id}/{ablator_idx.item()}",
            f"{reader_sae_id}/{reader_idx.item()}",
            weight=abs(weight),
            color=edge_color,
            title=f"Weight: {weight:.4f}",
        )

    timeprint("Building the plot")
    nt = Network("500px", "1000px", select_menu=True, cdn_resources="remote")
    timeprint("Calculating layout")
    nt.from_nx(G)
    nt.show_buttons(filter_=["physics"])

    nt.save_graph(
        f"{output_dir}/ablation_network_{datetime.now().strftime('%Y%m%d_%H%M%S')}.html"
    )
    timeprint("Plot has been saved")
