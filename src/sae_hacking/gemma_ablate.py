#!/usr/bin/env python3
import asyncio
from argparse import ArgumentParser, Namespace
from functools import partial

import aiohttp
import networkx as nx
import plotly.graph_objects as go
import torch
from beartype import beartype
from datasets import load_dataset
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
    return parser


@beartype
def compute_ablation_matrix(
    model: HookedSAETransformer,
    ablater_sae: SAE,
    reader_sae: SAE,
    prompt: str,
    device: str,
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

    def ablate_feature_hook(acts_BSe, hook, feature_id):
        acts_BSe[:, :, feature_id] = 0
        return acts_BSe

    # Get baseline activations first
    model.reset_hooks()
    model.reset_saes()
    _, baseline_cache = model.run_with_cache_with_saes(prompt, saes=[reader_sae])
    baseline_acts_BSE = baseline_cache[f"{reader_sae.cfg.hook_name}.hook_sae_acts_post"]
    baseline_acts_E = baseline_acts_BSE[0, -1, :]  # Take last sequence position
    print(f"{baseline_acts_E[28210]=}")

    # Initialize the ablation matrix
    e = abridge_ablations_to if abridge_ablations_to else ablater_sae.cfg.d_sae
    E = reader_sae.cfg.d_sae
    ablation_matrix_eE = torch.zeros((e, E), device="cpu")

    # Add the ablater SAE to the model
    model.add_sae(ablater_sae)
    hook_point = ablater_sae.cfg.hook_name + ".hook_sae_acts_post"

    # For each feature in the ablater SAE
    for ablater_idx in tqdm(range(e)):
        # Set up ablation hook for this feature
        ablation_hook = partial(ablate_feature_hook, feature_id=ablater_idx)
        model.add_hook(hook_point, ablation_hook, "fwd")

        # Run with this feature ablated
        _, ablated_cache = model.run_with_cache_with_saes(prompt, saes=[reader_sae])
        ablated_acts_BSE = ablated_cache[
            f"{reader_sae.cfg.hook_name}.hook_sae_acts_post"
        ]
        ablated_acts_E = ablated_acts_BSE[0, -1, :]  # Take last sequence position
        print(f"{ablated_acts_E[28210]=}")

        # Compute differences
        ablation_matrix_eE[ablater_idx, :] = baseline_acts_E - ablated_acts_E
        print(f"{ablation_matrix_eE[0,28210]=}")

        # Reset hooks for next iteration
        model.reset_hooks()

    return ablation_matrix_eE


@beartype
def graph_ablation_matrix(
    ablation_matrix_eE: torch.Tensor,
    ablater_sae_id: str,
    reader_sae_id: str,
    output_dir: str,
) -> None:
    """
    Creates and saves a network plot of the interactions.
    """

    # Get the size of the matrix
    n_ablater, n_reader = ablation_matrix_eE.shape

    # Find top 1% of edges by absolute value
    abs_matrix = torch.abs(ablation_matrix_eE)
    n_edges = int(0.01 * n_ablater * n_reader)
    _, flat_indices = torch.topk(abs_matrix.view(-1), n_edges)

    # Create graph
    G = nx.Graph()

    # Get all descriptions first
    ablater_indices = flat_indices.div(n_reader, rounding_mode="floor")
    reader_indices = flat_indices % n_reader

    ablater_descriptions = asyncio.run(
        get_all_descriptions(ablater_indices.tolist(), ablater_sae_id)
    )
    reader_descriptions = asyncio.run(
        get_all_descriptions(reader_indices.tolist(), reader_sae_id)
    )

    # Add nodes with attributes
    for i, desc in zip(ablater_indices, ablater_descriptions):
        G.add_node(f"A{i.item()}", description=desc, type="ablater")
    for i, desc in zip(reader_indices, reader_descriptions):
        G.add_node(f"R{i.item()}", description=desc, type="reader")

    # Add edges with weights
    for idx, (ablater_idx, reader_idx) in enumerate(
        zip(ablater_indices, reader_indices)
    ):
        weight = ablation_matrix_eE[ablater_idx, reader_idx].item()
        G.add_edge(
            f"A{ablater_idx.item()}",
            f"R{reader_idx.item()}",
            weight=weight,
            abs_weight=abs(weight),
        )

    # Get layout positions
    pos = nx.spring_layout(G)

    # Create edge traces
    edge_traces = []
    max_weight = max(abs(d["weight"]) for (u, v, d) in G.edges(data=True))

    for edge in G.edges(data=True):
        x0, y0 = pos[edge[0]]
        x1, y1 = pos[edge[1]]
        weight = edge[2]["weight"]
        width = 3 * abs(weight) / max_weight

        edge_trace = go.Scatter(
            x=[x0, x1, None],
            y=[y0, y1, None],
            line=dict(width=width, color="gray"),
            hoverinfo="text",
            text=f"Weight: {weight:.3f}",
            mode="lines",
        )
        edge_traces.append(edge_trace)

    # Create node traces for ablater and reader nodes separately
    ablater_nodes = [n for n in G.nodes() if G.nodes[n]["type"] == "ablater"]
    reader_nodes = [n for n in G.nodes() if G.nodes[n]["type"] == "reader"]

    def create_node_trace(nodes, color):
        return go.Scatter(
            x=[pos[node][0] for node in nodes],
            y=[pos[node][1] for node in nodes],
            mode="markers",
            hoverinfo="text",
            text=[
                f"Node: {node}\nDescription: {G.nodes[node]['description']}"
                for node in nodes
            ],
            marker=dict(size=10, color=color, line=dict(width=2)),
        )

    ablater_trace = create_node_trace(ablater_nodes, "red")
    reader_trace = create_node_trace(reader_nodes, "blue")

    # Create figure
    fig = go.Figure(data=[*edge_traces, ablater_trace, reader_trace])

    # Update layout
    fig.update_layout(
        showlegend=False,
        hovermode="closest",
        margin=dict(b=20, l=5, r=5, t=40),
        xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
    )

    # Save the plot
    fig.write_html(f"{output_dir}/ablation_network.html")


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
        device=device,
        abridge_ablations_to=args.abridge_ablations_to,
    )

    print("Analyzing results...")
    analyze_ablation_matrix(ablation_matrix_eE, ablater_sae, reader_sae)
    graph_ablation_matrix(
        ablation_matrix_eE,
        ablater_sae.cfg.neuronpedia_id,
        reader_sae.cfg.neuronpedia_id,
        "/tmp",
    )


if __name__ == "__main__":
    main(make_parser().parse_args())
