#!/usr/bin/env python3
import asyncio
from argparse import ArgumentParser, Namespace

import aiohttp
import numpy as np
import plotly.express as px
from beartype import beartype
from huggingface_hub import hf_hub_download
from scipy.cluster import hierarchy

# Gemma-scope based on https://colab.research.google.com/drive/17dQFYUYnuKnP6OwQPH9v_GSYUW5aj-Rp
# Neuronpedia API based on https://colab.research.google.com/github/jbloomAus/SAELens/blob/main/tutorials/tutorial_2_0.ipynb


@beartype
def make_parser() -> ArgumentParser:
    parser = ArgumentParser()
    parser.add_argument("--abridge", type=int)
    return parser


@beartype
async def get_description_async(idx: int, session: aiohttp.ClientSession) -> str:
    url = f"https://www.neuronpedia.org/api/feature/gemma-2-2b/20-gemmascope-res-16k/{idx}"
    async with session.get(url) as response:
        data = await response.json()
        try:
            return data["explanations"][0]["description"]
        except:
            print(data)
            raise


@beartype
async def get_all_descriptions(indices: list[int]) -> list[str]:
    async with aiohttp.ClientSession() as session:
        tasks = [get_description_async(idx, session) for idx in indices]
        return await asyncio.gather(*tasks)


# TODO Claude wrote this. It's the sort of leetcode thing that Claude is good at,
# but I should check it
def convert_linkage_to_treemap(Z, labels=None):
    """
    Convert scipy linkage matrix Z to lists of names and parents for plotly treemap

    Parameters:
    Z : ndarray
        The linkage matrix
    labels : list, optional
        The original observation labels. If None, will use '0', '1', etc.

    Returns:
    names : list
        List of all node names
    parents : list
        List of parent names for each node
    """
    n_samples = len(Z) + 1

    # If no labels provided, use indices as strings
    if labels is None:
        labels = [str(i) for i in range(n_samples)]

    names = labels.copy()  # Start with original labels
    parents = [""] * n_samples  # Original observations have no parents

    # Process each merger in the linkage matrix
    for i, row in enumerate(Z):
        # New cluster label
        new_cluster = f"Cluster_{i + n_samples}"

        # Add new cluster to names
        names.append(new_cluster)

        # Get indices of children
        left_child = int(row[0])
        right_child = int(row[1])

        # Convert indices to names
        left_name = names[left_child]
        right_name = names[right_child]

        # Set parent for both children
        idx_left = names.index(left_name)
        idx_right = names.index(right_name)
        parents[idx_left] = new_cluster
        parents[idx_right] = new_cluster

        # New cluster has no parent unless it gets merged later
        parents.append("")

    return names, parents


@beartype
def main(args: Namespace) -> None:
    path_to_params = hf_hub_download(
        repo_id="google/gemma-scope-2b-pt-res",
        filename="layer_20/width_16k/average_l0_71/params.npz",
        force_download=False,
    )

    params = np.load(path_to_params)
    decoder_vectors_EM = params["W_dec"]
    if args.abridge:
        decoder_vectors_EM = decoder_vectors_EM[0 : args.abridge]
    E = decoder_vectors_EM.shape[0]
    print(f"{decoder_vectors_EM.shape=}")
    Z = hierarchy.linkage(decoder_vectors_EM, "complete")

    descriptions = asyncio.run(get_all_descriptions(list(range(E))))

    names, parents = convert_linkage_to_treemap(
        Z, labels=[f"{i}: {d}" for i, d in enumerate(descriptions)]
    )
    fig = px.treemap(names=names, parents=parents)
    fig.update_traces(root_color="lightgrey")
    fig.update_layout(margin=dict(t=50, l=25, r=25, b=25))
    fig.write_html("/results/treemap.html")


if __name__ == "__main__":
    main(make_parser().parse_args())
