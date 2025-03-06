#!/usr/bin/env python3
from argparse import ArgumentParser, Namespace

import torch
from beartype import beartype
from tqdm import tqdm

from sae_hacking.graph_network import NeuronExplanationLoader
from sae_hacking.safetensor_utils import load_dict_with_tensors


@beartype
def find_pattern(
    tensor_dict: dict, treat_as_zero: float
) -> list[tuple[int, int, int, int]]:
    """
    We need to find 4-tuples (A, B, C, D) where:

        A, C must be from one set
        B, D must be from the other set
        Edges AB, AD, BC are positive
        Edge CD is negative

    Without losing any 4-tuples, we can assume A & C are ablators

    """
    # Get bipartite partition

    results = []

    # Precompute positive and negative neighbors for each node
    pos_neighbors = {ablator_node: set() for ablator_node in tensor_dict}
    neg_neighbors = {ablator_node: set() for ablator_node in tensor_dict}

    for ablator_node in tqdm(tensor_dict):
        reader_tensor = tensor_dict[ablator_node]

        # Find indices where values are positive or negative using tensor operations
        pos_indices = torch.where(reader_tensor > treat_as_zero)[0]
        neg_indices = torch.where(reader_tensor < -treat_as_zero)[0]

        # Convert to sets in one operation
        pos_neighbors[ablator_node] = set(pos_indices.tolist())
        neg_neighbors[ablator_node] = set(neg_indices.tolist())

    for A in tqdm(tensor_dict):
        # Find all B where AB is positive
        pos_B_from_A = pos_neighbors[A]

        for C in tensor_dict:
            if A == C:
                continue

            # Find all B where BC is positive
            pos_B_from_C = pos_neighbors[C]

            # Find common B nodes: these can be the 'B' in our 4-tuple
            common_B = pos_B_from_A.intersection(pos_B_from_C)

            # Find all D where AD is positive and CD is negative
            potential_D = pos_neighbors[A].intersection(neg_neighbors[C])

            # Generate all valid 4-tuples
            for B in common_B:
                for D in potential_D:
                    if B != D:  # Ensure distinct nodes
                        results.append((A, B, C, D))

    return results


@beartype
def make_parser() -> ArgumentParser:
    parser = ArgumentParser()
    parser.add_argument("--input-path", required=True)
    parser.add_argument("--treat-as-zero", default=0.0, type=float)
    parser.add_argument("--ablator-sae-neuronpedia-id", required=True)
    parser.add_argument("--reader-sae-neuronpedia-id", required=True)
    return parser


@beartype
def construct_url(id: str, idx: int) -> str:
    return f"https://www.neuronpedia.org/{id}/{idx}"


@beartype
def process_results(
    results: list[tuple[int, int, int, int]],
    ablator_sae_id: str,
    reader_sae_id: str,
    tensor_dict: dict,
) -> None:
    ablator_descriptions = NeuronExplanationLoader(ablator_sae_id)
    reader_descriptions = NeuronExplanationLoader(reader_sae_id)

    for result in results:
        a, b, c, d = result

        print(f"A{a}: {ablator_descriptions.get_explanation(a)}")
        print(f"  See {construct_url(ablator_sae_id, a)}")
        print(f"R{b}: {reader_descriptions.get_explanation(b)}")
        print(f"  See {construct_url(reader_sae_id, b)}")
        print(f"A{c}: {ablator_descriptions.get_explanation(c)}")
        print(f"  See {construct_url(ablator_sae_id, c)}")
        print(f"R{d}: {reader_descriptions.get_explanation(d)}")
        print(f"  See {construct_url(reader_sae_id, d)}")

        print(f"A{a}'s effect on R{b}: {tensor_dict[a][b]}")
        print(f"A{a}'s effect on R{d}: {tensor_dict[a][d]}")
        print(f"A{c}'s effect on R{b}: {tensor_dict[c][b]}")
        print(f"A{c}'s effect on R{d}: {tensor_dict[c][d]}")

        print()


@beartype
def main(args: Namespace) -> None:
    tensor_dict = load_dict_with_tensors(args.input_path)
    results = find_pattern(tensor_dict, args.treat_as_zero)
    print(f"{len(results)=}")
    process_results(
        results,
        args.ablator_sae_neuronpedia_id,
        args.reader_sae_neuronpedia_id,
        tensor_dict,
    )


if __name__ == "__main__":
    main(make_parser().parse_args())
