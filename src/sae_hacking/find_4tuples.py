#!/usr/bin/env python3
from argparse import ArgumentParser, Namespace

import torch
from beartype import beartype
from tqdm import tqdm

from sae_hacking.safetensor_utils import load_dict_with_tensors


@beartype
def find_pattern(tensor_dict: dict) -> list[tuple[int, int, int, int]]:
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
        pos_indices = torch.where(reader_tensor > 0)[0]
        neg_indices = torch.where(reader_tensor < 0)[0]

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
                        # Calculate min abs value for this 4-tuple
                        AB_value = abs(tensor_dict[A][B])
                        BC_value = abs(tensor_dict[C][B])
                        AD_value = abs(tensor_dict[A][D])
                        CD_value = abs(tensor_dict[C][D])
                        min_abs_value = min(AB_value, BC_value, AD_value, CD_value)

                        results.append((A, B, C, D, min_abs_value))

        # Sort by min abs value and keep top 1000 after each A iteration
        results.sort(
            key=lambda x: x[4], reverse=True
        )  # Assuming larger values are better
        results = results[:1000]

    # Final result should be just the tuples without the min_abs_value
    final_results = [(a, b, c, d) for a, b, c, d, _ in results]

    return final_results


@beartype
def make_parser() -> ArgumentParser:
    parser = ArgumentParser()
    parser.add_argument("--input-path", required=True)
    return parser


@beartype
def main(args: Namespace) -> None:
    tensor_dict = load_dict_with_tensors(args.input_path)
    find_pattern(tensor_dict)


if __name__ == "__main__":
    main(make_parser().parse_args())
