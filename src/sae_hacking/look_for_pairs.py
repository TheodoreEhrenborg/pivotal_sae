#!/usr/bin/env python3
from argparse import ArgumentParser, Namespace

import torch
from beartype import beartype
from tqdm import tqdm

from sae_hacking.neuronpedia_utils import NeuronExplanationLoader, construct_url
from sae_hacking.safetensor_utils import load_dict_with_tensors
from sae_hacking.timeprint import timeprint


@beartype
def find_similar_noncooccurring_pairs(
    tensor_dict: dict,
    cooccurrences_DD: torch.Tensor,
    top_n: int,
    cooccurrence_threshold: int,
    max_steps: int | None,
) -> list[tuple[int, int, float]]:
    """
    Find pairs of ablator latents that:
    1. Don't significantly co-occur (below cooccurrence_threshold)
    2. Have similar effects on reader SAEs (dot product above threshold)

    Returns a list of tuples (ablator1, ablator2, dot_product)
    """
    similar_pairs = []
    ablator_ids = sorted(list(tensor_dict.keys()))

    for key in tensor_dict:
        tensor_dict[key] = torch.sign(tensor_dict[key].cuda())

    # Check each pair of ablators
    for i, ablator1 in enumerate(tqdm(ablator_ids)):
        if max_steps is not None and i >= max_steps:
            timeprint(f"Reached maximum steps ({max_steps}). Stopping early.")
            break
        for ablator2 in ablator_ids[i + 1 :]:
            # Skip if they co-occur frequently
            if cooccurrences_DD[ablator1, ablator2] > cooccurrence_threshold:
                continue

            # Compute dot product of their effects on reader SAEs
            dot_prod = torch.dot(tensor_dict[ablator1], tensor_dict[ablator2]).item()

            similar_pairs.append((ablator1, ablator2, dot_prod))

        # Sort by dot product and abridge
        similar_pairs.sort(key=lambda x: x[2], reverse=True)
        similar_pairs = similar_pairs[:top_n]

    return similar_pairs


@beartype
def make_parser() -> ArgumentParser:
    parser = ArgumentParser()
    parser.add_argument("--input-path", required=True)
    parser.add_argument(
        "--cooccurrence-threshold",
        type=int,
        default=0,
        help="Throw away any pairs that co-occur more than this",
    )
    parser.add_argument("--ablator-sae-neuronpedia-id", required=True)
    parser.add_argument("--top-n", type=int, default=100, help="Keep top N results")
    parser.add_argument(
        "--max-steps", type=int, help="Maximum number of pair comparisons to perform"
    )
    return parser


@beartype
def process_results(
    results: list[tuple[int, int, float]],
    ablator_sae_id: str,
    cooccurrences: torch.Tensor,
    top_n: int,
) -> None:
    ablator_descriptions = NeuronExplanationLoader(ablator_sae_id)

    timeprint(f"Found {len(results)} similar non-co-occurring pairs")
    timeprint(f"Showing top {min(top_n, len(results))} results:")
    print()

    for i, (ablator1, ablator2, dot_prod) in enumerate(results[:top_n]):
        print(f"Pair {i + 1}: Ablator {ablator1} and Ablator {ablator2}")
        print(f"  Dot product: {dot_prod:.4f}")
        print(f"  Co-occurrence count: {cooccurrences[ablator1, ablator2]}")

        print(f"  Ablator {ablator1}: {ablator_descriptions.get_explanation(ablator1)}")
        print(f"  Ablator {ablator2}: {ablator_descriptions.get_explanation(ablator2)}")

        print(f"  URLs: {construct_url(ablator_sae_id, ablator1)}")
        print(f"        {construct_url(ablator_sae_id, ablator2)}")
        print()


@beartype
def main(args: Namespace) -> None:
    timeprint("Loading file")
    tensor_dict, cooccurrences_DD = load_dict_with_tensors(args.input_path)

    # Find similar non-co-occurring pairs
    timeprint("Finding similar non-co-occurring pairs...")
    results = find_similar_noncooccurring_pairs(
        tensor_dict,
        cooccurrences_DD,
        args.top_n,
        args.cooccurrence_threshold,
        max_steps=args.max_steps,
    )

    # Process and display results
    process_results(
        results, args.ablator_sae_neuronpedia_id, cooccurrences_DD, args.top_n
    )


if __name__ == "__main__":
    main(make_parser().parse_args())
