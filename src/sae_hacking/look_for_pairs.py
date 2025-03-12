#!/usr/bin/env python3
from argparse import ArgumentParser, Namespace

import torch
import torch.nn.functional as F
from beartype import beartype
from tqdm import tqdm

from sae_hacking.neuronpedia_utils import NeuronExplanationLoader, construct_url
from sae_hacking.safetensor_utils import load_dict_with_tensors


@beartype
def find_similar_noncooccurring_pairs(
    tensor_dict: dict,
    cooccurrences: torch.Tensor,
    cosine_threshold: float,
    cooccurrence_threshold: int,
) -> list[tuple[int, int, float]]:
    """
    Find pairs of ablator latents that:
    1. Don't significantly co-occur (below cooccurrence_threshold)
    2. Have similar effects on reader SAEs (cosine similarity above threshold)

    Returns a list of tuples (ablator1, ablator2, cosine_similarity)
    """
    similar_pairs = []
    ablator_ids = sorted(list(tensor_dict.keys()))

    # Check each pair of ablators
    for i, ablator1 in enumerate(tqdm(ablator_ids)):
        for ablator2 in ablator_ids[i + 1 :]:
            # Skip if they co-occur frequently
            if cooccurrences[ablator1, ablator2] > cooccurrence_threshold:
                continue

            # Compute cosine similarity of their effects on reader SAEs
            cosine_sim = F.cosine_similarity(
                tensor_dict[ablator1], tensor_dict[ablator2]
            ).item()

            # Keep if similarity is high enough
            if cosine_sim >= cosine_threshold:
                similar_pairs.append((ablator1, ablator2, cosine_sim))

    # Sort by similarity (highest first)
    similar_pairs.sort(key=lambda x: x[2], reverse=True)
    return similar_pairs


@beartype
def make_parser() -> ArgumentParser:
    parser = ArgumentParser()
    parser.add_argument(
        "--input-path",
        required=True,
    )
    parser.add_argument(
        "--cosine-threshold",
        type=float,
        default=0.8,
        help="Minimum cosine similarity to keep",
    )
    parser.add_argument(
        "--cooccurrence-threshold",
        type=int,
        default=0,
        help="Throw away any pairs that co-occur more than this",
    )
    parser.add_argument("--ablator-sae-neuronpedia-id", required=True)
    parser.add_argument("--reader-sae-neuronpedia-id", required=True)
    parser.add_argument("--top-n", type=int, default=100, help="Show top N results")
    return parser


@beartype
def process_results(
    results: list[tuple[int, int, float]],
    ablator_sae_id: str,
    reader_sae_id: str,
    tensor_dict: dict,
    cooccurrences: torch.Tensor,
    top_n: int,
) -> None:
    ablator_descriptions = NeuronExplanationLoader(ablator_sae_id)

    print(f"Found {len(results)} similar non-co-occurring pairs")
    print(f"Showing top {min(top_n, len(results))} results:")
    print()

    for i, (ablator1, ablator2, cosine_sim) in enumerate(results[:top_n]):
        print(f"Pair {i + 1}: Ablator {ablator1} and Ablator {ablator2}")
        print(f"  Cosine similarity: {cosine_sim:.4f}")
        print(f"  Co-occurrence count: {cooccurrences[ablator1, ablator2]}")

        print(f"  Ablator {ablator1}: {ablator_descriptions.get_explanation(ablator1)}")
        print(f"  Ablator {ablator2}: {ablator_descriptions.get_explanation(ablator2)}")

        print(f"  URLs: {construct_url(ablator_sae_id, ablator1)}")
        print(f"        {construct_url(ablator_sae_id, ablator2)}")
        print()


@beartype
def main(args: Namespace) -> None:
    print("Loading file")
    tensor_dict, cooccurrences = load_dict_with_tensors(args.input_path)

    # Find similar non-co-occurring pairs
    print("Finding similar non-co-occurring pairs...")
    results = find_similar_noncooccurring_pairs(
        tensor_dict, cooccurrences, args.cosine_threshold, args.cooccurrence_threshold
    )

    # Process and display results
    process_results(
        results,
        args.ablator_sae_neuronpedia_id,
        args.reader_sae_neuronpedia_id,
        tensor_dict,
        cooccurrences,
        args.top_n,
    )


if __name__ == "__main__":
    main(make_parser().parse_args())
