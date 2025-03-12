#!/usr/bin/env python3
from argparse import ArgumentParser, Namespace

import torch
import torch.nn.functional as F
from beartype import beartype
from tqdm import tqdm

from sae_hacking.neuronpedia_utils import NeuronExplanationLoader, construct_url
from sae_hacking.safetensor_utils import load_dict_with_tensors
from sae_hacking.timeprint import timeprint


@beartype
def find_similar_noncooccurring_pairs(
    tensor_dict: dict,
    cooccurrences_DD: torch.Tensor,
    cosine_threshold: float,
    cooccurrence_threshold: int,
    max_steps: int | None,
) -> list[tuple[int, int, float]]:
    """
    Find pairs of ablator latents that:
    1. Don't significantly co-occur (below cooccurrence_threshold)
    2. Have similar effects on reader SAEs (cosine similarity above threshold)

    Returns a list of tuples (ablator1, ablator2, cosine_similarity)
    """
    similar_pairs = []
    ablator_ids = sorted(list(tensor_dict.keys()))

    for ablator in tensor_dict:
        tensor_dict[ablator] = tensor_dict[ablator].cuda()

    D = len(tensor_dict)
    M = tensor_dict[ablator_ids[0]].shape[0]
    all_ablators_DM = torch.zeros(D, M, device="cuda")
    for i, ablator in enumerate(tqdm(ablator_ids)):
        all_ablators_DM[i] = tensor_dict[ablator]

    # Check each ablator against remaining ablators in batch
    for i, ablator1 in enumerate(tqdm(ablator_ids)):
        if max_steps is not None and i >= max_steps:
            timeprint(f"Reached maximum steps ({max_steps}). Stopping early.")
            break

        # Skip if there are no more ablators to compare
        if i + 1 >= len(ablator_ids):
            continue

        # Get all subsequent ablators for batch processing
        next_ablators = ablator_ids[i + 1 :]
        next_indices = [ablator_ids.index(a) for a in next_ablators]

        # Get current ablator tensor and reshape for batch comparison
        current_tensor = all_ablators_DM[i].unsqueeze(0)  # [1, M]

        # Get all comparison tensors
        comparison_tensors = all_ablators_DM[next_indices]  # [num_remaining, M]

        # Compute cosine similarities in one batch operation
        cosine_sims = F.cosine_similarity(
            current_tensor,  # [1, M]
            comparison_tensors,  # [num_remaining, M]
            dim=1,
        )

        # Process the results
        for j, ablator2 in enumerate(next_ablators):
            # Skip if they co-occur frequently
            if cooccurrences_DD[ablator1, ablator2] > cooccurrence_threshold:
                continue

            # Check if similarity is high enough
            sim_value = cosine_sims[j].item()
            if sim_value >= cosine_threshold:
                similar_pairs.append((ablator1, ablator2, sim_value))

    # Sort by similarity (highest first)
    similar_pairs.sort(key=lambda x: x[2], reverse=True)
    return similar_pairs


@beartype
def make_parser() -> ArgumentParser:
    parser = ArgumentParser()
    parser.add_argument("--input-path", required=True)
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
    parser.add_argument("--top-n", type=int, default=100, help="Show top N results")
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
    timeprint("Loading file")
    tensor_dict, cooccurrences_DD = load_dict_with_tensors(args.input_path)

    # Find similar non-co-occurring pairs
    timeprint("Finding similar non-co-occurring pairs...")
    results = find_similar_noncooccurring_pairs(
        tensor_dict,
        cooccurrences_DD,
        args.cosine_threshold,
        args.cooccurrence_threshold,
        max_steps=args.max_steps,
    )

    # Process and display results
    process_results(
        results, args.ablator_sae_neuronpedia_id, cooccurrences_DD, args.top_n
    )


if __name__ == "__main__":
    main(make_parser().parse_args())
