#!/usr/bin/env python3
from argparse import ArgumentParser, Namespace

import torch
import torch.nn.functional as F
from beartype import beartype
from tqdm import tqdm

from sae_hacking.neuronpedia_utils import NeuronExplanationLoader, construct_url
from sae_hacking.safetensor_utils import load_v2
from sae_hacking.timeprint import timeprint


@beartype
def find_similar_noncooccurring_pairs(
    effects_eE: torch.Tensor,
    cooccurrences_ee: torch.Tensor,
    top_n: int,
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
    num_ablators = effects_eE.shape[0]

    timeprint("Beginning to normalize")
    normalized_effects_eE = F.normalize(torch.sign(effects_eE), dim=1).cuda()
    timeprint("Done normalizing")

    # Process in batches for each ablator
    for i in tqdm(range(num_ablators)):
        if max_steps is not None and i >= max_steps:
            timeprint(f"Reached maximum steps ({max_steps}). Stopping early.")
            break

        # Get all ablators after the current one
        remaining_indices = torch.arange(i + 1, num_ablators, device=effects_eE.device)

        if len(remaining_indices) == 0:
            continue

        # Check cooccurrence threshold for all pairs at once
        valid_cooccurrence_mask = (
            cooccurrences_ee[i, remaining_indices] <= cooccurrence_threshold
        )
        valid_indices = remaining_indices[valid_cooccurrence_mask]

        if len(valid_indices) == 0:
            continue

        # Compute cosine similarity for all valid pairs at once
        cosine_sims_D = torch.matmul(
            normalized_effects_eE[valid_indices], normalized_effects_eE[i]
        )

        # Convert to CPU for processing
        valid_indices_cpu = valid_indices.cpu()
        cosine_sims_cpu_D = cosine_sims_D.cpu()

        # Add all valid pairs to the list
        for idx, j in enumerate(valid_indices_cpu):
            similar_pairs.append((i, int(j), float(cosine_sims_cpu_D[idx])))

        # Sort by cosine sim and keep only top_n
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
    parser.add_argument("--top-n", type=int, default=1000, help="Keep top N results")
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
    data = load_v2(args.input_path)

    effects_eE = data["effects_eE"]
    cooccurrences_ee = data["cooccurrences_ee"]

    # Find similar non-co-occurring pairs
    timeprint("Finding similar non-co-occurring pairs...")
    results = find_similar_noncooccurring_pairs(
        effects_eE,
        cooccurrences_ee,
        args.top_n,
        args.cooccurrence_threshold,
        max_steps=args.max_steps,
    )

    # Process and display results
    process_results(
        results, args.ablator_sae_neuronpedia_id, cooccurrences_ee, args.top_n
    )


if __name__ == "__main__":
    main(make_parser().parse_args())
