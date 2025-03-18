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
    cooccurrence_threshold: int,
    cosine_sim_threshold: float,
    max_steps: int | None,
) -> list[tuple[int, int, float]]:
    """
    Find pairs of ablator latents that:
    1. Don't significantly co-occur (below cooccurrence_threshold)
    2. Have similar effects on reader SAEs (cosine similarity above cosine_sim_threshold)

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

        # Skip if we're at the last ablator
        if i >= num_ablators - 1:
            continue

        # Compute cosine similarity with ALL ablators at once via matmul
        # This avoids indexing with large arrays
        all_cosine_sims_e = torch.matmul(
            normalized_effects_eE, normalized_effects_eE[i]
        )

        # Create a mask for remaining indices (all after current i)
        remaining_mask_e = torch.zeros(
            num_ablators, dtype=torch.bool, device=effects_eE.device
        )
        remaining_mask_e[i + 1 : num_ablators] = True

        # Apply cooccurrence threshold to those remaining
        valid_cooccurrences_e = cooccurrences_ee[i] <= cooccurrence_threshold

        # Apply cosine similarity threshold
        valid_cosine_sims_e = all_cosine_sims_e >= cosine_sim_threshold

        # Combine all conditions
        combined_mask_e = torch.logical_and(
            torch.logical_and(remaining_mask_e, valid_cooccurrences_e),
            valid_cosine_sims_e,
        )

        # Get the valid indices
        valid_indices = torch.where(combined_mask_e)[0]

        if len(valid_indices) == 0:
            continue

        # Get the corresponding cosine similarities
        cosine_sims_D = all_cosine_sims_e[valid_indices]

        # Convert to CPU for processing
        valid_indices_cpu = valid_indices.cpu()
        cosine_sims_cpu_D = cosine_sims_D.cpu()

        # Add all valid pairs to the list
        for idx in range(len(valid_indices_cpu)):
            similar_pairs.append(
                (i, int(valid_indices_cpu[idx]), float(cosine_sims_cpu_D[idx]))
            )

    # Sort by cosine similarity (highest first)
    similar_pairs.sort(key=lambda x: x[2], reverse=True)
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
    parser.add_argument(
        "--cosine-sim-threshold",
        type=float,
        default=0.0,
        help="Only keep pairs with cosine similarity above this threshold",
    )
    parser.add_argument("--ablator-sae-neuronpedia-id", required=True)
    parser.add_argument(
        "--max-steps", type=int, help="Maximum number of pair comparisons to perform"
    )
    return parser


@beartype
def process_results(
    results: list[tuple[int, int, float]],
    ablator_sae_id: str,
    cooccurrences: torch.Tensor,
) -> None:
    ablator_descriptions = NeuronExplanationLoader(ablator_sae_id)

    timeprint(f"Found {len(results)} similar non-co-occurring pairs")
    print()

    for i, (ablator1, ablator2, cosine_sim) in enumerate(results):
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
        args.cooccurrence_threshold,
        args.cosine_sim_threshold,
        max_steps=args.max_steps,
    )

    # Process and display results
    process_results(results, args.ablator_sae_neuronpedia_id, cooccurrences_ee)


if __name__ == "__main__":
    main(make_parser().parse_args())
