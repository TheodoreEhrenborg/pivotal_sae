#!/usr/bin/env python3
import datetime
import json
import os
from argparse import ArgumentParser, Namespace

import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F
from beartype import beartype
from jaxtyping import Float, jaxtyped
from tqdm import tqdm

from sae_hacking.neuronpedia_utils import NeuronExplanationLoader, construct_url
from sae_hacking.safetensor_utils import load_v2
from sae_hacking.timeprint import timeprint


@jaxtyped(typechecker=beartype)
def find_similar_noncooccurring_pairs(
    effects_eE: Float[torch.Tensor, "e E"],
    cooccurrences_ee: Float[torch.Tensor, "e e"],
    cooccurrence_threshold: int,
    cosine_sim_threshold: float,
    max_steps: int | None,
    just_these: list[int] | None,
    skip_torch_sign: bool,
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
    normalized_effects_eE = F.normalize(
        effects_eE if skip_torch_sign else torch.sign(effects_eE), dim=1
    ).cuda()
    timeprint("Done normalizing")

    # Process in batches for each ablator
    for i in tqdm(range(num_ablators)):
        # Skip if not in the list of specific ablators to process
        if just_these is not None and i not in just_these:
            continue

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

        # Apply cooccurrence threshold to those remaining
        valid_cooccurrences_e = cooccurrences_ee[i] <= cooccurrence_threshold

        # Apply cosine similarity threshold
        valid_cosine_sims_e = all_cosine_sims_e >= cosine_sim_threshold

        # Combine all conditions
        combined_mask_e = valid_cooccurrences_e & valid_cosine_sims_e.cpu()

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
        "--cooccurrence-path",
        help="If provided, load only the co-occurrence matrix from this path",
    )
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
        "--just-these",
        type=int,
        nargs="+",
        help="List of specific ablator indices to process, skipping all others",
    )
    parser.add_argument("--skip-torch-sign", action="store_true")
    parser.add_argument(
        "--max-steps", type=int, help="Maximum number of pair comparisons to perform"
    )
    return parser


@jaxtyped(typechecker=beartype)
def save_to_json(
    results: list[tuple[int, int, float]],
    ablator_sae_id: str,
    cooccurrences_ee: Float[torch.Tensor, "e e"],
    how_often_activated_e: Float[torch.Tensor, " e"],
    filename: str,
) -> None:
    ablator_descriptions = NeuronExplanationLoader(ablator_sae_id)

    # Create a dictionary to hold all the data
    output_data = {
        "meta": {"count": len(results), "ablator_sae_id": ablator_sae_id},
        "pairs": [],
    }

    # Process each result pair
    for ablator1, ablator2, cosine_sim in results:
        pair_data = {
            "ablator1": {
                "id": int(ablator1),
                "explanation": ablator_descriptions.get_explanation(ablator1),
                "activation_count": float(how_often_activated_e[ablator1]),
                "url": construct_url(ablator_sae_id, ablator1),
            },
            "ablator2": {
                "id": int(ablator2),
                "explanation": ablator_descriptions.get_explanation(ablator2),
                "activation_count": float(how_often_activated_e[ablator2]),
                "url": construct_url(ablator_sae_id, ablator2),
            },
            "cosine_similarity": float(cosine_sim),
            "cooccurrence_count": float(cooccurrences_ee[ablator1, ablator2]),
        }
        output_data["pairs"].append(pair_data)

    # Write the dictionary to a JSON file
    with open(filename, "w") as f:
        json.dump(output_data, f, indent=2)

    timeprint(f"Results saved to {filename}")


@beartype
def plot_similarity_histogram(
    results: list[tuple[int, int, float]], filename: str, title: str, log_scale: bool
) -> None:
    bins = 50
    # Extract the cosine similarities from the results
    similarities = [sim for _, _, sim in results]

    # Create the histogram
    plt.figure(figsize=(10, 6))
    plt.hist(similarities, bins=bins, color="skyblue", edgecolor="black", alpha=0.7)

    # Add labels and title
    plt.xlabel("Cosine Similarity")
    plt.ylabel("Frequency")
    plt.title(title)

    # Set y-axis to logarithmic scale if requested
    if log_scale:
        plt.yscale("log")

    # Save the plot
    plt.tight_layout()
    plt.savefig(filename)
    plt.close()

    timeprint(f"Histogram saved to {filename}")


@beartype
def main(args: Namespace) -> None:
    # Create the results directory if it doesn't exist
    os.makedirs("/results", exist_ok=True)

    # Generate filename with current timestamp
    current_time = datetime.datetime.now()
    base_filename = f"/results/{current_time.strftime('%Y%m%d_%H%M')}"
    json_filename = f"{base_filename}.json"
    histogram_filename = f"{base_filename}_histogram.png"

    # Print the output file's name at the very start
    print(f"Output will be saved to: {json_filename}")

    timeprint("Loading file")
    data = load_v2(args.input_path)

    effects_eE = data["effects_eE"]

    if args.cooccurrence_path:
        timeprint(f"Loading co-occurrence matrix from {args.cooccurrence_path}")
        cooccurrence_data = load_v2(args.cooccurrence_path)
        cooccurrences_ee = cooccurrence_data["cooccurrences_ee"]
    else:
        cooccurrences_ee = data["cooccurrences_ee"]

    # Find similar non-co-occurring pairs
    timeprint("Finding similar non-co-occurring pairs...")
    results = find_similar_noncooccurring_pairs(
        effects_eE,
        cooccurrences_ee,
        args.cooccurrence_threshold,
        args.cosine_sim_threshold,
        max_steps=args.max_steps,
        just_these=args.just_these,
        skip_torch_sign=args.skip_torch_sign,
    )

    save_to_json(
        results,
        args.ablator_sae_neuronpedia_id,
        cooccurrences_ee,
        data["how_often_activated_e"],
        json_filename,
    )

    plot_similarity_histogram(
        results,
        histogram_filename,
        title=f"Distribution of Cosine Similarities (threshold={args.cosine_sim_threshold})",
    )


if __name__ == "__main__":
    main(make_parser().parse_args())
