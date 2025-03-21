#!/usr/bin/env python3
import datetime
import os
from argparse import ArgumentParser, Namespace

import torch
import torch.nn.functional as F
from beartype import beartype
from jaxtyping import Float, jaxtyped
from sae_lens import SAE

from sae_hacking.neuronpedia_utils import NeuronExplanationLoader, construct_url
from sae_hacking.safetensor_utils import load_v2
from sae_hacking.timeprint import timeprint


@jaxtyped(typechecker=beartype)
def compute_decoder_similarities(
    decoder_eD: Float[torch.Tensor, "e D"],
    cooccurrences_ee: Float[torch.Tensor, "e e"],
    top_k: int,
) -> list[tuple[int, int, float]]:
    """
    Compute cosine similarities between decoder rows (features), mask out pairs
    with nonzero co-occurrence, and return the top_k highest similarities.

    Args:
        decoder_eD: Decoder matrix from the SAE (e features, D dimensions)
        cooccurrences_ee: Co-occurrence matrix (e by e)
        top_k: Number of top similar pairs to return

    Returns:
        List of tuples (feature1, feature2, cosine_similarity) for top_k pairs
    """
    timeprint("Normalizing decoder vectors")
    normalized_decoder_eD = F.normalize(decoder_eD, dim=1)

    timeprint("Computing cosine similarities")
    # Compute all cosine similarities
    cosine_sims_ee = torch.matmul(normalized_decoder_eD, normalized_decoder_eD.T)

    # Create mask for pairs with zero co-occurrence
    timeprint("Creating co-occurrence mask")
    mask_ee = cooccurrences_ee == 0

    # Set diagonal to False (don't want self-similarities)
    mask_ee.fill_diagonal_(False)

    # Apply mask to cosine similarities
    timeprint("Applying mask")
    masked_cosine_sims_ee = cosine_sims_ee.clone()
    masked_cosine_sims_ee[~mask_ee] = -1.0  # Set masked values to -1

    # Find indices of top_k highest values
    timeprint(f"Finding top {top_k} similarities")
    # Flatten to find top values
    flat_sims = masked_cosine_sims_ee.view(-1)
    top_values, top_indices = torch.topk(flat_sims, k=top_k)

    # Convert flat indices back to 2D
    e = decoder_eD.shape[0]
    row_indices = top_indices // e
    col_indices = top_indices % e

    # Create result list
    similar_pairs = []
    for i in range(top_k):
        feature1 = int(row_indices[i])
        feature2 = int(col_indices[i])
        similarity = float(top_values[i])
        similar_pairs.append((feature1, feature2, similarity))

    return similar_pairs


@beartype
def make_parser() -> ArgumentParser:
    parser = ArgumentParser()
    parser.add_argument(
        "--ablator-sae-release", default="gemma-scope-2b-pt-res-canonical"
    )
    parser.add_argument("--ablator-sae-id", default="layer_20/width_65k/canonical")
    parser.add_argument(
        "--cooccurrence-path", required=True, help="Path to co-occurrence matrix file"
    )
    parser.add_argument(
        "--top-k", type=int, default=1000, help="Number of top similarities to find"
    )
    return parser


@beartype
def process_results(
    results: list[tuple[int, int, float]], neuronpedia_id: str, filename: str
) -> None:
    """Process and save the results to a file."""
    feature_descriptions = NeuronExplanationLoader(neuronpedia_id)

    with open(filename, "w") as f:
        f.write(f"Top {len(results)} similar decoder vectors with zero co-occurrence\n")
        f.write("\n")

        for i, (feature1, feature2, cosine_sim) in enumerate(results):
            f.write(f"Pair {i + 1}: Feature {feature1} and Feature {feature2}\n")
            f.write(f"  Cosine similarity of decoder vectors: {cosine_sim:.4f}\n")

            # Get feature descriptions
            desc1 = feature_descriptions.get_explanation(feature1)
            desc2 = feature_descriptions.get_explanation(feature2)

            f.write(f"  Feature {feature1}: {desc1}\n")
            f.write(f"  Feature {feature2}: {desc2}\n")

            # Add URLs to Neuronpedia
            f.write(f"  URLs: {construct_url(neuronpedia_id, feature1)}\n")
            f.write(f"        {construct_url(neuronpedia_id, feature2)}\n")
            f.write("\n")

    timeprint(f"Results saved to {filename}")


@beartype
def main(args: Namespace) -> None:
    # Create results directory if it doesn't exist
    os.makedirs("/results", exist_ok=True)

    # Generate filename with timestamp
    current_time = datetime.datetime.now()
    filename = (
        f"/results/{current_time.strftime('%Y%m%d_%H%M')}_decoder_similarities.txt"
    )

    # Print output filename at start
    print(f"Output will be saved to: {filename}")

    # Get device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    timeprint("Loading SAE file")
    ablator_sae, ablator_sae_config, _ = SAE.from_pretrained(
        release=args.ablator_sae_release, sae_id=args.ablator_sae_id, device=device
    )

    # Get the neuronpedia_id from the sae_config
    neuronpedia_id = ablator_sae_config.neuronpedia_id

    # Extract decoder matrix
    decoder_eD = ablator_sae.W_dec

    timeprint("Loading co-occurrence matrix")

    cooccurrence_data = load_v2(args.cooccurrence_path)

    if "cooccurrences_ee" in cooccurrence_data:
        cooccurrences_ee = cooccurrence_data["cooccurrences_ee"]
    else:
        raise ValueError("Co-occurrence matrix not found in file")

    # Ensure dimensions match
    e = decoder_eD.shape[0]
    if cooccurrences_ee.shape != (e, e):
        raise ValueError(
            f"Dimension mismatch: decoder has {e} features but co-occurrence is {cooccurrences_ee.shape}"
        )

    # Compute similarities and find top pairs
    timeprint("Computing decoder similarities")
    top_pairs = compute_decoder_similarities(decoder_eD, cooccurrences_ee, args.top_k)

    # Process and save results
    timeprint("Processing results")
    process_results(top_pairs, neuronpedia_id, filename)

    timeprint("Done!")


if __name__ == "__main__":
    main(make_parser().parse_args())
