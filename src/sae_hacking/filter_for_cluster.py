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
def find_similar_feature_group(
    start_feature: int,
    decoder_eD: Float[torch.Tensor, "e D"],
    cooccurrences_ee: Float[torch.Tensor, "e e"],
    min_cosine_sim: float,
    group_size: int,
    neuronpedia_id: str,
) -> list[int]:
    """
    Find a group of features where each pair has cosine similarity >= min_cosine_sim
    and no pair has co-occurrence > 0, excluding features with explanation starting with "No explanation found".

    Args:
        start_feature: Index of the feature to start with
        decoder_eD: Decoder matrix from the SAE (e features, D dimensions)
        cooccurrences_ee: Co-occurrence matrix (e by e)
        min_cosine_sim: Minimum cosine similarity threshold
        group_size: Number of features to find (including start_feature)
        neuronpedia_id: ID for the Neuronpedia database

    Returns:
        List of feature indices that form a group with the required properties
    """
    timeprint(
        f"Finding a group of {group_size} features starting with feature {start_feature}"
    )

    # Load feature explanations
    feature_descriptions = NeuronExplanationLoader(neuronpedia_id)

    # Check if start feature has valid explanation
    start_feature_desc = feature_descriptions.get_explanation(start_feature)
    if start_feature_desc.startswith("No explanation found"):
        raise ValueError("start feature doesn't have a valid explanation")

    # Normalize decoder vectors
    normalized_decoder_eD = F.normalize(decoder_eD, dim=1)

    # Compute all cosine similarities
    cosine_sims_ee = torch.matmul(normalized_decoder_eD, normalized_decoder_eD.T)

    # Initialize result with the start feature
    selected_features = [start_feature]

    # Continue until we have enough features or no more candidates
    while len(selected_features) < group_size:
        # Filter for candidates that satisfy constraints with all selected features
        candidates = set(range(decoder_eD.shape[0]))
        candidates.difference_update(
            selected_features
        )  # Remove already selected features

        # Remove candidates with no explanation or "No explanation found"
        candidates = {
            idx
            for idx in candidates
            if not feature_descriptions.get_explanation(idx).startswith(
                "No explanation found"
            )
        }

        # Check cosine similarity and co-occurrence constraints for each candidate
        for selected in selected_features:
            # Remove candidates with cosine similarity < threshold
            low_sim_candidates = set(
                idx.item()
                for idx in torch.where(cosine_sims_ee[selected] < min_cosine_sim)[0]
            )
            candidates.difference_update(low_sim_candidates)

            # Remove candidates with non-zero co-occurrence
            nonzero_cooccur_candidates = set(
                idx.item() for idx in torch.where(cooccurrences_ee[selected] > 0)[0]
            )
            candidates.difference_update(nonzero_cooccur_candidates)

        if not candidates:
            timeprint(
                f"Could only find {len(selected_features)} features meeting criteria"
            )
            break

        # From remaining candidates, select the one with highest average cosine similarity to already selected
        best_candidate = None
        best_avg_sim = -1.0

        for candidate in candidates:
            sims = [
                cosine_sims_ee[candidate, selected].item()
                for selected in selected_features
            ]
            avg_sim = sum(sims) / len(sims)
            if avg_sim > best_avg_sim:
                best_avg_sim = avg_sim
                best_candidate = candidate

        if best_candidate is not None:
            selected_features.append(best_candidate)
            timeprint(
                f"Added feature {best_candidate} (avg sim: {best_avg_sim:.4f}), group size: {len(selected_features)}"
            )
        else:
            break

    return selected_features


@beartype
def make_parser() -> ArgumentParser:
    parser = ArgumentParser()
    parser.add_argument("--sae-release", default="gemma-scope-2b-pt-res-canonical")
    parser.add_argument("--sae-id", default="layer_20/width_65k/canonical")
    parser.add_argument(
        "--cooccurrence-path", required=True, help="Path to co-occurrence matrix file"
    )
    parser.add_argument(
        "--start-feature", type=int, required=True, help="Feature index to start with"
    )
    parser.add_argument(
        "--min-cosine-sim",
        type=float,
        default=0.4,
        help="Minimum cosine similarity threshold",
    )
    parser.add_argument(
        "--group-size",
        type=int,
        default=10,
        help="Number of features to find (including start feature)",
    )
    return parser


@beartype
def process_results(
    features: list[int],
    cosine_sims_ee: Float[torch.Tensor, "e e"],
    neuronpedia_id: str,
    filename: str,
) -> None:
    """Process and save the results to a file."""
    feature_descriptions = NeuronExplanationLoader(neuronpedia_id)

    with open(filename, "w") as f:
        f.write(
            f"Group of {len(features)} features with similar decoder vectors and zero co-occurrence\n\n"
        )

        # Print individual features
        f.write("FEATURES:\n")
        for i, feature in enumerate(features):
            desc = feature_descriptions.get_explanation(feature)
            f.write(f"Feature {i + 1}: {feature}\n")
            f.write(f"  Description: {desc}\n")
            f.write(f"  URL: {construct_url(neuronpedia_id, feature)}\n\n")

        # Print cosine similarities between all pairs
        f.write("PAIRWISE COSINE SIMILARITIES:\n")
        for i, feature1 in enumerate(features):
            for j, feature2 in enumerate(features):
                if i < j:  # Only upper triangle
                    sim = cosine_sims_ee[feature1, feature2].item()
                    f.write(f"Features {feature1} and {feature2}: {sim:.4f}\n")

    timeprint(f"Results saved to {filename}")


@beartype
def main(args: Namespace) -> None:
    # Create results directory if it doesn't exist
    os.makedirs("/results", exist_ok=True)

    # Generate filename with timestamp
    current_time = datetime.datetime.now()
    filename = f"/results/{current_time.strftime('%Y%m%d_%H%M')}_feature_group.txt"

    # Print output filename at start
    print(f"Output will be saved to: {filename}")

    # Get device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    timeprint("Loading SAE file")
    sae, sae_config, _ = SAE.from_pretrained(
        release=args.sae_release, sae_id=args.sae_id, device=device
    )

    # Get the neuronpedia_id from the sae_config
    neuronpedia_id = sae_config["neuronpedia_id"]

    # Extract decoder matrix
    decoder_eD = sae.W_dec

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

    # Validate start feature index
    if args.start_feature < 0 or args.start_feature >= e:
        raise ValueError(
            f"Start feature index {args.start_feature} is out of range [0, {e - 1}]"
        )

    # Find group of similar features
    timeprint("Finding group of similar features")
    feature_group = find_similar_feature_group(
        args.start_feature,
        decoder_eD,
        cooccurrences_ee,
        args.min_cosine_sim,
        args.group_size,
        neuronpedia_id,
    )

    # Compute cosine similarities for output
    normalized_decoder_eD = F.normalize(decoder_eD, dim=1)
    cosine_sims_ee = torch.matmul(normalized_decoder_eD, normalized_decoder_eD.T)

    # Process and save results
    timeprint("Processing results")
    process_results(feature_group, cosine_sims_ee, neuronpedia_id, filename)

    timeprint("Done!")


if __name__ == "__main__":
    main(make_parser().parse_args())
