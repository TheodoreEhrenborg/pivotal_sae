#!/usr/bin/env python3
from argparse import ArgumentParser, Namespace

import matplotlib.pyplot as plt
from beartype import beartype

from sae_hacking.safetensor_utils import load_dict_with_tensors
from sae_hacking.timeprint import timeprint


@beartype
def count_active_readers(
    tensor_dict: dict, threshold: float, exclude_over: float
) -> list[int]:
    """
    Count how many reader latents are active (above threshold) for each ablator latent.

    Args:
        tensor_dict: Dictionary mapping ablator IDs to their reader activation tensors
        threshold: Threshold above which a reader is considered active

    Returns:
        Tensor containing counts of active readers for each ablator
    """
    active_counts = []

    for ablator_id in sorted(tensor_dict.keys()):
        # Count values above threshold
        active_count = (tensor_dict[ablator_id].abs() > threshold).sum().item()
        if active_count <= exclude_over:
            active_counts.append(active_count)

    return active_counts


@beartype
def plot_histogram(active_counts: list[int], output_path: str, bins: int) -> None:
    """
    Plot and save a histogram of active reader counts.
    """
    plt.figure(figsize=(10, 6))

    plt.hist(active_counts, bins=bins, edgecolor="black")
    plt.xlabel("Number of Active Reader Latents")
    plt.ylabel("Count of Ablator Latents")
    plt.title("Distribution of Active Reader Latents per Ablator")

    # Add summary statistics
    plt.legend()

    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()


@beartype
def make_parser() -> ArgumentParser:
    parser = ArgumentParser()
    parser.add_argument("--input-path", required=True, help="Path to input tensor file")
    parser.add_argument(
        "--output-path", required=True, help="Path to save the histogram plot"
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.0,
        help="Threshold for considering a reader latent active",
    )
    parser.add_argument(
        "--bins", type=int, default=50, help="Number of bins in the histogram"
    )
    parser.add_argument("--exclude-over", type=float, default=5000.0)
    return parser


@beartype
def main(args: Namespace) -> None:
    timeprint("Loading tensor file...")
    tensor_dict, _ = load_dict_with_tensors(args.input_path)

    timeprint("Counting active readers for each ablator...")
    active_counts = count_active_readers(tensor_dict, args.threshold, args.exclude_over)

    timeprint("Creating histogram plot...")
    plot_histogram(active_counts, args.output_path, args.bins)

    timeprint(f"Plot saved to {args.output_path}")


if __name__ == "__main__":
    main(make_parser().parse_args())
