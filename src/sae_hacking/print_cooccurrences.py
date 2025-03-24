#!/usr/bin/env python3
from argparse import ArgumentParser, Namespace

from beartype import beartype

from sae_hacking.safetensor_utils import load_v2


@beartype
def make_parser() -> ArgumentParser:
    parser = ArgumentParser(
        description="Print the co-occurrence value between two features"
    )
    parser.add_argument(
        "cooccurrence_path", help="Path to the co-occurrence matrix file"
    )
    parser.add_argument("feature1", type=int, help="Index of the first feature")
    parser.add_argument("feature2", type=int, help="Index of the second feature")
    return parser


@beartype
def main(args: Namespace) -> None:
    data = load_v2(args.cooccurrence_path)
    cooccurrences_ee = data["cooccurrences_ee"]
    cooccurrence = float(cooccurrences_ee[args.feature1, args.feature2])
    print(
        f"Co-occurrence between feature {args.feature1} and feature {args.feature2}: {cooccurrence}"
    )


if __name__ == "__main__":
    main(make_parser().parse_args())
