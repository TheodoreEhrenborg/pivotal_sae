#!/usr/bin/env python3
from argparse import ArgumentParser, Namespace

from beartype import beartype

from sae_hacking.safetensor_utils import load_v2, save_v2


@beartype
def make_parser() -> ArgumentParser:
    parser = ArgumentParser()
    parser.add_argument("--input-path", required=True)
    parser.add_argument("--output-path", required=True)
    parser.add_argument("--abridge", type=int, required=True)
    return parser


@beartype
def main(args: Namespace) -> None:
    # Load tensors from input path
    tensors = load_v2(args.input_path)

    # Extract tensors
    effects_eE = tensors["effects_eE"]
    cooccurrences_ee = tensors["cooccurrences_ee"]
    how_often_activated_e = tensors["how_often_activated_e"]

    # Abridge tensors: keep only the first args.abridge elements
    abridged_effects_eE = effects_eE[: args.abridge]
    abridged_cooccurrences_ee = cooccurrences_ee[: args.abridge, : args.abridge]
    abridged_how_often_activated_e = how_often_activated_e[: args.abridge]

    # Save abridged tensors to output path
    save_v2(
        effects_eE=abridged_effects_eE,
        save_path=args.output_path,
        cooccurrences_ee=abridged_cooccurrences_ee,
        how_often_activated_e=abridged_how_often_activated_e,
    )

    print("Original tensor shapes:")
    print(f"  effects_eE: {effects_eE.shape}")
    print(f"  cooccurrences_ee: {cooccurrences_ee.shape}")
    print(f"  how_often_activated_e: {how_often_activated_e.shape}")

    print(f"Abridged tensor shapes (keeping first {args.abridge} elements):")
    print(f"  effects_eE: {abridged_effects_eE.shape}")
    print(f"  cooccurrences_ee: {abridged_cooccurrences_ee.shape}")
    print(f"  how_often_activated_e: {abridged_how_often_activated_e.shape}")


if __name__ == "__main__":
    main(make_parser().parse_args())
