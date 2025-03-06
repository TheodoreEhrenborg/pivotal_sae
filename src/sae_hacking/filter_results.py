#!/usr/bin/env python3
from argparse import ArgumentParser, Namespace

from beartype import beartype

from sae_hacking.safetensor_utils import load_dict_with_tensors, save_dict_with_tensors


@beartype
def make_parser() -> ArgumentParser:
    parser = ArgumentParser()
    parser.add_argument("--input_path", required=True)
    parser.add_argument("--output_path", required=True)
    parser.add_argument("--num-keys-to-keep", type=int, required=True)
    return parser


@beartype
def main(args: Namespace) -> None:
    tensors = load_dict_with_tensors(args.input_path)
    filtered_tensors = dict(list(tensors.items())[: args.num_keys_to_keep])
    save_dict_with_tensors(filtered_tensors, args.output_path)


if __name__ == "__main__":
    main(make_parser().parse_args())
