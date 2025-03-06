#!/usr/bin/env python3
from argparse import ArgumentParser, Namespace

from beartype import beartype

from sae_hacking.json_utils import load_dict_with_tensors_from_json
from sae_hacking.safetensor_utils import save_dict_with_tensors


@beartype
def make_parser() -> ArgumentParser:
    parser = ArgumentParser()
    parser.add_argument("input_path")
    parser.add_argument("output_path")
    return parser


@beartype
def main(args: Namespace) -> None:
    tensors = load_dict_with_tensors_from_json(args.input_path)
    save_dict_with_tensors(tensors, args.output_path)


if __name__ == "__main__":
    main(make_parser().parse_args())
