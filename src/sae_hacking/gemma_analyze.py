#!/usr/bin/env python3
from argparse import ArgumentParser, Namespace

from beartype import beartype

from sae_hacking.safetensor_utils import load_dict_with_tensors


@beartype
def make_parser() -> ArgumentParser:
    parser = ArgumentParser()
    parser.add_argument("--ablator-sae-neuronpedia-id", required=True)
    parser.add_argument("--reader-sae-neuronpedia-id", required=True)
    parser.add_argument("--n-edges", type=int, default=1000)
    parser.add_argument("--output-dir", type=str, required=True)
    parser.add_argument("--input-path", type=str, required=True)
    return parser


@beartype
def main(args: Namespace) -> None:
    ablation_results, cooccurrence_matrix = load_dict_with_tensors(args.input_path)
    assert cooccurrence_matrix is not None

    for ablator_latent1 in ablation_results:
        for ablator_latent2 in ablation_results:
            if ablator_latent1 == ablator_latent2:
                continue
            if cooccurrence_matrix[ablator_latent1, ablator_latent2]:
                continue
            # measure cosine sim of their effects on the reader SAEs
            # if the pair is high enough:
            #    add it to the list of ones to be kept


if __name__ == "__main__":
    main(make_parser().parse_args())
