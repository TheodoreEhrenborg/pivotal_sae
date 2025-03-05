#!/usr/bin/env python3
from argparse import ArgumentParser, Namespace

from beartype import beartype

from sae_hacking.graph_network import graph_ablation_matrix
from sae_hacking.json_utils import load_dict_with_tensors_from_json


@beartype
def make_parser() -> ArgumentParser:
    parser = ArgumentParser()
    parser.add_argument("--ablater-sae-neuronpedia-id", required=True)
    parser.add_argument("--reader-sae-neuronpedia-id", required=True)
    parser.add_argument("--n-edges", type=int, default=1000)
    parser.add_argument("--output-dir", type=str, required=True)
    parser.add_argument("--json-path", type=str, required=True)
    return parser


@beartype
def main(args: Namespace) -> None:
    ablation_results = load_dict_with_tensors_from_json(args.json_path)

    graph_ablation_matrix(
        ablation_results,
        args.ablater_sae_neuronpedia_id,
        args.reader_sae_neuronpedia_id,
        args.output_dir,
        args.n_edges,
    )


if __name__ == "__main__":
    main(make_parser().parse_args())
