#!/usr/bin/env python3
from argparse import ArgumentParser, Namespace

import numpy as np
from beartype import beartype
from huggingface_hub import hf_hub_download
from scipy.cluster import hierarchy

# Gemma-scope based on https://colab.research.google.com/drive/17dQFYUYnuKnP6OwQPH9v_GSYUW5aj-Rp
# Neuronpedia API based on https://colab.research.google.com/github/jbloomAus/SAELens/blob/main/tutorials/tutorial_2_0.ipynb


@beartype
def make_parser() -> ArgumentParser:
    parser = ArgumentParser()
    return parser


@beartype
def main(args: Namespace) -> None:
    path_to_params_A = hf_hub_download(
        repo_id="google/gemma-scope-2b-pt-res",
        filename="layer_20/width_16k/average_l0_71/params.npz",
        force_download=False,
    )

    params = np.load(path_to_params_A)
    decoder_vectors_EM = params["W_dec"]
    if args.abridge:
        decoder_vectors_EM = decoder_vectors_EM[0 : args.abridge]
    E = decoder_vectors_EM.shape[0]
    print(f"{decoder_vectors_EM.shape=}")
    Z = hierarchy.linkage(decoder_vectors_EM, "complete")


if __name__ == "__main__":
    main(make_parser().parse_args())
