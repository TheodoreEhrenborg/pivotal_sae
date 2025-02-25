#!/usr/bin/env python3

# Copied from https://colab.research.google.com/github/jbloomAus/SAELens/blob/main/tutorials/tutorial_2_0.ipynb

import argparse

import pandas as pd
from sae_lens.toolkit.pretrained_saes_directory import get_pretrained_saes_directory

parser = argparse.ArgumentParser()
parser.add_argument("--release")
args = parser.parse_args()

df = pd.DataFrame.from_records(
    {k: v.__dict__ for k, v in get_pretrained_saes_directory().items()}
).T
df.drop(
    columns=[
        "expected_var_explained",
        "expected_l0",
        "config_overrides",
        "conversion_func",
    ],
    inplace=True,
)
print(
    df
)  # Each row is a "release" which has multiple SAEs which may have different configs / match different hook points in a model.

if args.release:
    print(f"SAEs in the {args.release} release")
    for k, v in df.loc[df.release == args.release, "saes_map"].values[0].items():
        print(f"SAE id: {k} for hook point: {v}")
