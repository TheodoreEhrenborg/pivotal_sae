#!/usr/bin/env python3
from argparse import ArgumentParser, Namespace
from pathlib import Path

import torch
from beartype import beartype
from tqdm import tqdm

from tiny_stories_sae.common.activation_analysis import (
    Sample,
    prune,
    write_activation_json,
)
from tiny_stories_sae.common.obtain_activations import (
    get_llm_activation,
    normalize_activations,
)
from tiny_stories_sae.common.setting_up import make_base_parser, setup


@beartype
def main(user_args: Namespace):
    filtered_datasets, llm, sae, tokenizer = setup(
        user_args.sae_hidden_dim, user_args.cuda, False
    )
    sae.load_state_dict(torch.load(user_args.checkpoint, weights_only=True))
    if user_args.cuda:
        sae.cuda()
    sae.eval()

    strongest_activations = [[] for _ in range(user_args.sae_hidden_dim)]
    with torch.no_grad():
        for step, example in enumerate(tqdm(filtered_datasets["validation"])):
            if step > user_args.max_step:
                break
            activation = get_llm_activation(llm, example, user_args)
            norm_act = normalize_activations(activation)
            _, feat_magnitudes = sae(norm_act)
            for feature_idx in range(user_args.sae_hidden_dim):
                strengths = feat_magnitudes[0, :, feature_idx].tolist()
                strongest_activations[feature_idx].append(
                    Sample(
                        step=step,
                        feature_idx=feature_idx,
                        tokens=example["input_ids"],
                        strengths=strengths,
                        max_strength=max(strengths),
                    )
                )
            strongest_activations = [
                prune(sample_list, user_args.samples_to_keep)
                for sample_list in strongest_activations
            ]
    output_path = Path(user_args.checkpoint).with_suffix(".json")
    write_activation_json(output_path, strongest_activations, tokenizer)


@beartype
def make_parser() -> ArgumentParser:
    parser = make_base_parser()
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--samples_to_keep", type=int, default=10)
    return parser


if __name__ == "__main__":
    main(make_parser().parse_args())
