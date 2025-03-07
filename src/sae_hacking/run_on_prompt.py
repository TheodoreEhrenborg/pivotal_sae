#!/usr/bin/env python3

import time
from argparse import ArgumentParser, Namespace
from ast import literal_eval
from pathlib import Path

import torch
from beartype import beartype
from sae_lens import SAE, HookedSAETransformer
from transformers import AutoTokenizer

from sae_hacking.timeprint import timeprint


@beartype
def highlight_tokens_with_intensity(
    split_text: list[str], activations: torch.Tensor
) -> str:
    html_parts = []

    for token, activation in zip(split_text, activations, strict=True):
        activation = min(activation, 30)

        red = int(255 - (activation * 8))
        green = 255
        blue = int(255 - (activation * 8))

        color = f"#{red:02x}{green:02x}{blue:02x}"

        highlighted = f'<span style="background-color: {color};">{token}</span>'
        html_parts.append(highlighted)

    return "".join(html_parts)


@beartype
def create_html(
    split_text: list[str],
    activations: torch.Tensor,
    sae_id: str,
    sae_release: str,
    prompt: str,
    feature_idx: int,
) -> str:
    html_output = highlight_tokens_with_intensity(split_text, activations)

    full_html = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <title>Highlighted Text Example</title>
        <meta charset="UTF-8">
        <style>
            body {{
                font-family: Arial, sans-serif;
                font-size: 16px;
                line-height: 1.5;
                margin: 20px;
            }}
        </style>
    </head>
    <body>
        <h1>Green Intensity Highlighting Example</h1>
        <p>{html_output}</p>
        <hr>
        <p>{sae_id=}</p>
        <hr>
        <p>{sae_release=}</p>
        <hr>
        <p>{prompt}</p>
        <hr>
        <p>{feature_idx}</p>
        <hr>
        <p>Activations: {list(zip(split_text, activations.tolist(), strict=True))}</p>
    </body>
    </html>
    """

    return full_html


@beartype
def get_feature_activation_per_token(
    model: HookedSAETransformer,
    sae: SAE,
    feature_idx: int,
    prompt: str,
) -> torch.Tensor:
    """
    Returns an array showing how much a specific SAE feature activated on each token of the prompt.

    Args:
        model: The transformer model with SAE hooks
        sae: The SAE to analyze
        feature_idx: Index of the specific feature to track
        prompt: The input text prompt

    Returns:
        Tensor of shape [num_tokens] containing activation values for the specified feature
        across all tokens in the prompt
    """
    # Ensure the SAE uses its error term for accurate activation measurement
    sae.use_error_term = True

    # Reset the model and SAEs to ensure clean state
    model.reset_hooks()
    model.reset_saes()

    # Run the model with the SAE to get activations
    _, cache = model.run_with_cache_with_saes(prompt, saes=[sae])

    # Get the SAE activations from the cache
    # Shape: [batch_size, sequence_length, n_features]
    sae_acts = cache[f"{sae.cfg.hook_name}.hook_sae_acts_post"]

    # Extract activations for the specified feature across all tokens
    # Assuming batch_size is 1, we take the first batch with sae_acts[0]
    feature_acts = sae_acts[0, :, feature_idx]

    return feature_acts


@beartype
def make_parser() -> ArgumentParser:
    parser = ArgumentParser()
    parser.add_argument("--model", default="google/gemma-2-2b")
    parser.add_argument("--sae-release", required=True)
    parser.add_argument("--sae-id", required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--feature-idx", required=True, type=int)
    parser.add_argument("--prompt", required=True)
    parser.add_argument("--output-dir", required=True, type=Path)
    return parser


def maybe_get(old_value, name):
    strng = input(f"Enter new value for {name} (currently {old_value}): ")
    if strng == "":
        return old_value
    return literal_eval(strng)


def run_once(
    model: HookedSAETransformer,
    tokenizer: AutoTokenizer,
    sae_id: str,
    sae_release: str,
    device: str,
    prompt: str,
    feature_idx: int,
    output_dir: Path,
) -> None:
    sae, _, _ = SAE.from_pretrained(release=sae_release, sae_id=sae_id, device=device)
    timeprint("Loaded SAE")
    activations_S = get_feature_activation_per_token(model, sae, feature_idx, prompt)
    timeprint("Got activations")

    split_text = tokenizer.tokenize(prompt, add_special_tokens=True)

    html_output = create_html(
        split_text, activations_S, sae_id, sae_release, prompt, feature_idx
    )
    with open(output_dir / time.strftime("%Y%m%d_%H%M%S_prompt.html"), "w") as f:
        f.write(html_output)


@torch.inference_mode()
@beartype
def main(args: Namespace) -> None:
    timeprint("Starting")
    model = HookedSAETransformer.from_pretrained(args.model, device=args.device)
    timeprint("Loaded model")

    tokenizer = AutoTokenizer.from_pretrained(args.model)
    timeprint("Loaded tokenizer")

    sae_id = args.sae_id
    sae_release = args.sae_release
    prompt = args.prompt
    feature_idx = args.feature_idx
    while True:
        run_once(
            model,
            tokenizer,
            sae_id,
            sae_release,
            args.device,
            prompt,
            feature_idx,
            args.output_dir,
        )
        sae_id = maybe_get(sae_id, "sae_id")
        sae_release = maybe_get(sae_release, "sae_release")
        prompt = maybe_get(prompt, "prompt")
        feature_idx = maybe_get(feature_idx, "feature_idx")


if __name__ == "__main__":
    main(make_parser().parse_args())
