#!/usr/bin/env python3
import time
from argparse import ArgumentParser, Namespace
from pathlib import Path

import torch
from beartype import beartype
from coolname import generate_slug
from datasets import load_dataset
from jaxtyping import Float, jaxtyped
from sae_lens import SAE, HookedSAETransformer
from tqdm import tqdm
from transformers import AutoTokenizer

from sae_hacking.safetensor_utils import save_v2
from sae_hacking.timeprint import timeprint


@beartype
def make_parser() -> ArgumentParser:
    parser = ArgumentParser()
    parser.add_argument("--model", default="google/gemma-2-2b")
    parser.add_argument(
        "--ablator-sae-release", default="gemma-scope-2b-pt-res-canonical"
    )
    parser.add_argument("--ablator-sae-id", default="layer_20/width_65k/canonical")
    parser.add_argument("--max-tokens-in-prompt", type=int, default=125)
    parser.add_argument("--n-prompts", type=int, default=1)
    parser.add_argument("--save-frequency", type=int, default=240)
    return parser


@beartype
def generate_prompts(
    model: str, n_prompts: int, max_tokens_in_prompt: int
) -> list[str]:
    dataset = load_dataset("NeelNanda/pile-10k", split="train")
    tokenizer = AutoTokenizer.from_pretrained(model)
    prompts = [dataset[i]["text"] for i in range(n_prompts)]

    processed_prompts = []
    for prompt in prompts:
        tokenized_prompt_1S = tokenizer(prompt)["input_ids"]
        # Skip the BOS that the tokenizer adds
        processed_prompt = tokenizer.decode(
            tokenized_prompt_1S[1 : max_tokens_in_prompt + 1]
        )
        processed_prompts.append(processed_prompt)

    return processed_prompts


@jaxtyped(typechecker=beartype)
def compute_cooccurrences(
    model: HookedSAETransformer,
    ablator_sae: SAE,
    prompt: str,
    cooccurrences_ee: Float[torch.Tensor, "num_ablator_features num_ablator_features"],
) -> None:
    """
    - e: number of features in ablator SAE
    """
    timeprint(prompt)
    ablator_sae.use_error_term = True

    # Run the model with ablator SAE to get its activations
    model.reset_hooks()
    model.reset_saes()
    _, ablator_cache = model.run_with_cache_with_saes(prompt, saes=[ablator_sae])
    ablator_acts_1Se = ablator_cache[f"{ablator_sae.cfg.hook_name}.hook_sae_acts_post"]

    timeprint("Starting to update co-occurrence matrix")
    cooccurrences_ee += gather_co_occurrences2(ablator_acts_1Se)
    timeprint("Done updating co-occurrence matrix")


@torch.inference_mode()
@beartype
def main(args: Namespace) -> None:
    output_dir = f"/results/{time.strftime('%Y%m%d-%H%M%S')}{generate_slug()}"
    Path(output_dir).mkdir()
    timeprint(f"Writing to {output_dir}")
    device = "cuda"
    model = HookedSAETransformer.from_pretrained(args.model, device=device)

    ablator_sae, ablator_sae_config, _ = SAE.from_pretrained(
        release=args.ablator_sae_release, sae_id=args.ablator_sae_id, device=device
    )
    e = ablator_sae_config["d_sae"]
    prompts = generate_prompts(args.model, args.n_prompts, args.max_tokens_in_prompt)

    cooccurrences_ee = torch.zeros(e, e)
    for i, prompt in enumerate(tqdm(prompts)):
        timeprint("Computing co-occurrences...")
        compute_cooccurrences(model, ablator_sae, prompt, cooccurrences_ee)
        timeprint("Done computing co-occurrences")
        if i % args.save_frequency == 0 or i + 1 == len(prompts):
            save_v2(
                None,
                f"{output_dir}/{time.strftime('%Y%m%d-%H%M%S')}intermediate.safetensors.zst",
                cooccurrences_ee.to_dense(),
                None,
            )


@beartype
def gather_co_occurrences2(ablator_acts_1Se) -> torch.Tensor:
    # Convert to binary activation (1 where features are active, 0 otherwise)
    active_binary_Se = (ablator_acts_1Se[0] > 0).float().to_sparse()

    # Compute co-occurrences using matrix multiplication
    these_cooccurrences_ee = active_binary_Se.T @ active_binary_Se

    return these_cooccurrences_ee.cpu()


if __name__ == "__main__":
    main(make_parser().parse_args())
