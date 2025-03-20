#!/usr/bin/env python3
import time
from argparse import ArgumentParser, Namespace
from pathlib import Path

import torch
from beartype import beartype
from coolname import generate_slug
from datasets import IterableDataset, load_dataset
from jaxtyping import Float, jaxtyped
from sae_lens import SAE, HookedSAETransformer
from tqdm import tqdm
from transformers import AutoTokenizer

from sae_hacking.gemma_utils import gather_co_occurrences2
from sae_hacking.safetensor_utils import save_v2
from sae_hacking.timeprint import timeprint


@beartype
def generate_prompts(
    model: str, n_prompts: int, max_tokens_in_prompt: int, dataset_id: str
) -> IterableDataset:
    dataset = load_dataset(dataset_id, split="train", streaming=True)
    tokenizer = AutoTokenizer.from_pretrained(model)

    def preprocess_function(example):
        tokenized_prompt = tokenizer(example["text"])["input_ids"]
        # Skip the BOS that the tokenizer adds and limit to max_tokens_in_prompt
        processed_prompt = tokenizer.decode(
            tokenized_prompt[1 : max_tokens_in_prompt + 1]
        )
        return {"processed_text": processed_prompt}

    processed_dataset = dataset.map(preprocess_function)

    if n_prompts is not None:
        processed_dataset = processed_dataset.take(n_prompts)

    return processed_dataset


@beartype
def make_parser() -> ArgumentParser:
    parser = ArgumentParser()
    parser.add_argument("--model", default="google/gemma-2-2b")
    parser.add_argument(
        "--ablator-sae-release", default="gemma-scope-2b-pt-res-canonical"
    )
    parser.add_argument("--ablator-sae-id", default="layer_20/width_65k/canonical")
    parser.add_argument("--dataset-id", required=True)
    parser.add_argument("--max-tokens-in-prompt", type=int, default=125)
    parser.add_argument("--n-prompts", type=int, default=1)
    parser.add_argument("--save-frequency", type=int, default=240)
    return parser


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
    prompts = generate_prompts(
        args.model, args.n_prompts, args.max_tokens_in_prompt, args.dataset_id
    )

    cooccurrences_ee = torch.zeros(e, e)
    for i, prompt in enumerate(tqdm(prompts)):
        timeprint("Computing co-occurrences...")
        compute_cooccurrences(
            model, ablator_sae, prompt["processed_text"], cooccurrences_ee
        )
        timeprint("Done computing co-occurrences")
        if i % args.save_frequency == 0 or i + 1 == args.n_prompts:
            save_v2(
                None,
                f"{output_dir}/{time.strftime('%Y%m%d-%H%M%S')}intermediate.safetensors.zst",
                cooccurrences_ee.to_dense(),
                None,
            )


if __name__ == "__main__":
    main(make_parser().parse_args())
