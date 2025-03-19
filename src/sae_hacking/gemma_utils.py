#!/usr/bin/env python3


import torch
from beartype import beartype
from datasets import load_dataset
from transformers import AutoTokenizer


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


@beartype
def gather_co_occurrences2(ablator_acts_1Se) -> torch.Tensor:
    # Convert to binary activation (1 where features are active, 0 otherwise)
    active_binary_Se = (ablator_acts_1Se[0] > 0).float().to_sparse()

    # Compute co-occurrences using matrix multiplication
    these_cooccurrences_ee = active_binary_Se.T @ active_binary_Se

    return these_cooccurrences_ee.cpu()
