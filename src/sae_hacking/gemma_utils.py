#!/usr/bin/env python3


import torch
from beartype import beartype
from datasets import IterableDataset, load_dataset
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


@beartype
def generate_prompts2(
    model: str,
    n_prompts: int,
    max_tokens_in_prompt: int,
    dataset_id: str,
    batch_size: int,
) -> IterableDataset:
    dataset = load_dataset(dataset_id, split="train", streaming=True)
    tokenizer = AutoTokenizer.from_pretrained(model)

    if n_prompts is not None:
        abridged_dataset = dataset.take(n_prompts)

    def preprocess_function(examples):
        # Process a batch of examples at once
        tokenized_prompts = tokenizer(
            examples["text"],
            return_tensors="pt",
            padding="max_length",
            max_length=max_tokens_in_prompt,
            truncation=True,
        )["input_ids"]

        # Hack to change the dataset's batch size
        # https://discuss.huggingface.co/t/streaming-batched-data/21603
        return {k: [v] for k, v in examples.items()} | {
            "abridged_tensor": [tokenized_prompts]
        }

    processed_dataset = abridged_dataset.map(
        preprocess_function,
        batched=True,
        batch_size=batch_size,
        remove_columns=dataset.column_names,
    )

    return processed_dataset
