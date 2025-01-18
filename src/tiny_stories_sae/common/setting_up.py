#!/usr/bin/env python3

from argparse import ArgumentParser

from beartype import beartype
from datasets import DatasetDict, load_dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    GPT2TokenizerFast,
    GPTNeoForCausalLM,
)

from tiny_stories_sae.common.sae import SparseAutoEncoder


@beartype
def make_base_parser() -> ArgumentParser:
    parser = ArgumentParser()
    parser.add_argument("--cuda", action="store_true")
    parser.add_argument("--sae_hidden_dim", type=int, default=100)
    parser.add_argument("--max_step", type=float, default=float("inf"))
    return parser


@beartype
def make_dataset(tokenizer: GPT2TokenizerFast) -> DatasetDict:
    d = load_dataset("roneneldan/TinyStories")

    def tokenize(example):
        return {"input_ids": tokenizer(example["text"])["input_ids"]}

    tokenized_datasets = d.map(tokenize)
    return tokenized_datasets.filter(lambda x: len(x["input_ids"]) != 0)


@beartype
def setup(
    sae_hidden_dim: int, cuda: bool, no_internet: bool
) -> tuple[DatasetDict, GPTNeoForCausalLM, SparseAutoEncoder, GPT2TokenizerFast]:
    llm = AutoModelForCausalLM.from_pretrained(
        "roneneldan/TinyStories-33M", local_files_only=no_internet
    )
    if cuda:
        llm.cuda()
    tokenizer = AutoTokenizer.from_pretrained(
        "EleutherAI/gpt-neo-125M", local_files_only=no_internet
    )
    tokenizer.pad_token = tokenizer.eos_token
    filtered_datasets = make_dataset(tokenizer)
    sae = SparseAutoEncoder(sae_hidden_dim)
    if cuda:
        sae.cuda()
    return filtered_datasets, llm, sae, tokenizer
