#!/usr/bin/env python3
from argparse import ArgumentParser, Namespace

import torch
from beartype import beartype
from coolname import generate_slug
from datasets import DatasetDict
from jaxtyping import Float, jaxtyped
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    GPT2TokenizerFast,
    GPTNeoForCausalLM,
)

from sae_hacking.common.obtain_activations import (
    get_llm_activation,
    normalize_activations,
)
from sae_hacking.common.sae import TopkSparseAutoEncoder
from sae_hacking.common.setting_up import make_base_parser, make_dataset


@beartype
def setup(
    sae_hidden_dim: int, cuda: bool, no_internet: bool
) -> tuple[DatasetDict, GPTNeoForCausalLM, TopkSparseAutoEncoder, GPT2TokenizerFast]:
    # TODO DRY with the original setup function
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
    sae = TopkSparseAutoEncoder(sae_hidden_dim)
    if cuda:
        sae.cuda()
    return filtered_datasets, llm, sae, tokenizer


@beartype
def make_parser() -> ArgumentParser:
    parser = make_base_parser()
    parser.add_argument(
        "--only_count_tokens",
        action="store_true",
        help="Skip training and just log how many tokens were seen",
    )
    return parser


@beartype
def main(user_args: Namespace):
    output_dir = f"/results/{generate_slug()}"
    print(f"Writing to {output_dir}")
    writer = SummaryWriter(output_dir)

    filtered_datasets, llm, sae, _ = setup(
        user_args.sae_hidden_dim, user_args.cuda, False
    )

    lr = 1e-5
    optimizer = torch.optim.Adam(sae.parameters(), lr=lr)

    tokens_seen = 0
    for step, example in enumerate(tqdm(filtered_datasets["train"])):
        if step > user_args.max_step:
            break
        tokens_seen += len(example["input_ids"])
        writer.add_scalar("Tokens seen/train", tokens_seen, step)
        if user_args.only_count_tokens:
            continue
        optimizer.zero_grad()
        activation = get_llm_activation(llm, example, user_args)
        norm_act = normalize_activations(activation)
        sae_act = sae(norm_act)
        rec_loss = get_reconstruction_loss(norm_act, sae_act)
        rec_loss.backward()
        optimizer.step()
        if step % 5000 == 0:
            torch.save(sae.state_dict(), f"{output_dir}/{step}.pt")

        writer.add_scalar("act mean/train", activation.mean(), step)
        writer.add_scalar("act std/train", activation.std(), step)
        writer.add_scalar("lr", lr, step)
        writer.add_scalar("sae_hidden_dim", user_args.sae_hidden_dim, step)
        writer.add_scalar("norm act mean/train", norm_act.mean(), step)
        writer.add_scalar("norm act std/train", norm_act.std(), step)
        writer.add_scalar("sae act mean/train", sae_act.mean(), step)
        writer.add_scalar("sae act std/train", sae_act.std(), step)
        writer.add_scalar("Total loss/train", rec_loss, step)
        writer.add_scalar(
            "Total loss per element/train", rec_loss / torch.numel(norm_act), step
        )
        writer.add_scalar("Reconstruction loss/train", rec_loss, step)
        writer.add_scalar(
            "Reconstruction loss per element/train",
            rec_loss / torch.numel(norm_act),
            step,
        )

    writer.close()


@jaxtyped(typechecker=beartype)
def get_reconstruction_loss(
    act: Float[torch.Tensor, "1 seq_len 768"],
    sae_act: Float[torch.Tensor, "1 seq_len 768"],
) -> Float[torch.Tensor, ""]:
    # TODO DRY this
    return ((act - sae_act) ** 2).sum()


if __name__ == "__main__":
    main(make_parser().parse_args())
