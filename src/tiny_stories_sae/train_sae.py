#!/usr/bin/env python3
from argparse import ArgumentParser, Namespace

import torch
from beartype import beartype
from coolname import generate_slug
from jaxtyping import Float, jaxtyped
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from tiny_stories_sae.common.obtain_activations import (
    get_llm_activation,
    normalize_activations,
)
from tiny_stories_sae.common.setting_up import make_base_parser, setup


@beartype
def make_parser() -> ArgumentParser:
    parser = make_base_parser()
    parser.add_argument("--l1_coefficient", type=float, default=0.0)
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
        sae_act, feat_magnitudes = sae(norm_act)
        rec_loss = get_reconstruction_loss(norm_act, sae_act)
        l1_penalty, nonzero_proportion = get_l1_penalty_nonzero(feat_magnitudes)
        loss = rec_loss + user_args.l1_coefficient * l1_penalty
        loss.backward()
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
        writer.add_scalar("Total loss/train", loss, step)
        writer.add_scalar(
            "Total loss per element/train", loss / torch.numel(norm_act), step
        )
        writer.add_scalar("Reconstruction loss/train", rec_loss, step)
        writer.add_scalar(
            "Reconstruction loss per element/train",
            rec_loss / torch.numel(norm_act),
            step,
        )
        writer.add_scalar("L1 penalty/train", l1_penalty, step)
        writer.add_scalar(
            "L1 penalty per element/train", l1_penalty / torch.numel(norm_act), step
        )
        writer.add_scalar("L1 penalty strength", user_args.l1_coefficient, step)
        writer.add_scalar("Proportion of nonzero features", nonzero_proportion, step)

    writer.close()


@jaxtyped(typechecker=beartype)
def get_reconstruction_loss(
    act: Float[torch.Tensor, "1 seq_len 768"],
    sae_act: Float[torch.Tensor, "1 seq_len 768"],
) -> Float[torch.Tensor, ""]:
    return ((act - sae_act) ** 2).sum()


@jaxtyped(typechecker=beartype)
def get_l1_penalty_nonzero(
    feat_magnitudes: Float[torch.Tensor, "1 seq_len sae_hidden_dim"],
) -> tuple[Float[torch.Tensor, ""], Float[torch.Tensor, ""]]:
    # Sum over the SAE features (i.e. a 1-norm)
    # And then sum over seq_len and batch
    l1 = torch.linalg.vector_norm(feat_magnitudes, ord=1)
    l0 = torch.linalg.vector_norm(feat_magnitudes, ord=0)
    return l1, l0 / torch.numel(feat_magnitudes)


if __name__ == "__main__":
    main(make_parser().parse_args())
