#!/usr/bin/env python3
import time
from argparse import ArgumentParser, Namespace

import torch
from beartype import beartype
from coolname import generate_slug
from jaxtyping import Float, jaxtyped
from torch.utils.tensorboard import SummaryWriter
from tqdm import trange

from sae_hacking.common.sae import TopkSparseAutoEncoder2Child, TopkSparseAutoEncoder_v2
from sae_hacking.common.toy_dataset import ToyDataset


@beartype
def setup(
    sae_hidden_dim: int,
    cuda: bool,
    no_internet: bool,
    hierarchical: bool,
) -> TopkSparseAutoEncoder_v2 | TopkSparseAutoEncoder2Child:
    SAEClass = TopkSparseAutoEncoder2Child if hierarchical else TopkSparseAutoEncoder_v2
    print(SAEClass)
    sae = SAEClass(sae_hidden_dim)
    if cuda:
        sae.cuda()
    return sae


@beartype
def make_parser() -> ArgumentParser:
    parser = ArgumentParser()
    parser.add_argument("--cuda", action="store_true")
    parser.add_argument("--sae_hidden_dim", type=int, default=100)
    parser.add_argument("--max_step", type=int, default=100000)
    parser.add_argument("--lr", type=float, default=1e-5)
    parser.add_argument("--hierarchical", action="store_true")
    return parser


@beartype
def main(user_args: Namespace):
    output_dir = f"/results/{time.strftime('%Y%m%d-%H%M%S')}{generate_slug()}"
    print(f"Writing to {output_dir}")
    writer = SummaryWriter(output_dir)
    dataset = ToyDataset(seed=1)

    sae = setup(
        user_args.sae_hidden_dim,
        user_args.cuda,
        False,
        user_args.hierarchical,
    )

    lr = user_args.lr
    optimizer = torch.optim.Adam(sae.parameters(), lr=lr)

    for step in trange(user_args.max_step):
        example = dataset.generate()
        optimizer.zero_grad()
        reconstructed = sae(example)
        rec_loss = get_reconstruction_loss(reconstructed, example)
        rec_loss.backward()
        optimizer.step()
        if step % 5000 == 0:
            torch.save(sae.state_dict(), f"{output_dir}/{step}.pt")

        writer.add_scalar("lr", lr, step)
        writer.add_scalar("sae_hidden_dim", user_args.sae_hidden_dim, step)
        writer.add_scalar("Total loss/train", rec_loss, step)
        writer.add_scalar("Reconstruction loss/train", rec_loss, step)
        writer.add_scalar(
            "Reconstruction loss per element/train",
            rec_loss / torch.numel(example),
            step,
        )

    writer.close()


@jaxtyped(typechecker=beartype)
def get_reconstruction_loss(
    act: Float[torch.Tensor, "1 projected_dim"],
    sae_act: Float[torch.Tensor, "1 projected_dim"],
) -> Float[torch.Tensor, ""]:
    # TODO DRY this
    return ((act - sae_act) ** 2).sum()


if __name__ == "__main__":
    main(make_parser().parse_args())
