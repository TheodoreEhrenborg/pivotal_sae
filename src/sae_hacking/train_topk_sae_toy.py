#!/usr/bin/env python3
import time
from argparse import ArgumentParser, Namespace

import matplotlib.pyplot as plt
import seaborn as sns
import torch
import torch.nn.functional as F
from beartype import beartype
from coolname import generate_slug
from einops import rearrange, reduce
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
    parser.add_argument("--sae-hidden-dim", type=int, default=100)
    parser.add_argument("--dataset-num-features", type=int, default=100)
    parser.add_argument("--max-step", type=int, default=100000)
    parser.add_argument("--lr", type=float, default=1e-5)
    parser.add_argument("--hierarchical", action="store_true")
    parser.add_argument("--batch-size", type=int, default=1)
    return parser


@beartype
def main(user_args: Namespace):
    output_dir = f"/results/{time.strftime('%Y%m%d-%H%M%S')}{generate_slug()}"
    print(f"Writing to {output_dir}")
    writer = SummaryWriter(output_dir)
    dataset = ToyDataset(
        seed=1, num_features=user_args.dataset_num_features, cuda=user_args.cuda
    )

    sae = setup(
        user_args.sae_hidden_dim,
        user_args.cuda,
        False,
        user_args.hierarchical,
    )

    lr = user_args.lr
    optimizer = torch.optim.Adam(sae.parameters(), lr=lr)

    for step in trange(user_args.max_step):
        example, num_activated_features = dataset.generate(user_args.batch_size)
        optimizer.zero_grad()
        reconstructed, num_live_latents = sae(example)
        rec_loss = get_reconstruction_loss(reconstructed, example)
        rec_loss.backward()
        optimizer.step()
        if step % 5000 == 0:
            torch.save(sae.state_dict(), f"{output_dir}/{step}.pt")
            save_similarity_graph(sae, dataset, output_dir, step)

        writer.add_scalar(
            "Activated ground-truth features", num_activated_features, step
        )
        if step % 10 == 0:
            writer.add_scalar(
                "Min_{feature}( Max_{decoder vector} cosine sim)",
                min_max_cosine_similarity(sae, dataset),
                step,
            )
            writer.add_scalar(
                "Mean_{feature}( Max_{decoder vector} cosine sim)",
                mean_max_cosine_similarity(sae, dataset),
                step,
            )
        writer.add_scalar("lr", lr, step)
        num_dead_latents = user_args.sae_hidden_dim - num_live_latents
        writer.add_scalar("Num dead latents", num_dead_latents, step)
        writer.add_scalar(
            "Proportion of dead latents",
            num_dead_latents / user_args.sae_hidden_dim,
            step,
        )
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
    act: Float[torch.Tensor, "batch_size model_dim"],
    sae_act: Float[torch.Tensor, "batch_size model_dim"],
) -> Float[torch.Tensor, ""]:
    return ((act - sae_act) ** 2).mean()


@jaxtyped(typechecker=beartype)
def calculate_cosine_sim(
    decoder_weights: Float[torch.Tensor, "model_dim sae_dim"],
    all_child_vecs: Float[torch.Tensor, "total_num_children model_dim"],
) -> Float[torch.Tensor, "sae_dim total_num_children"]:
    return F.cosine_similarity(
        rearrange(decoder_weights, "model_dim sae_dim -> sae_dim 1 model_dim"),
        rearrange(
            all_child_vecs,
            "total_num_children model_dim -> 1 total_num_children model_dim",
        ),
        dim=2,
    )


@jaxtyped(typechecker=beartype)
def get_similarity(sae, dataset) -> Float[torch.Tensor, "sae_dim total_num_children"]:
    child_vecs = (
        rearrange(dataset.features, "n_features model_dim -> n_features 1 model_dim")
        + dataset.perturbations
    )
    all_child_vecs = rearrange(
        child_vecs,
        "n_features children_per_parent model_dim -> (n_features children_per_parent) model_dim",
    )
    return calculate_cosine_sim(sae.decoder.weight, all_child_vecs)


@jaxtyped(typechecker=beartype)
def min_max_cosine_similarity(sae, dataset) -> Float[torch.Tensor, ""]:
    similarity = get_similarity(sae, dataset)
    per_feature_sim = reduce(
        similarity, "sae_dim total_num_children -> total_num_children", "max"
    )
    return per_feature_sim.min()


@jaxtyped(typechecker=beartype)
def mean_max_cosine_similarity(sae, dataset) -> Float[torch.Tensor, ""]:
    similarity = get_similarity(sae, dataset)
    per_feature_sim = reduce(
        similarity, "sae_dim total_num_children -> total_num_children", "max"
    )
    return per_feature_sim.mean()


def save_similarity_graph(sae, dataset, output_dir, step):
    similarity = get_similarity(sae, dataset)

    # Create heatmap
    plt.figure(figsize=(12, 5))
    sns.heatmap(
        similarity.cpu().detach().numpy(),
        cmap="RdYlBu_r",
        center=0,
        vmin=-1,
        vmax=1,
        square=True,
    )

    # Add labels
    plt.title(f"Decoder Weights vs Child Vectors Similarity, step {step}")
    plt.xlabel("Child Vectors (Child 1 | Child 2)")
    plt.ylabel("Decoder Weight Vectors")

    plt.tight_layout()
    plt.savefig(
        f"{output_dir}/similarity_heatmap{step}.png", dpi=300, bbox_inches="tight"
    )
    plt.close()


if __name__ == "__main__":
    main(make_parser().parse_args())
