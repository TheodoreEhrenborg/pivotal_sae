#!/usr/bin/env python3
import time
from argparse import ArgumentParser, Namespace

import matplotlib.pyplot as plt
import seaborn as sns
import torch
import torch.nn.functional as F
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
    parser.add_argument("--sae-hidden-dim", type=int, default=100)
    parser.add_argument("--dataset-num-features", type=int, default=100)
    parser.add_argument("--max-step", type=int, default=100000)
    parser.add_argument("--lr", type=float, default=1e-5)
    parser.add_argument("--hierarchical", action="store_true")
    return parser


@beartype
def main(user_args: Namespace):
    output_dir = f"/results/{time.strftime('%Y%m%d-%H%M%S')}{generate_slug()}"
    print(f"Writing to {output_dir}")
    writer = SummaryWriter(output_dir)
    dataset = ToyDataset(seed=1, num_features=user_args.dataset_num_features)

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
            save_similarity_graph(sae, dataset, output_dir, step)

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
    return ((act - sae_act) ** 2).mean()

@jaxtyped(typechecker=beartype)
def calculate_cosine_sim(decoder_weights_t: Float[torch.Tensor, "n_features model_dim"], all_child_vecs : Float[torch.Tensor, "total_num_children model_dim"]) -> Float[torch.Tensor, "n_features total_num_children"]:
    return F.cosine_similarity(
        decoder_weights_t.unsqueeze(1), all_child_vecs.unsqueeze(0), dim=2
    )

def save_similarity_graph(sae, dataset, output_dir, step):
    decoder_weights = sae.decoder.weight

    # Get parent vectors
    parent_vecs = dataset.features

    # Create normalized child vectors
    child_vecs = []
    for i in range(dataset.N_CHILDREN_PER_PARENT):
        children = parent_vecs + dataset.perturbations[:, i, :]
        child_vecs.append(children)

    # Concatenate children only
    all_child_vecs = torch.cat(child_vecs, dim=0)

    similarity = calculate_cosine_sim(torch.transpose(decoder_weights, 0,1), all_child_vecs)

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
    plt.savefig(f"{output_dir}/similarity_heatmap{step}.png", dpi=300, bbox_inches="tight")
    plt.close()


if __name__ == "__main__":
    main(make_parser().parse_args())
