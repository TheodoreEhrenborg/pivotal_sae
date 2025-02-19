#!/usr/bin/env python3
import time
from argparse import ArgumentParser, Namespace

import matplotlib.pyplot as plt
import seaborn as sns
import torch
import torch.nn.functional as F
import yaml
from beartype import beartype
from coolname import generate_slug
from einops import rearrange, reduce, repeat
from jaxtyping import Bool, Float, jaxtyped
from safetensors.torch import save_model
from torch.utils.tensorboard import SummaryWriter
from tqdm import trange

from sae_hacking.common.sae import (
    TopkSparseAutoEncoder2Child_v2,
    TopkSparseAutoEncoder_v2,
)
from sae_hacking.common.toy_dataset import ToyDataset

SomeSAE = TopkSparseAutoEncoder2Child_v2 | TopkSparseAutoEncoder_v2


@beartype
def setup(
    sae_hidden_dim: int,
    cuda: bool,
    hierarchical: bool,
    model_dim: int,
    aux_loss_coeff: float,
    k: int,
) -> TopkSparseAutoEncoder_v2 | TopkSparseAutoEncoder2Child_v2:
    SAEClass = (
        TopkSparseAutoEncoder2Child_v2 if hierarchical else TopkSparseAutoEncoder_v2
    )
    # TODO Pretty sure this is broken for the non-hierarchical SAE
    sae = SAEClass(sae_hidden_dim, model_dim, aux_loss_coeff, k)
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
    parser.add_argument("--handcode-sae", action="store_true")
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--perturbation_size", type=float, default=0.2)
    parser.add_argument("--model-dim", type=int, default=10)
    parser.add_argument("--aux-loss-coeff", type=float, default=0.0)
    parser.add_argument("--dataset-k", type=int, default=3)
    parser.add_argument("--sae-k", type=int, default=3)
    return parser


@beartype
def main(args: Namespace):
    torch.manual_seed(args.seed)
    output_dir = f"/results/{time.strftime('%Y%m%d-%H%M%S')}{generate_slug()}"
    print(f"Writing to {output_dir}")
    writer = SummaryWriter(output_dir)

    with open(f"{output_dir}/args.yaml", "w") as f:
        yaml.dump(vars(args), f, default_flow_style=False)

    dataset = ToyDataset(
        args.dataset_num_features,
        args.cuda,
        args.perturbation_size,
        args.model_dim,
        args.dataset_k,
    )
    plot_feature_similarity(dataset, output_dir)

    sae = setup(
        args.sae_hidden_dim,
        args.cuda,
        args.hierarchical,
        args.model_dim,
        args.aux_loss_coeff,
        args.sae_k,
    )
    if args.handcode_sae:
        assert args.hierarchical
        handcode_sae(sae, dataset)

    lr = args.lr
    optimizer = torch.optim.Adam(sae.parameters(), lr=lr)

    for step in trange(args.max_step):
        sae.train()
        example, num_activated_features = dataset.generate(args.batch_size)
        optimizer.zero_grad()
        reconstructed, _, aux_loss = sae(example)
        rec_loss = get_reconstruction_loss(reconstructed, example)
        total_loss = args.aux_loss_coeff * aux_loss + rec_loss
        total_loss.backward()
        optimizer.step()
        sae.eval()
        if step % 5000 == 0:
            save_model(sae, f"{output_dir}/{step}.safetensors")
            save_similarity_graph(sae, dataset, output_dir, step, args.hierarchical)
            save_sorted_similarity_graph(
                sae, dataset, output_dir, step, args.hierarchical
            )
            if args.hierarchical:
                save_latent_similarity_graph(sae, output_dir, step)
                plot_norms(sae, step, output_dir, dataset)
            save_legible_similarity_graph(
                sae, dataset, output_dir, step, args.hierarchical
            )
            with torch.no_grad():
                val_example, _ = dataset.generate(10000)
                _, num_live_latents, _ = sae(val_example)
                log_dead_latents(
                    num_live_latents,
                    args.hierarchical,
                    writer,
                    step,
                    args.sae_hidden_dim,
                )

        # py-spy claims the next line is slow,
        # but I don't see an improvement when I take it out
        writer.add_scalar(
            "Activated ground-truth features",
            num_activated_features / args.batch_size,
            step,
        )
        # If we calculate this every step,
        # roughly 40% of the runtime is spent doing that
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
            if args.hierarchical:
                writer.add_scalar(
                    "Adjusted Min_{feature}( Max_{decoder vector} cosine sim)",
                    adjusted_min_max_cosine_similarity(sae, dataset),
                    step,
                )
                writer.add_scalar(
                    "Adjusted Mean_{feature}( Max_{decoder vector} cosine sim)",
                    adjusted_mean_max_cosine_similarity(sae, dataset),
                    step,
                )
                writer.add_scalar(
                    "Feature pair detection rate",
                    feature_pair_detection_rate(sae, dataset),
                    step,
                )
                writer.add_scalar(
                    "Adjusted feature pair detection rate",
                    adjusted_feature_pair_detection_rate(sae, dataset),
                    step,
                )
        writer.add_scalar("lr", lr, step)
        writer.add_scalar("sae_hidden_dim", args.sae_hidden_dim, step)
        writer.add_scalar("Total loss/train", total_loss, step)
        writer.add_scalar("Reconstruction loss/train", rec_loss, step)
        writer.add_scalar("Auxiliary loss/train", aux_loss, step)

    writer.close()


@jaxtyped(typechecker=beartype)
def get_reconstruction_loss(
    act: Float[torch.Tensor, "batch_size model_dim"],
    sae_act: Float[torch.Tensor, "batch_size model_dim"],
) -> Float[torch.Tensor, ""]:
    return ((act - sae_act) ** 2).mean()


@jaxtyped(typechecker=beartype)
def get_decoder_weights3(
    sae_model: TopkSparseAutoEncoder2Child_v2,
) -> Float[torch.Tensor, "model_dim expanded_sae_dim"]:
    return rearrange(
        [
            sae_model.decoder.weight
            + sae_model.decoder_child1.weight * sae_model.child1_parent_ratios,
            sae_model.decoder.weight
            + sae_model.decoder_child2.weight * sae_model.child2_parent_ratios,
        ],
        "copies model_dim sae_dim -> model_dim (sae_dim copies)",
        copies=2,
    )


@jaxtyped(typechecker=beartype)
def get_decoder_weights2(
    sae_model: SomeSAE,
) -> Float[torch.Tensor, "model_dim expanded_sae_dim"]:
    if isinstance(sae_model, TopkSparseAutoEncoder_v2):
        return sae_model.decoder.weight
    elif isinstance(sae_model, TopkSparseAutoEncoder2Child_v2):
        return rearrange(
            [
                sae_model.decoder.weight + sae_model.decoder_child1.weight,
                sae_model.decoder.weight + sae_model.decoder_child2.weight,
            ],
            "copies model_dim sae_dim -> model_dim (sae_dim copies)",
            copies=2,
        )
    else:
        raise TypeError(f"Unsupported model type: {type(sae_model)}")


@jaxtyped(typechecker=beartype)
def get_decoder_weights4(
    sae_model: SomeSAE,
) -> Float[torch.Tensor, "model_dim expanded_sae_dim"]:
    if isinstance(sae_model, TopkSparseAutoEncoder_v2):
        return sae_model.decoder.weight
    elif isinstance(sae_model, TopkSparseAutoEncoder2Child_v2):
        return rearrange(
            [
                sae_model.decoder.weight,
                sae_model.decoder.weight
                + sae_model.decoder_child1.weight * sae_model.child1_parent_ratios,
                sae_model.decoder.weight
                + sae_model.decoder_child2.weight * sae_model.child2_parent_ratios,
            ],
            "copies model_dim sae_dim -> model_dim (sae_dim copies)",
            copies=3,
        )
    else:
        raise TypeError(f"Unsupported model type: {type(sae_model)}")


@jaxtyped(typechecker=beartype)
def get_decoder_weights(
    sae_model: SomeSAE,
) -> Float[torch.Tensor, "model_dim expanded_sae_dim"]:
    if isinstance(sae_model, TopkSparseAutoEncoder_v2):
        return sae_model.decoder.weight
    elif isinstance(sae_model, TopkSparseAutoEncoder2Child_v2):
        return rearrange(
            [
                sae_model.decoder.weight,
                sae_model.decoder_child1.weight,
                sae_model.decoder_child2.weight,
                sae_model.decoder.weight + sae_model.decoder_child1.weight,
                sae_model.decoder.weight + sae_model.decoder_child2.weight,
                sae_model.decoder.weight
                + sae_model.decoder_child1.weight * sae_model.child1_parent_ratios,
                sae_model.decoder.weight
                + sae_model.decoder_child2.weight * sae_model.child2_parent_ratios,
                sae_model.encoder.weight.transpose(0, 1),
                sae_model.encoder_child1.weight.transpose(0, 1),
                sae_model.encoder_child2.weight.transpose(0, 1),
            ],
            "copies model_dim sae_dim -> model_dim (sae_dim copies)",
            copies=10,
        )
    else:
        raise TypeError(f"Unsupported model type: {type(sae_model)}")


@beartype
def adjusted_feature_pair_detection_rate(
    sae_model: TopkSparseAutoEncoder2Child_v2, dataset: ToyDataset
) -> float:
    successes_Bool_E = adjusted_feature_pair_detection_aux(sae_model, dataset)
    return float(successes_Bool_E.sum() / len(successes_Bool_E))


@jaxtyped(typechecker=beartype)
def adjusted_feature_pair_detection_aux(
    sae_model: TopkSparseAutoEncoder2Child_v2, dataset: ToyDataset
) -> Bool[torch.Tensor, " E"]:
    # TODO DRY this
    decoder_weights_MH = get_decoder_weights3(sae_model)
    E = decoder_weights_MH.shape[1] // 2

    all_child_vecs_CM = get_all_features(dataset)
    cosine_sim_HC = calculate_cosine_sim(decoder_weights_MH, all_child_vecs_CM)

    return find_feature_pair_successes(cosine_sim_HC, E)


@jaxtyped(typechecker=beartype)
def find_feature_pair_successes(
    cosine_sim_HC: Float[torch.Tensor, "H C"], E: int
) -> Bool[torch.Tensor, " {E}"]:
    successes_Bool_E = torch.zeros(E, dtype=torch.bool)

    for latent_idx in range(E):
        latent1_sims_C = cosine_sim_HC[2 * latent_idx]
        latent2_sims_C = cosine_sim_HC[2 * latent_idx + 1]

        closest_feature_to_latent1 = torch.argmax(latent1_sims_C)
        closest_feature_to_latent2 = torch.argmax(latent2_sims_C)

        # Check if these features are the two features in a pair
        if (
            closest_feature_to_latent1 != closest_feature_to_latent2
            and abs(closest_feature_to_latent1 - closest_feature_to_latent2) == 1
            and min(closest_feature_to_latent1, closest_feature_to_latent2) % 2 == 0
        ):
            successes_Bool_E[latent_idx] = True
    return successes_Bool_E


@beartype
def feature_pair_detection_rate(
    sae_model: TopkSparseAutoEncoder2Child_v2, dataset: ToyDataset
) -> float:
    decoder_weights = rearrange(
        [
            sae_model.decoder.weight + sae_model.decoder_child1.weight,
            sae_model.decoder.weight + sae_model.decoder_child2.weight,
        ],
        "copies model_dim sae_dim -> model_dim (sae_dim copies)",
        copies=2,
    )
    num_latent_pairs = decoder_weights.shape[1] // 2

    all_child_vecs = get_all_features(dataset)
    cosine_sim = calculate_cosine_sim(decoder_weights, all_child_vecs)

    successes = 0

    for k in range(num_latent_pairs):
        latent1_sims = cosine_sim[2 * k]
        latent2_sims = cosine_sim[2 * k + 1]

        closest_feature_to_latent1 = torch.argmax(latent1_sims)
        closest_feature_to_latent2 = torch.argmax(latent2_sims)

        # Check if these features are the two features in a pair
        if (
            closest_feature_to_latent1 != closest_feature_to_latent2
            and abs(closest_feature_to_latent1 - closest_feature_to_latent2) == 1
            and min(closest_feature_to_latent1, closest_feature_to_latent2) % 2 == 0
        ):
            successes += 1

    return successes / num_latent_pairs


@jaxtyped(typechecker=beartype)
def calculate_cosine_sim(
    decoder_weights: Float[torch.Tensor, "model_dim expanded_sae_dim"],
    all_child_vecs: Float[torch.Tensor, "total_num_children model_dim"],
) -> Float[torch.Tensor, "expanded_sae_dim total_num_children"]:
    return F.cosine_similarity(
        rearrange(
            decoder_weights,
            "model_dim expanded_sae_dim -> expanded_sae_dim 1 model_dim",
        ),
        rearrange(
            all_child_vecs,
            "total_num_children model_dim -> 1 total_num_children model_dim",
        ),
        dim=2,
    )


@jaxtyped(typechecker=beartype)
def get_all_features(
    dataset: ToyDataset,
) -> Float[torch.Tensor, "total_num_children model_dim"]:
    child_vecs = (
        rearrange(dataset.features, "n_features model_dim -> n_features 1 model_dim")
        + dataset.perturbations
    )
    return rearrange(
        child_vecs,
        "n_features children_per_parent model_dim -> (n_features children_per_parent) model_dim",
    )


@jaxtyped(typechecker=beartype)
def get_feature_v_feature_sim(
    dataset: ToyDataset,
) -> Float[torch.Tensor, "total_num_children total_num_children"]:
    # TODO Change name of total_num_children
    all_child_vecs = get_all_features(dataset)
    return F.cosine_similarity(
        rearrange(
            all_child_vecs,
            "total_num_children model_dim -> total_num_children 1 model_dim",
        ),
        rearrange(
            all_child_vecs,
            "total_num_children model_dim -> 1 total_num_children model_dim",
        ),
        dim=2,
    )


@jaxtyped(typechecker=beartype)
def get_similarity2(
    sae: SomeSAE, dataset: ToyDataset
) -> Float[torch.Tensor, "sae_dim total_num_children"]:
    all_child_vecs = get_all_features(dataset)
    return calculate_cosine_sim(get_decoder_weights2(sae), all_child_vecs)


@jaxtyped(typechecker=beartype)
def get_similarity4(
    sae: SomeSAE, dataset: ToyDataset
) -> Float[torch.Tensor, "sae_dim total_num_children"]:
    all_child_vecs = get_all_features(dataset)
    return calculate_cosine_sim(get_decoder_weights4(sae), all_child_vecs)


@jaxtyped(typechecker=beartype)
def get_similarity5(sae: SomeSAE) -> Float[torch.Tensor, "G G"]:
    decoder_weights_MG = get_decoder_weights4(sae)
    return F.cosine_similarity(
        decoder_weights_MG.unsqueeze(1), decoder_weights_MG.unsqueeze(2), dim=0
    )


# TODO total_num_children means child features, right?


@jaxtyped(typechecker=beartype)
def get_similarity(
    sae: SomeSAE, dataset: ToyDataset
) -> Float[torch.Tensor, "sae_dim total_num_children"]:
    all_child_vecs = get_all_features(dataset)
    return calculate_cosine_sim(get_decoder_weights(sae), all_child_vecs)


@jaxtyped(typechecker=beartype)
def min_max_cosine_similarity(sae, dataset) -> Float[torch.Tensor, ""]:
    similarity = get_similarity2(sae, dataset)
    per_feature_sim = reduce(
        similarity, "sae_dim total_num_children -> total_num_children", "max"
    )
    return per_feature_sim.min()


@jaxtyped(typechecker=beartype)
def adjusted_min_max_cosine_similarity(sae, dataset) -> Float[torch.Tensor, ""]:
    similarity = adjusted_get_similarity(sae, dataset)
    per_feature_sim = reduce(
        similarity, "sae_dim total_num_children -> total_num_children", "max"
    )
    return per_feature_sim.min()


@jaxtyped(typechecker=beartype)
def mean_max_cosine_similarity(sae, dataset) -> Float[torch.Tensor, ""]:
    similarity = get_similarity2(sae, dataset)
    per_feature_sim = reduce(
        similarity, "sae_dim total_num_children -> total_num_children", "max"
    )
    return per_feature_sim.mean()


@jaxtyped(typechecker=beartype)
def adjusted_mean_max_cosine_similarity(sae, dataset) -> Float[torch.Tensor, ""]:
    similarity = adjusted_get_similarity(sae, dataset)
    per_feature_sim = reduce(
        similarity, "sae_dim total_num_children -> total_num_children", "max"
    )
    return per_feature_sim.mean()


@jaxtyped(typechecker=beartype)
def adjusted_get_similarity(
    sae: SomeSAE, dataset: ToyDataset
) -> Float[torch.Tensor, "sae_dim total_num_children"]:
    all_child_vecs = get_all_features(dataset)
    return calculate_cosine_sim(get_decoder_weights3(sae), all_child_vecs)


@beartype
def plot_feature_similarity(
    dataset: ToyDataset,
    output_dir: str,
) -> None:
    similarity = get_feature_v_feature_sim(dataset)
    plt.figure(figsize=(12, 5))
    sns.heatmap(
        similarity.cpu().detach().numpy(),
        cmap="RdYlBu_r",
        center=0,
        vmin=-1,
        vmax=1,
        square=True,
        xticklabels=4,
        yticklabels=4,
    )

    plt.title("Cosine similarity between features")
    plt.xlabel("Features")
    plt.ylabel("Features")

    plt.tight_layout()
    plt.savefig(
        f"{output_dir}/feature_vs_feature_similarity_heatmap",
        dpi=300,
        bbox_inches="tight",
    )
    plt.close()


@beartype
def save_latent_similarity_graph(sae: SomeSAE, output_dir: str, step: int) -> None:
    similarity = get_similarity5(sae)

    LATENT_GROUP_SIZE = 3

    plt.figure(figsize=(12, 5))
    sns.heatmap(
        similarity.cpu().detach().numpy(),
        cmap="RdYlBu_r",
        center=0,
        vmin=-1,
        vmax=1,
        square=True,
        xticklabels=LATENT_GROUP_SIZE,
        yticklabels=LATENT_GROUP_SIZE,
    )

    num_rows = similarity.shape[0]
    num_cols = similarity.shape[1]
    assert num_rows == num_cols

    for i in range(LATENT_GROUP_SIZE, num_rows, LATENT_GROUP_SIZE):
        plt.axhline(y=i, color="black", linewidth=0.5)
    for i in range(LATENT_GROUP_SIZE, num_rows, LATENT_GROUP_SIZE):
        plt.axvline(x=i, color="black", linewidth=0.5)

    plt.title(f"Cosine Similarity of decoder weights vs themselves, step {step}")
    plt.xlabel(
        "Decoder Weight Vectors (0 mod 3 is parent weight, 1-2 mod 3 is parent weight + scaled child 1-2 weight)",
        fontsize=5,
    )
    plt.ylabel(
        "Decoder Weight Vectors (0 mod 3 is parent weight, 1-2 mod 3 is parent weight + scaled child 1-2 weight)",
        fontsize=5,
    )

    plt.tight_layout()
    plt.savefig(f"{output_dir}/latent_heatmap{step}.png", dpi=1200, bbox_inches="tight")
    plt.close()


@beartype
def save_legible_similarity_graph(
    sae: SomeSAE, dataset: ToyDataset, output_dir: str, step: int, hierarchical: bool
) -> None:
    similarity = get_similarity4(sae, dataset)

    LATENT_GROUP_SIZE = 3

    plt.figure(figsize=(12, 5))
    sns.heatmap(
        similarity.cpu().detach().numpy(),
        cmap="RdYlBu_r",
        center=0,
        vmin=-1,
        vmax=1,
        square=True,
        xticklabels=4,
        yticklabels=LATENT_GROUP_SIZE,
    )

    num_rows = similarity.shape[0]
    num_cols = similarity.shape[1]
    for i in range(2, num_cols, 2):
        plt.axvline(x=i, color="black", linewidth=0.5)
    if hierarchical:
        for i in range(LATENT_GROUP_SIZE, num_rows, LATENT_GROUP_SIZE):
            plt.axhline(y=i, color="black", linewidth=0.5)

    # Add labels
    plt.title(f"Cosine Similarity of decoder weights vs dataset features, step {step}")
    plt.xlabel("Toy dataset features (sibling features are consecutive)")
    plt.ylabel(
        "Decoder Weight Vectors (0 mod 3 is parent weight, 1-2 mod 3 is parent weight + scaled child 1-2 weight)",
        fontsize=5,
    )

    plt.tight_layout()
    plt.savefig(
        f"{output_dir}/v2_similarity_heatmap{step}.png", dpi=1200, bbox_inches="tight"
    )
    plt.close()


@beartype
def save_similarity_graph(
    sae: SomeSAE, dataset: ToyDataset, output_dir: str, step: int, hierarchical: bool
) -> None:
    similarity = get_similarity(sae, dataset)

    LATENT_GROUP_SIZE = 10

    # Create heatmap
    plt.figure(figsize=(12, 5))
    sns.heatmap(
        similarity.cpu().detach().numpy(),
        cmap="RdYlBu_r",
        center=0,
        vmin=-1,
        vmax=1,
        square=True,
        xticklabels=4,
        yticklabels=LATENT_GROUP_SIZE,
    )

    num_rows = similarity.shape[0]
    num_cols = similarity.shape[1]
    for i in range(2, num_cols, 2):
        plt.axvline(x=i, color="black", linewidth=0.5)
    if hierarchical:
        for i in range(LATENT_GROUP_SIZE, num_rows, LATENT_GROUP_SIZE):
            plt.axhline(y=i, color="black", linewidth=0.5)

    # Add labels
    plt.title(f"Cosine Similarity of decoder weights vs dataset features, step {step}")
    plt.xlabel("Toy dataset features (sibling features are consecutive)")
    plt.ylabel(
        "Decoder Weight Vectors (0 mod 5 is parent weight, 1-2 mod 5 is child 1-2, 3-4 mod 5 is parent weight + child 1-2 weight)",
        fontsize=5,
    )  # TODO No longer correct

    plt.tight_layout()
    plt.savefig(
        f"{output_dir}/similarity_heatmap{step}.png", dpi=1200, bbox_inches="tight"
    )
    plt.close()


@beartype
def log_dead_latents(
    num_live_latents: int | tuple[int, int, int],
    hierarchical: bool,
    writer: SummaryWriter,
    step: int,
    sae_hidden_dim: int,
) -> None:
    if hierarchical:
        num_live_parent_latents, num_live_child1_latents, num_live_child2_latents = (
            num_live_latents
        )
        num_dead_child1_latents = sae_hidden_dim - num_live_child1_latents
        writer.add_scalar("Num dead child1 latents", num_dead_child1_latents, step)
        writer.add_scalar(
            "Proportion of dead child1 latents",
            num_dead_child1_latents / sae_hidden_dim,
            step,
        )
        num_dead_child2_latents = sae_hidden_dim - num_live_child2_latents
        writer.add_scalar("Num dead child2 latents", num_dead_child2_latents, step)
        writer.add_scalar(
            "Proportion of dead child2 latents",
            num_dead_child2_latents / sae_hidden_dim,
            step,
        )
    else:
        num_live_parent_latents = num_live_latents
    num_dead_parent_latents = sae_hidden_dim - num_live_parent_latents
    writer.add_scalar("Num dead parent latents", num_dead_parent_latents, step)
    writer.add_scalar(
        "Proportion of dead parent latents",
        num_dead_parent_latents / sae_hidden_dim,
        step,
    )


@beartype
def handcode_sae(sae: TopkSparseAutoEncoder2Child_v2, dataset: ToyDataset) -> None:
    sae.encoder.weight.data = dataset.features.detach().clone()
    sae.decoder.weight.data = dataset.features.transpose(0, 1).detach().clone()

    norm_perturbations = (
        (
            dataset.perturbations
            / torch.linalg.vector_norm(dataset.perturbations, dim=2, keepdim=True)
        )
        .detach()
        .clone()
    )

    sae.encoder_child1.weight.data = norm_perturbations[:, 0]
    sae.decoder_child1.weight.data = norm_perturbations[:, 0].transpose(0, 1)
    sae.encoder_child2.weight.data = norm_perturbations[:, 1]
    sae.decoder_child2.weight.data = norm_perturbations[:, 1].transpose(0, 1)

    sae.encoder.bias.data = torch.zeros_like(sae.encoder.bias)
    sae.decoder.bias.data = torch.zeros_like(sae.decoder.bias)
    sae.encoder_child1.bias.data = torch.zeros_like(sae.encoder_child1.bias)
    sae.decoder_child1.bias.data = torch.zeros_like(sae.decoder_child1.bias)
    sae.encoder_child2.bias.data = torch.zeros_like(sae.encoder_child2.bias)
    sae.decoder_child2.bias.data = torch.zeros_like(sae.decoder_child2.bias)


@beartype
def plot_norms(
    sae: TopkSparseAutoEncoder2Child_v2, step: int, output_dir: str, dataset: ToyDataset
) -> None:
    successes_Bool_E = adjusted_feature_pair_detection_aux(sae, dataset)

    LATENT_GROUP_SIZE = 7
    weights_MG = rearrange(
        [
            sae.decoder.weight,
            sae.decoder_child1.weight,
            sae.decoder_child2.weight,
            sae.decoder_child1.weight * sae.child1_parent_ratios,
            sae.decoder_child2.weight * sae.child2_parent_ratios,
            sae.decoder.weight + sae.decoder_child1.weight * sae.child1_parent_ratios,
            sae.decoder.weight + sae.decoder_child2.weight * sae.child2_parent_ratios,
        ],
        "copies model_dim sae_dim -> model_dim (sae_dim copies)",
        copies=LATENT_GROUP_SIZE,
    )
    weights_norm_G = torch.linalg.vector_norm(weights_MG, dim=0)

    successes_Bool_G = repeat(
        successes_Bool_E, "E -> (E repeat)", repeat=LATENT_GROUP_SIZE
    )
    colors = ["blue" if x else "red" for x in successes_Bool_G.cpu()]

    plt.figure(figsize=(10, 6))
    plt.bar(range(len(weights_norm_G)), weights_norm_G.cpu().detach(), color=colors)

    for i in range(LATENT_GROUP_SIZE, len(weights_norm_G), LATENT_GROUP_SIZE):
        plt.axvline(x=i - 0.5, color="red", linestyle="--", alpha=0.5)

    plt.xlabel("Weight Index")
    plt.ylabel("Norm Value")
    plt.title("Weight Norms with Group Separators")
    plt.tight_layout()
    plt.savefig(f"{output_dir}/norms_{step}.png", bbox_inches="tight")
    plt.close()


@beartype
def save_sorted_similarity_graph(
    sae: SomeSAE, dataset: ToyDataset, output_dir: str, step: int, hierarchical: bool
) -> None:
    similarity_GC = get_similarity4(sae, dataset)
    LATENT_GROUP_SIZE = 3

    # Keep on CPU but stay in torch
    sim_GC = similarity_GC.cpu().detach()

    # Get dimensions
    n_latents = sim_GC.shape[0]
    n_features = sim_GC.shape[1]
    n_latent_groups = n_latents // LATENT_GROUP_SIZE
    n_feature_pairs = n_features // 2

    # For each latent group, calculate average child similarity for each feature pair
    matches = []
    available_features = set(range(n_feature_pairs))

    # Calculate matches for each latent group
    for group_idx in range(n_latent_groups):
        start_idx = group_idx * LATENT_GROUP_SIZE
        # Get average of child similarities (indices 1 and 2 within group)
        child_sims_2C = sim_GC[start_idx + 1 : start_idx + LATENT_GROUP_SIZE]
        child_avg_C = torch.mean(child_sims_2C, dim=0)

        # Reshape to feature pairs and get average across pairs
        feature_pair_avgs_A = torch.zeros(n_feature_pairs)
        for feat_idx in available_features:
            feat_start = feat_idx * 2
            feature_pair_avgs_A[feat_idx] = torch.mean(
                child_avg_C[feat_start : feat_start + 2]
            )

        # Find best available feature pair
        best_feature = max(
            available_features, key=lambda i: feature_pair_avgs_A[i].item()
        )
        matches.append((group_idx, best_feature))
        available_features.remove(best_feature)

    # Sort latent groups by their matched feature index
    matches.sort(key=lambda x: x[1])

    # Create new sorted similarity matrix
    new_sim_GC = torch.zeros_like(sim_GC)
    for new_group_idx, (old_group_idx, _) in enumerate(matches):
        # Place the latent group
        old_group_start = old_group_idx * LATENT_GROUP_SIZE
        new_group_start = new_group_idx * LATENT_GROUP_SIZE

        # Copy the group's similarities with all features
        new_sim_GC[new_group_start : new_group_start + LATENT_GROUP_SIZE, :] = sim_GC[
            old_group_start : old_group_start + LATENT_GROUP_SIZE, :
        ]

    # Convert to numpy for plotting
    final_sim_np_GC = new_sim_GC.numpy()

    # Plot
    plt.figure(figsize=(12, 5))
    sns.heatmap(
        final_sim_np_GC,
        cmap="RdYlBu_r",
        center=0,
        vmin=-1,
        vmax=1,
        square=True,
        xticklabels=4,
        yticklabels=LATENT_GROUP_SIZE,
    )

    # Add grid lines
    for i in range(2, n_features, 2):
        plt.axvline(x=i, color="black", linewidth=0.5)
    if hierarchical:
        for i in range(LATENT_GROUP_SIZE, n_latents, LATENT_GROUP_SIZE):
            plt.axhline(y=i, color="black", linewidth=0.5)

    # Add labels
    plt.title(f"Cosine Similarity of decoder weights vs dataset features, step {step}")
    plt.xlabel("Toy dataset features (sibling features are consecutive)")
    plt.ylabel(
        "Rearranged decoder Weight Vectors (0 mod 3 is parent weight, 1-2 mod 3 is parent weight + scaled child 1-2 weight)",
        fontsize=5,
    )

    plt.tight_layout()
    plt.savefig(
        f"{output_dir}/sorted_similarity_heatmap{step}.png",
        dpi=1200,
        bbox_inches="tight",
    )
    plt.close()


if __name__ == "__main__":
    main(make_parser().parse_args())
