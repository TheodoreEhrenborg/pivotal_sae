#!/usr/bin/env python3
import time
from argparse import ArgumentParser, Namespace
from pathlib import Path

import torch
from beartype import beartype
from coolname import generate_slug
from jaxtyping import Float, Int, jaxtyped
from sae_lens import SAE, HookedSAETransformer
from tqdm import tqdm

from sae_hacking.gemma_utils import generate_prompts2
from sae_hacking.safetensor_utils import save_v2
from sae_hacking.timeprint import timeprint


@beartype
def make_parser() -> ArgumentParser:
    parser = ArgumentParser()
    parser.add_argument("--model", default="google/gemma-2-2b")
    parser.add_argument(
        "--ablator-sae-release", default="gemma-scope-2b-pt-res-canonical"
    )
    parser.add_argument("--ablator-sae-id", default="layer_20/width_65k/canonical")
    parser.add_argument(
        "--reader-sae-release", default="gemma-scope-2b-pt-mlp-canonical"
    )
    parser.add_argument("--reader-sae-id", default="layer_21/width_65k/canonical")
    parser.add_argument("--max-tokens-in-prompt", type=int, default=125)
    parser.add_argument("--abridge-ablations-to", type=int, default=1000)
    parser.add_argument("--n-prompts", type=int, default=1)
    parser.add_argument("--save-frequency", type=int, default=240)
    parser.add_argument("--selected-feature", type=int, required=True)
    parser.add_argument(
        "--exclude-latent-threshold",
        type=float,
        default=0.1,
        help="Latents more frequent than this are excluded from ablation",
    )
    parser.add_argument("--dataset-id", required=True)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--never-save", action="store_true")
    return parser


@jaxtyped(typechecker=beartype)
def compute_ablation_matrix(
    model: HookedSAETransformer,
    ablator_sae: SAE,
    reader_sae: SAE,
    prompt_BS: Int[torch.Tensor, "batch seq_len"],
    ablation_results_eE: Float[
        torch.Tensor, "num_ablator_features num_reader_features"
    ],
    abridge_ablations_to: int,
    how_often_activated_e: Float[torch.Tensor, " num_ablator_features"],
    selected_feature: int,
) -> None:
    """
    - e: number of features in ablator SAE
    - E: number of features in reader SAE
    """
    ablator_sae.use_error_term = True
    reader_sae.use_error_term = True

    # First, run the model with ablator SAE to get its activations
    model.reset_hooks()
    model.reset_saes()
    _, ablator_cache = model.run_with_cache_with_saes(prompt_BS, saes=[ablator_sae])
    ablator_acts_BSe = ablator_cache[f"{ablator_sae.cfg.hook_name}.hook_sae_acts_post"]

    for batch_idx in range(prompt_BS.shape[0]):
        # Find the features with highest activation summed across all positions
        summed_acts_e = ablator_acts_BSe[batch_idx].sum(dim=0)
        tentative_top_features_k = torch.topk(
            summed_acts_e, k=abridge_ablations_to
        ).indices

        top_features_K = tentative_top_features_k[:abridge_ablations_to]
        assert len(top_features_K) == abridge_ablations_to

        if selected_feature not in top_features_K:
            continue
        print("Got one")

        how_often_activated_e[top_features_K] += 1

        # Get baseline activations for reader SAE
        model.reset_hooks()
        model.reset_saes()
        prompt_1S = prompt_BS[batch_idx].unsqueeze(0)
        _, baseline_cache = model.run_with_cache_with_saes(prompt_1S, saes=[reader_sae])
        baseline_acts_1SE = baseline_cache[
            f"{reader_sae.cfg.hook_name}.hook_sae_acts_post"
        ]
        baseline_acts_E = baseline_acts_1SE[0, -1, :]

        # Add the ablator SAE to the model
        model.add_sae(ablator_sae)
        hook_point = ablator_sae.cfg.hook_name + ".hook_sae_acts_post"

        # For each top feature in the ablator SAE
        for ablator_idx in top_features_K:
            # Set up ablation hook for this feature
            def ablation_hook(acts_BSe, hook):
                acts_BSe[:, :, ablator_idx] = 0
                return acts_BSe

            model.add_hook(hook_point, ablation_hook, "fwd")

            # Run with this feature ablated
            _, ablated_cache = model.run_with_cache_with_saes(
                prompt_1S, saes=[reader_sae]
            )
            ablated_acts_1SE = ablated_cache[
                f"{reader_sae.cfg.hook_name}.hook_sae_acts_post"
            ]
            ablated_acts_E = ablated_acts_1SE[0, -1, :]

            result = baseline_acts_E - ablated_acts_E

            ablator_idx_int = ablator_idx.item()
            ablation_results_eE[ablator_idx_int] += result.cpu()

            # Reset hooks for next iteration
            model.reset_hooks()


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
    reader_sae, reader_sae_config, _ = SAE.from_pretrained(
        release=args.reader_sae_release, sae_id=args.reader_sae_id, device=device
    )
    E = reader_sae_config["d_sae"]
    prompts = generate_prompts2(
        args.model,
        args.n_prompts,
        args.max_tokens_in_prompt,
        args.dataset_id,
        args.batch_size,
    )

    ablation_results_eE = torch.zeros(e, E)
    how_often_activated_e = torch.zeros(e).cuda()
    with tqdm(total=args.n_prompts) as pbar:
        print("HACK: Filtering out prompts unless they start with a comma")
        for i, batch in enumerate(prompts):
            any_comma = any(text.startswith(",") for text in batch["text"])
            if any_comma:
                compute_ablation_matrix(
                    model,
                    ablator_sae,
                    reader_sae,
                    batch["abridged_tensor"],
                    ablation_results_eE,
                    args.abridge_ablations_to,
                    how_often_activated_e,
                    args.selected_feature,
                )
            if (i % args.save_frequency == 0 or i + 1 == args.n_prompts) and (
                not args.never_save
            ):
                timeprint("Saving...")
                save_v2(
                    ablation_results_eE,
                    f"{output_dir}/{time.strftime('%Y%m%d-%H%M%S')}intermediate.safetensors.zst",
                    None,
                    how_often_activated_e,
                )
                timeprint("Saved")
            pbar.update(args.batch_size)


if __name__ == "__main__":
    main(make_parser().parse_args())
