#!/usr/bin/env python3
import time
from argparse import ArgumentParser, Namespace
from pathlib import Path

import torch
from beartype import beartype
from coolname import generate_slug
from jaxtyping import Float, jaxtyped
from sae_lens import SAE, HookedSAETransformer
from tqdm import tqdm

from sae_hacking.gemma_utils import gather_co_occurrences2, generate_prompts
from sae_hacking.safetensor_utils import save_v2
from sae_hacking.timeprint import timeprint

# Gemma-scope based on https://colab.research.google.com/drive/17dQFYUYnuKnP6OwQPH9v_GSYUW5aj-Rp
# Neuronpedia API based on https://colab.research.google.com/github/jbloomAus/SAELens/blob/main/tutorials/tutorial_2_0.ipynb


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
    parser.add_argument(
        "--exclude-latent-threshold",
        type=float,
        default=0.1,
        help="Latents more frequent than this are excluded from ablation",
    )
    return parser


@jaxtyped(typechecker=beartype)
def compute_ablation_matrix(
    model: HookedSAETransformer,
    ablator_sae: SAE,
    reader_sae: SAE,
    prompt: str,
    ablation_results_eE: Float[
        torch.Tensor, "num_ablator_features num_reader_features"
    ],
    abridge_ablations_to: int,
    cooccurrences_ee: Float[torch.Tensor, "num_ablator_features num_ablator_features"],
    how_often_activated_e: Float[torch.Tensor, " num_ablator_features"],
) -> None:
    """
    - e: number of features in ablator SAE
    - E: number of features in reader SAE
    """
    timeprint(prompt)
    ablator_sae.use_error_term = True
    reader_sae.use_error_term = True

    # First, run the model with ablator SAE to get its activations
    model.reset_hooks()
    model.reset_saes()
    _, ablator_cache = model.run_with_cache_with_saes(prompt, saes=[ablator_sae])
    ablator_acts_1Se = ablator_cache[f"{ablator_sae.cfg.hook_name}.hook_sae_acts_post"]

    timeprint("Starting to update co-occurrence matrix")
    cooccurrences_ee += gather_co_occurrences2(ablator_acts_1Se)
    timeprint("Done updating co-occurrence matrix")

    # Find the features with highest activation summed across all positions
    summed_acts_e = ablator_acts_1Se[0].sum(dim=0)
    tentative_top_features_k = torch.topk(summed_acts_e, k=abridge_ablations_to).indices

    top_features_K = tentative_top_features_k[:abridge_ablations_to]
    assert len(top_features_K) == abridge_ablations_to

    how_often_activated_e[top_features_K] += 1

    # Get baseline activations for reader SAE
    model.reset_hooks()
    model.reset_saes()
    _, baseline_cache = model.run_with_cache_with_saes(prompt, saes=[reader_sae])
    baseline_acts_1SE = baseline_cache[f"{reader_sae.cfg.hook_name}.hook_sae_acts_post"]
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
        _, ablated_cache = model.run_with_cache_with_saes(prompt, saes=[reader_sae])
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
    prompts = generate_prompts(args.model, args.n_prompts, args.max_tokens_in_prompt)

    ablation_results_eE = torch.zeros(e, E)
    cooccurrences_ee = torch.zeros(e, e)
    how_often_activated_e = torch.zeros(e)
    for i, prompt in enumerate(tqdm(prompts)):
        timeprint("Computing ablation matrix...")
        compute_ablation_matrix(
            model,
            ablator_sae,
            reader_sae,
            prompt,
            ablation_results_eE,
            args.abridge_ablations_to,
            cooccurrences_ee,
            how_often_activated_e,
        )
        timeprint("Done computing ablation matrix")
        if i % args.save_frequency == 0 or i + 1 == len(prompts):
            save_v2(
                ablation_results_eE,
                f"{output_dir}/{time.strftime('%Y%m%d-%H%M%S')}intermediate.safetensors.zst",
                cooccurrences_ee.to_dense(),
                how_often_activated_e,
            )


if __name__ == "__main__":
    main(make_parser().parse_args())
