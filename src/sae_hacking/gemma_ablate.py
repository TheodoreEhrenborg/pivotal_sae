#!/usr/bin/env python3
from argparse import ArgumentParser, Namespace
from functools import partial

from beartype import beartype
from sae_lens import SAE, HookedSAETransformer
from transformer_lens.utils import test_prompt

# Gemma-scope based on https://colab.research.google.com/drive/17dQFYUYnuKnP6OwQPH9v_GSYUW5aj-Rp
# Neuronpedia API based on https://colab.research.google.com/github/jbloomAus/SAELens/blob/main/tutorials/tutorial_2_0.ipynb


@beartype
def make_parser() -> ArgumentParser:
    parser = ArgumentParser()
    parser.add_argument("--model", default="gpt2-small")
    parser.add_argument("--sae-release", default="gpt2-small-res-jb")
    parser.add_argument("--sae-id", default="blocks.7.hook_resid_pre")
    return parser


def test_prompt_with_ablation(model, sae, prompt, answer, ablation_features):
    def ablate_feature_hook(feature_activations, hook, feature_ids, position=None):
        if position is None:
            feature_activations[:, :, feature_ids] = 0
        else:
            feature_activations[:, position, feature_ids] = 0

        return feature_activations

    ablation_hook = partial(ablate_feature_hook, feature_ids=ablation_features)

    model.add_sae(sae)
    hook_point = sae.cfg.hook_name + ".hook_sae_acts_post"
    model.add_hook(hook_point, ablation_hook, "fwd")

    test_prompt(prompt, answer, model)

    model.reset_hooks()
    model.reset_saes()


@beartype
def main(args: Namespace) -> None:
    device = "cuda"
    model = HookedSAETransformer.from_pretrained(args.model, device=device)

    # the cfg dict is returned alongside the SAE since it may contain useful information for analysing the SAE (eg: instantiating an activation store)
    # Note that this is not the same as the SAEs config dict, rather it is whatever was in the HF repo, from which we can extract the SAE config dict
    # We also return the feature sparsities which are stored in HF for convenience.
    sae, cfg_dict, sparsity = SAE.from_pretrained(
        release=args.sae_release,  # <- Release name
        sae_id=args.sae_id,  # <- SAE id (not always a hook point!)
        device=device,
    )
    model.reset_hooks(including_permanent=True)
    prompt = "In the beginning, God created the heavens and the"
    answer = "earth"
    test_prompt(prompt, answer, model)

    # Generate text with feature ablation
    print("Test Prompt with feature ablation and no error term")
    ablation_feature = 16873  # Replace with any feature index you're interested in. We use the religion feature
    sae.use_error_term = False
    test_prompt_with_ablation(model, sae, prompt, answer, ablation_feature)

    print("Test Prompt with feature ablation and error term")
    ablation_feature = 16873  # Replace with any feature index you're interested in. We use the religion feature
    sae.use_error_term = True
    test_prompt_with_ablation(model, sae, prompt, answer, ablation_feature)


if __name__ == "__main__":
    main(make_parser().parse_args())
