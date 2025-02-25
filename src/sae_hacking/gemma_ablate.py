#!/usr/bin/env python3
import asyncio
from argparse import ArgumentParser, Namespace
from collections import abc
from functools import partial

import aiohttp
import torch
from beartype import beartype
from jaxtyping import Int, jaxtyped
from sae_lens import SAE, HookedSAETransformer
from transformer_lens.utils import test_prompt

# Gemma-scope based on https://colab.research.google.com/drive/17dQFYUYnuKnP6OwQPH9v_GSYUW5aj-Rp
# Neuronpedia API based on https://colab.research.google.com/github/jbloomAus/SAELens/blob/main/tutorials/tutorial_2_0.ipynb


@beartype
def make_parser() -> ArgumentParser:
    parser = ArgumentParser()
    parser.add_argument("--model", default="google/gemma-2-2b")
    parser.add_argument(
        "--ablater-sae-release", default="gemma-scope-2b-pt-res-canonical"
    )
    parser.add_argument("--ablater-sae-id", default="layer_20/width_65k/canonical")
    parser.add_argument(
        "--reader-sae-release", default="gemma-scope-2b-pt-mlp-canonical"
    )
    parser.add_argument("--reader-sae-id", default="layer_21/width_65k/canonical")
    return parser


@beartype
def test_prompt_with_ablation(
    model, ablater_sae, prompt, answer, ablation_features, reader_sae: SAE
):
    def ablate_feature_hook(feature_activations, hook, feature_ids, position=None):
        if position is None:
            feature_activations[:, :, feature_ids] = 0
        else:
            feature_activations[:, position, feature_ids] = 0

        return feature_activations

    ablation_hook = partial(ablate_feature_hook, feature_ids=ablation_features)

    model.add_sae(ablater_sae)
    hook_point = ablater_sae.cfg.hook_name + ".hook_sae_acts_post"
    model.add_hook(hook_point, ablation_hook, "fwd")

    test_prompt(prompt, answer, model)
    _, cache = model.run_with_cache_with_saes(prompt, saes=[reader_sae])
    # TODO I think there's a way to look this up in the SAE config?
    vals, inds = torch.topk(
        cache["blocks.21.hook_mlp_out.hook_sae_acts_post"][0, -1, :], 5
    )
    descriptions = asyncio.run(get_all_descriptions(inds))
    for val, ind, description in zip(vals, inds, descriptions, strict=True):
        print(f"Feature {ind} fired {val:.2f}")
        print(f"Description: {description}")

    model.reset_hooks()
    model.reset_saes()


@jaxtyped(typechecker=beartype)
async def get_description_async(
    idx: int | Int[torch.Tensor, ""], session: aiohttp.ClientSession
) -> abc.Coroutine[None, None, str]:
    url = f"https://www.neuronpedia.org/api/feature/gemma-2-2b/21-gemmascope-mlp-65k/{idx}"
    async with session.get(url) as response:
        data = await response.json()
        try:
            return data["explanations"][0]["description"]
        except:
            print(data)
            raise


@jaxtyped(typechecker=beartype)
async def get_all_descriptions(
    indices: list[int] | Int[torch.Tensor, " length"],
) -> abc.Coroutine[None, None, list[str]]:
    async with aiohttp.ClientSession() as session:
        tasks = [get_description_async(idx, session) for idx in indices]
        return await asyncio.gather(*tasks)


@beartype
def main(args: Namespace) -> None:
    device = "cuda"
    model = HookedSAETransformer.from_pretrained(args.model, device=device)

    # the cfg dict is returned alongside the SAE since it may contain useful information for analysing the SAE (eg: instantiating an activation store)
    # Note that this is not the same as the SAEs config dict, rather it is whatever was in the HF repo, from which we can extract the SAE config dict
    # We also return the feature sparsities which are stored in HF for convenience.
    ablater_sae, cfg_dict, sparsity = SAE.from_pretrained(
        release=args.ablater_sae_release,  # <- Release name
        sae_id=args.ablater_sae_id,  # <- SAE id (not always a hook point!)
        device=device,
    )
    reader_sae, _, _ = SAE.from_pretrained(
        release=args.reader_sae_release, sae_id=args.reader_sae_id, device=device
    )
    model.reset_hooks(including_permanent=True)
    prompt = "In the beginning, God created the heavens and the"
    answer = "earth"
    test_prompt(prompt, answer, model)

    # Generate text with feature ablation
    print("Test Prompt with feature ablation and no error term")
    ablation_feature = 16873  # TODO Change this
    ablater_sae.use_error_term = False
    test_prompt_with_ablation(
        model, ablater_sae, prompt, answer, ablation_feature, reader_sae
    )

    print("Test Prompt with feature ablation and error term")
    ablater_sae.use_error_term = True
    test_prompt_with_ablation(
        model, ablater_sae, prompt, answer, ablation_feature, reader_sae
    )


if __name__ == "__main__":
    main(make_parser().parse_args())
