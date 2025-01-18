#!/usr/bin/env python3
from argparse import ArgumentParser, Namespace

import torch
from beartype import beartype
from jaxtyping import Float, Int, jaxtyped
from transformers import GenerationConfig, GPTNeoForCausalLM

from tiny_stories_sae.common.activation_analysis import get_annotated_text
from tiny_stories_sae.common.angle import get_rotation_between
from tiny_stories_sae.common.obtain_activations import (
    get_llm_activation_from_tensor,
    normalize_activations,
)
from tiny_stories_sae.common.sae import SparseAutoEncoder
from tiny_stories_sae.common.setting_up import setup


@jaxtyped(typechecker=beartype)
def generate(
    llm: GPTNeoForCausalLM,
    prompt_tokens: Int[torch.Tensor, "1 input_seq_len"],
    max_generation_length: int,
) -> Int[torch.Tensor, "1 output_seq_len"]:
    return llm.generate(
        prompt_tokens,
        max_length=max_generation_length,
        num_beams=1,
        generation_config=GenerationConfig(do_sample=True, temperature=1.0),
    )


@jaxtyped(typechecker=beartype)
def get_feature_strength(
    llm_activation: Float[torch.Tensor, "1 seq_len 768"],
    which_feature: int,
    sae: SparseAutoEncoder,
) -> Float[torch.Tensor, " seq_len"]:
    norm_act = normalize_activations(llm_activation)
    _, feat_magnitudes = sae(norm_act)
    return feat_magnitudes[0, :, which_feature]


@jaxtyped(typechecker=beartype)
def debug_angles(
    user_args: Namespace,
    sae: SparseAutoEncoder,
    decoder_vector: Float[torch.Tensor, "768"],
):
    if user_args.debug:
        with torch.no_grad():
            encoder_vector = sae.encoder.weight[user_args.which_feature, :]
        if user_args.cuda:
            encoder_vector = encoder_vector.cuda()
        print(
            "Rotation between encoder and decoder vectors for same feature",
            get_rotation_between(encoder_vector, decoder_vector),
        )
        onehot = torch.zeros(sae.sae_hidden_dim)
        onehot[user_args.which_feature] = 1
        if user_args.cuda:
            onehot = onehot.cuda()
        nudge = sae.decoder(onehot)
        assert nudge.shape == torch.Size([768]), nudge.shape
        print(
            "Rotation between nudge+bias and decoder_vec",
            get_rotation_between(nudge, decoder_vector),
        )


@jaxtyped(typechecker=beartype)
def debug_strengths(
    user_args: Namespace,
    activation: Float[torch.Tensor, "1 seq_len 768"],
    activation_with_nudge: Float[torch.Tensor, "1 seq_len 768"],
    sae: SparseAutoEncoder,
):
    if user_args.debug:
        print(
            "This feature's strength pre nudge",
            get_feature_strength(activation, user_args.which_feature, sae),
        )
        print(
            "This feature's strength post nudge",
            get_feature_strength(activation_with_nudge, user_args.which_feature, sae),
        )


@beartype
def main(user_args: Namespace):
    _, unmodified_llm, _, tokenizer = setup(
        user_args.sae_hidden_dim, user_args.cuda, user_args.no_internet
    )

    input_tokens = torch.tensor(
        tokenizer(user_args.initial_sample)["input_ids"]
    ).unsqueeze(0)
    if user_args.cuda:
        input_tokens = input_tokens.cuda()
    if user_args.print_unsteered:
        unsteered_output_tokens = generate(
            unmodified_llm, input_tokens, user_args.max_generation_length
        )
        print("Unsteered output:")
        print(tokenizer.decode(unsteered_output_tokens[0]))

    _, steered_llm, sae, _ = setup(
        user_args.sae_hidden_dim, user_args.cuda, user_args.no_internet
    )
    sae.load_state_dict(torch.load(user_args.checkpoint, weights_only=True))
    with torch.no_grad():
        decoder_vector = sae.decoder.weight[:, user_args.which_feature]
    debug_angles(user_args, sae, decoder_vector)
    norm_nudge = decoder_vector / torch.linalg.vector_norm(decoder_vector)

    def simple_nudge_hook(module, args, output):
        activation = output[0]
        activation_with_nudge = activation + user_args.feature_strength * norm_nudge
        debug_strengths(user_args, activation, activation_with_nudge, sae)
        return activation_with_nudge, output[1]

    # The SAE was trained on the 3rd hidden state,
    # so we intervene there---this is the state after the 2nd transformer.
    # test_layer.py confirms this is correct
    steered_llm.transformer.h[1].register_forward_hook(simple_nudge_hook)
    steered_output_tokens = generate(
        steered_llm, input_tokens, user_args.max_generation_length
    )
    print("Steered output:")
    print(tokenizer.decode(steered_output_tokens[0]))

    print()
    print(
        "Now feed the steered text into an unmodified LLM, "
        "and print how much the SparseAutoEncoder thinks the LLM activates on the feature:"
    )
    activation = get_llm_activation_from_tensor(unmodified_llm, steered_output_tokens)
    strengths = get_feature_strength(activation, user_args.which_feature, sae).tolist()
    print(
        get_annotated_text(
            tokenizer, steered_output_tokens[0].tolist(), strengths, max(strengths)
        )
    )


@beartype
def make_parser() -> ArgumentParser:
    parser = ArgumentParser()
    parser.add_argument("--cuda", action="store_true")
    parser.add_argument("--sae_hidden_dim", type=int, default=100)
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--which_feature", type=int, required=True)
    parser.add_argument("--feature_strength", type=float, default=10.0)
    parser.add_argument("--print_unsteered", action="store_true")
    parser.add_argument("--no_internet", action="store_true")
    parser.add_argument("--max_generation_length", type=int, default=1000)
    parser.add_argument("--initial_sample", type=str, default="Once upon a time")
    parser.add_argument(
        "--debug",
        action="store_true",
        help="""
            This script needed a lot of debugging. I've hidden
            the debug statements behind a flag. Probably in
            production I'd delete them to make the code cleaner""",
    )
    return parser


if __name__ == "__main__":
    main(make_parser().parse_args())
