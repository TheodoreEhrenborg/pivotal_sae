#!/usr/bin/env python3

import math
from argparse import Namespace

import torch
from beartype import beartype
from jaxtyping import Float, Int, jaxtyped
from transformers import GPTNeoForCausalLM


@jaxtyped(typechecker=beartype)
def get_llm_activation(
    model: GPTNeoForCausalLM, example: dict, user_args: Namespace
) -> Float[torch.Tensor, "1 seq_len 768"]:
    tokens_tensor = torch.tensor(example["input_ids"]).unsqueeze(0)
    if user_args.cuda:
        tokens_tensor = tokens_tensor.cuda()
    return get_llm_activation_from_tensor(model, tokens_tensor)


@jaxtyped(typechecker=beartype)
def get_llm_activation_from_tensor(
    model: GPTNeoForCausalLM,
    tokens_tensor: Int[torch.Tensor, "1 seq_len"],
) -> Float[torch.Tensor, "1 seq_len 768"]:
    with torch.no_grad():
        x = model(
            tokens_tensor,
            output_hidden_states=True,
        )
        assert len(x.hidden_states) == 5
        return x.hidden_states[2]


@jaxtyped(typechecker=beartype)
def normalize_activations(
    activation: Float[torch.Tensor, "1 seq_len 768"],
) -> Float[torch.Tensor, "1 seq_len 768"]:
    return (activation - activation.mean()) / activation.std() * math.sqrt(768)
