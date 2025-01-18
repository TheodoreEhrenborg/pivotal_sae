#!/usr/bin/env python3


import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


def test_layer():
    """Check that we will steer at the same layer
    that we trained the SAE on"""
    llm = AutoModelForCausalLM.from_pretrained("roneneldan/TinyStories-33M")
    result_dict = {}

    def hook(module, args, output):
        result_dict["output"] = output

    llm.transformer.h[1].register_forward_hook(hook)
    tokenizer = AutoTokenizer.from_pretrained("EleutherAI/gpt-neo-125M")
    tokenizer.pad_token = tokenizer.eos_token
    sample = "There once was a cat"
    input = torch.tensor(tokenizer(sample)["input_ids"]).unsqueeze(0)
    with torch.no_grad():
        x = llm(input, output_hidden_states=True)
    assert len(x.hidden_states) == 5
    assert torch.allclose(x.hidden_states[2], result_dict["output"][0]), (
        x.hidden_states[2] - result_dict["output"][0]
    )
