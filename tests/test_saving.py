#!/usr/bin/env python3


import random

import torch

from sae_hacking.gemma_ablate import (
    load_dict_with_tensors_from_json,
    save_dict_with_tensors_to_json,
)


def test_save_and_restore():
    d = {}
    for _ in range(50):
        i = random.randrange(100)
        d[i] = torch.randn(10)
    name = "/tmp/testing.json"
    save_dict_with_tensors_to_json(d, name)
    d2 = load_dict_with_tensors_from_json(name)
    assert list(d.keys()) == list(d2.keys())
    for k in d:
        assert torch.allclose(d[k], d2[k])
