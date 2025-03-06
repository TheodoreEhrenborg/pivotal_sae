#!/usr/bin/env python3


import random
import uuid

import torch

from sae_hacking.safetensor_utils import load_dict_with_tensors, save_dict_with_tensors


def test_save_and_restore():
    d = {}
    for _ in range(50):
        i = random.randrange(100)
        d[i] = torch.randn(10)
    name = f"/tmp/{uuid.uuid4()}.safetensors.zst"
    save_dict_with_tensors(d, name)
    d2 = load_dict_with_tensors(name)
    assert sorted(list(d.keys())) == sorted(list(d2.keys()))
    for k in d:
        assert torch.equal(d[k], d2[k])
