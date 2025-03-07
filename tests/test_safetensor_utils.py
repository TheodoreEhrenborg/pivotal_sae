#!/usr/bin/env python3


import random
import uuid

import pytest
import torch

from sae_hacking.safetensor_utils import load_dict_with_tensors, save_dict_with_tensors


@pytest.mark.parametrize("c", [None, torch.randn(7, 11)])
def test_save_and_restore(c):
    d = {}
    for _ in range(50):
        i = random.randrange(100)
        d[i] = torch.randn(10)
    name = f"/tmp/{uuid.uuid4()}.safetensors.zst"
    save_dict_with_tensors(d, name, c)
    d2, c2 = load_dict_with_tensors(name)
    assert sorted(list(d.keys())) == sorted(list(d2.keys()))
    for k in d:
        assert torch.equal(d[k], d2[k])
    if c is not None:
        assert torch.equal(c, c2)
    else:
        assert c is c2
