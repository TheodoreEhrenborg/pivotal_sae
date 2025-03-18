#!/usr/bin/env python3


import os
import random
import uuid

import pytest
import torch

from sae_hacking.safetensor_utils import (
    load_dict_with_tensors,
    load_v2,
    save_dict_with_tensors,
    save_v2,
)


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


def test_save_and_restore_v2():
    # Create sample tensors
    e, E = 11, 20  # Example dimensions
    effects_eE = torch.randn(e, E)
    cooccurrences_ee = torch.randn(e, e)
    how_often_activated_e = torch.randn(e)

    # Generate a temporary file path
    name = f"/tmp/{uuid.uuid4()}.safetensors.zst"

    try:
        # Save the tensors
        save_v2(effects_eE, name, cooccurrences_ee, how_often_activated_e)

        # Load the tensors
        loaded_dict = load_v2(name)

        # Check that we have exactly the expected keys
        assert set(loaded_dict.keys()) == {
            "effects_eE",
            "cooccurrences_ee",
            "how_often_activated_e",
        }

        # Check that each tensor matches the original
        assert torch.equal(effects_eE, loaded_dict["effects_eE"])
        assert torch.equal(cooccurrences_ee, loaded_dict["cooccurrences_ee"])
        assert torch.equal(how_often_activated_e, loaded_dict["how_often_activated_e"])

    finally:
        # Clean up the temporary file
        if os.path.exists(name):
            os.remove(name)
