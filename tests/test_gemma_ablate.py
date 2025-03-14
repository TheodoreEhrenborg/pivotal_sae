#!/usr/bin/env python3

import torch

from sae_hacking.gemma_ablate import update_co_occurrences, update_co_occurrences2


def test_co_occurrences():
    num_features = 4
    activations = torch.tensor(
        [
            [
                [1.0, -0.5, 0.0, 2.0],
                [0.0, 1.0, 2.0, -1.0],
                [3.0, 0.0, -0.5, 0.0],
                [0.0, 0.0, 0.0, 0.0],
                [1.0, 1.0, 1.0, 1.0],
            ]
        ]
    )

    cooccurrences1 = torch.zeros(num_features, num_features)
    cooccurrences2 = torch.zeros(num_features, num_features)

    update_co_occurrences(cooccurrences1, activations)
    update_co_occurrences2(cooccurrences2, activations)

    assert torch.allclose(cooccurrences1, cooccurrences2)
