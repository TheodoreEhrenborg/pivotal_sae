#!/usr/bin/env python3

import torch

from sae_hacking.gemma_ablate import gather_co_occurrences, gather_co_occurrences2


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


    r1 = gather_co_occurrences( activations)
    r2 = gather_co_occurrences2( activations)

    assert torch.allclose(r1, r2.to_dense())
