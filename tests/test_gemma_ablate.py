#!/usr/bin/env python3


import torch
from beartype import beartype

from sae_hacking.gemma_utils import gather_co_occurrences2


def test_co_occurrences():
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

    r1 = gather_co_occurrences(activations)
    r2 = gather_co_occurrences2(activations)

    assert torch.allclose(r1, r2.to_dense())


@beartype
def gather_co_occurrences(ablator_acts_1Se) -> torch.Tensor:
    e = ablator_acts_1Se.shape[2]
    these_cooccurrences_ee = torch.zeros(e, e)
    # Update co-occurrence matrix for ablator features
    # For each position in the sequence
    for pos in range(ablator_acts_1Se.shape[1]):
        # Get feature activations at this position
        pos_acts_e = ablator_acts_1Se[0, pos]
        # Find which features are active (> 0)
        active_features = torch.where(pos_acts_e > 0)[0]

        # Update co-occurrence counts for each pair of active features
        for i in range(len(active_features)):
            for j in range(len(active_features)):
                feat1, feat2 = active_features[i].item(), active_features[j].item()
                these_cooccurrences_ee[feat1, feat2] += 1
    return these_cooccurrences_ee
