#!/usr/bin/env python3

import torch


def test_batched_indices():
    n_reader = 101

    # Collect all values and their indices
    all_indices = []
    ablator_idxs = [3, 4, 6, 8, 1, 67, 5]
    for ablator_idx in ablator_idxs:
        indices = torch.arange(n_reader)
        all_indices.append(indices + ablator_idx * n_reader)

    all_indices = torch.cat(all_indices)

    all_indices2 = (
        (torch.tensor(ablator_idxs) * n_reader).unsqueeze(1) + torch.arange(n_reader)
    ).view(-1)
    assert torch.equal(all_indices, all_indices2)
