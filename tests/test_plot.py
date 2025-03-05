#!/usr/bin/env python3
import random

import torch

from sae_hacking.graph_network import graph_ablation_matrix


def test_plot():
    d = {}
    for _ in range(50):
        i = random.randrange(100)
        d[i] = torch.randn(10)
    graph_ablation_matrix(
        d,
        "gemma-2-2b/20-gemmascope-res-65k",
        "gemma-2-2b/21-gemmascope-mlp-65k",
        "/tmp",
        100,
    )
