#!/usr/bin/env python3
import torch

from sae_hacking.gemma_ablate import graph_ablation_matrix


def test_plot():
    graph_ablation_matrix(
        torch.randn(100, 100),
        "gemma-2-2b/20-gemmascope-res-65k",
        "gemma-2-2b/21-gemmascope-mlp-65k",
        "/tmp",
    )
