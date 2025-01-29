from typing import Optional

import torch
from beartype import beartype
from jaxtyping import Bool, Float, jaxtyped


class ToyDataset:
    features: Float[torch.Tensor, "n_features n_dim"]

    @beartype
    def __init__(self, seed: Optional[int] = None):
        if seed is not None:
            torch.manual_seed(seed)
        self.features = torch.randn(10, 100)

    @jaxtyped(typechecker=beartype)
    def generate(
        self, num_samples: int = 1
    ) -> Float[torch.Tensor, "num_samples n_dim"]:
        activations: Bool[torch.Tensor, "num_samples n_features"] = (
            torch.rand(num_samples, 10) < 0.2
        )
        activations_float: Float[torch.Tensor, "num_samples n_features"] = (
            activations.float()
        )
        result: Float[torch.Tensor, "num_samples n_dim"] = torch.matmul(
            activations_float, self.features
        )
        return result
