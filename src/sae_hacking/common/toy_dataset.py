from typing import Optional

import torch
from beartype import beartype
from jaxtyping import Bool, Float, jaxtyped


class ToyDataset:
    N_FEATURES = 10
    N_DIMS = 100
    N_CHILDREN = 2
    ACTIVATION_PROB = 0.2
    PERTURBATION_SIZE = 0.2

    features: Float[torch.Tensor, "n_features n_dim"]
    perturbations: Float[torch.Tensor, "n_features n_children n_dim"]

    @beartype
    def __init__(self, seed: Optional[int] = None):
        if seed is not None:
            torch.manual_seed(seed)
        self.features = torch.randn(self.N_FEATURES, self.N_DIMS)
        self.features = self.features / self.features.norm(dim=1, keepdim=True)

        raw_perturbations = torch.randn(self.N_FEATURES, self.N_CHILDREN, self.N_DIMS)
        self.perturbations = (
            self.PERTURBATION_SIZE
            * raw_perturbations
            / raw_perturbations.norm(dim=2, keepdim=True)
        )

    @jaxtyped(typechecker=beartype)
    def generate(
        self, num_samples: int = 1
    ) -> Float[torch.Tensor, "num_samples n_dim"]:
        activations: Bool[torch.Tensor, "num_samples n_features"] = (
            torch.rand(num_samples, self.N_FEATURES) < self.ACTIVATION_PROB
        )
        perturbation_choices = torch.randint(
            0, self.N_CHILDREN, (num_samples, self.N_FEATURES)
        )

        result = torch.zeros(num_samples, self.N_DIMS)
        for i in range(num_samples):
            for j in range(self.N_FEATURES):
                if activations[i, j]:
                    perturbed_feature = (
                        self.features[j]
                        + self.perturbations[j, perturbation_choices[i, j]]
                    )
                    result[i] += perturbed_feature

        return result
