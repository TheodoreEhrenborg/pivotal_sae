import torch
from beartype import beartype
from jaxtyping import Bool, Float, Int, jaxtyped


class ToyDataset:
    N_DIMS = 10
    N_CHILDREN_PER_PARENT = 2
    ACTIVATION_PROB = 0.03
    PERTURBATION_SIZE = 0.2

    features: Float[torch.Tensor, "n_features n_dim"]
    perturbations: Float[torch.Tensor, "n_features n_children n_dim"]

    @beartype
    def __init__(self, num_features: int, seed: int) -> None:
        torch.manual_seed(seed)
        self.n_features = num_features
        self.features = torch.randn(self.n_features, self.N_DIMS)
        # TODO "torch.norm is deprecated and may be removed in a future PyTorch release. Its documentation and behavior may be incorrect, and it is no longer actively maintained."
        self.features = self.features / self.features.norm(dim=1, keepdim=True)

        raw_perturbations = torch.randn(
            self.n_features, self.N_CHILDREN_PER_PARENT, self.N_DIMS
        )
        self.perturbations = (
            self.PERTURBATION_SIZE
            * raw_perturbations
            / raw_perturbations.norm(dim=2, keepdim=True)
        )

    @jaxtyped(typechecker=beartype)
    def generate(
        self, num_samples: int = 1
    ) -> tuple[Float[torch.Tensor, "num_samples {self.N_DIMS}"], Int[torch.Tensor, ""]]:
        active_features = 0
        # TODO This check really should make sure each of num_samples has >0 features
        while active_features == 0:
            activations: Bool[torch.Tensor, "num_samples n_features"] = (
                torch.rand(num_samples, self.n_features) < self.ACTIVATION_PROB
            )
            active_features = activations.sum()
        perturbation_choices = torch.randint(
            0, self.N_CHILDREN_PER_PARENT, (num_samples, self.n_features)
        )

        result = torch.zeros(num_samples, self.N_DIMS)
        for i in range(num_samples):
            for j in range(self.n_features):
                if activations[i, j]:
                    perturbed_feature = (
                        self.features[j]
                        + self.perturbations[j, perturbation_choices[i, j]]
                    )
                    result[i] += perturbed_feature

        return result, activations.sum()
