import torch
from einops import repeat, einsum
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
    def __init__(self, num_features: int, seed: int, cuda: bool) -> None:
        self.device = torch.device("cuda" if cuda else "cpu")
        torch.manual_seed(seed)
        self.n_features = num_features
        self.features = torch.randn(self.n_features, self.N_DIMS, device=self.device)
        # TODO "torch.norm is deprecated and may be removed in a future PyTorch release. Its documentation and behavior may be incorrect, and it is no longer actively maintained."
        self.features = self.features / self.features.norm(dim=1, keepdim=True)

        raw_perturbations = torch.randn(
            self.n_features, self.N_CHILDREN_PER_PARENT, self.N_DIMS, device=self.device
        )
        self.perturbations = (
            self.PERTURBATION_SIZE
            * raw_perturbations
            / raw_perturbations.norm(dim=2, keepdim=True)
        )

    @jaxtyped(typechecker=beartype)
    def generate(
        self, batch_size: int
    ) -> tuple[Float[torch.Tensor, "batch_size {self.N_DIMS}"], Int[torch.Tensor, ""]]:
        activations: Bool[torch.Tensor, "batch_size n_features"] = (
            torch.rand(batch_size, self.n_features, device=self.device)
            < self.ACTIVATION_PROB
        )
        perturbation_choices = torch.randint(
            0,
            self.N_CHILDREN_PER_PARENT,
            (batch_size, self.n_features),
            device=self.device,
        )

        result = compute_result2(
            activations,
            perturbation_choices,
            self.features,
            self.perturbations,
            self.device,
        )

        return result, activations.sum()


@jaxtyped(typechecker=beartype)
def compute_result(
    activations: Bool[torch.Tensor, "batch_size n_features"],
    perturbation_choices: Int[torch.Tensor, "batch_size n_features"],
    features: Float[torch.Tensor, "n_features model_dim"],
    perturbations: Float[torch.Tensor, "n_features n_children model_dim"],
    device: torch.device,
) -> Float[torch.Tensor, "batch_size model_dim"]:
    result = torch.zeros(activations.shape[0], features.shape[1], device=device)
    for i in range(activations.shape[0]):
        for j in range(activations.shape[1]):
            if activations[i, j]:
                perturbed_feature = (
                    features[j] + perturbations[j, perturbation_choices[i, j]]
                )
                result[i] += perturbed_feature
    return result

@jaxtyped(typechecker=beartype)
def compute_result2(
    activations: Bool[torch.Tensor, "batch_size n_features"],
    perturbation_choices: Int[torch.Tensor, "batch_size n_features"],
    features: Float[torch.Tensor, "n_features model_dim"],
    perturbations: Float[torch.Tensor, "n_features n_children model_dim"],
    device: torch.device,
) -> Float[torch.Tensor, "batch_size model_dim"]:
    batch_size, n_features = activations.shape
    feature_indices = torch.arange(n_features, device=device)

    # feature_indices is broadcast to match perturbation_choices,
    # and then they are iterated over in lockstep
    # See https://numpy.org/doc/2.2/user/basics.indexing.html#integer-array-indexing
    selected_perturbations = perturbations[feature_indices, perturbation_choices]

    model_dim = features.shape[1]
    assert selected_perturbations.shape == (batch_size, n_features, model_dim)

    perturbed_features = features + selected_perturbations

    result = einsum(
        perturbed_features,
        activations.float(),
        'batch_size n_features model_dim, batch_size n_features -> batch_size model_dim'
    )

    return result
