import torch
from beartype import beartype
from einops import einsum
from jaxtyping import Bool, Float, Int, jaxtyped


class ToyDataset:
    N_CHILDREN_PER_PARENT = 2
    ACTIVATION_PROB = 0.03

    features: Float[torch.Tensor, "n_features model_dim"]
    perturbations: Float[torch.Tensor, "n_features n_children model_dim"]

    @beartype
    def __init__(
        self, num_features: int, cuda: bool, perturbation_size: float, model_dim: int
    ) -> None:
        self.device = torch.device("cuda" if cuda else "cpu")
        self.n_features = num_features
        self.features = torch.randn(self.n_features, model_dim, device=self.device)
        self.features = self.features / torch.linalg.vector_norm(
            self.features, dim=1, keepdim=True
        )

        raw_perturbations = torch.randn(
            self.n_features, self.N_CHILDREN_PER_PARENT, model_dim, device=self.device
        )
        self.perturbations = (
            perturbation_size
            * raw_perturbations
            / torch.linalg.vector_norm(raw_perturbations, dim=2, keepdim=True)
        )

    @jaxtyped(typechecker=beartype)
    def generate(
        self, batch_size: int
    ) -> tuple[Float[torch.Tensor, "batch_size model_dim"], Int[torch.Tensor, ""]]:
        k = 3
        activations = torch.zeros(
            batch_size, self.n_features, dtype=torch.bool, device=self.device
        )
        perm = torch.stack(
            [
                torch.randperm(self.n_features, device=self.device)
                for _ in range(batch_size)
            ]
        )
        selected_indices = perm[:, :k]
        activations[torch.arange(batch_size).unsqueeze(1), selected_indices] = True

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
    """compute_result2 is the same but faster"""
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
        "batch_size n_features model_dim, batch_size n_features -> batch_size model_dim",
    )

    return result
