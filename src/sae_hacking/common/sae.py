#!/usr/bin/env python3


import torch
from beartype import beartype
from jaxtyping import Float, jaxtyped


class ReluSparseAutoEncoder(torch.nn.Module):
    @beartype
    def __init__(self, sae_hidden_dim: int):
        super().__init__()
        self.sae_hidden_dim = sae_hidden_dim
        llm_hidden_dim = 768
        self.encoder = torch.nn.Linear(llm_hidden_dim, sae_hidden_dim)
        self.decoder = torch.nn.Linear(sae_hidden_dim, llm_hidden_dim)

    @jaxtyped(typechecker=beartype)
    def forward(
        self, llm_activations: Float[torch.Tensor, "1 seq_len 768"]
    ) -> tuple[
        Float[torch.Tensor, "1 seq_len 768"],
        Float[torch.Tensor, "1 seq_len {self.sae_hidden_dim}"],
    ]:
        sae_activations = self.get_features(llm_activations)
        feat_magnitudes = get_feature_magnitudes(
            sae_activations, self.decoder.weight.transpose(0, 1)
        )
        reconstructed = self.decoder(sae_activations)
        return reconstructed, feat_magnitudes

    @jaxtyped(typechecker=beartype)
    def get_features(
        self, llm_activations: Float[torch.Tensor, "1 seq_len 768"]
    ) -> Float[torch.Tensor, "1 seq_len {self.sae_hidden_dim}"]:
        return torch.nn.functional.relu(self.encoder(llm_activations))


@jaxtyped(typechecker=beartype)
def get_feature_magnitudes(
    sae_activations: Float[torch.Tensor, "1 seq_len sae_hidden_dim"],
    decoder_weight: Float[torch.Tensor, "sae_hidden_dim 768"],
) -> Float[torch.Tensor, "1 seq_len sae_hidden_dim"]:
    decoder_magnitudes = torch.linalg.vector_norm(decoder_weight, dim=1, ord=2)
    result = sae_activations * decoder_magnitudes
    return result


class TopkSparseAutoEncoder(torch.nn.Module):
    @beartype
    def __init__(self, sae_hidden_dim: int):
        super().__init__()
        self.sae_hidden_dim = sae_hidden_dim
        llm_hidden_dim = 768
        self.encoder = torch.nn.Linear(llm_hidden_dim, sae_hidden_dim)
        self.decoder = torch.nn.Linear(sae_hidden_dim, llm_hidden_dim)
        self.k = 150

    @jaxtyped(typechecker=beartype)
    def forward(
        self, llm_activations: Float[torch.Tensor, "1 seq_len 768"]
    ) -> Float[torch.Tensor, "1 seq_len 768"]:
        pre_activations = self.encoder(llm_activations)
        topk = torch.topk(pre_activations, self.k)
        # Just zero out the parts of the decoder matrix that isn't in the topk
        # Later look at instead making the decoder matrix smaller with torch.gather
        # for efficiency
        sae_activations = torch.scatter(
            input=torch.zeros_like(pre_activations),
            dim=2,
            index=topk.indices,
            src=topk.values,
        )

        reconstructed = self.decoder(sae_activations)
        return reconstructed


class TopkSparseAutoEncoder2Child(torch.nn.Module):
    @beartype
    def __init__(self, sae_hidden_dim: int):
        super().__init__()
        self.sae_hidden_dim = sae_hidden_dim
        llm_hidden_dim = 768
        self.encoder = torch.nn.Linear(llm_hidden_dim, sae_hidden_dim)
        self.decoder = torch.nn.Linear(sae_hidden_dim, llm_hidden_dim)
        # Maybe I should use matrices and bias vectors directly instead
        # of Linear; it might be easier to have that level of control.
        self.encoder_child1 = torch.nn.Linear(llm_hidden_dim, sae_hidden_dim)
        self.decoder_child1 = torch.nn.Linear(sae_hidden_dim, llm_hidden_dim)
        self.encoder_child2 = torch.nn.Linear(llm_hidden_dim, sae_hidden_dim)
        self.decoder_child2 = torch.nn.Linear(sae_hidden_dim, llm_hidden_dim)
        self.k = 150
        # Let's guess P(child_i activates | parent activates) should be 50%
        self.per_child_k = self.k // 2

    @jaxtyped(typechecker=beartype)
    def forward(
        self, llm_activations: Float[torch.Tensor, "1 seq_len 768"]
    ) -> Float[torch.Tensor, "1 seq_len 768"]:
        pre_activations = self.encoder(llm_activations)
        topk = torch.topk(pre_activations, self.k)
        sae_activations = torch.scatter(
            input=torch.zeros_like(pre_activations),
            dim=2,
            index=topk.indices,
            src=topk.values,
        )

        # This is wasting compute and memory because we already know which indices
        # we're going to throw away
        pre_activations_child1 = self.encoder_child1(llm_activations)
        pre_activations_child2 = self.encoder_child2(llm_activations)

        # Filter down each child to only activate where the parent
        # does
        masked_activations_child1 = torch.where(
            sae_activations != 0.0,
            pre_activations_child1,
            torch.zeros_like(pre_activations_child1),
        )

        masked_activations_child2 = torch.where(
            sae_activations != 0.0,
            pre_activations_child2,
            torch.zeros_like(pre_activations_child2),
        )

        # Now take topk of each child

        topk_child1 = torch.topk(masked_activations_child1, self.per_child_k)
        topk_child2 = torch.topk(masked_activations_child2, self.per_child_k)

        final_activations_child1 = torch.scatter(
            input=torch.zeros_like(masked_activations_child1),
            dim=2,
            index=topk_child1.indices,
            src=topk_child1.values,
        )

        final_activations_child2 = torch.scatter(
            input=torch.zeros_like(masked_activations_child2),
            dim=2,
            index=topk_child2.indices,
            src=topk_child2.values,
        )

        reconstructed = (
            self.decoder(sae_activations)
            + self.decoder_child1(final_activations_child1)
            + self.decoder_child2(final_activations_child2)
        )
        return reconstructed


class TopkSparseAutoEncoder_v2(torch.nn.Module):
    @beartype
    def __init__(self, sae_hidden_dim: int, model_dim: int):
        super().__init__()
        self.encoder = torch.nn.Linear(model_dim, sae_hidden_dim)
        self.decoder = torch.nn.Linear(sae_hidden_dim, model_dim)
        self.k = 3

    @jaxtyped(typechecker=beartype)
    def forward(
        self, model_activations: Float[torch.Tensor, "batch_size model_dim"]
    ) -> tuple[Float[torch.Tensor, "batch_size model_dim"], int]:
        pre_activations = self.encoder(model_activations)
        topk = torch.topk(pre_activations, self.k)
        # Just zero out the parts of the decoder matrix that aren't in the topk
        # Later look at instead making the decoder matrix smaller with torch.gather
        # for efficiency
        sae_activations = torch.scatter(
            input=torch.zeros_like(pre_activations),
            dim=1,
            index=topk.indices,
            src=topk.values,
        )

        num_live_latents = len(topk.indices.unique())

        reconstructed = self.decoder(sae_activations)
        return reconstructed, num_live_latents


class TopkSparseAutoEncoder2Child_v2(torch.nn.Module):
    @beartype
    def __init__(self, sae_hidden_dim: int, model_dim: int):
        super().__init__()
        self.sae_hidden_dim = sae_hidden_dim
        self.encoder = torch.nn.Linear(model_dim, sae_hidden_dim)
        self.decoder = torch.nn.Linear(sae_hidden_dim, model_dim)
        # Maybe I should use matrices and bias vectors directly instead
        # of Linear; it might be easier to have that level of control.
        self.encoder_child1 = torch.nn.Linear(model_dim, sae_hidden_dim)
        self.decoder_child1 = torch.nn.Linear(sae_hidden_dim, model_dim)
        self.encoder_child2 = torch.nn.Linear(model_dim, sae_hidden_dim)
        self.decoder_child2 = torch.nn.Linear(sae_hidden_dim, model_dim)
        self.k = 3
        self.child1_parent_ratios = torch.zeros(sae_hidden_dim)
        self.child2_parent_ratios = torch.zeros(sae_hidden_dim)

    @jaxtyped(typechecker=beartype)
    def forward(
        self, model_activations: Float[torch.Tensor, "batch_size model_dim"]
    ) -> tuple[Float[torch.Tensor, "batch_size model_dim"], tuple[int, int, int]]:
        pre_activations = self.encoder(model_activations)
        topk = torch.topk(pre_activations, self.k)
        sae_activations = torch.scatter(
            input=torch.zeros_like(pre_activations),
            dim=1,
            index=topk.indices,
            src=topk.values,
        )

        num_live_parent_latents = len(topk.indices.unique())

        # This is wasting compute and memory because we already know which indices
        # we're going to throw away
        pre_activations_child1 = self.encoder_child1(model_activations)
        pre_activations_child2 = self.encoder_child2(model_activations)

        # Filter down each child to only activate where the parent
        # does
        masked_activations_child1 = torch.where(
            sae_activations != 0.0,
            pre_activations_child1,
            torch.zeros_like(pre_activations_child1),
        )
        masked_activations_child2 = torch.where(
            sae_activations != 0.0,
            pre_activations_child2,
            torch.zeros_like(pre_activations_child2),
        )

        # Compare children and keep only the winner where parent is active
        winners_mask = masked_activations_child1 > masked_activations_child2
        final_activations_child1 = torch.where(
            winners_mask,
            masked_activations_child1,
            torch.zeros_like(masked_activations_child1),
        )
        final_activations_child2 = torch.where(
            ~winners_mask,
            masked_activations_child2,
            torch.zeros_like(masked_activations_child2),
        )
        num_live_child1_latents = torch.sum(
            torch.any(final_activations_child1 != 0, dim=0)
        ).item()
        num_live_child2_latents = torch.sum(
            torch.any(final_activations_child2 != 0, dim=0)
        ).item()

        reconstructed = (
            self.decoder(sae_activations)
            + self.decoder_child1(final_activations_child1)
            + self.decoder_child2(final_activations_child2)
        )
        update_parent_child_ratio(
            sae_activations,
            final_activations_child1,
            final_activations_child2,
            self.child1_parent_ratios,
            self.child2_parent_ratios,
        )
        return reconstructed, (
            num_live_parent_latents,
            num_live_child1_latents,
            num_live_child2_latents,
        )


@jaxtyped(typechecker=beartype)
def update_parent_child_ratio(
    parent_activations: Float[torch.Tensor, "batch_size sae_dim"],
    child1_activations: Float[torch.Tensor, "batch_size sae_dim"],
    child2_activations: Float[torch.Tensor, "batch_size sae_dim"],
    child1_parent_ratios: Float[torch.Tensor, " sae_dim"],
    child2_parent_ratios: Float[torch.Tensor, " sae_dim"],
) -> None:
    # TODO Maybe this should be a method
    batch_size, sae_dim = parent_activations.shape
    EMA_COEFF = 0.01
    for b in range(batch_size):
        for s in range(sae_dim):
            if parent_activations[b, s] != 0 and child1_activations[b, s] != 0:
                child1_parent_ratios[s] = (1 - EMA_COEFF) * child1_parent_ratios[
                    s
                ] + EMA_COEFF * child1_activations[b, s] / parent_activations[b, s]
            if parent_activations[b, s] != 0 and child2_activations[b, s] != 0:
                child2_parent_ratios[s] = (1 - EMA_COEFF) * child2_parent_ratios[
                    s
                ] + EMA_COEFF * child2_activations[b, s] / parent_activations[b, s]


@jaxtyped(typechecker=beartype)
def update_parent_child_ratio2(
    parent_activations: Float[torch.Tensor, "batch_size sae_dim"],
    child1_activations: Float[torch.Tensor, "batch_size sae_dim"],
    child2_activations: Float[torch.Tensor, "batch_size sae_dim"],
    child1_parent_ratios: Float[torch.Tensor, " sae_dim"],
    child2_parent_ratios: Float[torch.Tensor, " sae_dim"],
) -> None:
    EMA_COEFF = 0.01

    # Create masks for non-zero parent and child activations
    parent_nonzero = parent_activations != 0
    child1_nonzero = child1_activations != 0
    child2_nonzero = child2_activations != 0

    # Combined masks
    mask1 = parent_nonzero & child1_nonzero
    mask2 = parent_nonzero & child2_nonzero

    # For each valid pair, update the EMA
    for b in range(parent_activations.shape[0]):
        # Update child1 ratios where valid
        valid_indices1 = mask1[b]
        if valid_indices1.any():
            child1_parent_ratios[valid_indices1] = (
                1 - EMA_COEFF
            ) * child1_parent_ratios[valid_indices1] + EMA_COEFF * (
                child1_activations[b, valid_indices1]
                / parent_activations[b, valid_indices1]
            )

        # Update child2 ratios where valid
        valid_indices2 = mask2[b]
        if valid_indices2.any():
            child2_parent_ratios[valid_indices2] = (
                1 - EMA_COEFF
            ) * child2_parent_ratios[valid_indices2] + EMA_COEFF * (
                child2_activations[b, valid_indices2]
                / parent_activations[b, valid_indices2]
            )
