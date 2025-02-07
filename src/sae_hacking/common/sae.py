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
        # TODO The device should be configurable
        self.child1_parent_ratios = torch.ones(sae_hidden_dim).cuda()
        self.child2_parent_ratios = torch.ones(sae_hidden_dim).cuda()

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
        with torch.no_grad():
            update_parent_child_ratio3(
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
    for s in range(sae_dim):
        child1_new_ratios = []
        child2_new_ratios = []
        for b in range(batch_size):
            if parent_activations[b, s] != 0 and child1_activations[b, s] != 0:
                child1_new_ratios.append(
                    child1_activations[b, s] / parent_activations[b, s]
                )
            if parent_activations[b, s] != 0 and child2_activations[b, s] != 0:
                child2_new_ratios.append(
                    child2_activations[b, s] / parent_activations[b, s]
                )
        if child1_new_ratios:
            # Adjust the moving average by more if there were more nonzero activations
            # in this batch
            child1_parent_ratios[s] = (
                1 - EMA_COEFF * len(child1_new_ratios)
            ) * child1_parent_ratios[s] + EMA_COEFF * sum(child1_new_ratios)
        if child2_new_ratios:
            child2_parent_ratios[s] = (
                1 - EMA_COEFF * len(child2_new_ratios)
            ) * child2_parent_ratios[s] + EMA_COEFF * sum(child2_new_ratios)


@jaxtyped(typechecker=beartype)
def update_parent_child_ratio3(
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

    # Calculate ratios where both parent and child are non-zero
    new_ratios1 = torch.where(
        mask1,
        child1_activations / parent_activations,
        torch.zeros_like(child1_activations),
    )
    new_ratios2 = torch.where(
        mask2,
        child2_activations / parent_activations,
        torch.zeros_like(child2_activations),
    )

    # Take mean across batch dimension for non-zero entries
    valid_counts1 = mask1.sum(dim=0)
    valid_counts2 = mask2.sum(dim=0)

    # Update ratios using EMA, but only where we have valid new ratios
    batch_means1 = torch.sum(new_ratios1, dim=0) / torch.clamp(valid_counts1, min=1)
    batch_means2 = torch.sum(new_ratios2, dim=0) / torch.clamp(valid_counts2, min=1)

    # Update only where we had valid entries
    update_mask1 = valid_counts1 > 0
    update_mask2 = valid_counts2 > 0

    child1_parent_ratios[update_mask1] = (
        1 - EMA_COEFF * valid_counts1[update_mask1]
    ) * child1_parent_ratios[update_mask1] + EMA_COEFF * batch_means1[
        update_mask1
    ] * valid_counts1[update_mask1]
    child2_parent_ratios[update_mask2] = (
        1 - EMA_COEFF * valid_counts2[update_mask2]
    ) * child2_parent_ratios[update_mask2] + EMA_COEFF * batch_means2[
        update_mask2
    ] * valid_counts2[update_mask2]
