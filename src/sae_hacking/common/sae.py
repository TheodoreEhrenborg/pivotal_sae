#!/usr/bin/env python3
import torch
import torch.nn.functional as F
from beartype import beartype
from jaxtyping import Bool, Float, jaxtyped


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
    def __init__(
        self,
        sae_hidden_dim: int,
        model_dim: int,
        aux_loss_threshold: float,
        aux_loss_coeff: float,
    ):
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
        self.aux_loss_threshold = aux_loss_threshold
        self.use_aux_loss = aux_loss_coeff != 0

    @jaxtyped(typechecker=beartype)
    def forward(
        self, model_activations: Float[torch.Tensor, "batch_size model_dim"]
    ) -> tuple[
        Float[torch.Tensor, "batch_size model_dim"],
        tuple[int, int, int],
        Float[torch.Tensor, ""],
    ]:
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
        if self.training:
            with torch.no_grad():
                update_parent_child_ratio3(
                    sae_activations,
                    final_activations_child1,
                    final_activations_child2,
                    self.child1_parent_ratios,
                    self.child2_parent_ratios,
                )

        aux_loss = (
            auxiliary_loss(
                sae_activations,
                model_activations,
                winners_mask,
                final_activations_child1,
                final_activations_child2,
                self.decoder.weight,
                self.decoder_child1.weight,
                self.decoder_child2.weight,
                self.aux_loss_threshold,
            )
            if self.use_aux_loss
            else torch.tensor(0.0)
        )

        return (
            reconstructed,
            (
                num_live_parent_latents,
                num_live_child1_latents,
                num_live_child2_latents,
            ),
            aux_loss,
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
            child1_parent_ratios[s] = (1 - EMA_COEFF) * child1_parent_ratios[
                s
            ] + EMA_COEFF * sum(child1_new_ratios) / len(child1_new_ratios)
        if child2_new_ratios:
            child2_parent_ratios[s] = (1 - EMA_COEFF) * child2_parent_ratios[
                s
            ] + EMA_COEFF * sum(child2_new_ratios) / len(child2_new_ratios)


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

    child1_parent_ratios[update_mask1] = (1 - EMA_COEFF) * child1_parent_ratios[
        update_mask1
    ] + EMA_COEFF * batch_means1[update_mask1]
    child2_parent_ratios[update_mask2] = (1 - EMA_COEFF) * child2_parent_ratios[
        update_mask2
    ] + EMA_COEFF * batch_means2[update_mask2]


@jaxtyped(typechecker=beartype)
def auxiliary_loss_reference(
    # TODO Fix the test
    sae_activations: Float[torch.Tensor, "batch_size n_features"],
    model_activations: Float[torch.Tensor, "batch_size model_dim"],
    winners_mask: Bool[torch.Tensor, "batch_size n_features"],
    final_activations_child1: Float[torch.Tensor, "batch_size n_features"],
    final_activations_child2: Float[torch.Tensor, "batch_size n_features"],
    decoder_weight: Float[torch.Tensor, "model_dim n_features"],
    decoder_child1_weight: Float[torch.Tensor, "model_dim n_features"],
    decoder_child2_weight: Float[torch.Tensor, "model_dim n_features"],
) -> Float[torch.Tensor, ""]:
    batch_size = sae_activations.shape[0]
    aux_loss = torch.tensor(0.0, device=model_activations.device)

    # Get active feature indices per batch element
    for batch_idx in range(batch_size):
        # Get features active for this batch element
        active_indices = torch.nonzero(sae_activations[batch_idx]).squeeze()

        for feat_idx in active_indices:
            # Get parent weights and activation for this feature
            parent_weights = decoder_weight[:, feat_idx]
            parent_scale = sae_activations[batch_idx, feat_idx]
            scaled_parent = parent_weights * parent_scale

            # Determine which child won for this feature
            is_child1_winner = winners_mask[batch_idx, feat_idx]

            if is_child1_winner:
                child_weights = decoder_child1_weight[:, feat_idx]
                child_scale = final_activations_child1[batch_idx, feat_idx]
            else:
                child_weights = decoder_child2_weight[:, feat_idx]
                child_scale = final_activations_child2[batch_idx, feat_idx]

            scaled_child = child_weights * child_scale

            # Calculate cosine similarity between parent and (parent + child)
            combined_weights = scaled_parent + scaled_child
            cos_sim = F.cosine_similarity(scaled_parent, combined_weights, dim=0)

            # Add to loss if similarity is too low
            if cos_sim < 0.5:
                aux_loss += 0.5 - cos_sim

    # Average across batch
    aux_loss = aux_loss / batch_size

    return aux_loss


@jaxtyped(typechecker=beartype)
def auxiliary_loss(
    sae_activations: Float[torch.Tensor, "batch_size n_features"],
    # TODO Unused variable
    model_activations: Float[torch.Tensor, "batch_size model_dim"],
    winners_mask: Bool[torch.Tensor, "batch_size n_features"],
    final_activations_child1: Float[torch.Tensor, "batch_size n_features"],
    final_activations_child2: Float[torch.Tensor, "batch_size n_features"],
    decoder_weight: Float[torch.Tensor, "model_dim n_features"],
    decoder_child1_weight: Float[torch.Tensor, "model_dim n_features"],
    decoder_child2_weight: Float[torch.Tensor, "model_dim n_features"],
    aux_loss_threshold: float,
) -> Float[torch.Tensor, ""]:
    # Expand activations for broadcasting
    sae_acts_expanded = sae_activations.unsqueeze(-2)  # [batch_size, 1, n_features]

    # Scale parent weights by activations
    scaled_parent = (
        decoder_weight.unsqueeze(0) * sae_acts_expanded
    )  # [batch_size, model_dim, n_features]

    # Scale child weights by activations and winner mask
    child1_acts = final_activations_child1.unsqueeze(-2)  # [batch_size, 1, n_features]
    child2_acts = final_activations_child2.unsqueeze(-2)  # [batch_size, 1, n_features]

    scaled_child1 = (
        decoder_child1_weight.unsqueeze(0) * child1_acts
    )  # [batch_size, model_dim, n_features]
    scaled_child2 = (
        decoder_child2_weight.unsqueeze(0) * child2_acts
    )  # [batch_size, model_dim, n_features]

    # Select appropriate child based on winners_mask
    winners_mask_expanded = winners_mask.unsqueeze(1)  # [batch_size, 1, n_features]
    scaled_child = torch.where(winners_mask_expanded, scaled_child1, scaled_child2)

    # Calculate combined weights
    combined_weights = (
        scaled_parent + scaled_child
    )  # [batch_size, model_dim, n_features]

    # Calculate cosine similarity using torch.nn.functional
    cos_sim = F.cosine_similarity(
        scaled_parent.transpose(1, 2),  # [batch_size, n_features, model_dim]
        combined_weights.transpose(1, 2),  # [batch_size, n_features, model_dim]
        dim=2,  # compute similarity along model_dim
    )  # [batch_size, n_features]

    # Calculate loss only for active features
    active_mask = sae_activations > 0
    similarity_loss = torch.maximum(
        torch.tensor(0.0, device=cos_sim.device), aux_loss_threshold - cos_sim
    )
    masked_loss = similarity_loss * active_mask

    # Average across batch and features
    total_active = active_mask.sum()
    aux_loss = masked_loss.sum() / (total_active + 1e-8)

    return aux_loss
