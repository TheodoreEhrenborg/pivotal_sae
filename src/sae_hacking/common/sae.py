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
        # Let's guess P(child_i activates | parent activates) should be 50%
        self.per_child_k = self.k // 2

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
            pre_activations_child1,
            sae_activations != 0,
            torch.zeros_like(pre_activations_child1),
        )

        masked_activations_child2 = torch.where(
            pre_activations_child2,
            sae_activations != 0,
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
