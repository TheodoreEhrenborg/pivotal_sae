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
