#!/usr/bin/env python3

import torch
import zstandard
from beartype import beartype
from safetensors.torch import save


@beartype
def save_dict_with_tensors(tensor_dict: dict, save_path: str) -> None:
    """
    Save a dictionary containing tensors to a safetensors file
    Non-tensor values are stored as tensor metadata.

    Args:
        tensor_dict: Dictionary containing tensors
        save_path: Path to save the safetensors file
    """
    # Make sure the save_path has .zst extension
    assert save_path.endswith(".zst")

    for _, value in tensor_dict.items():
        assert isinstance(value, torch.Tensor)

    # Save tensors with metadata
    uncompressed_data = save(tensor_dict)

    compressor = zstandard.ZstdCompressor()
    compressed_data = compressor.compress(uncompressed_data)

    with open(save_path, "wb") as f_out:
        f_out.write(compressed_data)
