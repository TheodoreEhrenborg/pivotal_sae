#!/usr/bin/env python3
import os

import torch
import zstandard
from beartype import beartype
from safetensors.torch import save_file


@beartype
def save_dict_with_tensors(tensor_dict: dict, save_path: str) -> None:
    """
    Save a dictionary containing tensors to a safetensors file
    Non-tensor values are stored as tensor metadata.

    Args:
        tensor_dict: Dictionary containing tensors
        save_path: Path to save the safetensors file
    """

    for _, value in tensor_dict.items():
        assert isinstance(value, torch.Tensor)

    # Make sure the save_path has .zst extension
    assert save_path.endswith(".zst")
    # Use a temporary file for the uncompressed version
    safetensors_path = save_path[:-4]  # Remove .zst

    # Save tensors with metadata
    save_file(tensor_dict, safetensors_path)

    with open(safetensors_path, "rb") as f_in:
        uncompressed_data = f_in.read()

    compressor = zstandard.ZstdCompressor()
    compressed_data = compressor.compress(uncompressed_data)

    with open(save_path, "wb") as f_out:
        f_out.write(compressed_data)

    # Remove the temporary uncompressed file
    os.remove(safetensors_path)
