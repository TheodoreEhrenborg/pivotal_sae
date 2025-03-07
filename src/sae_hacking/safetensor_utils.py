#!/usr/bin/env python3

import safetensors.torch
import torch
import zstandard
from beartype import beartype


@beartype
def save_dict_with_tensors(
    tensor_dict: dict, save_path: str, cooccurrences: torch.Tensor | None
) -> None:
    """
    Save a dictionary containing tensors to a safetensors file
    Non-tensor values are stored as tensor metadata.

    Args:
        tensor_dict: Dictionary containing tensors
        save_path: Path to save the safetensors file
    """
    assert save_path.endswith(".safetensors.zst")

    tensors = {}
    for key, value in tensor_dict.items():
        assert isinstance(value, torch.Tensor)
        tensors[str(key)] = value

    if cooccurrences is not None:
        tensors["cooccurrences"] = cooccurrences

    # Save tensors with metadata
    uncompressed_data = safetensors.torch.save(tensors)

    compressor = zstandard.ZstdCompressor()
    compressed_data = compressor.compress(uncompressed_data)

    with open(save_path, "wb") as f_out:
        f_out.write(compressed_data)


@beartype
def load_dict_with_tensors(load_path: str) -> tuple[dict, torch.Tensor | None]:
    """
    Load a dictionary containing tensors from a safetensors file.
    This function is the inverse of save_dict_with_tensors.

    Args:
        load_path: Path to the safetensors file

    Returns:
        Dictionary containing the loaded tensors
    """
    assert load_path.endswith(".safetensors.zst")

    # Read the compressed data
    with open(load_path, "rb") as f_in:
        compressed_data = f_in.read()

    # Decompress the data
    decompressor = zstandard.ZstdDecompressor()
    uncompressed_data = decompressor.decompress(compressed_data)

    # Load the tensors from the uncompressed data
    tensor_dict = safetensors.torch.load(uncompressed_data)

    results = {}
    for key, value in tensor_dict.items():
        if key != "cooccurrences":
            results[int(key)] = value

    return results, tensor_dict.get("cooccurrences")
