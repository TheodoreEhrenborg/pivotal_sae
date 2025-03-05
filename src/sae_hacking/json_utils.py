#!/usr/bin/env python3

import json
from pathlib import Path

import torch
import zstandard
from beartype import beartype


@beartype
def load_dict_with_tensors_from_json(load_path: str) -> dict:
    path = Path(load_path)

    if load_path.endswith(".json.zst"):
        with open(path, "rb") as compressed_file:
            decompressor = zstandard.ZstdDecompressor()
            print("Decompressing to bytes")
            json_str = decompressor.decompress(compressed_file.read())
            print("Converting to json")
            json_dict = json.loads(json_str)
    else:
        with open(path, "r") as f:
            print("Reading json from file")
            json_dict = json.load(f)

    print("Making tensor dictionary")
    result_dict = {}

    for key, value in json_dict.items():
        result_dict[int(key)] = torch.tensor(value)

    print("Done making tensor dictionary")
    return result_dict


@beartype
def save_dict_with_tensors_to_json(
    tensor_dict: dict, save_path: str, compress: bool
) -> None:
    """
    Save a dictionary containing tensors to a JSON file, with optional compression.

    Args:
        tensor_dict: Dictionary containing tensors and other serializable values
        save_path: Path to save the JSON file
        compress: If True, compress the output using Zstandard and append .zst extension
    """
    json_dict = {}

    for key, value in tensor_dict.items():
        if isinstance(value, torch.Tensor):
            assert value.dim() == 1, f"Tensor for key '{key}' is not 1D"
            json_dict[key] = value.tolist()
        else:
            json_dict[key] = value

    if compress:
        # Make sure the save_path has .zst extension
        assert save_path.endswith(".zst"), "Please use the .zst extension"

        # Convert to JSON string
        json_data = json.dumps(json_dict)
        # Compress the JSON data and write to file
        compressor = zstandard.ZstdCompressor()
        compressed_data = compressor.compress(json_data.encode("utf-8"))
        with open(save_path, "wb") as f:
            f.write(compressed_data)
    else:
        # Save as regular JSON
        assert not save_path.endswith(".zst"), "Did you mean to compress?"
        with open(save_path, "w") as f:
            json.dump(json_dict, f)
