#!/usr/bin/env python3

import json
import os

import requests
from beartype import beartype

from sae_hacking.timeprint import timeprint


@beartype
class NeuronExplanationLoader:
    def __init__(self, combined_id: str):
        """
        Initialize the loader with a combined model/sae id string.
        Downloads and caches data if not already present.

        Args:
            combined_id (str): Combined identifier (e.g., "gemma-2-2b/20-gemmascope-res-65k")
        """
        self.model_id, self.sae_id = self._parse_combined_id(combined_id)
        self.cache_path = f"/tmp/neuron_explanations_{self.model_id}_{self.sae_id}.json"
        self.explanations = self._preprocess(self._load_or_download_data())

    def _preprocess(self, data: list) -> dict:
        return {int(item["index"]): item["description"] for item in data}

    def _parse_combined_id(self, combined_id: str) -> tuple[str, str]:
        """
        Parse the combined ID into model_id and sae_id.

        Args:
            combined_id (str): The combined identifier string

        Returns:
            Tuple of (model_id, sae_id)
        """
        model_id, sae_id = combined_id.split("/")
        return model_id, sae_id

    def _load_or_download_data(self) -> list[dict]:
        """
        Load data from cache if it exists, otherwise download and cache it.

        Returns:
            Dict containing the explanations data
        """
        if os.path.exists(self.cache_path):
            with open(self.cache_path, "r") as f:
                return json.load(f)

        timeprint("Downloading data from neuronpedia")
        url = "https://www.neuronpedia.org/api/explanation/export"
        params = {"modelId": self.model_id, "saeId": self.sae_id}

        response = requests.get(url, params=params)
        response.raise_for_status()

        data = response.json()

        with open(self.cache_path, "w") as f:
            json.dump(data, f)

        timeprint("Done downloading data")
        return data

    def get_explanation(self, index: int) -> str:
        """
        Get the explanation for a specific neuron index.

        Args:
            index (str): The neuron index to look up

        Returns:
            Dict containing the explanation data for the specified index

        """
        return self.explanations.get(index, f"No explanation found for index {index}")


@beartype
def construct_url(id: str, idx: int) -> str:
    return f"https://www.neuronpedia.org/{id}/{idx}"
