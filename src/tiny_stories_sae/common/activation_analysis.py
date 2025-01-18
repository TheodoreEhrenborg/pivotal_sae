#!/usr/bin/env python3

import json
from dataclasses import asdict, dataclass
from pathlib import Path

from beartype import beartype
from transformers import GPT2TokenizerFast

blocks = [chr(x) for x in range(9601, 9609)]


@beartype
@dataclass
class Sample:
    step: int
    feature_idx: int
    tokens: list[int]
    strengths: list[float]
    max_strength: float


@beartype
def prune(sample_list: list[Sample], samples_to_keep: int) -> list[Sample]:
    return sorted(sample_list, reverse=True, key=lambda sample: sample.max_strength)[
        :samples_to_keep
    ]


@beartype
def format_token(
    tokenizer: GPT2TokenizerFast, token: int, strength: float, max_strength: float
) -> str:
    assert strength >= 0, strength
    rank = int(7 * strength / max_strength) if max_strength != 0 else 0
    assert 0 <= rank <= 7, rank
    return f"{tokenizer.decode(token)} {blocks[rank]}"


@beartype
def get_dict(tokenizer: GPT2TokenizerFast, sample: Sample) -> dict:
    results = asdict(sample)
    # This merges them into one string
    results["text"] = tokenizer.decode(sample.tokens)
    results["annotated_text"] = get_annotated_text(
        tokenizer, sample.tokens, sample.strengths, sample.max_strength
    )
    return results


@beartype
def get_annotated_text(
    tokenizer: GPT2TokenizerFast,
    tokens: list[int],
    strengths: list[float],
    max_strength: float,
) -> str:
    return "".join(
        format_token(tokenizer, token, strength, max_strength)
        for token, strength in zip(tokens, strengths, strict=True)
    )


@beartype
def write_activation_json(
    output_path: Path,
    strongest_activations: list[list[Sample]],
    tokenizer: GPT2TokenizerFast,
):
    num_dead_features = 0
    for sample_list in strongest_activations:
        if max(map(lambda x: x.max_strength, sample_list)) == 0:
            num_dead_features += 1
    print("Proportion of dead features", num_dead_features / len(strongest_activations))

    with open(output_path, "w") as f:
        json.dump(
            [
                get_dict(tokenizer, sample)
                for sample_list in strongest_activations
                for sample in sample_list
            ],
            f,
            indent=2,
            ensure_ascii=False,
        )
