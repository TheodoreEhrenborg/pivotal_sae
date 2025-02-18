Work in progress

## Installation

### With uv

This repo uses [uv](https://github.com/astral-sh/uv) for packaging,

1. Install with `curl -LsSf https://astral.sh/uv/install.sh | sh`
1. Run scripts using `uv run`, e.g. `uv run src/sae_hacking/train_sae.py -h`.
   The first time you call uv, it will download all the necessary dependencies.

### With docker

uv doesn't work well on machines that don't follow the Filesystem Hierarchy Standard (e.g. NixOS).
To run uv in this case, use the provided Dockerfile:

1. Build the image with `./build.sh`
1. Enter the container with `./run.sh`. If you have GPUs, instead use `./run.sh --gpus all`
1. To mount a results directory, use `./run.sh -v /absolute/host/path/to/results/:/results`
1. Then inside the container you can run `uv run ...` as before

## Running tests

`uv run pytest tests`

## History

Based on https://github.com/TheodoreEhrenborg/tiny_stories_sae, except with many files deleted

# Dimension naming convention
- 1: singleton dimension (for broadcasting)
- A: Number of parent features in the dataset
- B: batch size
- C: Number of child features in the dataset
- F: SAE hidden dim
- G: total number of SAE latents (i.e. a multiple of F)
- M: model dimension
