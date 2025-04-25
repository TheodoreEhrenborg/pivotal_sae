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
- D: Default---no convention, but should be locally unique
- E: SAE hidden dim
- F: reserved for torch.nn.functional
- G: some multiple of E
- H: total number of SAE children (multiple of E)
- K: k in topk
- M: model dimension
- S: sequence length

# TODO Scripts
All scripts can run on a single 4090. 
It may be necessary to have a lot of RAM available 
(>200 GB), and a lot of disk space (>100 GB).


uv run src/sae_hacking/train_topk_sae_toy.py --lr 1e-4 --sae-hidden-dim 30 --dataset-num-features 30 --batch-size 100 --cuda --hierarchical --perturbation_size 0.4 --model-dim 50

Unusual combinations of hyperparameters are not guaranteed to run.


uv run src/sae_hacking/gemma_selective_ablate.py


uv run src/sae_hacking/look_for_pairs.py

uv run src/sae_hacking/gemma_cooccurrences.py



uv run src/sae_hacking/prompt_server.py


uv run src/sae_hacking/prompt_client.py
