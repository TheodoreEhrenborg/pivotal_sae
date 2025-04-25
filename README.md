# Pivotal SAE

This is my code
from when I was a fellow at [Pivotal Research](https://www.pivotal-research.org/),
in Q1 2025.

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

`run.sh` is meant for local development.
Some of the scripts are memory hungry.
To prevent your dev machine from freezing,
`run.sh` limits the container's CPUs to 1 and the RAM to 10 GB.

If you instead run the code on [Vast.ai](https://vast.ai),
the image is called
[theodoreehrenborg/ubuntu-uv](https://hub.docker.com/r/theodoreehrenborg/ubuntu-uv). Note that Vast.ai will not call `./run.sh`.

## Running tests

`uv run pytest tests`

## Dimension naming convention

This repo uses
[Shazeer typing](https://www.kolaayonrinde.com/blog/2025/01/01/shazeer-typing.html):

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

## Scripts

Not a comprehensive list.

Unusual combinations of hyperparameters are not guaranteed to run.

### Hierarchical SAEs

All scripts can run on a single 4090.

Example of training a hierarchical SAE:

```bash
uv run src/sae_hacking/train_topk_sae_toy.py --lr 1e-4 --sae-hidden-dim 30 --dataset-num-features 30 --batch-size 100 --cuda --hierarchical --perturbation_size 0.4 --model-dim 50
```

### Looking for feature absorption

It may be necessary to have a lot of RAM available
(>200 GB), and a lot of disk space (>100 GB).

First gather data on which latents cooccur with which other latents:

```bash
uv run src/sae_hacking/gemma_cooccurrences.py --max-tokens 100 --ablator-sae-release gemma-scope-2b-pt-mlp-canonical --n-prompts 2000000 --save-frequency 30000 --dataset-id monology/pile-uncopyrighted --batch-size 5
```

Then gather data on latents' downstream effects:

```bash
uv run src/sae_hacking/gemma_selective_ablate.py --abridge-ablations-to 100 --max-tokens 100 --ablator-sae-release gemma-scope-2b-pt-mlp-canonical --reader-sae-release gemma-scope-2b-pt-res-canonical --n-prompts 1000000 --save-frequency 10000 --dataset-id monology/pile-uncopyrighted --selected-features 3000 9000 15000 21000 27000 33000 39000 45000 51000 57000 63000 --batch-size 5
```

Finally filter to find non-cooccuring latents with similar downstream effects:

```bash
uv run src/sae_hacking/look_for_pairs.py --input-path /results/ablations.safetensors.zst --ablator-sae-neuronpedia-id gemma-2-2b/20-gemmascope-mlp-65k --cosine-sim-threshold -0.2 --cooccurrence-path /results/cooccurrences.safetensors --skip-torch-sign --just-these 15000
```

### Highlighting prompts

Here's how to start the prompt server:

```bash
uv run src/sae_hacking/prompt_server.py
```

And then in a different window, you can send prompts to the server to get annotated with how strongly the latents activate:

```bash
uv run src/sae_hacking/prompt_client.py --sae-release gemma-scope-2b-pt-mlp-canonical --sae-id layer_20/width_65k/canonical --prompt "testing 1 2 3" --output-dir /results/prompts --feature-idx 1000
```

## History

Cloned from [here](https://github.com/TheodoreEhrenborg/tiny_stories_sae)
