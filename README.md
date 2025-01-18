# TinyStories SAE

Train a sparse autoencoder using your laptop on this [TinyStories](https://huggingface.co/roneneldan/TinyStories-33M) model

Docs are [here](https://sae.ehrenborg.dev/).
The rest of this readme is software engineering details.

## Installation

### With uv

This repo uses [uv](https://github.com/astral-sh/uv) for packaging,

1. Install with `curl -LsSf https://astral.sh/uv/install.sh | sh`
1. Run scripts using `uv run`, e.g. `uv run src/tiny_stories_sae/train_sae.py -h`.
   The first time you call uv, it will download all the necessary dependencies.

### With docker

uv doesn't work well on machines that don't follow that Filesystem Hierarchy Standard (e.g. NixOS).
To run uv in this case, use the provided Dockerfile:

1. Build the image with `./build.sh`
1. Enter the container with `./run.sh`. If you have GPUs, instead use `./run.sh --gpus all`
1. To mount a results directory, use `./run.sh -v /absolute/host/path/to/results/:/results`
1. Then inside the container you can run `uv run ...` as before

## Available scripts

### train_sae.py

Trains a sparse autoencoder
on activations from the language model `roneneldan/TinyStories-33M`.

Example usage:

```bash
uv run src/tiny_stories_sae/train_sae.py \
  --cuda --l1_coefficient 50 \
  --sae_hidden_dim 10000 --max_step 105000
```

### steer.py

This script uses the LM `roneneldan/TinyStories-33M` to generate text,
but it adds a fixed vector (one of the autoencoder's features)
to the activations,
skewing the text responses towards a certain topic.

Example usage:

```bash
uv run src/tiny_stories_sae/steer.py \
  --checkpoint path/to/sparse_autoencoder.pt \
  --which_feature 1 --sae_hidden_dim 10000 \
  --feature_strength 5 --cuda
```

### gather_high_activations.py

This runs the LM on the validation set
and tracks how strongly the various
autoencoder features activate.
It saves a list of validation examples
that made the features activate the most.

Example usage:

```bash
uv run src/tiny_stories_sae/gather_high_activations.py \
  --checkpoint path/to/sparse_autoencoder.pt \ 
  --cuda --sae_hidden_dim 10000 
```

### gather_high_activations_llm.py

The same as `gather_high_activations.py`,
except it instead tracks how strongly
the LM's neurons activate.

Example usage:

```bash
uv run src/tiny_stories_sae/gather_high_activations_llm.py \
  --cuda --output_file path/to/log.json --make_positive_by abs
```

### call_openai.py

Given a log file produced by
`gather_high_activations.py` or `gather_high_activations_llm.py`,
this script sends the examples to GPT-4,
and asks GPT-4 to look for a pattern
(for each feature/neuron separately)
and judge how clear the pattern is.
This requires an `OPENAI_API_KEY` in `.env`.

Example usage:

```bash
uv run src/tiny_stories_sae/call_openai.py \
  --feature_lower 0 --feature_upper 100 \
  --path_to_feature_strengths path/to/log.json
```

### plot.py and combined_plot.py

Given GPT-4's ratings, these scripts plots them.
`combined_plot.py` can graph multiple ratings in different colors.

Example usage:

```bash
uv run src/tiny_stories_sae/plot.py \
  --response_json results/gpt4_api/20241001-123456 \
  --xlabel "Clearness (5 is most clear)" \
  --title "GPT-4o's ranking of 100 sparse autoencoder features"
```

and

```bash
uv run src/tiny_stories_sae/combined_plot.py \
  --response_jsons results/1.json results/2.json \
  --labels "1" "2" 
```

## Running tests

`uv run pytest tests`
