#!/usr/bin/env sh
docker run -it --rm \
    $@ \
    -v $HOME/.cache/huggingface:/root/.cache/huggingface \
    -v $(pwd):/code \
    -v $HOME/.cache/uv:/root/.cache/uv \
    -v $HOME/.local/share/uv:/root/.local/share/uv \
    $(cat docker_name) \
    sh -c "/bin/bash"
