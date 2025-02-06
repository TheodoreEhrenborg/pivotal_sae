#!/usr/bin/env sh
if [ -z "$command" ]; then
	command=/bin/bash
fi
FISH_MOUNT=""
if [ -d "$HOME/.local/share/fish" ]; then
	FISH_MOUNT="-v $HOME/.local/share/fish:/root/.local/share/fish"
fi
docker run -it --rm \
	$@ \
	-v $HOME/.cache/huggingface:/root/.cache/huggingface \
	-v $(pwd):/code \
	-v $HOME/.cache/uv:/root/.cache/uv \
	-v $HOME/.local/share/uv:/root/.local/share/uv $FISH_MOUNT \
	$(cat docker_name) \
	sh -c "$command"
