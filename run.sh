#!/usr/bin/env sh
if [ -z "$command" ]; then
	command=/bin/bash
fi
FISH_MOUNT=""
if [ -d "$HOME/.local/share/fish" ]; then
	FISH_MOUNT="-v $HOME/.local/share/fish:/root/.local/share/fish"
fi
# --memory-swap is RAM+swap,
# so by setting it to the same value as --memory, we disable swap
docker run -it --rm \
	--memory=10g \
	--memory-swap=10g \
	--cpus=1 \
	$@ \
	-v $HOME/.cache/huggingface:/root/.cache/huggingface \
	-v $HOME/.config/vastai:/root/.config/vastai \
	-v $(pwd):/code \
	-v $HOME/.cache/uv:/root/.cache/uv \
	-v $HOME/.local/share/uv:/root/.local/share/uv $FISH_MOUNT \
	$(cat docker_name) \
	sh -c "$command"
