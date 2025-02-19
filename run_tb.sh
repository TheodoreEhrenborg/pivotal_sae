#!/usr/bin/env bash
set -e
PORT=${PORT:-6010}
if [ -z "$1" ]; then
	echo "Usage: ./run_tb.sh /path/to/results <other docker args>"
	exit 1
fi
full_path=$(realpath "$1")
shift 1
command="/root/.local/bin/uv run tensorboard --logdir=$full_path --port $PORT --host 0.0.0.0"
command=$command ./run.sh -v "$full_path":"$full_path" -p "$PORT":"$PORT" "$@"
