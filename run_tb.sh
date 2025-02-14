#!/usr/bin/env bash
set -e
PORT=${PORT:-6010}
if [ -z "$1" ]; then
    echo "Usage: ./run_tb.sh /path/to/results"
    exit 1
fi
command="/root/.local/bin/uv run tensorboard --logdir=/results/ --port $PORT --host 0.0.0.0"
command=$command ./run.sh -v "$(realpath "$1")":/results -p "$PORT":"$PORT"
