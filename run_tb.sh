#!/usr/bin/env bash
set -ue
PORT=${PORT:-6010}
command="/root/.cargo/bin/uv run tensorboard --logdir=/results/ --port $PORT --host 0.0.0.0"
command=$command ./run.sh -v "$(realpath $1)":/results -p $PORT:$PORT
