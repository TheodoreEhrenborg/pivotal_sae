#!/usr/bin/env bash
command="tensorboard --logdir=/results/ --port 6010 --host 0.0.0.0"
command=$command ./run.sh -v "$(realpath ~/projects/results/sae_results)":/results -p 6010:6010
