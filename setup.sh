#!/usr/bin/env bash
set -e
apt-get update
apt-get install -y fish curl nvtop htop
curl -LsSf https://astral.sh/uv/install.sh | sh
