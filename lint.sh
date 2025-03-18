#!/usr/bin/env bash

if command -v uv >/dev/null 2>&1; then
	uvx ruff format
	uvx ruff check --fix
else
	ruff format
	ruff check --fix
fi
