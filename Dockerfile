FROM ubuntu:22.04

RUN apt-get update && apt-get install -y fish curl
RUN curl -LsSf https://astral.sh/uv/install.sh | sh

WORKDIR /code
