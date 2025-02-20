# This dockerfile is meant for local development.
# For running on a vast.ai server, it's more reliable
# to start with ubuntu:24.04 and then run setup.sh
FROM ubuntu:24.04
WORKDIR /code
COPY setup.sh setup.sh
RUN ./setup.sh
