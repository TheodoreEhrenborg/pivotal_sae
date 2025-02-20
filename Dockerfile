FROM ubuntu:24.04
WORKDIR /code
COPY setup.sh setup.sh
RUN ./setup.sh
