# Python Version
ARG VARIANT="3.11-bullseye"

FROM mcr.microsoft.com/vscode/devcontainers/python:0-${VARIANT}

RUN apt-get update && export DEBIAN_FRONTEND=noninteractive \
    && apt-get -y install --no-install-recommends \
    gdal-bin \
    build-essential \
    libgdal-dev \
    libproj-dev \
    python3-gdal

ENV PIP_USE_PEP517=true
ENV CPLUS_INCLUDE_PATH=/usr/include/gdal
ENV C_INCLUDE_PATH=/usr/include/gdal