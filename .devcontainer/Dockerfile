# Python Version
ARG VARIANT="3.11-bullseye"
FROM mcr.microsoft.com/vscode/devcontainers/python:0-${VARIANT}

RUN apt-get update && apt-get install -y \
    gdal-bin \
    libgdal-dev \
    libproj-dev \
    python3-gdal