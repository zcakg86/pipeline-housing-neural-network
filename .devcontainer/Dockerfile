# Python Version
ARG VARIANT="3.11-bullseye"
FROM mcr.microsoft.com/vscode/devcontainers/python:0-${VARIANT}

RUN apt-get update && export DEBIAN_FRONTEND=noninteractive \
    && apt-get -y install --no-install-recommends gdal-bin \
    libgdal-dev \
<<<<<<< HEAD
    libproj-dev \
    python3-gdal
=======
    libproj-dev 
>>>>>>> ab6dccc54c6dde2e79ebc56292715e7f72886f4f
