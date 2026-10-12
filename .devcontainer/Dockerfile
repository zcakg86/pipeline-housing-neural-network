# GitHub Codespaces and local Dev Containers share this development image.
ARG VARIANT="3.11-bookworm"
FROM mcr.microsoft.com/devcontainers/python:1-${VARIANT}

# LightGBM wheels require the OpenMP runtime. The remaining tools support the
# Maven wrapper and the repository's Make targets without a Docker-in-Docker
# daemon or cloud-vendor CLI in the image.
 
# Remove stale/invalid Yarn apt source, then install only the tools we need.
RUN -f /etc/apt/sources.list.d/yarn.list \
    && apt-get update \
    && DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends \
        curl \
        libgomp1 \
        make \
    && rm -rf /var/lib/apt/lists/*
