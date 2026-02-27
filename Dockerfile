# Production Dockerfile for Enhanced Real Estate Price Model V2
# Multi-stage build for optimized image size

# Stage 1: Builder
ARG VARIANT=3.12-bullseye
ARG TARGETPLATFORM=linux/amd64
FROM --platform=${TARGETPLATFORM} python:${VARIANT} AS builder

# Set working directory
WORKDIR /build

# Install system dependencies for building
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    gcc \
    g++ \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir --user -r requirements.txt

# Stage 2: Runtime
ARG VARIANT=3.12-bullseye
ARG TARGETPLATFORM=linux/amd64
FROM --platform=${TARGETPLATFORM} python:${VARIANT}

# Set environment variables
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    TZ=America/Los_Angeles

# Create non-root user
RUN useradd -m -u 1000 modeluser && \
    mkdir -p /app /app/data /app/outputs /app/outputs/models && \
    chown -R modeluser:modeluser /app

# Set working directory
WORKDIR /app

# Copy Python packages from builder
COPY --from=builder /root/.local /home/modeluser/.local

# Copy application code
COPY --chown=modeluser:modeluser src/ ./src/
COPY --chown=modeluser:modeluser *.py ./
COPY --chown=modeluser:modeluser *.md ./
COPY --chown=modeluser:modeluser requirements.txt ./

# Copy data directory structure (but not large data files)
COPY --chown=modeluser:modeluser data/community_map.json ./data/

# Create necessary directories
RUN mkdir -p /app/data/market_indicators && \
    chown -R modeluser:modeluser /app

# Switch to non-root user
USER modeluser

# Add local Python packages to PATH
ENV PATH=/home/modeluser/.local/bin:$PATH

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD python -c "import torch; import pandas; print('OK')" || exit 1

# Default command (can be overridden)
CMD ["python", "test_installation.py"]
