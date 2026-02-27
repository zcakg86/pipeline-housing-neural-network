# Docker Deployment Guide

## Overview

This guide covers deploying the Enhanced Real Estate Price Model V2 using Docker containers.

## Container Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                     Docker Compose Stack                     │
├─────────────────────────────────────────────────────────────┤
│                                                               │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐      │
│  │   Training   │  │  API Server  │  │  Scheduler   │      │
│  │  Container   │  │  Container   │  │  Container   │      │
│  │              │  │              │  │              │      │
│  │ - Train      │  │ - FastAPI    │  │ - Cron-like  │      │
│  │ - Evaluate   │  │ - REST API   │  │ - Auto       │      │
│  │ - Save       │  │ - Predict    │  │   Retrain    │      │
│  └──────────────┘  └──────────────┘  └──────────────┘      │
│         │                 │                  │               │
│         └─────────────────┴──────────────────┘               │
│                           │                                  │
│                    ┌──────▼──────┐                          │
│                    │   Volumes   │                          │
│                    │  - data/    │                          │
│                    │  - outputs/ │                          │
│                    │  - cache/   │                          │
│                    └─────────────┘                          │
└─────────────────────────────────────────────────────────────┘
```

## Quick Start

### 1. Build Containers

```bash
# Build all containers
docker-compose build

# Or build specific service
docker-compose build train
docker-compose build api
```

### 2. Train Initial Model

```bash
# Run training container
docker-compose run --rm train

# This will:
# - Load your data from ./data/
# - Fetch market indicators
# - Train the model
# - Save to ./outputs/models/
```

### 3. Start API Server

```bash
# Start API in background
docker-compose up -d api

# Check logs
docker-compose logs -f api

# API available at http://localhost:8000
```

### 4. Start Scheduler (Optional)

```bash
# Start automated retraining scheduler
docker-compose up -d scheduler

# Check logs
docker-compose logs -f scheduler
```

## Container Services

### Training Container

**Purpose**: Train and evaluate models

**Usage**:
```bash
# Train new model
docker-compose run --rm train

# Run with custom script
docker-compose run --rm train python compare_models.py

# Interactive shell
docker-compose run --rm train bash
```

**Volumes**:
- `./data` → `/app/data` (read/write)
- `./outputs` → `/app/outputs` (read/write)

### API Container

**Purpose**: Serve predictions via REST API

**Endpoints**:
- `GET /` - API info
- `GET /health` - Health check
- `POST /predict` - Single prediction
- `POST /predict/batch` - Batch predictions
- `POST /predict/csv` - Upload CSV for predictions
- `GET /model/info` - Model information
- `GET /model/retrain/status` - Check if retraining needed

**Usage**:
```bash
# Start API
docker-compose up -d api

# Test health
curl http://localhost:8000/health

# Make prediction
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{
    "listing_id": "L001",
    "lat": 47.6062,
    "lng": -122.3321,
    "sqft": 1800,
    "sqft_lot": 5000,
    "beds": 3,
    "h3_07": "8728d5cd3ffffff",
    "list_date": "2025-02-26"
  }'

# Stop API
docker-compose stop api
```

**Environment Variables**:
- `API_HOST` - Host to bind (default: 0.0.0.0)
- `API_PORT` - Port to bind (default: 8000)
- `MODEL_PATH` - Path to models directory

### Scheduler Container

**Purpose**: Automated retraining

**Features**:
- Checks daily at 2 AM (configurable)
- Triggers retraining if:
  - Model > 30 days old
  - Performance degraded (>15% error)
- Uses 5-year sliding window

**Usage**:
```bash
# Start scheduler
docker-compose up -d scheduler

# View logs
docker-compose logs -f scheduler

# Run once (no scheduling)
docker-compose run --rm scheduler python scheduler.py --mode once

# Custom schedule
docker-compose run --rm scheduler python scheduler.py --mode scheduled --time 03:00

# Interval mode (every 12 hours)
docker-compose run --rm scheduler python scheduler.py --mode interval --interval 12
```

## Data Management

### Required Data Files

Before running containers, ensure you have:

```
data/
├── sales_202025.csv          # Historical sales data (REQUIRED)
├── community_map.json        # H3 to community mapping (REQUIRED)
└── local_inventory.csv       # Optional inventory data
```

### Volume Mounts

```yaml
volumes:
  - ./data:/app/data              # Data directory
  - ./outputs:/app/outputs        # Model outputs
  - model-cache:/home/modeluser/.cache  # Cache (named volume)
```

### Backing Up Models

```bash
# Backup outputs directory
tar -czf models-backup-$(date +%Y%m%d).tar.gz outputs/

# Restore from backup
tar -xzf models-backup-20250226.tar.gz
```

## API Usage Examples

### Python Client

```python
import requests

API_URL = "http://localhost:8000"

# Single prediction
listing = {
    "listing_id": "L001",
    "lat": 47.6062,
    "lng": -122.3321,
    "sqft": 1800,
    "sqft_lot": 5000,
    "beds": 3,
    "h3_07": "8728d5cd3ffffff",
    "list_date": "2025-02-26"
}

response = requests.post(f"{API_URL}/predict", json=listing)
prediction = response.json()

print(f"Predicted Price: ${prediction['predicted_price']:,.0f}")
print(f"95% CI: ${prediction['price_lower_95']:,.0f} - ${prediction['price_upper_95']:,.0f}")
print(f"Confidence: {prediction['confidence']}")

# Batch predictions
listings = [listing, ...]  # Multiple listings
response = requests.post(f"{API_URL}/predict/batch", json={"listings": listings})
predictions = response.json()

# CSV upload
with open('listings.csv', 'rb') as f:
    response = requests.post(f"{API_URL}/predict/csv", files={'file': f})
predictions = response.json()
```

### cURL Examples

```bash
# Health check
curl http://localhost:8000/health

# Single prediction
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d @listing.json

# Model info
curl http://localhost:8000/model/info

# Check retraining status
curl http://localhost:8000/model/retrain/status

# Upload CSV
curl -X POST http://localhost:8000/predict/csv \
  -F "file=@listings.csv"
```

## Production Deployment

### Environment Configuration

Create `.env` file:

```bash
# API Configuration
API_HOST=0.0.0.0
API_PORT=8000

# Model Configuration
MODEL_PATH=/app/outputs/models
PERFORMANCE_THRESHOLD=15.0
TIME_THRESHOLD_DAYS=30
SLIDING_WINDOW_YEARS=5

# Timezone
TZ=America/Los_Angeles

# Resource Limits
TRAIN_CPU_LIMIT=4
TRAIN_MEMORY_LIMIT=8G
API_CPU_LIMIT=2
API_MEMORY_LIMIT=4G
```

### Using with docker-compose

```bash
# Load environment variables
docker-compose --env-file .env up -d
```

### Scaling API Service

```bash
# Run multiple API instances
docker-compose up -d --scale api=3

# Use nginx for load balancing
```

### HTTPS/SSL

Add nginx reverse proxy:

```yaml
# docker-compose.yml
services:
  nginx:
    image: nginx:alpine
    ports:
      - "443:443"
      - "80:80"
    volumes:
      - ./nginx.conf:/etc/nginx/nginx.conf:ro
      - ./ssl:/etc/nginx/ssl:ro
    depends_on:
      - api
```

## Monitoring

### Container Health

```bash
# Check container status
docker-compose ps

# View logs
docker-compose logs -f

# Specific service logs
docker-compose logs -f api

# Resource usage
docker stats
```

### API Monitoring

```bash
# Health endpoint
curl http://localhost:8000/health

# Model status
curl http://localhost:8000/model/info

# Retraining status
curl http://localhost:8000/model/retrain/status
```

### Log Management

```bash
# View recent logs
docker-compose logs --tail=100 api

# Follow logs
docker-compose logs -f scheduler

# Export logs
docker-compose logs > logs-$(date +%Y%m%d).txt
```

## Troubleshooting

### Container Won't Start

```bash
# Check logs
docker-compose logs train

# Rebuild container
docker-compose build --no-cache train

# Check permissions
ls -la data/ outputs/
```

### Model Not Loading

```bash
# Check if model exists
ls -la outputs/models/

# Train a model first
docker-compose run --rm train

# Check API logs
docker-compose logs api
```

### Out of Memory

```bash
# Increase memory limits in docker-compose.yml
deploy:
  resources:
    limits:
      memory: 16G  # Increase from 8G
```

### Permission Errors

```bash
# Fix ownership
sudo chown -R $USER:$USER data/ outputs/

# Or run as root (not recommended)
docker-compose run --user root train
```

## Maintenance

### Update Containers

```bash
# Pull latest code
git pull

# Rebuild containers
docker-compose build

# Restart services
docker-compose down
docker-compose up -d
```

### Clean Up

```bash
# Stop all services
docker-compose down

# Remove volumes (WARNING: deletes data)
docker-compose down -v

# Remove old images
docker image prune -a

# Clean everything
docker system prune -a --volumes
```

### Backup Strategy

```bash
# Daily backup script
#!/bin/bash
DATE=$(date +%Y%m%d)
tar -czf backup-$DATE.tar.gz data/ outputs/
aws s3 cp backup-$DATE.tar.gz s3://my-bucket/backups/
```

## Advanced Configuration

### Custom Dockerfile

Modify `Dockerfile` for custom requirements:

```dockerfile
# Add custom dependencies
RUN pip install custom-package

# Add system packages
RUN apt-get update && apt-get install -y custom-tool
```

### Multi-Stage Build

Already implemented for optimized image size:
- Builder stage: Compiles dependencies
- Runtime stage: Minimal production image

### GPU Support

For GPU acceleration:

```yaml
# docker-compose.yml
services:
  train:
    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia
              count: 1
              capabilities: [gpu]
```

Requires:
- NVIDIA Docker runtime
- CUDA-compatible PyTorch

## CI/CD Integration

### GitHub Actions Example

```yaml
# .github/workflows/docker.yml
name: Build and Push Docker Images

on:
  push:
    branches: [main]

jobs:
  build:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v2
      
      - name: Build images
        run: docker-compose build
      
      - name: Run tests
        run: docker-compose run --rm train python test_installation.py
      
      - name: Push to registry
        run: |
          docker tag real-estate-model:latest myregistry/real-estate-model:latest
          docker push myregistry/real-estate-model:latest
```

## Summary

You now have a complete containerized deployment:

- ✅ **Training container** - Train models on demand
- ✅ **API container** - Serve predictions via REST API
- ✅ **Scheduler container** - Automated retraining
- ✅ **Volume management** - Persistent data and models
- ✅ **Health checks** - Monitor container health
- ✅ **Resource limits** - Control CPU/memory usage
- ✅ **Production-ready** - Logging, monitoring, scaling

## Quick Commands Reference

```bash
# Build
docker-compose build

# Train model
docker-compose run --rm train

# Start API
docker-compose up -d api

# Start scheduler
docker-compose up -d scheduler

# View logs
docker-compose logs -f

# Stop all
docker-compose down

# Clean up
docker-compose down -v
docker system prune -a
```

## Next Steps

1. Build containers: `docker-compose build`
2. Train initial model: `docker-compose run --rm train`
3. Start API: `docker-compose up -d api`
4. Test API: `curl http://localhost:8000/health`
5. Start scheduler: `docker-compose up -d scheduler`
6. Monitor: `docker-compose logs -f`
