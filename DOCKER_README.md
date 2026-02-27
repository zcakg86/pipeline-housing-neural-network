# Docker Deployment - Quick Reference

## 🚀 Quick Start (3 Commands)

```bash
# 1. Build containers
make build

# 2. Train model
make train

# 3. Start API
make api
```

Your API is now running at http://localhost:8000

## 📦 What's Included

### Containers
- **Training** - Train and evaluate models
- **API** - REST API for predictions (FastAPI)
- **Scheduler** - Automated retraining

### Features
- ✅ Production-ready Docker setup
- ✅ REST API with FastAPI
- ✅ Automated retraining scheduler
- ✅ Health checks and monitoring
- ✅ Volume management for data persistence
- ✅ Resource limits (CPU/memory)
- ✅ Non-root user for security

## 🛠️ Common Commands

### Using Make (Recommended)

```bash
# Build
make build              # Build all containers
make build-clean        # Build without cache

# Training
make train              # Train model
make test               # Run tests
make compare            # Compare models

# API
make api                # Start API server
make api-logs           # View API logs
make api-health         # Check API health
make api-test           # Test prediction

# Scheduler
make scheduler          # Start scheduler
make retrain-check      # Run retraining check once

# Management
make logs               # View all logs
make stop               # Stop all services
make clean              # Clean up containers
make status             # Show status

# Full deployment
make deploy             # Build + Train + Start API + Scheduler
```

### Using Docker Compose

```bash
# Build
docker-compose build

# Train
docker-compose run --rm train

# Start services
docker-compose up -d api
docker-compose up -d scheduler

# Logs
docker-compose logs -f

# Stop
docker-compose down
```

## 🔧 Configuration

### Environment Variables

Create `.env` file:

```bash
# API
API_HOST=0.0.0.0
API_PORT=8000

# Model
PERFORMANCE_THRESHOLD=15.0
TIME_THRESHOLD_DAYS=30
SLIDING_WINDOW_YEARS=5

# Resources
TRAIN_CPU_LIMIT=4
TRAIN_MEMORY_LIMIT=8G
```

### Data Files Required

```
data/
├── sales_202025.csv          # Historical sales (REQUIRED)
└── community_map.json        # H3 mapping (REQUIRED)
```

## 📡 API Usage

### Health Check

```bash
curl http://localhost:8000/health
```

### Single Prediction

```bash
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
```

### Batch Predictions

```bash
curl -X POST http://localhost:8000/predict/batch \
  -H "Content-Type: application/json" \
  -d '{
    "listings": [
      {...},
      {...}
    ]
  }'
```

### CSV Upload

```bash
curl -X POST http://localhost:8000/predict/csv \
  -F "file=@listings.csv"
```

### Model Info

```bash
curl http://localhost:8000/model/info
```

## 🔄 Retraining

### Automatic (Scheduler)

```bash
# Start scheduler (runs daily at 2 AM)
make scheduler

# View logs
docker-compose logs -f scheduler
```

### Manual

```bash
# Run retraining check once
make retrain-check

# Or train new model
make train
```

### Retraining Triggers

Model retrains automatically when:
- Model is > 30 days old
- Validation error > 15%
- Manually triggered

## 📊 Monitoring

### Container Status

```bash
make status
# Shows: container status + resource usage
```

### Logs

```bash
# All services
make logs

# Specific service
make api-logs
make scheduler-logs

# Last 100 lines
make logs-tail

# Export to file
make logs-export
```

### API Monitoring

```bash
# Health
make api-health

# Model info
make api-info

# Retraining status
make api-retrain-status
```

## 💾 Backup & Restore

### Backup

```bash
# Create backup
make backup

# Manual backup
tar -czf backup.tar.gz data/ outputs/
```

### Restore

```bash
# Extract backup
tar -xzf backup.tar.gz
```

## 🐛 Troubleshooting

### Container won't start

```bash
# Check logs
make logs

# Rebuild
make build-clean

# Check permissions
ls -la data/ outputs/
```

### Model not loading

```bash
# Train model first
make train

# Check if model exists
ls -la outputs/models/

# Check API logs
make api-logs
```

### Out of memory

Edit `docker-compose.yml`:
```yaml
deploy:
  resources:
    limits:
      memory: 16G  # Increase
```

### Permission errors

```bash
# Fix ownership
sudo chown -R $USER:$USER data/ outputs/

# Or check container user
docker-compose run --rm train whoami
```

## 🔒 Security

### Non-root User

Containers run as non-root user `modeluser` (UID 1000)

### Network Isolation

Services communicate via internal `model-network`

### Volume Permissions

```bash
# Set correct permissions
chmod 755 data/ outputs/
chown -R 1000:1000 data/ outputs/
```

## 📈 Production Deployment

### 1. Build and Test

```bash
make build
make test
make train
```

### 2. Deploy Services

```bash
make deploy
# Builds, trains, and starts all services
```

### 3. Verify

```bash
make api-health
make api-test
make status
```

### 4. Monitor

```bash
make logs
make monitor  # Continuous monitoring
```

## 🔄 Updates

```bash
# Pull latest code
git pull

# Rebuild and restart
make update
```

## 🧹 Cleanup

```bash
# Stop services
make stop

# Remove containers
make clean

# Remove everything (including volumes)
make clean-all
```

## 📚 Documentation

- **DOCKER_GUIDE.md** - Complete Docker documentation
- **MODEL_V2_GUIDE.md** - Model documentation
- **QUICKSTART.md** - Getting started guide
- **API Docs** - http://localhost:8000/docs (when API running)

## 🆘 Support

### Check Status

```bash
make status
```

### View Logs

```bash
make logs
```

### Test Installation

```bash
make test
```

### Interactive Shell

```bash
make shell
```

## 📋 Complete Workflow

### Initial Setup

```bash
# 1. Build containers
make build

# 2. Test installation
make test

# 3. Train initial model
make train

# 4. Compare performance (if you have V1)
make compare
```

### Production Deployment

```bash
# 1. Deploy all services
make deploy

# 2. Verify API
make api-health
make api-test

# 3. Monitor
make logs
```

### Daily Operations

```bash
# Check status
make status

# View logs
make logs-tail

# Backup
make backup

# Update
make update
```

## 🎯 Next Steps

1. ✅ Build containers: `make build`
2. ✅ Train model: `make train`
3. ✅ Start API: `make api`
4. ✅ Test API: `make api-test`
5. ✅ Start scheduler: `make scheduler`
6. ✅ Monitor: `make logs`

## 💡 Tips

- Use `make help` to see all commands
- Use `make deploy` for one-command deployment
- Use `make monitor` for continuous monitoring
- Use `make backup` regularly
- Check `make api-health` to verify API status
- View API docs at http://localhost:8000/docs

## 🚀 Production Checklist

- [ ] Build containers: `make build`
- [ ] Run tests: `make test`
- [ ] Train model: `make train`
- [ ] Start API: `make api`
- [ ] Test API: `make api-test`
- [ ] Start scheduler: `make scheduler`
- [ ] Set up backups: `make backup`
- [ ] Configure monitoring: `make monitor`
- [ ] Review logs: `make logs`
- [ ] Document API endpoint for your team

---

**Quick Help**: Run `make help` for all available commands
