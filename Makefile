# Makefile for Real Estate Price Model V2
# Convenient commands for Docker operations

.PHONY: help build train api scheduler test clean logs stop restart

# Default target
help:
	@echo "Real Estate Price Model V2 - Docker Commands"
	@echo ""
	@echo "Available commands:"
	@echo "  make build       - Build all Docker containers"
	@echo "  make train       - Train the model"
	@echo "  make api         - Start API server"
	@echo "  make scheduler   - Start retraining scheduler"
	@echo "  make test        - Run installation tests"
	@echo "  make logs        - View logs from all services"
	@echo "  make stop        - Stop all services"
	@echo "  make restart     - Restart all services"
	@echo "  make clean       - Clean up containers and volumes"
	@echo "  make shell       - Open shell in training container"
	@echo "  make predict     - Run prediction script"
	@echo "  make compare     - Compare model performance"
	@echo ""
	@echo "API commands:"
	@echo "  make api-logs    - View API logs"
	@echo "  make api-health  - Check API health"
	@echo "  make api-info    - Get model info"
	@echo ""
	@echo "Data commands:"
	@echo "  make backup      - Backup models and data"
	@echo "  make restore     - Restore from backup"
	@echo ""

# Build containers
build:
	@echo "Building Docker containers..."
	docker-compose build

# Build without cache
build-clean:
	@echo "Building Docker containers (no cache)..."
	docker-compose build --no-cache

# Train model
train:
	@echo "Training model..."
	docker-compose run --rm train python main_train_v2.py

# Start API server
api:
	@echo "Starting API server..."
	docker-compose up -d api
	@echo "API available at http://localhost:8000"
	@echo "View logs with: make api-logs"

# Start scheduler
scheduler:
	@echo "Starting retraining scheduler..."
	docker-compose up -d scheduler
	@echo "View logs with: docker-compose logs -f scheduler"

# Run tests
test:
	@echo "Running installation tests..."
	docker-compose run --rm train python test_installation.py

# View logs
logs:
	docker-compose logs -f

# API logs
api-logs:
	docker-compose logs -f api

# Scheduler logs
scheduler-logs:
	docker-compose logs -f scheduler

# Stop services
stop:
	@echo "Stopping all services..."
	docker-compose stop

# Restart services
restart:
	@echo "Restarting services..."
	docker-compose restart

# Clean up
clean:
	@echo "Cleaning up containers..."
	docker-compose down
	@echo "To remove volumes as well, run: docker-compose down -v"

# Deep clean (removes volumes)
clean-all:
	@echo "WARNING: This will remove all containers, volumes, and cached data!"
	@read -p "Are you sure? [y/N] " -n 1 -r; \
	echo; \
	if [[ $$REPLY =~ ^[Yy]$$ ]]; then \
		docker-compose down -v; \
		docker system prune -f; \
	fi

# Shell access
shell:
	@echo "Opening shell in training container..."
	docker-compose run --rm train bash

# Run prediction
predict:
	@echo "Running prediction script..."
	docker-compose run --rm train python predict_listings.py

# Compare models
compare:
	@echo "Comparing model performance..."
	docker-compose run --rm train python compare_models.py

# API health check
api-health:
	@echo "Checking API health..."
	@curl -s http://localhost:8000/health | python -m json.tool

# API model info
api-info:
	@echo "Getting model information..."
	@curl -s http://localhost:8000/model/info | python -m json.tool

# API retrain status
api-retrain-status:
	@echo "Checking retraining status..."
	@curl -s http://localhost:8000/model/retrain/status | python -m json.tool

# Test API prediction
api-test:
	@echo "Testing API prediction..."
	@curl -X POST http://localhost:8000/predict \
		-H "Content-Type: application/json" \
		-d '{"listing_id":"TEST001","lat":47.6062,"lng":-122.3321,"sqft":1800,"sqft_lot":5000,"beds":3,"h3_07":"8728d5cd3ffffff","list_date":"2025-02-26"}' \
		| python -m json.tool

# Backup
backup:
	@echo "Creating backup..."
	@mkdir -p backups
	@tar -czf backups/backup-$$(date +%Y%m%d-%H%M%S).tar.gz data/ outputs/
	@echo "Backup created in backups/"

# Status
status:
	@echo "Container status:"
	@docker-compose ps
	@echo ""
	@echo "Resource usage:"
	@docker stats --no-stream

# Full deployment
deploy: build train api scheduler
	@echo ""
	@echo "Deployment complete!"
	@echo "API: http://localhost:8000"
	@echo "Health: http://localhost:8000/health"
	@echo ""
	@echo "View logs: make logs"
	@echo "Stop services: make stop"

# Development mode (with live reload)
dev:
	@echo "Starting development mode..."
	docker-compose up train

# Run scheduler once
retrain-check:
	@echo "Running retraining check..."
	docker-compose run --rm scheduler python scheduler.py --mode once

# Install local dependencies (for development)
install:
	pip install -r requirements.txt

# Format code
format:
	@echo "Formatting code..."
	@docker-compose run --rm train python -m black src/ *.py

# Lint code
lint:
	@echo "Linting code..."
	@docker-compose run --rm train python -m pylint src/

# Update containers
update:
	@echo "Updating containers..."
	git pull
	docker-compose build
	docker-compose up -d

# Show container logs (last 100 lines)
logs-tail:
	docker-compose logs --tail=100

# Export logs
logs-export:
	@mkdir -p logs
	@docker-compose logs > logs/logs-$$(date +%Y%m%d-%H%M%S).txt
	@echo "Logs exported to logs/"

# Monitor (continuous status updates)
monitor:
	@watch -n 5 'docker-compose ps && echo "" && docker stats --no-stream'
