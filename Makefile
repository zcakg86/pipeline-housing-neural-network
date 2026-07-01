# Makefile for House Price Model + Java App

.PHONY: help train export java-setup java-dev java-build java-run java-kill \
        docker-build docker-train clean

JAVA_APP_DIR = java-app/house-price-app
MODEL_DIR    = outputs/models

# ── Help ──────────────────────────────────────────────────────────────────────
help:
	@echo ""
	@echo "House Price Model + Java App"
	@echo ""
	@echo "  Model training (Python):"
	@echo "    make train          Train the model (local Python)"
	@echo "    make docker-build   Build Docker training image"
	@echo "    make docker-train   Train inside Docker container"
	@echo "    make export         Export trained model to ONNX for Java"
	@echo ""
	@echo "  Java app:"
	@echo "    make java-setup     First-time setup (copy model artifacts)"
	@echo "    make java-dev       Run Java app in dev mode (hot reload)"
	@echo "    make java-build     Build Java app JAR"
	@echo "    make java-run       Run built JAR"
	@echo "    make java-kill      Kill process on port 8080"
	@echo ""
	@echo "  Utilities:"
	@echo "    make clean          Remove build artifacts"
	@echo ""

# ── Python model training ─────────────────────────────────────────────────────
sys-info:
	@echo "=== System Info ==="
	@echo "OS:            $$(uname -s 2>/dev/null || echo Windows)"
	@echo "Make Version:  $(MAKE_VERSION)"
	@echo "Shell:         $(SHELL)"
	@python3 --version 2>&1 || echo "Python 3 not installed"
	@docker --version 2>&1 || echo "Docker not installed"
	@python3 -m unittest discover
train:
	@echo "Training model..."
	python3 main_train_v4_h3l8.py

export:
	@echo "Exporting model to ONNX..."
	python3 export_model_for_java.py
	@echo "Copying artifacts to Java resources..."
	cp -r java-app/model-artifacts/* $(JAVA_APP_DIR)/src/main/resources/model-artifacts/
	@echo "Done. Artifacts ready in $(JAVA_APP_DIR)/src/main/resources/model-artifacts/"

# ── Docker training ───────────────────────────────────────────────────────────
docker-build:
	@echo "Building Docker training image..."
	docker-compose build --platform linux/amd64 train

docker-train:
	@echo "Training model in Docker..."
	docker-compose run --rm train python main_train_v4_h3l9.py

# ── Java app ──────────────────────────────────────────────────────────────────
java-setup:
	@echo "Setting up Java app..."
	@if [ ! -f java-app/model-artifacts/model.onnx ]; then \
		echo "  No ONNX model found. Running export first..."; \
		python3 export_model_for_java.py; \
	fi
	@mkdir -p $(JAVA_APP_DIR)/src/main/resources/model-artifacts
	cp -r java-app/model-artifacts/* $(JAVA_APP_DIR)/src/main/resources/model-artifacts/
	@if [ ! -f $(JAVA_APP_DIR)/.env ]; then \
		echo "  Creating .env from template..."; \
		cp $(JAVA_APP_DIR)/src/main/resources/application-secrets.properties.template $(JAVA_APP_DIR)/.env.example; \
		echo "  ⚠  Add your API keys to $(JAVA_APP_DIR)/.env"; \
	fi
	@echo "Setup complete. Run 'make java-dev' to start."

java-kill:
	@lsof -ti :8080 | xargs kill -9 2>/dev/null && echo "Killed process on port 8080" || echo "Nothing running on port 8080"

# Fetch Rentcast data from terminal (tracks call count, max 40 remaining this month)
# Usage: make rentcast-fetch LIMIT=500
# Each call costs 1 API credit. You have 40 remaining.
RENTCAST_COUNTER_FILE = .rentcast_call_count
RENTCAST_MAX_CALLS    = 40
LIMIT ?= 500

rentcast-fetch:
	@count=$$(cat $(RENTCAST_COUNTER_FILE) 2>/dev/null || echo 0); \
	if [ $$count -ge $(RENTCAST_MAX_CALLS) ]; then \
		echo "❌ API call limit reached ($$count/$(RENTCAST_MAX_CALLS)). Edit $(RENTCAST_COUNTER_FILE) to reset."; \
		exit 1; \
	fi; \
	echo "📡 Fetching Rentcast via RentcastFetcher (call $$((count+1))/$(RENTCAST_MAX_CALLS), limit=$(LIMIT))..."; \
	. $(JAVA_APP_DIR)/.env 2>/dev/null; \
	/opt/homebrew/opt/openjdk/bin/java \
		-DRENTCAST_API_KEY=$${RENTCAST_API_KEY:-} \
		-cp "$(JAVA_APP_DIR)/target/quarkus-app/lib/main/*:$(JAVA_APP_DIR)/target/quarkus-app/app/*" \
		com.houseprices.cli.RentcastFetcher $(LIMIT)

rentcast-status:
	@count=$$(cat $(RENTCAST_COUNTER_FILE) 2>/dev/null || echo 0); \
	echo "Rentcast API calls used this month: $$count/$(RENTCAST_MAX_CALLS) ($$(($(RENTCAST_MAX_CALLS)-count)) remaining)"

java-dev:
	@echo "Starting Java app in dev mode at http://localhost:8080"
	JAVA_HOME=/opt/homebrew/opt/openjdk mvn quarkus:dev -f $(JAVA_APP_DIR)/pom.xml

java-build:
	@echo "Building Java app..."
	JAVA_HOME=/opt/homebrew/opt/openjdk mvn package -DskipTests -f $(JAVA_APP_DIR)/pom.xml

java-run:
	@echo "Running Java app..."
	@set -a && [ -f $(JAVA_APP_DIR)/.env ] && . $(JAVA_APP_DIR)/.env; set +a; \
	/opt/homebrew/opt/openjdk/bin/java \
		-DRENTCAST_API_KEY=$${RENTCAST_API_KEY:-} \
		-DZILLOW_API_KEY=$${ZILLOW_API_KEY:-} \
		-jar $(JAVA_APP_DIR)/target/quarkus-app/quarkus-run.jar

# ── Utilities ─────────────────────────────────────────────────────────────────
clean:
	@echo "Cleaning build artifacts..."
	JAVA_HOME=/opt/homebrew/opt/openjdk mvn clean -q -f $(JAVA_APP_DIR)/pom.xml
	find . -name "__pycache__" -type d -exec rm -rf {} + 2>/dev/null || true
	find . -name "*.pyc" -delete 2>/dev/null || true
	@echo "Done."
