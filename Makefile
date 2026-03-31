# Makefile for House Price Model + Java App

.PHONY: help train export java-setup java-dev java-build java-run \
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
	@echo ""
	@echo "  Utilities:"
	@echo "    make clean          Remove build artifacts"
	@echo ""

# ── Python model training ─────────────────────────────────────────────────────
train:
	@echo "Training model..."
	python3 main_train_v4_h3l9.py

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

java-dev:
	@echo "Starting Java app in dev mode at http://localhost:8080"
	mvn quarkus:dev -f $(JAVA_APP_DIR)/pom.xml

java-build:
	@echo "Building Java app..."
	mvn package -DskipTests -f $(JAVA_APP_DIR)/pom.xml

java-run:
	@echo "Running Java app..."
	java -jar $(JAVA_APP_DIR)/target/quarkus-app/quarkus-run.jar

# ── Utilities ─────────────────────────────────────────────────────────────────
clean:
	@echo "Cleaning build artifacts..."
	cd $(JAVA_APP_DIR) && mvn clean -q
	find . -name "__pycache__" -type d -exec rm -rf {} + 2>/dev/null || true
	find . -name "*.pyc" -delete 2>/dev/null || true
	@echo "Done."
