# Makefile for House Price Model + Java App

.PHONY: help test train train-lightgbm export export-lightgbm retrain-deploy java-setup java-dev java-build java-run java-kill \
        rentcast-fetch rentcast-status clean

JAVA_APP_DIR = java-app/house-price-app
MODEL_DIR    = outputs/models
PYTHON_ENV   = PYTHONPATH=src
TRAIN_SEED  ?= 42
MVNW         = $(JAVA_APP_DIR)/mvnw

# Dev Containers provide JAVA_HOME. On macOS, fall back to the versioned
# Homebrew JDK; on other hosts derive it from the selected java executable.
ifeq ($(origin JAVA_HOME), undefined)
  ifeq ($(shell uname -s),Darwin)
    JAVA_HOME := /opt/homebrew/opt/openjdk@21/libexec/openjdk.jdk/Contents/Home
  else
    JAVA_HOME := $(shell dirname $$(dirname $$(readlink -f $$(command -v java))))
  endif
endif

# ── Help ──────────────────────────────────────────────────────────────────────
help:
	@echo ""
	@echo "House Price Model + Java App"
	@echo ""
	@echo "  Model training (Python):"
	@echo "    make train          Train the model (local Python)"
	@echo "    make train-lightgbm Train LightGBM with the same prepared features"
	@echo "    make export         Export trained model to ONNX for Java"
	@echo "    make export-lightgbm Export latest LightGBM model for Java"
	@echo "    make retrain-deploy Retrain and deploy both water-aware models"
	@echo ""
	@echo "  Java app:"
	@echo "    make java-setup     First-time setup (copy model artifacts)"
	@echo "    make java-dev       Run Java app in dev mode (hot reload)"
	@echo "    make java-build     Build Java app JAR"
	@echo "    make java-run       Run built JAR"
	@echo "    make java-kill      Kill process on port 8080"
	@echo "    make rentcast-fetch Fetch up to 500 RentCast records (one API call)"
	@echo "    make rentcast-status Show the persistent RentCast call counter"
	@echo ""
	@echo "  Utilities:"
	@echo "    make test           Run Python and Java tests"
	@echo "    make clean          Remove build artifacts"
	@echo ""

# ── Python model training ─────────────────────────────────────────────────────
sys-info:
	@echo "=== System Info ==="
	@echo "OS:            $$(uname -s 2>/dev/null || echo Windows)"
	@echo "Make Version:  $(MAKE_VERSION)"
	@echo "Shell:         $(SHELL)"
	@python3 --version 2>&1 || echo "Python 3 not installed"
	@$(JAVA_HOME)/bin/java -version 2>&1 || echo "Java 21 not installed"

test:
	$(PYTHON_ENV) python3 -m unittest discover -s tests -q
	JAVA_HOME=$(JAVA_HOME) $(MVNW) -q test -f $(JAVA_APP_DIR)/pom.xml
train:
	@echo "Training model..."
	$(PYTHON_ENV) python3 main_train.py --seed=$(TRAIN_SEED)

train-lightgbm:
	@echo "Training LightGBM model..."
	$(PYTHON_ENV) python3 main_lightgbm.py

export:
	@echo "Exporting model to ONNX..."
	$(PYTHON_ENV) python3 export_model_for_java.py
	@echo "Staged a versioned neural bundle under outputs/deployment/."

export-lightgbm:
	@echo "Exporting latest LightGBM model to ONNX..."
	$(PYTHON_ENV) python3 export_lightgbm_for_java.py

retrain-deploy: train train-lightgbm
	$(PYTHON_ENV) python3 deploy_models_for_java.py
	@echo "Both water-aware models are retrained and deployed to Java resources."

# ── Java app ──────────────────────────────────────────────────────────────────
java-setup:
	@echo "Setting up Java app..."
	@if [ ! -f $(JAVA_APP_DIR)/src/main/resources/model-artifacts/model.onnx ]; then \
		echo "  No complete deployment found. Exporting and atomically deploying both models..."; \
		$(PYTHON_ENV) python3 deploy_models_for_java.py; \
	fi
	@if [ ! -f $(JAVA_APP_DIR)/.env ]; then \
		echo "  Creating .env from template..."; \
		cp $(JAVA_APP_DIR)/src/main/resources/application-secrets.properties.template $(JAVA_APP_DIR)/.env.example; \
		echo "  ⚠  Add your API keys to $(JAVA_APP_DIR)/.env"; \
	fi
	@echo "Setup complete. Run 'make java-dev' to start."

java-kill:
	@lsof -ti :8080 | xargs kill -9 2>/dev/null && echo "Killed process on port 8080" || echo "Nothing running on port 8080"

# Fetch RentCast data through the charge-guarded Zsh wrapper.
# Usage: make rentcast-fetch LIMIT=500 MAX_CALLS=1 OFFSET=0
RENTCAST_COUNTER_FILE = $(JAVA_APP_DIR)/data/rentcast/.state/call_count
RENTCAST_TOTAL_LIMIT  = 45
LIMIT ?= 500
MAX_CALLS ?= 1
OFFSET ?= 0

rentcast-fetch:
	@cd $(JAVA_APP_DIR) && ./fetch_rentcast.zsh --confirm \
		--limit=$(LIMIT) --max-calls=$(MAX_CALLS) --offset=$(OFFSET)

rentcast-status:
	@if [ -f $(RENTCAST_COUNTER_FILE) ]; then \
		count=$$(cat $(RENTCAST_COUNTER_FILE)); \
		echo "RentCast API calls reserved: $$count/$(RENTCAST_TOTAL_LIMIT) ($$(($(RENTCAST_TOTAL_LIMIT)-count)) remaining)"; \
	else \
		echo "RentCast counter not initialized. The first guarded fetch will infer prior usage from saved responses."; \
	fi

java-dev:
	@echo "Starting Java app in dev mode at http://localhost:8080"
	JAVA_HOME=$(JAVA_HOME) $(MVNW) quarkus:dev -f $(JAVA_APP_DIR)/pom.xml

java-build:
	@echo "Building Java app..."
	JAVA_HOME=$(JAVA_HOME) $(MVNW) package -DskipTests -f $(JAVA_APP_DIR)/pom.xml

java-run:
	@echo "Running Java app..."
	@set -a && [ -f $(JAVA_APP_DIR)/.env ] && . $(JAVA_APP_DIR)/.env; set +a; \
	$(JAVA_HOME)/bin/java \
		-DRENTCAST_API_KEY=$${RENTCAST_API_KEY:-} \
		-DZILLOW_API_KEY=$${ZILLOW_API_KEY:-} \
		-jar $(JAVA_APP_DIR)/target/quarkus-app/quarkus-run.jar

# ── Utilities ─────────────────────────────────────────────────────────────────
clean:
	@echo "Cleaning build artifacts..."
	JAVA_HOME=$(JAVA_HOME) $(MVNW) clean -q -f $(JAVA_APP_DIR)/pom.xml
	find . -name "__pycache__" -type d -exec rm -rf {} + 2>/dev/null || true
	find . -name "*.pyc" -delete 2>/dev/null || true
	@echo "Done."
