PYTHON ?= python
PIP ?= pip
APP_DIR := web_app
APP_FILE := $(APP_DIR)/app.py
LOG_MODEL_FILE := $(APP_DIR)/log_model.py
SMOKE_TEST_FILE := scripts/smoke_test.py
TRACKING_URI := sqlite:///$(APP_DIR)/mlflow.db
MODEL_RUN_ID ?= 78447675102e40d38a3b5d4267717b11
APP_PORT ?= 5000
MLFLOW_UI_PORT ?= 5001
MODEL_PORT ?= 5003

.PHONY: help install compile lint test ci run predict runs mlflow-ui log-model serve-model clean

help:
	@echo "Available targets:"
	@echo "  make install       Install project dependencies"
	@echo "  make compile       Check Python files compile"
	@echo "  make lint          Run lightweight validation checks"
	@echo "  make test          Run API smoke tests"
	@echo "  make ci            Run checks suitable for CI"
	@echo "  make run           Start the Flask app"
	@echo "  make predict       Send a sample prediction request"
	@echo "  make runs          Show recent tracked runs from the API"
	@echo "  make mlflow-ui     Start the MLflow UI"
	@echo "  make log-model     Log a serveable MLflow model"
	@echo "  make serve-model   Serve the logged MLflow model"
	@echo "  make clean         Remove Python cache files"

install:
	$(PIP) install -r requirements.txt

compile:
	$(PYTHON) -m py_compile $(APP_FILE) $(LOG_MODEL_FILE)

lint: compile
	@test -f $(APP_DIR)/roberta-sequence-classification-9.onnx || echo "Note: $(APP_DIR)/roberta-sequence-classification-9.onnx is not committed; prediction tests will run in degraded mode."
	@test -f $(APP_DIR)/mlflow.db || echo "Note: $(APP_DIR)/mlflow.db will be created when the app or MLflow runs."

test:
	$(PYTHON) $(SMOKE_TEST_FILE)

ci: lint test

run:
	$(PYTHON) $(APP_FILE)

predict:
	curl -X POST http://127.0.0.1:$(APP_PORT)/predict \
		-H "Content-Type: application/json" \
		-d '{"text":"This product was excellent and very easy to use."}'

runs:
	curl http://127.0.0.1:$(APP_PORT)/runs

mlflow-ui:
	mlflow ui --backend-store-uri $(TRACKING_URI) --port $(MLFLOW_UI_PORT)

log-model:
	$(PYTHON) $(LOG_MODEL_FILE)

serve-model:
	MLFLOW_TRACKING_URI=$(TRACKING_URI) mlflow models serve \
		-m runs:/$(MODEL_RUN_ID)/model \
		-p $(MODEL_PORT) \
		--env-manager local

clean:
	find . -type d -name "__pycache__" -prune -exec rm -rf {} +
	find . -type f -name "*.pyc" -delete
