# Customer Churn Prediction & Retention Intelligence Platform
# Convenience targets — every target maps to a command documented in the README.

.DEFAULT_GOAL := help
PY ?= .venv/bin/python
PIP ?= $(PY) -m pip

.PHONY: help install data validate split train train-quick evaluate-test calibrate explain mlflow-ui api dashboard test lint fmt typecheck clean docker-up docker-down

help:  ## Show this help
	@grep -hE '^[a-zA-Z_-]+:.*?## ' $(MAKEFILE_LIST) | awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-14s\033[0m %s\n", $$1, $$2}'

install:  ## Create venv and install the package with dev + api + mlops extras
	python3 -m venv .venv
	$(PIP) install -U pip
	$(PIP) install -e ".[dev,api,mlops,frontend]"

data:  ## Phase 1: download/verify the raw dataset (SHA-256 checked)
	$(PY) -m churn.data.ingest

validate:  ## Phase 2: run the data validation contract
	$(PY) -m churn.data.validate

split:  ## Phase 2: coerce dtypes then split 60/20/20 (NaN preserved)
	$(PY) -m churn.data.split

train:  ## Phase 3: tune 10 configs, select on validation, freeze. Logs to MLflow
	$(PY) -m churn.models.run_phase3

train-quick:  ## Phase 3 with a tiny budget (smoke test / CI)
	$(PY) -m churn.models.run_phase3 --quick --no-mlflow

evaluate-test:  ## Phase 3: one-shot held-out test evaluation (requires a freeze)
	$(PY) -m churn.models.evaluate_test

calibrate:  ## Phase 4: fit sigmoid/isotonic on train, select on validation
	$(PY) -m churn.models.calibration

explain:  ## Phase 4: build the SHAP report and figures
	$(PY) -m churn.explain.build_report

mlflow-ui:  ## Open the MLflow tracking UI
	$(PY) -m mlflow ui --backend-store-uri sqlite:///mlflow.db --port 5000

api:  ## Serve the FastAPI backend on :8000
	$(PY) -m uvicorn churn.api.main:app --host 0.0.0.0 --port 8000

dashboard:  ## Serve the Streamlit dashboard on :8501 (needs the API running)
	$(PY) -m streamlit run src/churn/dashboard/app.py --server.port 8501 --server.address 0.0.0.0

test:  ## Run the test suite with coverage
	$(PY) -m pytest --cov=src/churn --cov=api --cov-report=term-missing -q

lint:  ## Lint and format-check
	.venv/bin/ruff check .
	.venv/bin/ruff format --check .

fmt:  ## Auto-fix lint issues and format
	.venv/bin/ruff check --fix .
	.venv/bin/ruff format .

typecheck:  ## Static type checking
	.venv/bin/mypy src/churn

docker-up:  ## Start the full stack (API + Postgres + MLflow + dashboard)
	docker compose up --build -d
	docker compose ps

docker-down:  ## Stop the stack and remove volumes
	docker compose down -v

clean:  ## Remove caches and generated artifacts
	rm -rf .pytest_cache .ruff_cache .mypy_cache htmlcov .coverage coverage.xml
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
