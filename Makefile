# Disaster Text Classifier — Common tasks

PYTHON := python3
VENV   := .venv
PIP    := $(VENV)/bin/pip
PYTHON_VENV := $(VENV)/bin/python

.PHONY: all install train run test lint format deploy clean

all: install train

install:
	$(PYTHON) -m venv $(VENV)
	$(PIP) install -r requirements.txt

train:
	$(PYTHON_VENV) scripts/train_model.py

test:
	$(PYTHON_VENV) scripts/predict_test.py

run:
	$(PYTHON_VENV) -m uvicorn app.main:app --reload --port 8080

lint:
	$(PYTHON_VENV) -m flake8 app/ scripts/ tests/ --max-line-length=100 --extend-ignore=E203,W503
	$(PYTHON_VENV) -m mypy app/ --ignore-missing-imports --follow-imports=silent

format:
	$(PYTHON_VENV) -m black --line-length=100 app/ scripts/ tests/
	$(PYTHON_VENV) -m isort --profile=black app/ scripts/ tests/

evaluate:
	$(PYTHON_VENV) scripts/evaluate_model.py

latency:
	$(PYTHON_VENV) scripts/latency_test.py

deploy:
	gcloud run deploy disaster-text-service \
	  --source . \
	  --platform managed \
	  --region us-east1 \
	  --allow-unauthenticated \
	  --min-instances 0 \
	  --max-instances 3 \
	  --memory 512Mi \
	  --cpu 1

clean:
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name "*.pyc" -delete
	find . -type f -name "predictions.csv" -delete
