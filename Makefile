# Production Distributed LLM Engineering Guide Makefile

.PHONY: help install install-dev setup test lint format type-check clean docker-build docker-run benchmark azure-setup

# Default target
help:
	@echo "Production Distributed LLM Engineering Guide"
	@echo "=============================="
	@echo ""
	@echo "Available targets:"
	@echo "  install      - Install production dependencies"
	@echo "  install-dev  - Install development dependencies"
	@echo "  setup        - Complete development setup"
	@echo "  test         - Run tests"
	@echo "  lint         - Run linting"
	@echo "  format       - Format code"
	@echo "  type-check   - Run type checking"
	@echo "  clean        - Clean build artifacts"
	@echo "  docker-build - Build Docker image"
	@echo "  docker-run   - Run Docker container"
	@echo "  benchmark    - Run performance benchmarks"
	@echo "  azure-setup  - Setup Azure ML resources"

# Installation targets
install:
	pip install -r requirements.txt

install-dev: install
	pip install pytest black ruff mypy pre-commit jupyter ipython
	pre-commit install

setup: install-dev
	python scripts/setup.py --dev

# Development targets
test:
	pytest tests/ -v --cov=src --cov-report=html --cov-report=term

lint:
	ruff check src/ tests/ examples/ benchmarks/
	black --check src/ tests/ examples/ benchmarks/

format:
	black src/ tests/ examples/ benchmarks/
	ruff check --fix src/ tests/ examples/ benchmarks/

type-check:
	mypy src/ tests/ examples/

# Clean targets
clean:
	find . -type f -name "*.pyc" -delete
	find . -type d -name "__pycache__" -delete
	find . -type d -name "*.egg-info" -exec rm -rf {} +
	rm -rf build/ dist/ .coverage htmlcov/ .pytest_cache/ .mypy_cache/

# Docker targets
docker-build:
	docker build -t distributed-llm-guide:latest .

docker-run:
	docker run -it --gpus all -p 8000:8000 distributed-llm-guide:latest

# Benchmarking
benchmark:
	python benchmarks/performance_benchmark.py --model-path microsoft/phi-2

# Azure targets
azure-setup:
	@echo "Setting up Azure ML resources..."
	@read -p "Enter Azure Subscription ID: " subscription_id; \
	python scripts/setup.py --azure --subscription-id $$subscription_id

# Training examples
train-code-model:
	python examples/train_code_model.py --model_name microsoft/CodeGPT-small-py --use_lora

train-multimodal:
	python examples/train_multimodal_model.py --vision_model openai/clip-vit-base-patch32

# Serving
serve-model:
	python src/serving/model_server.py --model-path ./models/trained-model

# Monitoring
start-monitoring:
	python src/monitoring/model_monitor.py --config monitoring_config.yaml

# CI/CD helpers
ci-install:
	pip install -r requirements.txt
	pip install pytest black ruff mypy

ci-test: ci-install test lint type-check

ci-build: ci-test docker-build

# Release helpers
release-patch:
	bump2version patch

release-minor:
	bump2version minor

release-major:
	bump2version major