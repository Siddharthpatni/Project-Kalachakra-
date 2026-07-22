.PHONY: help install install-all test lint typecheck format run-api run-dashboard clean

help: ## Show this help
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | sort | awk 'BEGIN {FS = ":.*?## "}; {printf "\033[36m%-20s\033[0m %s\n", $$1, $$2}'

install: ## Install core dependencies
	pip install -e ".[dev]"

install-all: ## Install all dependencies (all 50 domains)
	pip install -e ".[all,dev]"

test: ## Run all tests
	pytest tests/ -v --cov=kalachakra --cov-report=html --cov-report=term-missing

test-unit: ## Run unit tests only
	pytest tests/unit/ -v

test-integration: ## Run integration tests
	pytest tests/integration/ -v

test-benchmarks: ## Run benchmark tests
	python -m kalachakra.benchmarks.runner

lint: ## Lint code with ruff
	ruff check src/ tests/

format: ## Format code with ruff
	ruff format src/ tests/

typecheck: ## Type check with mypy
	mypy src/kalachakra/ --strict

check: lint typecheck test ## Run all checks (lint + typecheck + test)

run-api: ## Start FastAPI development server
	uvicorn kalachakra.api.main:app --reload --host 0.0.0.0 --port 8000

run-dashboard: ## Start React dashboard
	cd dashboard && npm run dev

docker-build: ## Build Docker image
	docker-compose build

docker-up: ## Start all services
	docker-compose up -d

docker-down: ## Stop all services
	docker-compose down

clean: ## Clean build artifacts
	rm -rf build/ dist/ *.egg-info .pytest_cache .mypy_cache .ruff_cache htmlcov/
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
