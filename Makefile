.PHONY: help install dev-install lint format type-check test clean pre-commit

help:  ## Show this help message
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | sort | awk 'BEGIN {FS = ":.*?## "}; {printf "\033[36m%-20s\033[0m %s\n", $$1, $$2}'

install:  ## Install production dependencies
	poetry install --only main

dev-install:  ## Install all dependencies including dev dependencies
	poetry install --with dev

lint:  ## Run linting with ruff
	poetry run ruff check .

format:  ## Format code with ruff
	poetry run ruff format .

type-check:  ## Run type checking with mypy
	poetry run mypy spatial_competition_pettingzoo/

test:  ## Run tests
	poetry run pytest

clean:  ## Clean up cache and temporary files
	find . -type d -name "__pycache__" -exec rm -rf {} +
	find . -type f -name "*.pyc" -delete
	rm -rf .coverage htmlcov/ .pytest_cache/ .mypy_cache/ .ruff_cache/

pre-commit:  ## Install pre-commit hooks
	poetry run pre-commit install

check:  ## Run all checks (lint, type-check, test)
	$(MAKE) lint
	$(MAKE) type-check
	$(MAKE) test
