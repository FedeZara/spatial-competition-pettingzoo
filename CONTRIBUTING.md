# Contributing to Spatial Competition PettingZoo

Thank you for your interest in contributing to this project!

## Development Setup

1. **Install Poetry** (if you haven't already):
   ```bash
   curl -sSL https://install.python-poetry.org | python3 -
   ```

2. **Clone the repository**:
   ```bash
   git clone https://github.com/FedeZara/spatial-competition-pettingzoo.git
   cd spatial-competition-pettingzoo
   ```

3. **Install dependencies**:
   ```bash
   make dev-install
   # or manually: poetry install --with dev
   ```

4. **Install pre-commit hooks**:
   ```bash
   make pre-commit
   # or manually: poetry run pre-commit install
   ```

## Development Workflow

1. **Run all checks**:
   ```bash
   make check  # runs linting, type checking, and tests
   ```

2. **Individual commands**:
   ```bash
   make lint        # Check code style
   make format      # Format code
   make type-check  # Run mypy
   make test        # Run tests
   make test-cov    # Run tests with coverage
   ```

3. **Clean up**:
   ```bash
   make clean  # Remove cache files
   ```

## Code Standards

- **Python**: Compatible with Python 3.10-3.12
- **Formatting**: We use Ruff for both linting and formatting
- **Type hints**: Required for all public functions
- **Testing**: Write tests for new functionality
- **Pre-commit**: All commits should pass pre-commit hooks

## Submitting Changes

1. Fork the repository
2. Create a feature branch from `main`
3. Make your changes
4. Ensure all tests pass (`make check`)
5. Commit your changes (pre-commit hooks will run)
6. Push to your fork and submit a pull request

## Questions?

Feel free to open an issue for questions or discussion!
