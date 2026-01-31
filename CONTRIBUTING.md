# Contributing to Spatial Competition PettingZoo

Thank you for your interest in contributing to this project!

## Development Setup

1. **Install mise** (if you haven't already):
   ```bash
   curl https://mise.run | sh
   ```

2. **Clone the repository**:
   ```bash
   git clone https://github.com/FedeZara/spatial-competition-pettingzoo.git
   cd spatial-competition-pettingzoo
   ```

3. **Activate mise** (installs Poetry automatically):
   ```bash
   mise trust
   mise install
   ```

4. **Install dependencies**:
   ```bash
   mise run dev-install
   ```

5. **Install pre-commit hooks**:
   ```bash
   mise run pre-commit
   ```

## Development Workflow

1. **Run all checks**:
   ```bash
   mise run check  # runs linting, type checking, and tests
   ```

2. **Individual commands**:
   ```bash
   mise run lint        # Check code style
   mise run format      # Format code
   mise run type-check  # Run mypy
   mise run test        # Run tests
   ```

3. **Clean up**:
   ```bash
   mise run clean  # Remove cache files
   ```

4. **List all available tasks**:
   ```bash
   mise tasks
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
4. Ensure all tests pass (`mise run check`)
5. Commit your changes (pre-commit hooks will run)
6. Push to your fork and submit a pull request

## Questions?

Feel free to open an issue for questions or discussion!
