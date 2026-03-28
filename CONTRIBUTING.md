# Contributing to llamaindex-ambientmeta

Thanks for your interest in contributing!

## Development Setup

```bash
# Clone the repo
git clone https://github.com/AmbientMeta/llamaindex-ambientmeta.git
cd llamaindex-ambientmeta

# Create a virtual environment
python -m venv .venv
source .venv/bin/activate

# Install with dev dependencies
pip install -e ".[dev]"
```

## Running Tests

```bash
pytest tests/ -v
```

Tests use mocked SDK clients — no AmbientMeta API key is needed to run them.

## Code Style

This project uses [ruff](https://docs.astral.sh/ruff/) for linting and formatting.

```bash
# Check
ruff check .
ruff format --check .

# Fix
ruff check --fix .
ruff format .
```

## Submitting a PR

1. Fork the repo and create a feature branch
2. Make your changes
3. Add tests for new functionality
4. Ensure `pytest` and `ruff check` pass
5. Open a pull request with a clear description of the change

## Reporting Issues

Use [GitHub Issues](https://github.com/AmbientMeta/llamaindex-ambientmeta/issues) to report bugs or request features.
