#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.." || exit 1

echo "==> Linting"
ruff check src tests

echo "==> Type checking"
mypy src

echo "==> Running safe unit tests with coverage"
pytest tests/ -m "not smoke and not integration and not slow and not requires_model" --timeout=60 --cov=src --cov-report=term-missing --cov-report=html
