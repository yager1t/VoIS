#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.." || exit 1

echo "==> Linting"
ruff check src tests

echo "==> Type checking"
mypy src

echo "==> Running tests with coverage"
pytest tests/ --cov=src --cov-report=term-missing --cov-report=html
