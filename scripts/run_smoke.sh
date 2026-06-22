#!/usr/bin/env bash
set -euo pipefail

# Run smoke tests with real hardware/OS interaction.
# Requires microphone access and may inject text unless dry-run mode is used.
cd "$(dirname "$0")/.."
pytest tests/smoke -v --run-smoke --timeout=30
