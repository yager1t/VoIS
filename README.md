# Voice-to-Cursor

Local-first desktop dictation: press a global hotkey, speak, and get clean text inserted at the current cursor position.

## Installation

```bash
pip install -r requirements-dev.txt
```

## Run

```bash
python -m src.main
```

Optional custom configuration:

```bash
python -m src.main --config /path/to/.env
```

## Configuration

Settings are loaded from environment variables and an optional `.env` file.
See `src/config.py` for available options.

## Architecture

See [`docs/architecture.md`](docs/architecture.md) for a high-level overview.

## Development

```bash
ruff check src tests
mypy src
pytest tests/ -q
```
