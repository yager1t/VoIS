# Voice-to-Cursor

![CI](https://github.com/yager1t/VoIS/actions/workflows/ci.yml/badge.svg)

Local-first Windows desktop dictation: press a global hotkey, speak, and get clean text inserted at the current cursor position.

- Push-to-talk or toggle recording modes.
- Silence trimming with WebRTC VAD.
- Local ASR with faster-whisper.
- Optional LLM post-processing through Ollama for punctuation, casing, and cleanup.
- System tray icon with Start/Stop, Settings, and Exit (no persistent console window).
- Settings window for hotkey, model, language, device, LLM options, and dry-run mode.
- Recording indicator via tray icon tooltip and balloon notifications.
- Windows text injection via `SendInput` with optional clipboard fallback.
- Dry-run mode for safe testing.

> **Platform note:** The MVP is Windows-only. macOS and Linux support are planned for a later phase.

## Installation

```bash
pip install -r requirements-dev.txt
```

The first run downloads the Whisper model into `models/`; this may take a few minutes depending on the model size.

## Quick start

Run in **dry-run mode** first to verify that audio capture, VAD, ASR, and optional LLM post-processing work without typing text into another application:

```bash
python -m src.main --dry-run
```

The application starts as a system tray icon. Hold the hotkey (default `f9`), speak, and release. The transcribed text is printed to the console instead of injected.

Once everything looks good, run the real mode:

```bash
python -m src.main
```

Hold `f9`, speak, and release to insert the transcription at the cursor.

## CLI options

```bash
python -m src.main --model base --language en --device cpu --hotkey f10 --dry-run
```

| Flag | Description |
|------|-------------|
| `--config PATH` | Path to an optional `.env` configuration file. |
| `--model SIZE` | Whisper model size: `tiny`, `base`, `small`, `medium`, `large`. |
| `--language CODE` | ASR language code, or `auto` for detection. |
| `--device {cpu,cuda}` | Device to run the ASR model on. |
| `--hotkey KEY` | Override the global hotkey, e.g. `f9` or `<ctrl>+f9`. |
| `--toggle` | Disable push-to-talk; each press toggles recording on/off. |
| `--dry-run` | Print transcribed text instead of injecting it. |
| `--llm-enabled` | Enable LLM post-processing (also controlled by `.env`). |
| `--llm-model NAME` | Ollama model name for post-processing, e.g. `llama3`. |

## LLM setup

LLM post-processing is optional and runs through a local Ollama server.

1. Install [Ollama](https://ollama.com/).
2. Pull a model, for example:

   ```bash
   ollama pull llama3
   ```

3. Enable LLM post-processing in `.env` or the settings window:

   ```bash
   LLM_ENABLED=true
   LLM_MODEL=llama3
   ```

When enabled, raw transcripts are sent to `http://localhost:11434/api/chat` for cleanup before injection. If Ollama is unreachable or returns an error, the raw transcript is used unchanged.

## Smoke test

A convenience script runs the pipeline for a limited time:

```bash
python scripts/smoke_test.py --duration 30 --dry-run
```

By default it runs in dry-run mode. Add `--no-dry-run` to perform real text injection during the test.

## Configuration

Settings are loaded from environment variables and an optional `.env` file.
See `src/config.py` for available options.

## Architecture

See [`docs/architecture.md`](docs/architecture.md) for a high-level overview.

## Development

```bash
ruff check src tests
mypy src
pytest tests/unit tests/integration -m "not smoke and not slow and not requires_model" --cov=src --cov-fail-under=80 --timeout=60
```

The project currently has **155 unit tests**, **7 integration tests**, and **1 smoke test** (163 total) with **94% code coverage**.

AI assistants should read [`docs/ai_working_guide.md`](docs/ai_working_guide.md)
before running commands or editing code. The guide documents the safe test
commands and the rules for avoiding accidental real ASR model loads.

See [`docs/architecture.md`](docs/architecture.md) for the high-level design and
[`docs/testing.md`](docs/testing.md) for detailed testing instructions.
