# Architecture

This document provides a high-level overview of the Voice-to-Cursor AI Dictation System.

For the full approved plan, see [`voice_to_cursor_prompt.md`](../voice_to_cursor_prompt.md).

## Phase 1 — Infrastructure

The current phase establishes the project skeleton:

- `src/config.py` — pydantic-settings based configuration.
- `src/logging_config.py` — rotating file + console logging via loguru.
- `src/app.py` — application lifecycle skeleton (`start` / `stop`).
- `src/main.py` — CLI entry point.

## Upcoming phases

1. Audio capture layer
2. Global hotkey manager
3. Voice activity detection (VAD)
4. ASR integration (Whisper / Voxtral)
5. LLM post-processing (optional)
6. Keyboard injection
7. Adaptive dictionary & learning
