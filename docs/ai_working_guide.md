# AI Working Guide

This file is the operating guide for AI assistants working on this repository.
Read it before editing code or running commands.

## Project intent

Voice-to-Cursor is a local-first Windows desktop dictation tool. The critical
path is:

1. listen for a global hotkey;
2. record microphone audio;
3. trim silence with WebRTC VAD;
4. transcribe with faster-whisper;
5. inject text at the active cursor, or print it in dry-run mode.

Prefer small, reversible changes that keep this flow easy to reason about.

## Safety rules for commands

- Do not run raw `pytest tests/` during normal development.
- Use targeted tests first, for example:
  `python -m pytest tests/unit/test_model_manager.py -q`
- For broader checks, use:
  `python -m pytest tests/ -m "not smoke and not integration and not slow and not requires_model" --timeout=60`
- Do not run smoke scripts, microphone capture, hotkey listeners, text injection,
  or real ASR model loading unless the user explicitly asks for it.
- Do not download Whisper/faster-whisper models from unit tests.
- Do not start multiple full test runs in parallel.
- If a command times out, check for surviving `python.exe` processes before
  retrying.

## ASR and model rules

The ASR stack is intentionally lazy-loaded. Unit tests must patch the wrappers in
`src.asr.model_manager`:

- `_download_model`
- `_create_whisper_model`

Do not patch `faster_whisper.download_model` or instantiate
`faster_whisper.WhisperModel` directly in unit tests. That can import heavy ML
dependencies, touch the network, allocate large memory, or leave orphaned Python
processes after timeout.

Any test that intentionally loads a real model must be marked with
`@pytest.mark.requires_model` and must not run in the default test command.

## Runtime risk areas

- `App.toggle_recording()` owns toggle-mode behavior. In toggle mode, the first
  hotkey press starts capture and the next press stops, transcribes, and injects.
- `AudioBuffer` is bounded by `audio_max_record_seconds`; do not remove that
  limit without replacing it with another memory guard.
- `VADProvider.split_on_silence()` must keep actual neighboring silence frames as
  context; it must not duplicate speech frames.
- Text injection changes require Windows-specific testing and should default to
  dry-run verification first.

## Review checklist

Before handing work back:

1. Run focused tests for changed modules.
2. Run `ruff check` on changed code if practical.
3. Run `mypy src` when public types changed.
4. Mention any skipped smoke, hardware, model, or integration tests.
5. Document remaining risks in the final response.
