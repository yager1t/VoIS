# Architecture

This document describes the current architecture of the Voice-to-Cursor AI Dictation System.

For the original product brief, see [`voice_to_cursor_prompt.md`](../voice_to_cursor_prompt.md).  
For the approved implementation plan, see [`C:\Users\Jury\.kimi\plans\thor-wonder-man-raven.md`](../.kimi/plans/thor-wonder-man-raven.md).  
For day-to-day safety rules when editing or running commands, see [`docs/ai_working_guide.md`](ai_working_guide.md).

## Current status

The MVP is complete and working on Windows. The critical path is implemented end-to-end:

```
Global hotkey  →  Audio capture  →  VAD trim  →  ASR (faster-whisper)  →  Text injection
```

## Data flow

1. `HotkeyManager` listens for the configured global hotkey (default `f9`).
2. On press it notifies `App`, which clears `AudioBuffer` and starts `AudioCapture`.
3. While the key is held, `AudioCapture` streams microphone chunks into `AudioBuffer`.
4. On release, `App` stops capture, runs VAD-based silence trimming, and sends the trimmed audio to `ASRProvider`.
5. `FasterWhisperProvider` transcribes the audio (model is lazy-loaded on first use).
6. `App` passes the resulting text to `TextInjector`.
7. `WindowsTextInjector` types the text at the active cursor via `SendInput`, or falls back to clipboard paste when configured.

In `--dry-run` mode the text is logged instead of injected.

## Module breakdown

### `src/main.py`
CLI entry point. Parses arguments, loads `Settings` from environment/`.env`, applies CLI overrides, configures logging, and starts `App`.

### `src/app.py`
Application orchestrator. Owns the hotkey manager, audio capture, buffer, VAD provider, ASR provider, and text injector. Implements both push-to-talk and toggle modes.

### `src/config.py`
Pydantic-settings based configuration. Includes hotkey, audio, ASR, VAD, injection, and dry-run settings.

### `src/logging_config.py`
Loguru configuration with console and rotating file sinks.

### `src/audio/`
- `capture.py` — `AudioCapture` wraps `sounddevice.InputStream` with push-to-talk / toggle semantics.
- `buffer.py` — `AudioBuffer` is a thread-safe accumulator with an optional `max_seconds` bound to prevent unbounded memory growth.
- `vad.py` — `VADProvider` interface and `WebRTCVADProvider` implementation using `webrtcvad`. Includes `split_on_silence` for trimming leading/trailing silence while keeping neighboring context frames.

### `src/asr/`
- `base.py` — `ASRProvider` interface and `TranscriptionResult` dataclass.
- `model_manager.py` — `ModelManager` resolves model paths and handles downloads. Model download and instantiation are wrapped in patchable functions (`_download_model`, `_create_whisper_model`) so unit tests can avoid importing heavy ML packages.
- `whisper_provider.py` — `FasterWhisperProvider` transcribes audio via `faster-whisper`. The model is lazy-loaded.

### `src/hotkey/`
- `base.py` — `HotkeyManager` interface and `parse_hotkey` helper.
- `windows.py` — `PynputHotkeyManager` implements global hotkeys on Windows using `pynput`, supporting push-to-talk and toggle modes.

### `src/injection/`
- `base.py` — `TextInjector` interface.
- `windows.py` — `WindowsTextInjector` uses `ctypes`/`SendInput` with `KEYEVENTF_UNICODE` for direct text injection, with an optional clipboard fallback.

## Safety boundaries

- `AudioBuffer` is bounded by `audio_max_record_seconds`.
- ASR model loading is lazy and isolated behind `ModelManager` wrappers.
- Heavy imports (`faster_whisper`, `webrtcvad`) are deferred where possible.
- Real hardware/OS interaction (microphone, hotkeys, injection) is gated behind smoke tests and scripts, not unit tests.

## Roadmap

Completed:
- [x] Infrastructure, audio capture, VAD, ASR, hotkeys, text injection, pipeline integration.
- [x] Unit test coverage > 90 %.

Planned:
- [ ] System tray / settings UI (v0.2)
- [ ] LLM post-processing (v0.2)
- [ ] User dictionary and adaptive learning (v0.3)
- [ ] Streaming ASR and latency optimization (v0.4)
- [ ] macOS and Linux support (v0.5)
