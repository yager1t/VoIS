# Architecture

This document describes the current architecture of the Voice-to-Cursor AI Dictation System.

For the original product brief, see [`voice_to_cursor_prompt.md`](../voice_to_cursor_prompt.md).  
For the approved implementation plan, see [`C:\Users\Jury\.kimi\plans\aquaman-phantom-stranger-rictor.md`](file:///C:/Users/Jury/.kimi/plans/aquaman-phantom-stranger-rictor.md).  
For day-to-day safety rules when editing or running commands, see [`docs/ai_working_guide.md`](ai_working_guide.md).

## Current status

The MVP is complete and working on Windows. Phase 1 (LLM post-processing), Phase 2
(system tray), and Phase 3 (settings window) of v0.2 are implemented. The critical path is implemented end-to-end:

```
Global hotkey  →  Audio capture  →  VAD trim  →  ASR (faster-whisper)  →  Post-processing  →  Text injection
```

The application now runs with a Qt event loop. `App` executes its hotkey/audio loop in a
background `QThread`, while the main thread owns `QApplication` and the `QSystemTrayIcon`.

## Data flow

1. `HotkeyManager` listens for the configured global hotkey (default `f9`).
2. On press it notifies `App`, which clears `AudioBuffer` and starts `AudioCapture`.
3. While the key is held, `AudioCapture` streams microphone chunks into `AudioBuffer`.
4. On release, `App` stops capture, runs VAD-based silence trimming, and sends the trimmed audio to `ASRProvider`.
5. `FasterWhisperProvider` transcribes the audio (model is lazy-loaded on first use).
6. `App` passes the raw transcript to a `PostProcessor`. The default implementation is a deterministic `TextFormatter`; when `llm_enabled` is true an `LLMPostProcessor` backed by an Ollama LLM is used instead.
7. `App` passes the post-processed text to `TextInjector`.
8. `WindowsTextInjector` types the text at the active cursor via `SendInput`, or falls back to clipboard paste when configured.

In `--dry-run` mode the text is logged instead of injected.

## Module breakdown

### `src/main.py`
CLI entry point. Parses arguments, loads `Settings` from environment/`.env`, applies CLI overrides, configures logging, creates the Qt application, and starts `App` in a background `QThread`.

### `src/app.py`
Application orchestrator. Owns the hotkey manager, audio capture, buffer, VAD provider, ASR provider, and text injector. Implements both push-to-talk and toggle modes. Exposes optional callback attributes (`recording_started`, `recording_stopped`, `text_injected`) so the tray UI can react to pipeline events without making `App` depend on Qt.

### `src/ui/`
- `tray.py` — `TrayIcon` is a `QSystemTrayIcon` with a context menu (Start/Stop toggle, Settings, Exit) and a recording-state indicator. It wires the `App` callbacks to icon and balloon-notification updates.
- `settings_window.py` — `SettingsWindow` is a `QWidget` form for editing hotkey, push-to-talk, ASR model, language, device, LLM options, and dry-run mode. On save it writes the current configuration back to the project `.env` file and emits `settings_saved`.

### `src/config.py`
Pydantic-settings based configuration. Includes hotkey, audio, ASR, LLM post-processing, VAD, injection, and dry-run settings.

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

### `src/postprocess/`
- `base.py` — `PostProcessor` interface with `process(text, context)`.
- `formatter.py` — `TextFormatter` applies deterministic cleanup (whitespace collapse, capitalization, terminal punctuation).
- `llm_client.py` — `OllamaClient` and `LLMPostProcessor` improve transcripts via the Ollama `/api/chat` endpoint, falling back to the raw text on any error.
- `__init__.py` — `create_post_processor(settings)` factory chooses between LLM and deterministic formatting based on `settings.llm_enabled`.

### `src/injection/`
- `base.py` — `TextInjector` interface.
- `windows.py` — `WindowsTextInjector` uses `ctypes`/`SendInput` with `KEYEVENTF_UNICODE` for direct text injection, with an optional clipboard fallback.

## Safety boundaries

- `AudioBuffer` is bounded by `audio_max_record_seconds`.
- ASR model loading is lazy and isolated behind `ModelManager` wrappers.
- Heavy imports (`faster_whisper`, `webrtcvad`) are deferred where possible.
- Real hardware/OS interaction (microphone, hotkeys, injection) is gated behind smoke tests and scripts, not unit tests.

## Threading model

`QApplication` must run on the main (GUI) thread. `App.start()` blocks on a hotkey/audio
loop, so `main()` wraps `App` in a `_Worker(QObject)` and moves it to a dedicated
`QThread`. The worker thread calls `App.start()`; the main thread remains available for
tray input and the Qt event loop. On exit, `aboutToQuit` stops `App`, quits the worker
thread, and waits up to five seconds for a clean shutdown.

## Roadmap

Completed:
- [x] Infrastructure, audio capture, VAD, ASR, hotkeys, text injection, pipeline integration.
- [x] Unit test coverage > 90 %.
- [x] LLM post-processing layer (Phase 1 of v0.2).
- [x] System tray icon with PyQt6 (Phase 2 of v0.2).
- [x] Settings window with PyQt6 (Phase 3 of v0.2).

Planned:
- [ ] Recording indicator polish and dictation notifications (Phase 4 of v0.2).
- [ ] User dictionary and adaptive learning (v0.3).
- [ ] Streaming ASR and latency optimization (v0.4).
- [ ] macOS and Linux support (v0.5).
