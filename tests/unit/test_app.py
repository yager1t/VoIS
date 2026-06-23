"""Unit tests for the application orchestrator."""

from __future__ import annotations

from collections.abc import Callable
from contextlib import contextmanager
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from src.app import App, trim_silence
from src.asr.base import TranscriptionResult
from src.config import Settings
from src.dictionary.base import ContextMode

SAMPLE_RATE = 16000


@pytest.fixture
def settings(tmp_path) -> Settings:
    """Return minimal settings for App construction."""
    return Settings(
        data_dir=tmp_path / "data",
        models_dir=tmp_path / "models",
        vocab_dir=tmp_path / "vocab",
    )


@pytest.fixture
def mock_vad() -> MagicMock:
    """Return a VAD mock that leaves audio unchanged."""
    vad = MagicMock()
    vad.split_on_silence.return_value = []
    return vad


def _make_final_transcriber_factory(
    instances: list[MagicMock],
) -> Callable[..., MagicMock]:
    """Return a factory for FinalTranscriber mocks that invoke callbacks on start."""

    def _factory(*args: object, **kwargs: object) -> MagicMock:
        instance = MagicMock()
        instance._final_text = "final transcript"
        callback = args[3] if len(args) > 3 else kwargs.get("on_result")

        def _start() -> None:
            if callback is not None:
                callback(instance._final_text)

        instance.start.side_effect = _start
        instances.append(instance)
        return instance

    return _factory


@contextmanager
def _build_app(
    settings: Settings,
    mock_vad: MagicMock,
    *,
    streaming: bool = False,
    warmup: bool = False,
):
    """Construct an App instance with mocked platform dependencies."""

    def _make_event() -> MagicMock:
        event_mock = MagicMock()
        _set = False

        def _set_fn() -> None:
            nonlocal _set
            _set = True

        def _is_set() -> bool:
            return _set

        event_mock.set.side_effect = _set_fn
        event_mock.is_set.side_effect = _is_set
        return event_mock

    event_mock = _make_event()
    with (
        patch("src.app.WebRTCVADProvider", return_value=mock_vad),
        patch("src.app.threading.Event", return_value=event_mock),
        patch("src.app.create_hotkey_manager") as mock_hotkey_factory,
        patch("src.app.create_text_injector") as mock_injector_factory,
        patch("src.app.AudioCapture") as mock_capture_cls,
        patch("src.app.AudioBuffer") as mock_buffer_cls,
        patch("src.app.VocabularyManager") as mock_vocab_cls,
        patch("src.app.TextCorrector") as mock_corrector_cls,
        patch("src.app.VocabularyLearner") as mock_learner_cls,
        patch("src.app.StreamingTranscriber") as mock_streaming_cls,
        patch("src.app.FinalTranscriber") as mock_final_cls,
    ):
        mock_hotkey = MagicMock()
        mock_hotkey_factory.return_value = mock_hotkey
        mock_injector = MagicMock()
        mock_injector_factory.return_value = mock_injector
        mock_capture = MagicMock()
        mock_capture_cls.return_value = mock_capture
        mock_buffer = MagicMock()
        mock_buffer_cls.return_value = mock_buffer
        mock_vocab = MagicMock()
        mock_vocab_cls.return_value = mock_vocab
        mock_corrector = MagicMock()
        mock_corrector.correct.side_effect = lambda text, context=None: text
        mock_corrector_cls.return_value = mock_corrector
        mock_learner = MagicMock()
        mock_learner_cls.return_value = mock_learner
        mock_streaming = MagicMock()
        mock_streaming_cls.return_value = mock_streaming
        final_instances: list[MagicMock] = []
        mock_final_cls.side_effect = _make_final_transcriber_factory(final_instances)

        settings.asr_warmup_at_start = warmup
        if streaming:
            settings.streaming_enabled = True

        app = App(settings)
        # Expose mocks for assertions
        app._hotkey_mock = mock_hotkey
        app._injector_mock = mock_injector
        app._capture_mock = mock_capture
        app._buffer_mock = mock_buffer
        app._vocab_mock = mock_vocab
        app._corrector_mock = mock_corrector
        app._learner_mock = mock_learner
        app._streaming_mock = mock_streaming
        app._streaming_cls_mock = mock_streaming_cls
        app._final_transcriber_instances = final_instances
        app._final_transcriber_cls_mock = mock_final_cls
        yield app


@pytest.fixture
def app(settings: Settings, mock_vad: MagicMock) -> App:
    """Return an App instance with mocked platform dependencies."""
    with _build_app(settings, mock_vad) as app:
        yield app


@pytest.fixture
def streaming_app(settings: Settings, mock_vad: MagicMock) -> App:
    """Return an App instance with streaming transcription enabled."""
    with _build_app(settings, mock_vad, streaming=True) as app:
        yield app


def test_start_recording_clears_buffer_and_starts_capture(app: App) -> None:
    """start_recording should reset the buffer and begin audio capture."""
    app.start_recording()

    app._buffer_mock.clear.assert_called_once()
    app._capture_mock.start.assert_called_once()


def test_start_recording_invokes_callback(app: App) -> None:
    """start_recording should invoke the recording_started callback."""
    callback = MagicMock()
    app.recording_started = callback

    app.start_recording()

    callback.assert_called_once_with()


def test_toggle_recording_starts_then_stops_and_transcribes(app: App) -> None:
    """toggle_recording should start first, then stop/transcribe on next press."""
    audio = np.ones(SAMPLE_RATE, dtype=np.float32)
    app._capture_mock.is_recording.side_effect = [False, True]
    app._buffer_mock.get.return_value = audio
    app.vad.split_on_silence.return_value = [audio]
    asr_mock = MagicMock()
    asr_mock.transcribe.return_value.text = "toggle text"
    app._asr = asr_mock

    app.toggle_recording()
    app.toggle_recording()

    app._capture_mock.start.assert_called_once()
    app._capture_mock.stop.assert_called_once()
    asr_mock.transcribe.assert_called_once()
    app._injector_mock.inject_with_delay.assert_called_once()


def test_stop_recording_transcribes_and_injects(app: App, settings: Settings) -> None:
    """stop_recording should transcribe captured audio and inject the text."""
    audio = np.ones(SAMPLE_RATE, dtype=np.float32)
    app._buffer_mock.get.return_value = audio
    # VAD returns the audio unchanged
    app.vad.split_on_silence.return_value = [audio]

    asr_mock = MagicMock()
    asr_mock.transcribe.return_value.text = "hello world"
    app._asr = asr_mock
    app.post_processor.process = MagicMock(return_value="Hello world.")

    app.stop_recording()

    app._capture_mock.stop.assert_called_once()
    app._buffer_mock.clear.assert_called_once()
    asr_mock.transcribe.assert_called_once()
    app._corrector_mock.correct.assert_called_once_with("hello world", ContextMode.general)
    app.post_processor.process.assert_called_once_with("hello world")
    app._injector_mock.inject_with_delay.assert_called_once_with(
        "Hello world.",
        settings.injection_delay_ms,
    )


def test_stop_recording_invokes_callbacks(app: App) -> None:
    """stop_recording should invoke recording_stopped and text_injected."""
    audio = np.ones(SAMPLE_RATE, dtype=np.float32)
    app._buffer_mock.get.return_value = audio
    app.vad.split_on_silence.return_value = [audio]

    asr_mock = MagicMock()
    asr_mock.transcribe.return_value.text = "hello world"
    app._asr = asr_mock
    app.post_processor.process = MagicMock(return_value="Hello world.")

    stopped = MagicMock()
    text_injected = MagicMock()
    app.recording_stopped = stopped
    app.text_injected = text_injected

    app.stop_recording()

    stopped.assert_called_once_with()
    text_injected.assert_called_once_with("Hello world.")


def test_stop_recording_dry_run_skips_injection(app: App, settings: Settings) -> None:
    """In dry-run mode stop_recording should log text without injecting."""
    settings.dry_run = True
    audio = np.ones(SAMPLE_RATE, dtype=np.float32)
    app._buffer_mock.get.return_value = audio
    app.vad.split_on_silence.return_value = [audio]

    asr_mock = MagicMock()
    asr_mock.transcribe.return_value.text = "dry run text"
    app._asr = asr_mock
    app.post_processor.process = MagicMock(return_value="Dry run text.")

    app.stop_recording()

    app._injector_mock.inject_with_delay.assert_not_called()
    asr_mock.transcribe.assert_called_once()
    app.post_processor.process.assert_called_once_with("dry run text")


def test_stop_recording_no_audio_skips_asr(app: App) -> None:
    """stop_recording should short-circuit when no audio was captured."""
    app._buffer_mock.get.return_value = np.array([], dtype=np.float32)
    asr_mock = MagicMock()
    app._asr = asr_mock

    stopped = MagicMock()
    text_injected = MagicMock()
    app.recording_stopped = stopped
    app.text_injected = text_injected

    app.stop_recording()

    asr_mock.transcribe.assert_not_called()
    app._injector_mock.inject_with_delay.assert_not_called()
    stopped.assert_called_once_with()
    text_injected.assert_not_called()


def test_stop_recording_no_speech_after_vad_skips_injection(app: App) -> None:
    """stop_recording should skip ASR/injection when VAD finds no speech."""
    audio = np.ones(SAMPLE_RATE, dtype=np.float32)
    app._buffer_mock.get.return_value = audio
    app.vad.split_on_silence.return_value = []

    asr_mock = MagicMock()
    app._asr = asr_mock

    stopped = MagicMock()
    text_injected = MagicMock()
    app.recording_stopped = stopped
    app.text_injected = text_injected

    app.stop_recording()

    asr_mock.transcribe.assert_not_called()
    app._injector_mock.inject_with_delay.assert_not_called()
    stopped.assert_called_once_with()
    text_injected.assert_not_called()


def test_stop_sets_event_and_stops_hotkey(app: App) -> None:
    """stop() should signal shutdown and stop the hotkey listener."""
    app._running = True

    app.stop()

    assert app._running is False
    assert app._shutdown_event.is_set()
    app._hotkey_mock.stop.assert_called_once()
    app._capture_mock.stop.assert_called_once()


def test_is_running_reflects_state(app: App) -> None:
    """is_running should reflect the internal running flag."""
    app._running = False
    assert app.is_running() is False

    app._running = True
    assert app.is_running() is True


def test_start_loop_exits_after_stop(app: App) -> None:
    """The main loop should terminate once stop() is called."""
    call_count = 0

    def side_effect(*_args, **_kwargs) -> None:
        nonlocal call_count
        call_count += 1
        if call_count >= 2:
            app.stop()

    app._shutdown_event.wait.side_effect = side_effect

    app.start()

    assert app._running is False
    app._hotkey_mock.start.assert_called_once()
    app._hotkey_mock.stop.assert_called_once()


def test_trim_silence_returns_empty_when_no_speech() -> None:
    """trim_silence should return an empty array when no speech is detected."""
    vad = MagicMock()
    vad.split_on_silence.return_value = []
    audio = np.ones(SAMPLE_RATE, dtype=np.float32)

    result = trim_silence(audio, SAMPLE_RATE, vad)

    assert result.size == 0
    vad.split_on_silence.assert_called_once()


def test_trim_silence_concatenates_segments() -> None:
    """trim_silence should concatenate VAD speech segments."""
    vad = MagicMock()
    seg1 = np.ones(100, dtype=np.float32)
    seg2 = np.ones(50, dtype=np.float32)
    vad.split_on_silence.return_value = [seg1, seg2]

    result = trim_silence(np.ones(SAMPLE_RATE, dtype=np.float32), SAMPLE_RATE, vad)

    assert result.size == 150
    assert np.array_equal(result, np.concatenate([seg1, seg2], dtype=np.float32))


def test_transcribe_audio_handles_empty_result(app: App) -> None:
    """transcribe_audio should return an empty string when ASR yields no text."""
    asr_mock = MagicMock()
    asr_mock.transcribe.return_value.text = ""
    app._asr = asr_mock

    result = app.transcribe_audio(np.zeros(SAMPLE_RATE, dtype=np.float32))

    assert result == ""
    asr_mock.transcribe.assert_called_once()


def test_inject_text_ignores_empty_string(app: App) -> None:
    """inject_text should not call the injector for empty strings."""
    app.inject_text("")

    app._injector_mock.inject_with_delay.assert_not_called()


def test_start_handles_already_running(app: App) -> None:
    """start() should return cleanly when the app is already running."""
    app._running = True

    app.start()

    app._hotkey_mock.start.assert_not_called()
    assert app._running is True


def test_stop_called_on_exception_in_start(app: App) -> None:
    """stop() should be called when start() raises an unexpected exception."""
    app._shutdown_event.wait.side_effect = RuntimeError("loop error")

    with pytest.raises(RuntimeError, match="loop error"):
        app.start()

    app._hotkey_mock.stop.assert_called_once()
    app._capture_mock.stop.assert_called_once()
    assert app._running is False


def test_stop_recording_corrects_when_dictionary_enabled(app: App) -> None:
    """When dictionary_enabled is true, stop_recording should run the corrector."""
    app.settings.dictionary_enabled = True
    app._corrector_mock.correct.side_effect = None
    app._corrector_mock.correct.return_value = "corrected hello world"

    audio = np.ones(SAMPLE_RATE, dtype=np.float32)
    app._buffer_mock.get.return_value = audio
    app.vad.split_on_silence.return_value = [audio]

    asr_mock = MagicMock()
    asr_mock.transcribe.return_value.text = "hello world"
    app._asr = asr_mock
    app.post_processor.process = MagicMock(return_value="Corrected hello world.")

    app.stop_recording()

    app._corrector_mock.correct.assert_called_once_with("hello world", ContextMode.general)
    app.post_processor.process.assert_called_once_with("corrected hello world")


def test_stop_recording_skips_correction_when_dictionary_disabled(app: App) -> None:
    """When dictionary_enabled is false, stop_recording should skip the corrector."""
    app.settings.dictionary_enabled = False

    audio = np.ones(SAMPLE_RATE, dtype=np.float32)
    app._buffer_mock.get.return_value = audio
    app.vad.split_on_silence.return_value = [audio]

    asr_mock = MagicMock()
    asr_mock.transcribe.return_value.text = "hello world"
    app._asr = asr_mock
    app.post_processor.process = MagicMock(return_value="Hello world.")

    app.stop_recording()

    app._corrector_mock.correct.assert_not_called()
    app.post_processor.process.assert_called_once_with("hello world")


def test_stop_recording_learns_from_text_when_enabled(app: App) -> None:
    """When learning is enabled, stop_recording should feed post-processed text to the learner."""
    app.settings.dictionary_learning_enabled = True

    audio = np.ones(SAMPLE_RATE, dtype=np.float32)
    app._buffer_mock.get.return_value = audio
    app.vad.split_on_silence.return_value = [audio]

    asr_mock = MagicMock()
    asr_mock.transcribe.return_value.text = "hello kubernetes"
    app._asr = asr_mock
    app.post_processor.process = MagicMock(return_value="Hello Kubernetes.")

    app.stop_recording()

    app._learner_mock.learn_from_text.assert_called_once_with("Hello Kubernetes.")


def test_stop_recording_skips_learning_when_disabled(app: App) -> None:
    """When learning is disabled, stop_recording should not call the learner."""
    app.settings.dictionary_learning_enabled = False

    audio = np.ones(SAMPLE_RATE, dtype=np.float32)
    app._buffer_mock.get.return_value = audio
    app.vad.split_on_silence.return_value = [audio]

    asr_mock = MagicMock()
    asr_mock.transcribe.return_value.text = "hello world"
    app._asr = asr_mock
    app.post_processor.process = MagicMock(return_value="Hello world.")

    app.stop_recording()

    app._learner_mock.learn_from_text.assert_not_called()


def test_record_correction_delegates_to_learner(app: App) -> None:
    """App.record_correction should forward to the learner."""
    app.record_correction("foo", "bar")

    app._learner_mock.record_correction.assert_called_once_with("foo", "bar")


def _mock_asr(app: App) -> MagicMock:
    """Install a mock ASR provider on the app and return it."""
    asr_mock = MagicMock()
    app._asr = asr_mock
    return asr_mock


def test_capture_callback_routes_to_buffer_and_streamer(streaming_app: App) -> None:
    """Audio chunks should reach both the buffer and the streaming transcriber."""
    _mock_asr(streaming_app)
    streaming_app.start_recording()
    callback = streaming_app._capture_mock.set_callback.call_args.args[0]
    chunk = np.ones(160, dtype=np.float32)

    callback(chunk)

    streaming_app._buffer_mock.append.assert_called_once_with(chunk)
    streaming_app._streaming_mock.add_audio.assert_called_once_with(chunk)


def test_start_recording_creates_streaming_transcriber(streaming_app: App) -> None:
    """When streaming is enabled, start_recording creates and starts the transcriber."""
    _mock_asr(streaming_app)
    streaming_app.start_recording()

    streaming_app._streaming_cls_mock.assert_called_once_with(
        streaming_app.settings,
        streaming_app.asr,
        streaming_app.vad,
    )
    streaming_app._streaming_mock.start.assert_called_once()


def test_stop_recording_uses_streaming_results(streaming_app: App) -> None:
    """When streaming results exist, they are used instead of fallback transcription."""
    from src.asr.base import TranscriptionResult

    _mock_asr(streaming_app)
    audio = np.ones(SAMPLE_RATE, dtype=np.float32)
    streaming_app._buffer_mock.get.return_value = audio
    streaming_app.vad.split_on_silence.return_value = [audio]
    streaming_app._streaming_mock.get_results.return_value = [
        TranscriptionResult(text="hello", is_final=True),
        TranscriptionResult(text="world", is_final=True),
    ]
    streaming_app.post_processor.process = MagicMock(return_value="Hello world.")

    streaming_app.start_recording()
    streaming_app.stop_recording()

    streaming_app._streaming_mock.stop.assert_called_once()
    streaming_app._streaming_mock.get_results.assert_called_once()
    streaming_app._asr.transcribe.assert_not_called()
    streaming_app._injector_mock.inject_with_delay.assert_called_once_with(
        "Hello world.",
        streaming_app.settings.injection_delay_ms,
    )


def test_stop_recording_streaming_fallback_when_empty(streaming_app: App) -> None:
    """When streaming yields no final text, fallback to full transcription."""
    asr_mock = _mock_asr(streaming_app)
    audio = np.ones(SAMPLE_RATE, dtype=np.float32)
    streaming_app._buffer_mock.get.return_value = audio
    streaming_app.vad.split_on_silence.return_value = [audio]
    streaming_app._streaming_mock.get_results.return_value = []
    asr_mock.transcribe.return_value.text = "fallback text"
    streaming_app.post_processor.process = MagicMock(return_value="Fallback text.")

    streaming_app.start_recording()
    streaming_app.stop_recording()

    asr_mock.transcribe.assert_called_once()
    streaming_app._injector_mock.inject_with_delay.assert_called_once_with(
        "Fallback text.",
        streaming_app.settings.injection_delay_ms,
    )


def test_stop_recording_non_streaming_uses_existing_flow(app: App, settings: Settings) -> None:
    """When streaming is disabled, stop_recording uses the existing transcription flow."""
    asr_mock = _mock_asr(app)
    audio = np.ones(SAMPLE_RATE, dtype=np.float32)
    app._buffer_mock.get.return_value = audio
    app.vad.split_on_silence.return_value = [audio]
    asr_mock.transcribe.return_value.text = "non streaming text"
    app.post_processor.process = MagicMock(return_value="Non streaming text.")

    app.stop_recording()

    asr_mock.transcribe.assert_called_once()
    app._streaming_cls_mock.assert_not_called()
    app._final_transcriber_cls_mock.assert_not_called()
    app._injector_mock.inject_with_delay.assert_called_once_with(
        "Non streaming text.",
        settings.injection_delay_ms,
    )


def test_stop_recording_emits_text_finalized_when_streaming_enabled(
    streaming_app: App,
) -> None:
    """When streaming is enabled, a background final transcription emits text_finalized."""
    _mock_asr(streaming_app)
    audio = np.ones(SAMPLE_RATE, dtype=np.float32)
    streaming_app._buffer_mock.get.return_value = audio
    streaming_app.vad.split_on_silence.return_value = [audio]
    streaming_app._streaming_mock.get_results.return_value = [
        TranscriptionResult(text="hello", is_final=True),
        TranscriptionResult(text="world", is_final=True),
    ]
    streaming_app.post_processor.process = MagicMock(
        side_effect={"hello world": "Hello world.", "final transcript": "Final text."}.get
    )

    text_finalized = MagicMock()
    streaming_app.text_finalized = text_finalized

    streaming_app.start_recording()
    streaming_app.stop_recording()

    assert len(streaming_app._final_transcriber_instances) == 1
    final_mock = streaming_app._final_transcriber_instances[-1]
    final_mock.start.assert_called_once()
    streaming_app._injector_mock.inject_with_delay.assert_called_once_with(
        "Hello world.",
        streaming_app.settings.injection_delay_ms,
    )
    text_finalized.assert_called_once_with("Final text.")


def test_stop_recording_skips_text_finalized_when_disabled(streaming_app: App) -> None:
    """When final_transcription_enabled is false, no FinalTranscriber is started."""
    _mock_asr(streaming_app)
    streaming_app.settings.final_transcription_enabled = False
    audio = np.ones(SAMPLE_RATE, dtype=np.float32)
    streaming_app._buffer_mock.get.return_value = audio
    streaming_app.vad.split_on_silence.return_value = [audio]
    streaming_app._streaming_mock.get_results.return_value = [
        TranscriptionResult(text="hello", is_final=True),
    ]
    streaming_app.post_processor.process = MagicMock(return_value="Hello.")

    text_finalized = MagicMock()
    streaming_app.text_finalized = text_finalized

    streaming_app.start_recording()
    streaming_app.stop_recording()

    streaming_app._final_transcriber_cls_mock.assert_not_called()
    text_finalized.assert_not_called()


def test_start_recording_cancels_pending_final_transcriber(streaming_app: App) -> None:
    """Starting a new recording should stop any pending final transcription."""
    _mock_asr(streaming_app)
    audio = np.ones(SAMPLE_RATE, dtype=np.float32)
    streaming_app._buffer_mock.get.return_value = audio
    streaming_app.vad.split_on_silence.return_value = [audio]
    streaming_app._streaming_mock.get_results.return_value = [
        TranscriptionResult(text="hello", is_final=True),
    ]
    streaming_app.post_processor.process = MagicMock(return_value="Hello.")

    streaming_app.start_recording()
    streaming_app.stop_recording()

    assert len(streaming_app._final_transcriber_instances) == 1
    first_final = streaming_app._final_transcriber_instances[-1]

    streaming_app.start_recording()

    first_final.stop.assert_called_once()


def _capture_thread(target: Callable[..., object], *args: object, **_kwargs: object) -> MagicMock:
    """Capture thread construction details and return a dummy thread mock."""
    thread_mock = MagicMock()
    thread_mock._target = target
    thread_mock._args = args
    thread_mock.start.side_effect = lambda: target(*args)
    return thread_mock


def test_warmup_triggered_at_start(settings: Settings, mock_vad: MagicMock) -> None:
    """When asr_warmup_at_start is true, warmup is started in a background thread."""
    created_threads: list[MagicMock] = []

    def _tracking_thread(
        target: Callable[..., object], *args: object, **_kwargs: object
    ) -> MagicMock:
        thread_mock = _capture_thread(target, *args)
        created_threads.append(thread_mock)
        return thread_mock

    with (
        _build_app(settings, mock_vad, warmup=True) as app,
        patch(
            "src.app.threading.Thread", side_effect=_tracking_thread
        ) as thread_mock,
    ):
        _ = _mock_asr(app)

        app.start_recording()

        thread_mock.assert_called_once()
        assert len(created_threads) == 1
        dummy_thread = created_threads[0]
        dummy_thread.start.assert_called_once()
        assert dummy_thread._target == app._warmup_asr
        assert app._asr_warmed_up is True


def test_warmup_runs_asr_warmup(settings: Settings, mock_vad: MagicMock) -> None:
    """_warmup_asr should call asr.warmup() and mark warmup complete."""
    with _build_app(settings, mock_vad) as app:
        asr_mock = _mock_asr(app)

        app._warmup_asr()

        asr_mock.warmup.assert_called_once()
        assert app._asr_warmed_up is True


def test_warmup_skipped_when_disabled(settings: Settings, mock_vad: MagicMock) -> None:
    """When asr_warmup_at_start is false, warmup is not triggered."""
    with _build_app(settings, mock_vad) as app:
        asr_mock = _mock_asr(app)

        app.start_recording()

        asr_mock.warmup.assert_not_called()
        assert app._asr_warmed_up is False
