# Checklist Fix Plan

This file tracks issues found while verifying `docs/checklist.md`.

## Open Issues

- [ ] Local `main` is ahead of `origin/main`.
  - Impact: checklist section 9 is not satisfied for GitHub synchronization.
  - Evidence: `git status --short --branch` reported `main...origin/main
    [ahead 31]`.
  - Next step: push after fixes and successful verification.

- [ ] Some recent commit bodies do not use the exact `Change-Id:` spelling.
  - Impact: checklist section 9 is partially satisfied.
  - Evidence: some commits use `Change-ID:` or omit the field.
  - Next step: decide whether to normalize future commits only or rewrite local
    history before publishing.

- [ ] Backup folder date does not match the current environment date.
  - Impact: checklist section 8 is partially satisfied.
  - Evidence: latest backups are under `backups/2026-06-22`, while the current
    environment date is 2026-06-23.
  - Next step: create any new backups under the current date if further risky
    edits require them.

## Resolved Issues

- [x] `ruff format --check src tests benchmarks` failed.
  - Resolution: ran `ruff format src tests benchmarks`; the format check now
    passes.

- [x] `pytest tests/unit/test_settings_window.py -v --timeout=60` failed.
  - Resolution: made `src.ui` package exports lazy so importing
    `src.ui.settings_window` no longer imports `src.ui.tray` and real
    `PyQt6.QtGui`.

- [x] `asr_streaming_beam_size` was not applied in the streaming ASR path.
  - Resolution: `StreamingTranscriber` now calls `transcribe_streaming()` with
    the configured streaming beam size, and `FasterWhisperProvider` applies that
    value.

- [x] `docs/checklist.md` referenced a non-existent streaming test path.
  - Resolution: updated the path to `tests/unit/test_streaming_transcriber.py`.

- [x] The checklist "CI-safe" command excluded integration tests while naming
  `tests/integration`.
  - Resolution: removed `not integration` from that command.

- [x] CI workflow did not fully match checklist commands.
  - Resolution: updated GitHub Actions to lint `benchmarks` and collect coverage
    for both `src` and `benchmarks`.

- [x] Coverage numbers in docs were stale after the latest verification.
  - Resolution: updated README/testing docs to the latest safe-suite result:
    290 passed, 92.64% coverage.
