@echo off
setlocal

REM Run smoke tests with real hardware/OS interaction.
REM Requires microphone access and may inject text unless dry-run mode is used.
cd /d "%~dp0\.."
pytest tests/smoke -v --run-smoke --timeout=30
