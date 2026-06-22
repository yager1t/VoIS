# Testing Guide

This project uses [pytest](https://docs.pytest.org/) for unit tests and
[pytest-cov](https://pypi.org/project/pytest-cov/) for coverage reporting.

## Running unit tests

From the repository root:

```bash
pytest tests/unit -m "not smoke and not integration and not slow and not requires_model" --timeout=60
```

For a quieter output:

```bash
pytest tests/unit -q
```

Do not use raw `pytest tests/` for normal AI-assisted development. The default
safe run excludes smoke, integration, slow, and real-model tests so it cannot
open microphone hooks, inject text, download ASR models, or leave long-running
Python workers.

## Running with coverage

To run unit tests and print a missing-line coverage report:

```bash
pytest tests/unit -m "not smoke and not integration and not slow and not requires_model" --cov=src --cov-report=term-missing --timeout=60
```

To also generate an HTML coverage report:

```bash
pytest tests/unit --cov=src --cov-report=term-missing --cov-report=html
```

Alternatively use the convenience scripts:

```bash
# Bash (Git Bash / WSL / Linux / macOS)
./scripts/run_tests.sh

# Windows Command Prompt
scripts\run_tests.bat
```

## Running integration tests

Integration tests exercise the composed audio pipeline and end-to-end
application flow using mocked OS dependencies and synthetic audio fixtures.
They are marked with `@pytest.mark.integration` and excluded from the default
safe run:

```bash
pytest tests/integration -q --timeout=60
```

## Running smoke tests

Smoke tests require real hardware (microphone) and OS interaction (global
hotkeys, text injection). They are not run as part of the default unit test
suite. To run them explicitly once supported and approved:

```bash
pytest tests/ -m smoke
```

Real ASR model tests must be marked `requires_model` and run only when explicitly
needed:

```bash
pytest tests/ -m requires_model --timeout=300
```

## Current coverage

Coverage captured on 2026-06-22 after completing the integration-test expansion.

| Name                          | Stmts | Miss | Branch | BrPart | Cover |
|-------------------------------|------:|-----:|-------:|-------:|------:|
| `src\__init__.py`             |     1 |    0 |      0 |      0 |  100% |
| `src\app.py`                  |   105 |    6 |     24 |      3 |   93% |
| `src\asr\__init__.py`         |     4 |    0 |      0 |      0 |  100% |
| `src\asr\base.py`             |    20 |    0 |      0 |      0 |  100% |
| `src\asr\model_manager.py`    |    38 |    4 |      8 |      0 |   91% |
| `src\asr\whisper_provider.py` |    47 |    0 |      8 |      1 |   98% |
| `src\audio\__init__.py`       |     4 |    0 |      0 |      0 |  100% |
| `src\audio\buffer.py`         |    51 |    5 |     16 |      4 |   87% |
| `src\audio\capture.py`        |    77 |    1 |     20 |      2 |   97% |
| `src\audio\vad.py`            |    59 |    4 |     18 |      2 |   92% |
| `src\config.py`               |    27 |    0 |      0 |      0 |  100% |
| `src\hotkey\__init__.py`      |     9 |    0 |      0 |      0 |  100% |
| `src\hotkey\base.py`          |    32 |    1 |      6 |      1 |   95% |
| `src\hotkey\windows.py`       |    97 |   13 |     36 |      8 |   80% |
| `src\injection\__init__.py`   |     9 |    0 |      0 |      0 |  100% |
| `src\injection\base.py`       |    10 |    0 |      0 |      0 |  100% |
| `src\injection\windows.py`    |    69 |    5 |     22 |      6 |   88% |
| `src\logging_config.py`       |    12 |    0 |      0 |      0 |  100% |
| `src\main.py`                 |    44 |    0 |     14 |      0 |  100% |
| **TOTAL**                     | **715**| **39**| **172**| **27**| **92%** |

The overall unit-test coverage is **92%**, exceeding the configured `fail_under = 80`
threshold. There are **112 unit tests** and **7 integration tests**; combined
they complete in under two seconds.
