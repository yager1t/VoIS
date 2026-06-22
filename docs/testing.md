# Testing Guide

This project uses [pytest](https://docs.pytest.org/) for unit tests and
[pytest-cov](https://pypi.org/project/pytest-cov/) for coverage reporting.

## Running unit tests

From the repository root:

```bash
pytest tests/
```

For a quieter output:

```bash
pytest tests/ -q
```

## Running with coverage

To run all tests and print a missing-line coverage report:

```bash
pytest tests/ --cov=src --cov-report=term-missing
```

To also generate an HTML coverage report:

```bash
pytest tests/ --cov=src --cov-report=term-missing --cov-report=html
```

Alternatively use the convenience scripts:

```bash
# Bash (Git Bash / WSL / Linux / macOS)
./scripts/run_tests.sh

# Windows Command Prompt
scripts\run_tests.bat
```

## Running smoke tests

Smoke tests require real hardware (microphone) and OS interaction (global
hotkeys, text injection). They are not run as part of the default unit test
suite. To run them explicitly once supported:

```bash
pytest tests/ -m smoke
```

## Baseline coverage

Coverage was captured on 2026-06-22 after adding the test infrastructure.

| Module                        | Stmts | Miss | Branch | BrPart | Cover |
|-------------------------------|------:|-----:|-------:|-------:|------:|
| src\__init__.py               |     1 |    0 |      0 |      0 |  100% |
| src\app.py                    |    96 |    8 |     20 |      4 |   90% |
| src\asr\__init__.py           |     4 |    0 |      0 |      0 |  100% |
| src\asr\base.py               |    20 |    0 |      0 |      0 |  100% |
| src\asr\model_manager.py      |    34 |   12 |      8 |      0 |   67% |
| src\asr\whisper_provider.py   |    47 |    0 |      8 |      1 |   98% |
| src\audio\__init__.py         |     4 |    0 |      0 |      0 |  100% |
| src\audio\buffer.py           |    38 |    3 |     10 |      3 |   88% |
| src\audio\capture.py          |    77 |   59 |     20 |      0 |   19% |
| src\audio\vad.py              |    53 |   20 |     16 |      0 |   54% |
| src\config.py                 |    26 |    0 |      0 |      0 |  100% |
| src\hotkey\__init__.py        |     9 |    0 |      0 |      0 |  100% |
| src\hotkey\base.py            |    32 |    1 |      6 |      1 |   95% |
| src\hotkey\windows.py         |    97 |   13 |     36 |      8 |   80% |
| src\injection\__init__.py     |     9 |    0 |      0 |      0 |  100% |
| src\injection\base.py         |    10 |    0 |      0 |      0 |  100% |
| src\injection\windows.py      |    69 |    5 |     22 |      6 |   88% |
| src\logging_config.py         |    12 |   12 |      0 |      0 |    0% |
| src\main.py                   |    44 |   44 |     14 |      0 |    0% |
| **TOTAL**                     | **682** | **177** | **160** | **23** | **70%** |

The overall coverage is currently **70%**, which is below the configured
`fail_under = 80` threshold. Future Phase 1 tasks will add tests to raise the
overall coverage above the required 80%.
