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

Smoke tests exercise the composed App pipeline end-to-end. They are marked with
`@pytest.mark.smoke` and skipped by default. The automated harness in
`tests/smoke/test_smoke.py` stubs platform dependencies so it does not open a
real microphone, register global hotkeys, or inject text, but it still requires
explicit opt-in via the `--run-smoke` flag:

```bash
pytest tests/smoke -v --run-smoke --timeout=30
```

Convenience scripts are provided for manual runs:

```bash
# Bash (Git Bash / WSL / Linux / macOS)
./scripts/run_smoke.sh

# Windows Command Prompt
scripts\run_smoke.bat
```

For interactive manual smoke testing you can still use the legacy script, but
new automated runs should prefer `tests/smoke/test_smoke.py`:

```bash
python scripts/smoke_test.py --duration 30 --dry-run
```

Real ASR model tests must be marked `requires_model` and run only when explicitly
needed:

```bash
pytest tests/ -m requires_model --timeout=300
```

## Current coverage

Coverage captured on 2026-06-22 after completing the settings-window implementation
(Phase 3 of v0.2).

| Name                             | Stmts | Miss | Branch | BrPart | Cover |
|----------------------------------|------:|-----:|-------:|-------:|------:|
| `src\__init__.py`                |     1 |    0 |      0 |      0 |  100% |
| `src\app.py`                     |   121 |    6 |     28 |      3 |   94% |
| `src\asr\__init__.py`            |     4 |    0 |      0 |      0 |  100% |
| `src\asr\base.py`                |    20 |    0 |      0 |      0 |  100% |
| `src\asr\model_manager.py`       |    38 |    4 |      8 |      0 |   91% |
| `src\asr\whisper_provider.py`    |    47 |    0 |      8 |      1 |   98% |
| `src\audio\__init__.py`          |     4 |    0 |      0 |      0 |  100% |
| `src\audio\buffer.py`            |    51 |    5 |     16 |      4 |   87% |
| `src\audio\capture.py`           |    77 |    1 |     20 |      2 |   97% |
| `src\audio\vad.py`               |    59 |    0 |     18 |      0 |  100% |
| `src\config.py`                  |    30 |    0 |      0 |      0 |  100% |
| `src\hotkey\__init__.py`         |     9 |    0 |      0 |      0 |  100% |
| `src\hotkey\base.py`             |    32 |    1 |      6 |      1 |   95% |
| `src\hotkey\windows.py`          |    97 |   13 |     36 |      8 |   80% |
| `src\injection\__init__.py`      |     9 |    0 |      0 |      0 |  100% |
| `src\injection\base.py`          |    10 |    0 |      0 |      0 |  100% |
| `src\injection\windows.py`       |    69 |    5 |     22 |      6 |   88% |
| `src\logging_config.py`          |    12 |    0 |      0 |      0 |  100% |
| `src\main.py`                    |    75 |    6 |     14 |      1 |   92% |
| `src\postprocess\__init__.py`    |    10 |    0 |      2 |      0 |  100% |
| `src\postprocess\base.py`        |     5 |    0 |      0 |      0 |  100% |
| `src\postprocess\formatter.py`   |    13 |    0 |      4 |      0 |  100% |
| `src\postprocess\llm_client.py`  |    27 |    0 |      0 |      0 |  100% |
| `src\ui\__init__.py`             |     4 |    0 |      0 |      0 |  100% |
| `src\ui\settings_window.py`      |    90 |    0 |      6 |      0 |  100% |
| `src\ui\tray.py`                 |    69 |    5 |     18 |      5 |   89% |
| **TOTAL**                        | **983**| **46**| **206**| **31**| **93%** |

The overall code coverage is **93%** (93.02% precise), exceeding the configured `fail_under = 80`
threshold. There are **151 unit tests**, **7 integration tests**, and **1 smoke
test (159 total); the safe suites complete in under two seconds.

## CI

Continuous integration is configured in `.github/workflows/ci.yml`. The workflow
runs on pushes and pull requests to `master` and `main`, linting with `ruff`,
type-checking with `mypy`, and executing the unit and integration test suites
with coverage on Python 3.11 and 3.12. Smoke tests are not run in CI because they
require the `--run-smoke` opt-in flag.
