# AI Development Rules

These rules apply to every AI-assisted change in this repository. Also read
`docs/ai_working_guide.md` before running commands or editing code.

## Core principle

Every change must be:

- reversible;
- versioned;
- traceable;
- tested at the right scope;
- documented when behavior or workflow changes.

## Versioning and change records

Use semantic versioning:

- MAJOR for breaking changes;
- MINOR for new features;
- PATCH for fixes.

Every AI-generated change should include an AI change id when committed or
documented:

```text
YYYYMMDD-model-hash
```

Example:

```text
20260622-kimi-8f3a21
```

Commit messages should use this shape:

```text
[feat/fix/refactor/test/docs] short description

AI: yes/no
Model: codex / kimi / gpt / local-llm
Change-ID: xxx
```

## Backup and rollback

Before large AI modifications:

1. Make sure the current git state is understood.
2. Prefer a git commit or a clearly named backup archive.
3. Keep rollback possible within a few minutes.

Do not delete files, reset history, or rewrite architecture without explicit
user approval.

## Safe command policy

Default test commands must be safe for a Windows desktop machine:

```bash
python -m pytest tests/ -m "not smoke and not integration and not slow and not requires_model" --timeout=60
```

Do not run raw `pytest tests/` during AI work. Do not start multiple full test
runs in parallel.

Never run commands that can open real microphone capture, global keyboard hooks,
text injection, model downloads, or real ASR model loading unless the user
explicitly asks for that kind of test.

If a test or tool command times out, inspect for surviving `python.exe`
processes before retrying.

## ASR-specific restrictions

Unit tests must not instantiate `faster_whisper.WhisperModel` or download
Whisper models. Patch these wrappers instead:

- `src.asr.model_manager._download_model`
- `src.asr.model_manager._create_whisper_model`

Any real model test must be marked `requires_model` and excluded from the
default test run.

## Validation pipeline

Use the smallest useful validation set first:

1. focused unit tests for changed modules;
2. `ruff check` for changed code or `src tests`;
3. `mypy src` when public types or module contracts changed;
4. smoke tests only with explicit approval.

## AI change log

For substantial changes, create or update an entry under:

```text
ai-changes/YYYY-MM-DD-change-N.md
```

Include:

- reason;
- changed files;
- validation run;
- risk;
- rollback plan.
