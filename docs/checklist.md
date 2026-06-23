# Чек-лист проверки и тестов Voice-to-Cursor

Чек-лист для ручной и автоматической верификации перед коммитом, релизом или после значимых изменений. Предполагается запуск на Windows в виртуальном окружении проекта (`.venv`).

---

## 1. Подготовка окружения

- [ ] Активировано виртуальное окружение `.venv`
- [ ] Установлены зависимости: `pip install -r requirements.txt -r requirements-dev.txt`
- [ ] Установлены Windows-зависимости (`webrtcvad-wheels>=2.0.10`, `PyQt6`, `pynput`)
- [ ] Нет незафиксированных изменений перед началом работы (`git status`)
- [ ] Создан pre-change backup в `backups/YYYY-MM-DD/pre_change_<name>.tar.gz`

---

## 2. Статические проверки

Запускать из корня проекта.

```bash
ruff check src tests benchmarks
ruff format --check src tests benchmarks
mypy src
```

- [ ] `ruff check` проходит без ошибок
- [ ] `ruff format --check` проходит без ошибок (или `ruff format src tests benchmarks` применён)
- [ ] `mypy src` проходит без ошибок и без новых `note` критичных для CI

---

## 3. Безопасный набор тестов (CI-эквивалент)

```bash
pytest tests/unit tests/integration \
  -m "not smoke and not slow and not integration and not requires_model" \
  --cov=src --cov=benchmarks --cov-report=term-missing \
  --timeout=60
```

- [ ] Все тесты проходят
- [ ] Покрытие кода `>= 80%`
- [ ] Нет таймаутов (`pytest-timeout`)
- [ ] Нет падений, связанных с попыткой скачать/загрузить Whisper-модель

> **Важно:** юнит-тесты должны патчить `src.asr.model_manager._download_model` и `_create_whisper_model`. Никогда не создавать настоящий `WhisperModel` в тестах.

---

## 4. Проверка v0.4 Streaming ASR

### 4.1. Настройки

- [ ] `src/config.py` содержит:
  - `streaming_enabled: bool`
  - `streaming_chunk_seconds: float`
  - `asr_streaming_beam_size: int`
  - `asr_warmup_at_start: bool`
  - `final_transcription_enabled: bool`
  - `replace_streaming_with_final: bool`

### 4.2. App / оркестратор

- [ ] `start_recording()` создаёт и запускает `StreamingTranscriber`, если `streaming_enabled=True`
- [ ] `stop_recording()` останавливает `StreamingTranscriber` и получает результаты
- [ ] При включённом стриминге быстрый результат сразу инжектируется
- [ ] `FinalTranscriber` запускается в фоне после инжекта стриминг-текста
- [ ] Callback `text_finalized(final_text)` вызывается после финальной транскрипции
- [ ] При `streaming_enabled=False` сохраняется старое синхронное поведение
- [ ] `start_recording()` отменяет/ждёт предыдущий `FinalTranscriber`

### 4.3. Поставщики ASR

- [ ] `ASRProvider.warmup()` объявлен как абстрактный
- [ ] `FasterWhisperProvider.warmup()` транскрибирует 1 секунду тишины
- [ ] `transcribe_streaming()` и `transcribe()` корректно обрабатывают `beam_size`

### 4.4. Юнит-тесты

- [ ] `tests/unit/asr/test_streaming.py` проходит
- [ ] `tests/unit/asr/test_final_transcriber.py` проходит
- [ ] `tests/unit/test_app.py` проверяет новые callback'и и настройки

---

## 5. Проверка UI

```bash
pytest tests/unit/test_settings_window.py -v --timeout=60
```

- [ ] Окно настроек открывается без исключений
- [ ] В окне есть группа "Streaming ASR"
- [ ] Все новые контролы сохраняют значения в `Settings`
- [ ] Загрузка настроек корректно заполняет контролы
- [ ] `replace_streaming_with_final` отключён/помечен как зарезервированный

> GUI-smoke (реальное открытие окна, клики, микрофон) запускать только по явному запросу.

---

## 6. Бенчмарки латентности

```bash
python benchmarks/run_latency.py --mode both
pytest tests/unit/benchmarks -v --timeout=60
```

- [ ] `LatencyBenchmark` возвращает корректные метрики (`time_to_first_partial`, `time_to_final`, `rtf`, `transcript`)
- [ ] Mock-режим работает без реальной модели
- [ ] Реальные бенчмарки помечены `@pytest.mark.slow` и `@pytest.mark.requires_model`
- [ ] По умолчанию реальные бенчмарки не запускаются

---

## 7. Smoke и интеграционные тесты (только вручную / по запросу)

```bash
# Не запускать автоматически в CI/AI без явного разрешения
pytest tests/smoke -m smoke --timeout=120
pytest tests/integration -m integration --timeout=120
```

- [ ] Запуск приложения без падений
- [ ] Захват аудио с микрофона работает
- [ ] Горячая клавиша запускает/останавливает запись
- [ ] Текст инжектируется в активное окно
- [ ] Streaming ASR даёт промежуточный/финальный результат

---

## 8. Документация и метаданные

- [ ] `README.md` актуален (функции, версия, инструкции)
- [ ] `docs/architecture.md` отражает текущую архитектуру
- [ ] `docs/testing.md` содержит актуальные команды и покрытие
- [ ] Версия в `pyproject.toml` и `src/__init__.py` совпадают
- [ ] Создан AI change log: `ai-changes/YYYY-MM-DD-change-N.md`
- [ ] Создан post-change backup: `backups/YYYY-MM-DD/post_change_<name>.tar.gz`

---

## 9. Git / коммит

- [ ] Рабочее дерево чистое: `git status` показывает `nothing to commit`
- [ ] Коммит имеет осмысленное сообщение в формате проекта
- [ ] В сообщении есть `Change-Id:`
- [ ] Коммит(ы) не содержат случайных файлов (`.coverage`, кэш, бэкапы)
- [ ] `.gitignore` актуален

---

## 10. Предрелизная сводка

| Проверка | Команда | Минимум | Факт |
|---|---|---|---|
| Линтер | `ruff check src tests benchmarks` | 0 ошибок | |
| Форматер | `ruff format --check src tests benchmarks` | 0 изменений | |
| Типизация | `mypy src` | 0 ошибок | |
| Юнит + интеграция | `pytest tests/unit tests/integration -m "not smoke and not slow and not requires_model" --cov=src --cov=benchmarks --timeout=60` | 80% | |
| Версия | `grep version pyproject.toml src/__init__.py` | совпадает | |

---

## Быстрый старт для AI-агента

```bash
cd /c/Users/Jury/Desktop/KimiCode/Vo-IS
source .venv/Scripts/activate
ruff check src tests benchmarks
mypy src
pytest tests/unit tests/integration -m "not smoke and not slow and not requires_model" --cov=src --cov=benchmarks --timeout=60
```
