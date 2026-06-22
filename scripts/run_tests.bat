@echo off
setlocal

cd /d "%~dp0\.."

echo ==^> Linting
call ruff check src tests
if errorlevel 1 exit /b 1

echo ==^> Type checking
call mypy src
if errorlevel 1 exit /b 1

echo ==^> Running tests with coverage
call pytest tests/ --cov=src --cov-report=term-missing --cov-report=html
if errorlevel 1 exit /b 1
