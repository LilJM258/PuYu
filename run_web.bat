@echo off
cd /d "%~dp0"
if not exist ".venv\Scripts\python.exe" (
  echo Please follow README-web.md to install dependencies first.
  pause
  exit /b 1
)
".venv\Scripts\python.exe" run_web.py
if errorlevel 1 pause
