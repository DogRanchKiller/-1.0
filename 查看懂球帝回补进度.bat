@echo off
chcp 65001 >nul
cd /d "%~dp0"
set PYTHONIOENCODING=utf-8
.venv\Scripts\python -m src.jobs.dongqiudi_progress
echo.
pause
