@echo off
chcp 65001 >nul
cd /d "%~dp0"
echo Downloading football-data.co.uk history. About 30 minutes. You can close this window any time; the next run continues.
echo.
.venv\Scripts\python -m src.collect.football_data
echo.
pause
