@echo off
chcp 65001 >nul
cd /d "%~dp0"
set PYTHONIOENCODING=utf-8
rem 1. 把懂球帝每小时新抓的赔率导入数据库（约 40 秒）
.venv\Scripts\python -m src.clean.dongqiudi_import
rem 2. 算未来 7 天的预测，存进预测存档，生成网页
.venv\Scripts\python -m src.predict.preview
if errorlevel 1 (
  echo.
  echo Something went wrong. Copy the message above and send it to Claude.
  pause
  exit /b 1
)
rem 3. 用默认浏览器打开
start "" "data\reports\preview.html"
