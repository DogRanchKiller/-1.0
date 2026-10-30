@echo off
chcp 65001 >nul
cd /d "%~dp0"
set PYTHONIOENCODING=utf-8
rem 先重新生成一次数据状态页（不弹窗，问题都写在网页上），再用默认浏览器打开
.venv\Scripts\python -c "from src.jobs.status_page import main; main(notify=lambda *a: None)"
start "" "data\reports\status.html"
