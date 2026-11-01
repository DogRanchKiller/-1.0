@echo off
chcp 65001 >nul
cd /d "%~dp0"
set PYTHONIOENCODING=utf-8
rem 发一条测试推送“足球预测系统：测试消息”到微信，成功或失败都用中文说明原因
rem 要先双击“设置微信推送.bat”设置好口令
.venv\Scripts\python -m src.notify.send test
echo.
pause
