@echo off
chcp 65001 >nul
cd /d "%~dp0"
set PYTHONIOENCODING=utf-8
rem 设置微信推送：粘贴 pushplus 网站上的 token（屏幕上不显示），写进 config\secrets.yaml 的 pushplus.token
rem secrets.yaml 不会上传 GitHub；里面原有的其他内容保留
.venv\Scripts\python -m src.notify.setup
echo.
pause
