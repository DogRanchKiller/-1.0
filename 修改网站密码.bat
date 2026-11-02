@echo off
chcp 65001 >nul
cd /d "%~dp0"
set PYTHONIOENCODING=utf-8
rem 修改局域网网站的登录密码：输入新密码（屏幕上不显示），或直接回车随机生成一个。
rem 新密码写进 data\site_server\网站地址和密码.txt；已经登录的设备都要重新登录。
.venv\Scripts\python -m src.web.serve_auth change
echo.
pause
