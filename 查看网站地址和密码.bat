@echo off
chcp 65001 >nul
cd /d "%~dp0"
set PYTHONIOENCODING=utf-8
rem 查看局域网网站的地址和密码：还没有密码就自动生成一个，重新算一遍电脑的局域网地址，
rem 写进 data\site_server\网站地址和密码.txt 并用记事本打开。
.venv\Scripts\python -m src.web.serve_auth show
if errorlevel 1 pause
