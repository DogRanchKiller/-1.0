@echo off
chcp 65001 >nul
cd /d "%~dp0"
set PYTHONIOENCODING=utf-8
rem 开启局域网访问：加一条只在“专用网络”、只放行网站端口的防火墙规则，打开网站服务任务，
rem 检查现在的 Wi-Fi 是不是“公用网络”（是就提示你去设置里改，不替你改）。会弹窗要管理员权限。
rem 具体做什么见 src\jobs\lan_access.ps1 开头的说明；关掉用“关闭局域网访问.bat”。
powershell -NoProfile -ExecutionPolicy Bypass -File "src\jobs\lan_access.ps1"
