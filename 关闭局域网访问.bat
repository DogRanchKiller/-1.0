@echo off
chcp 65001 >nul
cd /d "%~dp0"
set PYTHONIOENCODING=utf-8
rem 关闭局域网访问：删掉防火墙规则，停掉并关掉“足球预测_网站服务”任务。会弹窗要管理员权限。
rem 想再打开就双击“开启局域网访问.bat”。
powershell -NoProfile -ExecutionPolicy Bypass -File "src\jobs\lan_access.ps1" -Close
