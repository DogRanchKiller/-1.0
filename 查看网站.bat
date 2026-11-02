@echo off
chcp 65001 >nul
setlocal
pushd "%~dp0"
set PYTHONIOENCODING=utf-8
rem 先生成静态网站；失败时停下，不打开旧页面冒充本次结果。
.venv\Scripts\python.exe -m src.jobs.build_site
if errorlevel 1 goto failed
rem 按配置里的输出位置打开首页，不把网站路径写死在启动器里。
.venv\Scripts\python.exe -c "import os; from src.common.config import load_settings, resolve_path; os.startfile(str(resolve_path(load_settings()['site']['output_dir']) / 'index.html'))"
if errorlevel 1 goto failed
popd
endlocal
exit /b 0

:failed
echo.
echo 网站生成或打开失败，请把上面的错误信息发给助手。
pause
popd
endlocal
exit /b 1
