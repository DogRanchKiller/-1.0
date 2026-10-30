# 把定时任务登记到 Windows 任务计划程序。重复运行会覆盖旧的登记，不会重复。
# 运行方法（在项目文件夹里，用 PowerShell）：
#     powershell -ExecutionPolicy Bypass -File src\jobs\register_tasks.ps1
# 取消登记：
#     Unregister-ScheduledTask -TaskName 足球预测_每日备份 -Confirm:$false

$root = (Resolve-Path "$PSScriptRoot\..\..").Path   # 项目文件夹
$python = Join-Path $root ".venv\Scripts\python.exe"

# 每日备份：每天 03:30；电脑当时关着，开机后尽快补跑（需求文档第 16 节）
$action = New-ScheduledTaskAction -Execute $python -Argument "-m src.jobs.backup" -WorkingDirectory $root
$trigger = New-ScheduledTaskTrigger -Daily -At 03:30
$settings = New-ScheduledTaskSettingsSet -StartWhenAvailable -ExecutionTimeLimit (New-TimeSpan -Hours 1)
Register-ScheduledTask -TaskName "足球预测_每日备份" -Action $action -Trigger $trigger -Settings $settings `
    -Description "备份数据库和原始页面" -Force | Out-Null

Get-ScheduledTask -TaskName "足球预测_*" | Select-Object TaskName, State
