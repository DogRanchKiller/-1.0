# 仅登记两个新增来源日更任务；不重新登记预测、网站、备份或推送。
# -PlanOnly 输出实际将登记的内容，不修改 Windows 任务。
param([switch]$PlanOnly, [string]$PythonPath)

$ErrorActionPreference = 'Stop'
[Console]::OutputEncoding = [System.Text.UTF8Encoding]::new($false)
$refreshRoot = (Resolve-Path "$PSScriptRoot\..\..").Path
if (-not $PythonPath) { $PythonPath = Join-Path $refreshRoot '.venv\Scripts\python.exe' }
if (-not (Test-Path -LiteralPath $PythonPath -PathType Leaf)) { throw "找不到项目 Python：$PythonPath" }
$refreshRunner = Join-Path (Split-Path -Parent $PythonPath) 'pythonw.exe'
if (-not (Test-Path -LiteralPath $refreshRunner -PathType Leaf)) { throw "找不到后台运行 Python：$refreshRunner" }
$refreshConfigPath = Join-Path $refreshRoot 'config\settings.yaml'
# 内置PowerShell 5.1会移除-c参数内的双引号；Python字面量用单引号以兼容两种Shell。
$readConfig = "import json,sys,yaml; s=yaml.safe_load(open(sys.argv[1],encoding='utf-8')); print(json.dumps({'data_refresh':s['data_refresh'],'cups_enabled':s['cup_refresh']['enabled']}))"
$configText = & $PythonPath -X utf8 -c $readConfig $refreshConfigPath
if ($LASTEXITCODE -ne 0) { throw '读取维护任务配置失败' }
$refreshConfig = ($configText | ConvertFrom-Json)
$refreshLimit = [int]$refreshConfig.data_refresh.task_limit_minutes
if ($refreshLimit -le 0 -or $refreshLimit -gt 120) { throw '日更任务限时必须在1至120分钟内' }
$refreshEntries = @(
    @{task_name='足球预测_联赛收盘更新'; module='src.jobs.refresh_football_data'; at=$refreshConfig.data_refresh.football_data_at}
)
if ($refreshConfig.cups_enabled) {
    $refreshEntries += @{task_name='足球预测_杯赛赛程更新'; module='src.jobs.refresh_cups'; at=$refreshConfig.data_refresh.cups_at}
}
$refreshPlans = @($refreshEntries | ForEach-Object {
    if ($_.at -notmatch '^([01][0-9]|2[0-3]):[0-5][0-9]$') { throw '任务时间必须是北京时间HH:mm' }
    [pscustomobject]@{
        task_name=$_.task_name; module=$_.module; at=$_.at
        executable=$refreshRunner; arguments="-X utf8 -m $($_.module)"
        working_directory=$refreshRoot; execution_limit_minutes=$refreshLimit
        multiple_instances='IgnoreNew'; start_when_available=$true
    }
})
if ($PlanOnly) {
    ConvertTo-Json -InputObject $refreshPlans -Depth 4
    return
}
# 配置中的日程按北京时间定义，不能在其他时区悄悄登记为当地时间。
if ((Get-TimeZone).Id -ne 'China Standard Time') { throw '此登记脚本要求Windows时区为北京时间' }
foreach ($refreshPlan in $refreshPlans) {
    $existingTask = Get-ScheduledTask -TaskName $refreshPlan.task_name -ErrorAction SilentlyContinue
    if ($existingTask -and $existingTask.State -eq 'Running') { throw "任务正在运行，未覆盖登记：$($refreshPlan.task_name)" }
    if ($existingTask) {
        $existingActions = @($existingTask.Actions)
        if ($existingActions.Count -ne 1 -or $existingActions[0].WorkingDirectory -ne $refreshRoot -or
            $existingActions[0].Execute -ne $refreshPlan.executable -or
            $existingActions[0].Arguments -ne $refreshPlan.arguments) {
            throw "同名任务内容不同，未覆盖登记：$($refreshPlan.task_name)"
        }
    }
}
foreach ($refreshPlan in $refreshPlans) {
    $refreshAction = New-ScheduledTaskAction -Execute $refreshPlan.executable -Argument $refreshPlan.arguments -WorkingDirectory $refreshRoot
    $refreshTrigger = New-ScheduledTaskTrigger -Daily -At $refreshPlan.at
    $refreshSettings = New-ScheduledTaskSettingsSet -StartWhenAvailable -MultipleInstances IgnoreNew `
        -ExecutionTimeLimit (New-TimeSpan -Minutes $refreshLimit)
    Register-ScheduledTask -TaskName $refreshPlan.task_name -Action $refreshAction -Trigger $refreshTrigger -Settings $refreshSettings `
        -Description '低频更新当前赛季已登记来源，验证后导入；失败由心跳和状态页提示' -Force | Out-Null
}
ConvertTo-Json -InputObject $refreshPlans -Depth 4
