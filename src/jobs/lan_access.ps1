# 开启 / 关闭局域网网站的访问（需求文档第 8 节“局域网网站”、第 16 节“安全”）。
# 双击根目录的“开启局域网访问.bat”“关闭局域网访问.bat”运行这个文件；会弹窗要管理员权限（改防火墙必须要）。
#
# 开启（不带参数）：
#   1. 添加一条 Windows 防火墙规则（名字固定，见 settings.yaml 的 site_server.firewall_rule）：
#      只在“专用网络”下生效、只放行网站端口的 TCP 入站、只允许同一网段的设备。重复运行只更新这条规则，不会加出第二条。
#   2. 网站服务任务如果被“关闭局域网访问”关掉过，重新打开它并启动。
#   3. 看现在连的网络是“公用”还是“专用”：是公用就提示你去 Windows 设置里改（不替你改）。
#   4. 看有没有以前留下的、挡住 Python 的防火墙规则（以前弹窗时点了“取消”会留下），有就提示。
# 关闭（-Close）：删掉这条防火墙规则，停掉并关掉网站服务任务（不删除登记，“开启”时会重新打开）。
#
# 文件必须保存为带 BOM 的 UTF-8，否则 Windows 自带的 PowerShell 5.1 会把中文读乱。

param([switch]$Close)

$ErrorActionPreference = 'Stop'
$root = (Resolve-Path "$PSScriptRoot\..\..").Path   # 项目文件夹
$python = Join-Path $root ".venv\Scripts\python.exe"

function Wait-Enter {
    Write-Host ""
    Read-Host "按回车关闭这个窗口" | Out-Null
}

# ---------- 1. 没有管理员权限就重新以管理员身份打开自己（会弹出“是否允许”的窗口） ----------
$me = [Security.Principal.WindowsPrincipal][Security.Principal.WindowsIdentity]::GetCurrent()
if (-not $me.IsInRole([Security.Principal.WindowsBuiltInRole]::Administrator)) {
    $argList = @('-NoProfile', '-ExecutionPolicy', 'Bypass', '-File', "`"$PSCommandPath`"")
    if ($Close) { $argList += '-Close' }
    try {
        Start-Process -FilePath "powershell.exe" -ArgumentList $argList -Verb RunAs | Out-Null
        Write-Host "已经打开一个新的窗口（管理员），请在那个窗口里看结果。"
    } catch {
        Write-Host "没有拿到管理员权限（弹窗里点了“否”？），什么都没改。要开启请重新双击，弹窗时点“是”。"
        Wait-Enter
    }
    exit 0
}

try {
    Set-Location $root
    # ---------- 2. 从 settings.yaml 读端口、规则名、任务名（中文是 \uXXXX 写法，不会因为控制台编码读乱） ----------
    $info = (& $python -m src.web.serve_auth firewall-info) | ConvertFrom-Json
    if (-not $info -or -not $info.port -or -not $info.rule -or -not $info.task) {
        throw "没读到 settings.yaml 里 site_server 的端口、规则名或任务名"
    }
    $port = [int]$info.port
    $rule = [string]$info.rule
    $task = [string]$info.task

    if ($Close) {
        # ---------- 关闭 ----------
        $existing = Get-NetFirewallRule -DisplayName $rule -ErrorAction SilentlyContinue
        if ($existing) {
            $existing | Remove-NetFirewallRule
            Write-Host "已删除防火墙规则“$rule”：手机、别的电脑连不进来了。"
        } else {
            Write-Host "防火墙规则“$rule”本来就没有。"
        }
        $t = Get-ScheduledTask -TaskName $task -ErrorAction SilentlyContinue
        if ($t) {
            Stop-ScheduledTask -TaskName $task -ErrorAction SilentlyContinue
            Disable-ScheduledTask -TaskName $task | Out-Null
            Write-Host "已停掉并关掉定时任务“$task”（以后登录 Windows 也不会自动启动）。想再打开就双击“开启局域网访问.bat”。"
        } else {
            Write-Host "定时任务“$task”还没有登记过。"
        }
        Wait-Enter
        exit 0
    }

    # ---------- 开启：防火墙规则 ----------
    $existing = Get-NetFirewallRule -DisplayName $rule -ErrorAction SilentlyContinue
    if ($existing) {
        # 已经有了：只把设置改成现在应有的样子（端口改过也能跟上），不加第二条
        $existing | Set-NetFirewallRule -Direction Inbound -Action Allow -Profile Private -Enabled True
        $existing | Get-NetFirewallPortFilter | Set-NetFirewallPortFilter -Protocol TCP -LocalPort $port
        $existing | Get-NetFirewallAddressFilter | Set-NetFirewallAddressFilter -RemoteAddress LocalSubnet
        Write-Host "防火墙规则“$rule”已经有了，已核对：只在专用网络、只放行 TCP $port 端口、只允许同一网段。"
    } else {
        New-NetFirewallRule -DisplayName $rule -Direction Inbound -Action Allow -Protocol TCP -LocalPort $port `
            -Profile Private -RemoteAddress LocalSubnet `
            -Description "足球预测系统的局域网网站（只在专用网络、只放行这个端口）。删掉它：双击“关闭局域网访问.bat”" | Out-Null
        Write-Host "已添加防火墙规则“$rule”：只在专用网络、只放行 TCP $port 端口、只允许同一网段。"
    }

    # ---------- 开启：公用网络、域网络下明确挡住这个端口 ----------
    # 以前运行 Python 时 Windows 弹窗点“允许”会留下“Python 任何端口都放行（含公用网络）”的规则，
    # 上面那条只限专用网络的规则管不住它。Windows 防火墙里“阻止”比“允许”优先，所以再加一条只挡这个端口的阻止规则：
    # 电脑连到公共 Wi-Fi 时别人连不进网站，其他 Python 程序不受影响。“关闭”时也保留它（留着没有坏处）。
    $blockRule = "$rule（公用网络阻止）"
    $blockExisting = Get-NetFirewallRule -DisplayName $blockRule -ErrorAction SilentlyContinue
    if ($blockExisting) {
        $blockExisting | Set-NetFirewallRule -Direction Inbound -Action Block -Profile 'Public,Domain' -Enabled True
        $blockExisting | Get-NetFirewallPortFilter | Set-NetFirewallPortFilter -Protocol TCP -LocalPort $port
    } else {
        New-NetFirewallRule -DisplayName $blockRule -Direction Inbound -Action Block -Protocol TCP -LocalPort $port `
            -Profile 'Public,Domain' `
            -Description "足球预测系统：电脑连公用网络时挡住网站端口（以前给 Python 的“全部放行”规则管不住，这条优先）" | Out-Null
    }
    Write-Host "已核对防火墙规则“$blockRule”：连公用网络时挡住 TCP $port 端口。"

    # ---------- 开启：网站服务任务 ----------
    $t = Get-ScheduledTask -TaskName $task -ErrorAction SilentlyContinue
    if ($t) {
        if ($t.State -eq 'Disabled') { Enable-ScheduledTask -TaskName $task | Out-Null }
        Start-ScheduledTask -TaskName $task
        Write-Host "定时任务“$task”已打开并启动（已经在运行的不会开第二个）。"
    } else {
        Write-Host "注意：定时任务“$task”还没有登记，网站服务器不会自动运行。请告诉 Claude 登记一下。"
    }

    # ---------- 开启：现在的网络是公用还是专用 ----------
    $profiles = @(Get-NetConnectionProfile -ErrorAction SilentlyContinue)
    $public = @($profiles | Where-Object { $_.NetworkCategory -eq 'Public' })
    if ($profiles.Count -eq 0) {
        Write-Host ""
        Write-Host "现在好像没有连上任何网络（Wi-Fi 或网线）。连上以后手机才能打开网站。"
    }
    foreach ($p in $public) {
        Write-Host ""
        Write-Host "【要你动手】现在连的网络“$($p.Name)”被 Windows 当成“公用网络”，手机连不进来。"
        Write-Host "  如果这是你家里的 Wi-Fi，请这样改成“专用网络”："
        Write-Host "  开始菜单 → 设置 → 网络和 Internet → 点上面正在用的 WLAN（或以太网）→ 属性"
        Write-Host "  → “网络配置文件类型”选“专用网络”。改完不用重新运行这个程序。"
        Write-Host "  咖啡店、酒店、公司访客网这类公共 Wi-Fi 不要改，也不要在那里用这个网站。"
    }
    if ($profiles.Count -gt 0 -and $public.Count -eq 0) {
        Write-Host "现在连的网络是“专用网络”，手机连同一个 Wi-Fi 就能打开。"
    }

    # ---------- 开启：有没有挡住 Python 的旧规则 ----------
    # 第一次运行 Python 网络程序时 Windows 会弹窗问“是否允许”，点了“取消”会留下一条“阻止”规则，它比上面的“允许”优先
    $blocks = @(Get-NetFirewallRule -Direction Inbound -Action Block -Enabled True -ErrorAction SilentlyContinue |
        Where-Object { ($_ | Get-NetFirewallApplicationFilter).Program -match 'python' })
    if ($blocks.Count -gt 0) {
        Write-Host ""
        Write-Host "【可能要你动手】防火墙里有 $($blocks.Count) 条挡住 Python 的旧规则（以前弹窗时点了“取消”留下的），会让手机连不进来："
        $blocks | ForEach-Object { Write-Host "    $($_.DisplayName)（$($_.Profile)）" }
        Write-Host "  手机打不开时：开始菜单搜“高级安全 Windows Defender 防火墙”→ 入站规则 → 找到上面这几条 → 右键“禁用规则”。"
        Write-Host "  拿不准就把这段话发给 Claude。"
    }

    Write-Host ""
    Write-Host "好了。网址和密码：双击“查看网站地址和密码.bat”。"
} catch {
    Write-Host ""
    Write-Host "出错了：$($_.Exception.Message)"
    Write-Host "请把这个窗口里的字发给 Claude。"
}
Wait-Enter
