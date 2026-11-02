"""只读预演日更任务登记：不能碰六个既有任务，也不能启用推送。"""
import json
from pathlib import Path
import shutil
import subprocess
import sys

import pytest


@pytest.mark.parametrize("shell_name", ["pwsh", "powershell"])
def test_registration_plan_is_limited_to_two_source_jobs(shell_name):
    # Windows内置5.1与PowerShell 7传递原生命令引号的行为不同，两者都要真实预演。
    shell = shutil.which(shell_name)
    if not shell:
        pytest.skip("此环境没有PowerShell；真实Windows验收另做")
    root = Path(__file__).resolve().parents[1]
    script = root / "src/jobs/register_data_tasks.ps1"
    assert script.exists(), "应提供仅登记新来源任务的安全入口"
    proc = subprocess.run([shell, "-NoProfile", "-ExecutionPolicy", "Bypass", "-File", str(script),
        "-PlanOnly", "-PythonPath", sys.executable], capture_output=True, encoding="utf-8", errors="replace", timeout=20)
    assert proc.returncode == 0, proc.stderr
    plans = json.loads(proc.stdout.lstrip("\ufeff"))
    assert len(plans) == 2
    assert {p["module"] for p in plans} == {"src.jobs.refresh_football_data", "src.jobs.refresh_cups"}
    assert {p["task_name"] for p in plans} == {"足球预测_联赛收盘更新", "足球预测_杯赛赛程更新"}
    assert all(Path(p["working_directory"]).resolve() == root for p in plans)
    assert all(p["multiple_instances"] == "IgnoreNew" and p["start_when_available"] for p in plans)
    assert all(p["execution_limit_minutes"] > 0 for p in plans)
    assert all(Path(p["executable"]).name.lower() == "pythonw.exe" for p in plans)
    assert len({p["at"] for p in plans}) == 2
