"""定时任务的“心跳”文件（需求文档第 9 节“完整链路的分工和接口”）。

为什么要有：定时任务在后台跑，电脑睡眠、程序出错、任务没登记，都会让它悄悄不跑。
每个定时任务每次运行都写一个小文件，记“这次什么时候开始、结束、成功没有、做了什么”，
数据状态页和推送程序读这些文件，就知道哪个任务很久没成功了。

文件放在 settings.yaml 的 jobs.state_dir（默认 data/jobs/），一个任务一个文件：<任务名>.json，内容例如：
    {"任务": "scheduled_predict", "开始": "2026-11-01 10:00:00", "结束": "2026-11-01 10:01:05",
     "成功": true, "摘要": {"新存": 3}, "错误": null, "上次成功": "2026-11-01 10:01:05"}
时间都是 UTC，写成 "YYYY-MM-DD HH:MM:SS"。"上次成功" 在这次失败时保留上一次成功的时间。

用法（在定时任务的 main 里）：
    with job_run("scheduled_predict") as run:
        ...干活...
        run.summary["新存"] = 3      # 想让状态页显示的几个数
出错时文件里记下错误，再把错误照样抛出去（程序照样以失败结束，任务计划程序也能看到）。
"""
import json
import os
import traceback
from contextlib import contextmanager
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path

from src.common.config import load_settings, resolve_path

TIME_FMT = "%Y-%m-%d %H:%M:%S"


def _now_text() -> str:
    """现在的 UTC 时间，写成文本。"""
    return datetime.now(timezone.utc).strftime(TIME_FMT)


def state_dir(settings: dict | None = None) -> Path:
    """心跳文件放在哪个文件夹（settings.yaml 的 jobs.state_dir）。"""
    settings = settings or load_settings()
    return resolve_path(settings["jobs"]["state_dir"])


def read_state(name: str, folder: Path | None = None) -> dict | None:
    """读一个任务的心跳文件；没有这个文件（任务从没跑过）或文件坏了返回 None。"""
    path = (folder or state_dir()) / f"{name}.json"
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None


def read_all(folder: Path | None = None) -> dict[str, dict]:
    """读文件夹里全部心跳文件：{任务名: 内容}。读不了的文件跳过。"""
    folder = folder or state_dir()
    out = {}
    if folder.exists():
        for p in sorted(folder.glob("*.json")):
            s = read_state(p.stem, folder)
            if s is not None:
                out[p.stem] = s
    return out


def _write(path: Path, data: dict) -> None:
    """先写临时文件再改名：写到一半断电时，不会留下半个文件（别的程序读到的总是完整的旧文件或新文件）。"""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(data, ensure_ascii=False, indent=1), encoding="utf-8")
    os.replace(tmp, path)


@dataclass
class JobRun:
    """一次运行。summary 里放想让状态页显示的几个数（要能写成 JSON）。"""
    name: str
    started: str
    summary: dict = field(default_factory=dict)


@contextmanager
def job_run(name: str, folder: Path | None = None):
    """包住一次定时任务的运行，开始、结束各写一次心跳文件。"""
    folder = folder or state_dir()
    path = folder / f"{name}.json"
    last_ok = (read_state(name, folder) or {}).get("上次成功")
    run = JobRun(name=name, started=_now_text())
    # 开始时先写一次（"结束" 为空 = 正在跑；如果之后一直是空的，说明程序中途被杀掉了）
    _write(path, {"任务": name, "开始": run.started, "结束": None, "成功": None,
                  "摘要": {}, "错误": None, "上次成功": last_ok})
    try:
        yield run
    except BaseException as e:
        _write(path, {"任务": name, "开始": run.started, "结束": _now_text(), "成功": False,
                      "摘要": run.summary, "错误": "".join(traceback.format_exception_only(type(e), e)).strip(),
                      "上次成功": last_ok})
        raise
    end = _now_text()
    _write(path, {"任务": name, "开始": run.started, "结束": end, "成功": True,
                  "摘要": run.summary, "错误": None, "上次成功": end})
