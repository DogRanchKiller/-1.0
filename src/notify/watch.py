"""看定时任务的心跳文件，判断每个任务现在好不好（需求文档第 9 节接口 2）。

数据状态页的“定时任务”栏目和故障推送（compose.py）都用这里，所以两边说法、界限一样。
界限在 settings.yaml 的 notify.jobs（每个任务多久没成功算故障）和 notify.running_stale_minutes。

三种故障：
- failed：上次运行失败了（心跳文件“成功”是 false）。
- stuck：“结束”是空的，而且开始了很久（超过 running_stale_minutes 分钟）：卡住了，或者中途电脑关机、程序被关掉。
- stale：上次成功离现在太久（超过这个任务的 stale_hours）：可能任务没登记、电脑一直关着或睡眠。
没有心跳文件 = 这个任务还没运行过（比如还没登记），只显示“还没运行过”，不算故障。
"""
from dataclasses import dataclass, field
from datetime import datetime, timezone

from src.clean.dongqiudi_reader import BEIJING
from src.jobs.heartbeat import TIME_FMT

FAULT_NAMES = {"failed": "上次运行失败", "stuck": "卡住或中途被关掉", "stale": "太久没成功"}
MAX_ERROR_CHARS = 200     # 错误说明太长时只留前面这么多字（页面、手机上看得下）


@dataclass
class Fault:
    type: str        # failed / stuck / stale
    text: str        # 给人看的说明


@dataclass
class JobStatus:
    job: str                       # 任务名（scheduled_predict 等）
    name: str                      # 中文名（settings 的 notify.jobs.<任务>.name）
    state: str                     # never 还没运行过 / running 正在跑 / ok 上次成功 / failed 上次失败
    text: str                      # 一句话说明：最近一次运行、上次成功
    faults: list[Fault] = field(default_factory=list)


def _parse(text) -> datetime | None:
    """心跳文件里的时间文字 → 带 UTC 时区的时间；空的或格式不对返回 None。"""
    if not text:
        return None
    try:
        return datetime.strptime(str(text), TIME_FMT).replace(tzinfo=timezone.utc)
    except ValueError:
        return None


def bj(t: datetime) -> str:
    """UTC 时间 → 北京时间文字（月-日 时:分）。"""
    return t.astimezone(BEIJING).strftime("%m-%d %H:%M")


def ago(t: datetime, now: datetime) -> str:
    """多久以前（和数据状态页同样的说法）。"""
    minutes = (now - t).total_seconds() / 60
    if minutes < 90:
        return f"{minutes:.0f} 分钟前"
    if minutes < 48 * 60:
        return f"{minutes / 60:.1f} 小时前"
    return f"{minutes / 60 / 24:.1f} 天前"


def short_error(err, secret: str | None = None) -> str:
    """错误说明：只留最后一行（异常的类型和说明），太长截断；口令一类的密钥换成 ***。"""
    text = str(err or "").strip().splitlines()[-1:] or [""]
    text = text[0]
    if secret:
        text = text.replace(secret, "***")
    return text if len(text) <= MAX_ERROR_CHARS else text[:MAX_ERROR_CHARS] + "…"


def summary_text(summary) -> str:
    """心跳文件的“摘要”写成一句话，例如“新存 3，T-24 2”。只取数字和短文字，列表、字典不写（太长）。"""
    if not isinstance(summary, dict):
        return ""
    parts = [f"{k} {v}" for k, v in summary.items() if isinstance(v, (int, float, str)) and not isinstance(v, bool)
             and len(str(v)) <= 40]
    return "，".join(parts[:6])


def job_status(job: str, hb: dict | None, now: datetime, settings: dict, secret: str | None = None) -> JobStatus:
    """一个任务现在的状态。hb 是心跳文件的内容（heartbeat.read_state 读的，没有文件是 None）。"""
    cfg = settings["notify"]
    jc = cfg["jobs"].get(job) or {}
    name = jc.get("name") or job
    if hb is None:
        return JobStatus(job, name, "never", "还没运行过（还没登记定时任务，或登记了还没到时间）")
    started, ended, last_ok = _parse(hb.get("开始")), _parse(hb.get("结束")), _parse(hb.get("上次成功"))
    ok = hb.get("成功")
    faults = []
    run = f"最近一次 {bj(started)} 开始（{ago(started, now)}）" if started else "最近一次开始时间不明"
    if ended is None:
        state = "running"
        run += "，还没结束"
        limit = float(cfg["running_stale_minutes"])
        if started and (now - started).total_seconds() / 60 > limit:
            faults.append(Fault("stuck", f"开始后超过 {limit:g} 分钟还没结束：可能卡住了，或者中途电脑关机、程序被关掉"))
    elif ok is True:
        state = "ok"
        run += "，成功"
    else:
        state = "failed"
        run += f"，失败：{short_error(hb.get('错误'), secret) or '没有写原因'}"
        faults.append(Fault("failed", f"上次运行失败：{short_error(hb.get('错误'), secret) or '没有写原因'}"))
    stale = jc.get("stale_hours")
    if last_ok:
        run += f"。上次成功 {bj(last_ok)}（{ago(last_ok, now)}）"
        if stale is not None and (now - last_ok).total_seconds() / 3600 > float(stale):
            faults.append(Fault("stale", f"超过 {float(stale):g} 小时没成功过（上次成功 {bj(last_ok)}）："
                                         "可能电脑关着或睡眠、任务没登记，或者一直在出错"))
    else:
        run += "。还没成功过"
    s = summary_text(hb.get("摘要"))
    if s:
        run += f"。摘要：{s}"
    return JobStatus(job, name, state, run, faults)
