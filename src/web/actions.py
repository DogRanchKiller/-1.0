"""网站上的操作按钮：登录后在数据状态页上点一下，让这台电脑跑一个现成的程序（需求文档第 8 节“网站上的操作按钮”、第 16 节“安全”）。

为什么有：以前要在电脑上双击 .bat；现在手机上登录网站也能点。程序照样在这台电脑上跑，网站服务器只是替你启动它。

四个按钮（白名单 ACTIONS，写死在下面；网页只能传“按钮名”，传不进任何命令、参数、路径）：
    refresh_data  立即刷新数据      python -m src.jobs.refresh_data       （和“足球预测_懂球帝赔率”定时任务同一个程序）
    predict       立即重新出预测    python -m src.jobs.scheduled_predict  （和“足球预测_定时预测”同一个程序）
    build_site    重新生成网站      python -m src.jobs.build_site         （和“足球预测_生成网站”同一个程序）
    test_push     发一条测试推送    python -m src.notify.send test        （和“发一条测试推送.bat”一样，只发测试消息，不打开真推送）

撞车怎么办（不排队、不硬跑，回一句话就完）：
- 前两个写数据库：点的时候不等待地看一眼数据库旁边的几把锁（src/jobs/data_lock.py）和定时预测的锁文件，
  有人在用就回“正在更新，稍后再试”（busy）。这两个按钮之间也互相挡。真正的锁还是子程序自己拿，和定时任务一样。
- 后两个不写数据库，但和定时任务写同一些文件（data/site/、data/notify/state.json）：看那个定时任务的心跳文件，
  “结束”是空的、开始不久，就算它正在跑，同样回 busy。
- 同一个按钮已经在跑：回“正在跑”（running），不再启动第二个。

怎么跑：后台开子程序 [和服务器同一个 python, "-m", 模块, 写死的参数]，不经过命令行解释器（shell），不弹黑窗口；
网页请求马上回“已开始”，另开一个线程等它跑完。每个按钮有最长时间（settings.yaml 的 actions.timeout_minutes），到点结束子程序。

网页上能看到什么：只有几句写死的中文和退出码（ActionRunner.status）。程序的原始输出（可能带路径、口令）只写进
logs/site_actions.log，写之前先脱敏（scrub）：路径换成 <项目> 和 ~，config/secrets.yaml 里的值、token= 之类后面的值换成 ***。

只用 Python 自带的库。
"""
from __future__ import annotations

import html
import json
import logging
import logging.handlers
import math
import os
import re
import subprocess
import sys
import threading
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable

from src.common.config import PROJECT_ROOT, load_secrets, resolve_path
from src.jobs.data_lock import lock_busy
from src.jobs.heartbeat import TIME_FMT, read_state

LOG = logging.getLogger("site_server")       # 和网站服务器同一个访问日志：只记“谁点了哪个按钮、结果代号”
NAME = re.compile(r"[a-z_]{1,32}")            # 按钮名只能是小写字母和下划线（网址 /api/action/<名字> 里的那一段）
SOURCES = ("dongqiudi", "cups", "football_data")   # 三个来源各自的锁（refresh_data、refresh_cups、refresh_football_data 用）

# 结果代号 → 给人看的一句话（网页上只出现这几句；没有脚本时跳回 /status?done=代号，服务器按代号选话，网址里别的字不显示）
RESULT_TEXT = {
    "started": "已开始，跑完后这里会写结果。",
    "running": "正在跑，这次没有再启动一个。",
    "busy": "正在更新，稍后再试。",
    "failed": "没能启动，详细原因写在 logs/site_actions.log。",
}


# ---------- 怎么判断“跑完了”还是“出错了” ----------
# 输入：退出码、程序的输出（只在服务器里看，不发到网页）；输出：(状态, 一句写死的中文)。
# 只看输出里固定的开头（各程序自己写死的那几句），不把输出原样搬上网页。

def judge_plain(code: int, out: str) -> tuple[str, str]:
    """退出码 0 = 跑完，别的 = 出错（refresh_data、build_site 出错时会抛错，退出码是 1）。"""
    if code == 0:
        return "done", "程序正常结束。"
    return "error", "程序出错了，详细原因写在 logs/site_actions.log。"


def judge_predict(code: int, out: str) -> tuple[str, str]:
    """定时预测拿不到锁（定时任务正在跑）时不算出错：它会打印“定时预测：跳过（…）”、退出码 0。"""
    if code == 0 and "定时预测：跳过" in out:
        return "done", "这次跳过了：定时预测正在跑，等它跑完就有新预测。"
    return judge_plain(code, out)


def judge_push(code: int, out: str) -> tuple[str, str]:
    """测试推送不管成没成功退出码都是 0，要看它打印的【成功】/【没成功】和后面固定的那几句。"""
    if code != 0:
        return judge_plain(code, out)
    if "【成功】" in out:
        return "done", "发送成功：看看微信里有没有收到“足球预测系统：测试消息”。"
    if "还没设置口令" in out:
        return "error", "还没设置口令：在电脑上双击“设置微信推送.bat”填好以后再试。"
    if "这次没发" in out:
        return "error", "这次没发：到了每分钟或每天的推送条数上限，过一会儿再试。"
    if "网络不通" in out:
        return "error", "没发出去：网络不通，看看电脑能不能上网。"
    return "error", "没发出去，详细原因写在 logs/site_actions.log。"


@dataclass(frozen=True)
class Action:
    """白名单里的一个按钮。module、args 写死在这个文件里，网页改不了。"""
    name: str                        # 按钮名（网址里的那一段）
    label: str                       # 按钮上写的字
    note: str                        # 按钮下面的一句说明
    module: str                      # python -m 后面的模块
    args: tuple[str, ...] = ()       # 写死的参数
    writes_db: bool = False          # 写数据库：要先看数据更新锁和定时预测锁，两个写库的按钮互相挡
    heartbeat: str | None = None     # 不写数据库、但和这个定时任务写同一些文件：看它的心跳（data/jobs/<名字>.json）
    judge: Callable[[int, str], tuple[str, str]] = judge_plain


ACTIONS: dict[str, Action] = {a.name: a for a in (
    Action("refresh_data", "立即刷新数据", "抓懂球帝的赛程、比分和赔率并导入数据库，和定时任务“足球预测_懂球帝赔率”是同一个程序，要几分钟。",
           "src.jobs.refresh_data", writes_db=True),
    Action("predict", "立即重新出预测", "给到点的比赛出 T-24 / T-6 / T-1 预测，和定时任务“足球预测_定时预测”是同一个程序；还没到点的比赛不会出。",
           "src.jobs.scheduled_predict", writes_db=True, judge=judge_predict),
    Action("build_site", "重新生成网站", "重新生成 data/site/ 里的静态网页（双击“查看网站.bat”看的那份），和定时任务“足球预测_生成网站”是同一个程序。",
           "src.jobs.build_site", heartbeat="site"),
    Action("test_push", "发一条测试推送", "和双击“发一条测试推送.bat”一样，只发一条测试消息；不会打开每天的推送，口令照旧用 .bat 填。",
           "src.notify.send", args=("test",), heartbeat="notify", judge=judge_push),
)}


@dataclass
class RunState:
    """一个按钮最近一次的情况（只在内存里；网站服务器重启后从“没跑过”重新算）。"""
    state: str = "idle"              # idle 没跑过 / running 已开始 / done 跑完 / error 出错
    started: datetime | None = None  # UTC
    finished: datetime | None = None
    exit_code: int | None = None
    message: str = ""
    proc: object | None = field(default=None, repr=False)


# ---------- 日志脱敏 ----------
# 写进 logs/site_actions.log 之前过一遍：
#   1. config/secrets.yaml 里的每个值（例如 pushplus 口令）换成 ***（和 src/notify/send.mask 同一个做法：原样找、原样换）；
#   2. token / password / secret / key 这类词后面跟着的值换成 ***（防止别的口令从输出里漏出来）；
#   3. 网址 ? 后面的部分去掉（里面可能带口令）；
#   4. 项目文件夹的路径换成 <项目>，用户文件夹（C:\Users\你的名字）换成 ~，别的 Users\名字 也换掉。
_KEY_VALUE = re.compile(r"(?i)(?<![a-z0-9])([a-z_]*(?:token|password|passwd|secret|api_?key|access_?key)[a-z_]*)"
                        r"(\s*[:=]\s*|\s+)([\"']?)[^\s\"',;&]{4,}")
_URL_QUERY = re.compile(r"(https?://[^\s?\"'<>]+)\?[^\s\"'<>]*")
_USER_DIR = re.compile(r"(?i)\b([a-z]:[\\/]+(?:users|documents and settings)[\\/]+)[^\\/\s\"']+")


def _secret_values(secrets: object) -> set[str]:
    """secrets.yaml 里所有的值（嵌套的也算）。太短的（少于 4 个字）不算，免得把普通的字也换掉。"""
    out: set[str] = set()
    if isinstance(secrets, dict):
        for v in secrets.values():
            out |= _secret_values(v)
    elif isinstance(secrets, (list, tuple)):
        for v in secrets:
            out |= _secret_values(v)
    elif isinstance(secrets, (str, int)) and not isinstance(secrets, bool):
        text = str(secrets).strip()
        if len(text) >= 4:
            out.add(text)
    return out


def _path_forms(p: Path) -> list[str]:
    """一个文件夹在输出里可能出现的几种写法：反斜杠、正斜杠。"""
    s = str(p)
    return sorted({s, s.replace("\\", "/"), s.replace("/", "\\")}, key=len, reverse=True)


def scrub(text: str, secret_values: set[str] = frozenset(), root: Path = PROJECT_ROOT, home: Path | None = None) -> str:
    """把要写进日志的文字脱敏（见上面的 1–4）。"""
    text = "" if text is None else str(text)
    for v in sorted(secret_values, key=len, reverse=True):
        text = text.replace(v, "***")
    text = _KEY_VALUE.sub(lambda m: m.group(1) + m.group(2) + m.group(3) + "***", text)
    text = _URL_QUERY.sub(r"\1?***", text)
    home = Path.home() if home is None else home
    for folder, mark in ((root, "<项目>"), (home, "~")):
        for form in _path_forms(folder):
            if len(form) > 3:                                   # 不替换 C:\ 这种太短的
                text = re.sub(re.escape(form), lambda m, mark=mark: mark, text, flags=re.I)
    return _USER_DIR.sub(r"\1<用户>", text)


# ---------- 跑按钮 ----------

def _bj(t: datetime | None) -> str:
    """UTC 时间 → 北京时间的 “11-04 21:05:30”（网页上显示用）。"""
    from src.web.serve_auth import BEIJING
    return t.astimezone(BEIJING).strftime("%m-%d %H:%M:%S") if t else ""


def _iso(t: datetime | None) -> str | None:
    return t.isoformat(timespec="seconds") if t else None


class ActionRunner:
    """网站服务器里管按钮的。一个服务器一个，几个请求线程一起用（内部有锁）。

    settings：完整的 settings.yaml。下面几个参数测试时换成假的：
      actions：白名单（默认 ACTIONS）；popen：开子程序的函数（默认 subprocess.Popen）；
      cwd：子程序在哪个文件夹里跑（默认项目文件夹）；python：用哪个 python（默认和服务器同一个 sys.executable）；
      now：现在几点（UTC）；secrets：读口令的函数（脱敏用，默认 load_secrets）。
    """

    def __init__(self, settings: dict, actions: dict[str, Action] | None = None, popen: Callable = subprocess.Popen,
                 cwd: Path | None = None, python: str | None = None, now: Callable[[], datetime] | None = None,
                 secrets: Callable[[], dict] = load_secrets):
        cfg = settings["actions"]
        self.settings = settings
        self.actions = dict(ACTIONS if actions is None else actions)
        self.popen = popen
        self.cwd = Path(cwd) if cwd is not None else PROJECT_ROOT
        self.python = python or sys.executable
        self.now = now or (lambda: datetime.now(timezone.utc))
        self.secrets = secrets
        self.log_path = resolve_path(cfg["log_file"])
        self.log_max_bytes = int(cfg["log_max_bytes"])
        self.log_backups = int(cfg["log_backups"])
        self.tail_chars = int(cfg["log_tail_chars"])
        self.poll_seconds = max(1, int(cfg["poll_seconds"]))
        self.busy_seconds = float(cfg["busy_minutes"]) * 60
        self.timeouts: dict[str, float] = {}
        for name in self.actions:
            minutes = float((cfg.get("timeout_minutes") or {}).get(name, "nan"))
            if not math.isfinite(minutes) or minutes <= 0:
                raise ValueError(f"settings.yaml 的 actions.timeout_minutes 里没写 {name}，或者不是正数")
            self.timeouts[name] = minutes * 60
        self._lock = threading.Lock()
        self._runs: dict[str, RunState] = {name: RunState() for name in self.actions}
        self._log_lock = threading.Lock()

    # ----- 日志 -----
    def _log(self, text: str) -> None:
        """写一行到 logs/site_actions.log（满了换新文件）。写之前脱敏；写不进去只在访问日志里记一句，不影响按钮。"""
        try:
            secret_values = _secret_values(self.secrets())
        except Exception:   # noqa: BLE001 —— secrets.yaml 坏了也要能写日志（这时只按规则 2–4 脱敏）
            secret_values = set()
        line = f"{datetime.now(timezone.utc).strftime(TIME_FMT)} UTC {scrub(text, secret_values, self.cwd)}"
        with self._log_lock:
            try:
                self.log_path.parent.mkdir(parents=True, exist_ok=True)
                handler = logging.handlers.RotatingFileHandler(self.log_path, maxBytes=self.log_max_bytes,
                                                               backupCount=self.log_backups, encoding="utf-8")
                try:
                    handler.setFormatter(logging.Formatter("%(message)s"))
                    handler.emit(logging.makeLogRecord({"msg": line, "levelno": logging.INFO, "levelname": "INFO"}))
                finally:
                    handler.close()
            except OSError as err:
                LOG.warning("操作按钮的日志写不进去（%s）", type(err).__name__)

    # ----- 有没有别的程序正在用 -----
    def _heartbeat_running(self, job: str) -> bool:
        """定时任务 job 的心跳文件：“结束”是空的、开始不到 actions.busy_minutes 分钟 = 正在跑。
        太久以前开始、一直没结束的，多半是中途被杀掉了（心跳来不及写“结束”），不算。"""
        from src.jobs.heartbeat import state_dir
        st = read_state(job, state_dir(self.settings))
        if not st or st.get("结束") is not None or not st.get("开始"):
            return False
        try:
            began = datetime.strptime(st["开始"], TIME_FMT).replace(tzinfo=timezone.utc)
        except (TypeError, ValueError):
            return False
        age = (datetime.now(timezone.utc) - began).total_seconds()
        return -60 <= age < self.busy_seconds

    def _predict_lock_held(self) -> bool:
        """定时预测的锁文件（schedule.lock_file）在、而且没过期（schedule.lock_stale_minutes）= 定时预测正在跑。"""
        cfg = self.settings["schedule"]
        try:
            age = time.time() - resolve_path(cfg["lock_file"]).stat().st_mtime
        except FileNotFoundError:
            return False
        return age <= float(cfg["lock_stale_minutes"]) * 60

    def external_busy(self, act: Action) -> bool:
        """服务器外面（定时任务、双击的 .bat）是不是正在跑会和这个按钮撞车的程序。只看一眼，不等待、不拿锁。"""
        if act.writes_db:
            if lock_busy(self.settings) or any(lock_busy(self.settings, src) for src in SOURCES):
                return True
            if self._predict_lock_held():
                return True
        if act.heartbeat and self._heartbeat_running(act.heartbeat):
            return True
        return False

    # ----- 点按钮 -----
    def start(self, name: str) -> str:
        """点了按钮 name。回结果代号：started 已开始 / running 这个按钮正在跑 / busy 别的程序正在更新 / failed 没能启动。
        name 不在白名单里抛 KeyError（服务器先用 NAME 和白名单挡过，这里再挡一次）。"""
        act = self.actions[name]
        with self._lock:          # 整个判断和启动都在锁里：两个请求同时来，只有一个能启动
            run = self._runs[name]
            if run.state == "running":
                return "running"
            if act.writes_db and any(self._runs[n].state == "running" and a.writes_db
                                     for n, a in self.actions.items() if n != name):
                self._log(f"按钮 {name}：没启动（另一个写数据库的按钮正在跑）")
                return "busy"
            try:
                busy = self.external_busy(act)
            except (OSError, ValueError, KeyError) as err:
                self._log(f"按钮 {name}：没启动，看锁时出错（{type(err).__name__}：{err}）")
                return "failed"
            if busy:
                self._log(f"按钮 {name}：没启动（正在更新）")
                return "busy"
            argv = [self.python, "-m", act.module, *act.args]
            env = {**os.environ, "PYTHONUTF8": "1", "PYTHONIOENCODING": "utf-8"}
            flags = getattr(subprocess, "CREATE_NO_WINDOW", 0) if os.name == "nt" else 0
            try:
                # 参数是列表、shell=False：不经过命令行解释器，参数里有什么字都不会被当成命令
                proc = self.popen(argv, cwd=str(self.cwd), env=env, stdin=subprocess.DEVNULL,
                                  stdout=subprocess.PIPE, stderr=subprocess.STDOUT, shell=False,
                                  creationflags=flags)
            except OSError as err:
                self._runs[name] = RunState("error", self.now(), self.now(), None, "没能启动程序，详细原因写在 logs/site_actions.log。")
                self._log(f"按钮 {name}：没能启动（{type(err).__name__}：{err}）")
                return "failed"
            self._runs[name] = RunState("running", self.now(), proc=proc)
            self._log(f"按钮 {name}：开始，运行 python -m {act.module} {' '.join(act.args)}".rstrip())
        threading.Thread(target=self._watch, args=(name, proc), name=f"action-{name}", daemon=True).start()
        return "started"

    def _watch(self, name: str, proc) -> None:
        """另一个线程：等子程序跑完（最多 actions.timeout_minutes），超时就结束它；然后记下结果。"""
        act = self.actions[name]
        limit = self.timeouts[name]
        began = time.monotonic()
        timed_out = False
        try:
            out, _ = proc.communicate(timeout=limit)
        except subprocess.TimeoutExpired:
            timed_out = True
            proc.kill()
            try:
                out, _ = proc.communicate(timeout=10)
            except subprocess.TimeoutExpired:      # 子程序又开了别的程序、还占着输出：不再等
                out = b""
        except Exception as err:   # noqa: BLE001 —— 故意的：等的过程出意外也要把状态改掉，不能一直“正在跑”
            out = f"等程序结束时出错：{type(err).__name__}".encode("utf-8")
            try:
                proc.kill()
            except OSError:
                pass
            proc.wait()
        code = proc.returncode
        text = (out or b"").decode("utf-8", errors="replace")
        if timed_out:
            state, message = "error", f"超过 {limit / 60:g} 分钟还没跑完，已经结束了它。"
        else:
            try:
                state, message = act.judge(code, text)
            except Exception:   # noqa: BLE001 —— 判断出错按“出错”记
                state, message = "error", "程序出错了，详细原因写在 logs/site_actions.log。"
        with self._lock:
            run = self._runs[name]
            self._runs[name] = RunState(state, run.started, self.now(), code, message)
        tail = text[-self.tail_chars:] if self.tail_chars > 0 else ""
        self._log(f"按钮 {name}：结束，{state}，退出码 {code}，用了 {time.monotonic() - began:.0f} 秒"
                  f"{'（超时，已结束）' if timed_out else ''}\n--- 输出的最后一段 ---\n{tail.rstrip()}\n---")

    # ----- 给网页看 -----
    def is_running(self, name: str) -> bool:
        with self._lock:
            return self._runs[name].state == "running"

    def display(self, name: str) -> str:
        """一个按钮现在的情况，写成一句话（页面上、/api/actions 都用这一句，两边一样）。"""
        with self._lock:
            run = self._runs[name]
        if run.state == "running":
            return f"已开始（北京时间 {_bj(run.started)}），正在跑……"
        if run.state == "done":
            return f"跑完了（北京时间 {_bj(run.finished)}）：{run.message}"
        if run.state == "error":
            code = "" if run.exit_code is None else f"（退出码 {run.exit_code}）"
            return f"出错（北京时间 {_bj(run.finished)}）：{run.message}{code}"
        return "这次服务器启动以后还没在这里点过。"

    def status(self) -> dict:
        """GET /api/actions 的内容：每个按钮的 state、开始和结束时间（UTC）、退出码、一句话。不带程序输出。"""
        out = []
        for name, act in self.actions.items():
            with self._lock:
                run = self._runs[name]
            out.append({"name": name, "label": act.label, "state": run.state, "started_at": _iso(run.started),
                        "finished_at": _iso(run.finished), "exit_code": run.exit_code, "text": self.display(name)})
        return {"actions": out, "poll_seconds": self.poll_seconds}


# ---------- 数据状态页上的“操作”一块 ----------
# 服务器发 /status 时插进去（status.html 文件本身不改）。没有脚本时每个按钮是一个普通表单（POST），
# 提交后服务器跳回 /status?done=<结果代号>&action=<按钮名>；有脚本时 site.js 拦下表单、用 fetch 发同样的东西。
# 页面里没有内联脚本、没有 onclick=（CSP 只许 /static/site.js）；样式写在 <style> 里（CSP 允许页面自带的样式）。
PANEL_CSS = """<style>
.act-panel{background:var(--card,#101c16);border:1px solid var(--line,#1d3027);border-radius:14px;padding:14px 16px;margin:14px 0 18px}
.act-panel h2{margin:0 0 6px}
.act-row{display:flex;flex-wrap:wrap;gap:6px 14px;align-items:center;padding:10px 0;border-top:1px solid var(--line,#1d3027)}
.act-row:first-of-type{border-top:0}
.act-row form{margin:0}
.act-row button{font:inherit;font-size:15px;font-weight:600;padding:9px 16px;border-radius:10px;cursor:pointer;
 border:1px solid var(--green,#34d399);color:var(--green,#34d399);background:transparent;min-width:9em}
.act-row button:disabled{opacity:.5;cursor:default}
.act-note{flex:1 1 260px;font-size:13px;color:var(--sub,#94a99d)}
.act-state{flex:1 1 100%;font-size:13.5px}
.act-state[data-state="running"]{color:var(--amber,#fbbf24)}
.act-state[data-state="done"]{color:var(--green,#34d399)}
.act-state[data-state="error"]{color:var(--red,#f87171)}
.act-notice{margin:0 0 8px;padding:8px 12px;border-radius:10px;border:1px solid var(--line,#1d3027);font-size:14px}
</style>"""
STATUS_SCRIPT = '<script src="/static/site.js" defer></script>'


def notice_text(done: str | None, name: str | None, runner: ActionRunner) -> str | None:
    """跳回 /status?done=…&action=… 时上面写的那句话。代号、按钮名都不在固定的几个里面就不写（网址里的字不会照搬到页面上）。"""
    if done not in RESULT_TEXT or name not in runner.actions:
        return None
    return f"“{runner.actions[name].label}”：{RESULT_TEXT[done]}"


def panel_html(runner: ActionRunner, csrf: str, notice: str | None = None) -> str:
    """“操作”一块的网页代码。所有字都先转义；csrf 是这次登录的防伪口令（serve_auth.csrf_token）。"""
    e = html.escape
    rows = ""
    for name, act in runner.actions.items():
        with runner._lock:
            state = runner._runs[name].state
        disabled = " disabled" if state == "running" else ""
        rows += (f'<div class="act-row" data-action="{e(name)}">'
                 f'<form method="post" action="/api/action/{e(name)}" class="act-form">'
                 f'<input type="hidden" name="csrf" value="{e(csrf)}">'
                 f'<button type="submit"{disabled}>{e(act.label)}</button></form>'
                 f'<span class="act-note">{e(act.note)}</span>'
                 f'<span class="act-state" data-state="{e(state)}" aria-live="polite">{e(runner.display(name))}</span></div>')
    top = f'<p class="act-notice" role="status">{e(notice)}</p>' if notice else ""
    return (f'{PANEL_CSS}<section class="act-panel" id="actions" data-poll-seconds="{runner.poll_seconds}">'
            f'<h2>操作</h2>{top}'
            f'<p class="small">点一下让这台电脑马上跑一次，和定时任务、双击 .bat 是同一个程序；不用等它跑完，结果会写在按钮下面。'
            f'别的程序正在更新数据时不会启动，会写“正在更新，稍后再试”。</p>{rows}</section>')


_HEAD_END = re.compile(rb"</head\s*>", re.I)
_FIRST_H2 = re.compile(rb"<h2\b", re.I)
_WRAP = re.compile(rb"<div\b[^>]*\bclass=[\"']wrap[\"'][^>]*>", re.I)
_BODY = re.compile(rb"<body\b[^>]*>", re.I)


def inject_panel(page: bytes, panel: str) -> bytes:
    """把“操作”一块插进数据状态页：第一个小标题（各项检查）前面；没有小标题就插在正文开头。
    <head> 里加上本站的脚本 /static/site.js（拦表单、问进度用；没有它按钮照样能用）。"""
    block = panel.encode("utf-8")
    m = _FIRST_H2.search(page) or _WRAP.search(page) or _BODY.search(page)
    if m is None:
        page = page + block
    elif m.re is _FIRST_H2:
        page = page[:m.start()] + block + page[m.start():]
    else:
        page = page[:m.end()] + block + page[m.end():]
    h = _HEAD_END.search(page)
    if h is not None:
        page = page[:h.start()] + STATUS_SCRIPT.encode("utf-8") + page[h.start():]
    return page


def missing_status_page() -> bytes:
    """数据状态页还没生成时，也要能看到按钮：给一个最简单的页面（正文在 .wrap 里，site.js 认它）。"""
    from src.web.theme import VIEWPORT
    return (f'<!doctype html><html lang="zh-CN"><head><meta charset="utf-8">{VIEWPORT}<title>足球预测 数据状态</title>'
            f'<style>:root{{color-scheme:dark}}body{{margin:0;padding:24px 16px;background:#0a1310;color:#e2ebe5;'
            f'font-family:"Segoe UI","PingFang SC","Microsoft YaHei",sans-serif}}.wrap{{max-width:900px;margin:0 auto}}'
            f'.small{{color:#94a99d;font-size:13px}}</style></head><body><div class="wrap"><h1>数据状态</h1>'
            f'<p class="small">数据状态页还没生成：每小时自动生成一次，过一会儿再来看。</p></div></body></html>').encode("utf-8")


def runner_from_settings(settings: dict) -> ActionRunner | None:
    """按 settings.yaml 造按钮管理器。没有 actions: 这一节，或者 actions.enabled 是 false：回 None（没有按钮，接口 404）。
    只有一节 actions: 但没写 enabled 时当作 true（需求文档：总开关默认 true）。"""
    cfg = settings.get("actions")
    if not isinstance(cfg, dict) or not cfg.get("enabled", True):
        return None
    return ActionRunner(settings)
