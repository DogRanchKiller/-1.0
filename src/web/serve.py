"""局域网网站服务器：带登录，把生成好的网页发给同一个 Wi-Fi 里的手机、电脑（需求文档第 8 节“局域网网站”、第 16 节“安全”）。

运行：.venv\\Scripts\\python -m src.web.serve
平时不用手动运行：定时任务“足球预测_网站服务”在你登录 Windows 时自动启动（src/jobs/register_tasks.ps1），不弹黑窗口。

它做什么：
- 现读现出（settings.yaml 的 live.dynamic: true，src/web/live_site.py）：登录后打开网页时，现场用只读方式读数据库，
  用和“足球预测_生成网站”同一套函数生成网页（读到的数据在内存里留 live.cache_seconds 秒）。读不出来时继续用上一次的数据，
  一次都没读到过就发 data/site/ 里生成好的静态文件。网页本身不跑模型、不写数据库、不联网。
- 即时比分（settings.yaml 的 live.live_poller，默认 false）：打开时服务器在后台开一个线程跑 WD3 的即时比分程序
  （src.collect.dongqiudi_live），它会联网请求懂球帝，把比分写进 scores.json，并往 football.db 的 fetch_log 表里写抓取记录。
  关着时服务器不联网、不写数据库。
- 数据接口（/api/...，返回 JSON，也要登录）：/api/version、/api/live、/api/matches、/api/match/<编号>，字段见需求文档第 8 节。
- /static/ 下发本站自己的脚本和样式（src/web/static/ 里的文件），网页只引用它们，不引用外部的东西。
- live.dynamic: false 时和以前完全一样：只把 data/site/ 下已经生成好的网页文件发出去。
- /status 发 data/reports/status.html（数据状态页）。现读现出时上面另插一块“操作”：四个按钮（src/web/actions.py，
  需求文档第 8 节“网站上的操作按钮”）。按钮用 POST /api/action/<名字>，进度用 GET /api/actions；都要登录，
  POST 还要过同源检查和防伪口令（CSRF，serve_auth.csrf_token）。settings.yaml 的 actions.enabled: false 时没有按钮、两个地址都 404。
- 没登录的一律先跳到登录页；登录成功给浏览器一个随机“会话 cookie”（相当于一张临时通行证），30 天内不用再登。
- 每个网页顶上插一条小栏“首页 · 数据状态 · 退出登录”（只在发出去时插，网页文件本身不改）。

几个词：
- 端口：一台电脑上可以同时跑很多网络程序，端口号用来区分它们，网址里冒号后面的数字（8790）就是端口。
- 0.0.0.0：监听这台电脑所有网卡上的地址，同一个 Wi-Fi 的手机才连得进来；127.0.0.1 只有本机自己能连。
- cookie：网站存在浏览器里的一小段文字，浏览器每次访问都带上，服务器靠它认出“这是登录过的人”。
- 路径穿越：坏人在网址里写 ../../ 之类，想让服务器把网站文件夹外面的文件（例如密码文件）发出来。这里层层挡住。

只用 Python 自带的库，不装新库。
"""
from __future__ import annotations

import argparse
import errno
import hashlib
import html
import importlib
import ipaddress
import json
import logging
import logging.handlers
import os
import re
import secrets
import socket
import sys
import threading
import time
from dataclasses import dataclass, field
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Callable
from urllib.parse import parse_qs, quote, unquote, urlsplit

from src.common.config import load_settings, resolve_path
from src.web import actions as site_actions
from src.web import serve_auth
from src.web.theme import VIEWPORT

LOG = logging.getLogger("site_server")
COOKIE = "fp_session"                  # 会话 cookie 的名字
ALLOW = "GET, HEAD, POST"              # 只接受这三种请求方法（POST 只用于登录 /login 和操作按钮 /api/action/<名字>）
REPORTS_PREFIX = "/review/reports/"    # 周报、月报的网址前缀（战绩页的相对链接 ../review/reports/ 在服务器下就是这个）
STATIC_PREFIX = "/static/"             # 本站自己的脚本、样式（src/web/static/ 里的文件）
STATIC_DIR = Path(__file__).resolve().parent / "static"
API_PREFIX = "/api/"                   # 数据接口（JSON）
MATCH_API = re.compile(r"/api/match/(\d{1,12})")
ACTION_PREFIX = "/api/action/"         # 操作按钮：POST /api/action/<名字>（名字要在 src/web/actions.py 的白名单里）
ACTIONS_API = "/api/actions"           # 操作按钮的进度（GET，JSON）
JSON_TYPE = "application/json; charset=utf-8"
# Windows 的设备名：文件名叫这些时会打开设备而不是文件，一律不认
WIN_DEVICES = {"con", "prn", "aux", "nul", *(f"com{i}" for i in range(10)), *(f"lpt{i}" for i in range(10))}
# 文件类型。不靠系统注册表猜（Windows 上有时会把 .js、.css 认错）
CONTENT_TYPES = {
    ".html": "text/html; charset=utf-8", ".htm": "text/html; charset=utf-8",
    ".css": "text/css; charset=utf-8", ".js": "text/javascript; charset=utf-8",
    ".json": "application/json; charset=utf-8", ".txt": "text/plain; charset=utf-8",
    ".svg": "image/svg+xml", ".png": "image/png", ".jpg": "image/jpeg", ".jpeg": "image/jpeg",
    ".gif": "image/gif", ".webp": "image/webp", ".ico": "image/x-icon",
    ".woff": "font/woff", ".woff2": "font/woff2",
}
# 每个响应都带的安全响应头
SECURITY_HEADERS = {
    "X-Content-Type-Options": "nosniff",               # 浏览器不许自己乱猜文件类型
    "X-Frame-Options": "DENY",                         # 别的网站不能把这里的页面嵌进去（防“点击劫持”）
    # 内容安全策略（CSP）：网页只能用本站自己的脚本文件（/static/site.js），不能用写在网页里的内联脚本、外部脚本；
    # 只能连本站（connect-src 'self'）；图片只能是本站的或者网页里内嵌的；别的网站不能把这里嵌进去（frame-ancestors）；
    # 网页里不能用 <base> 改掉链接的起点（base-uri 'none'）；表单只能提交给本站（form-action 'self'，登录表单就是）。
    # 'unsafe-inline' 只给样式：网页里的 <style> 和概率条宽度这种 style="..." 要用它，样式不能执行程序。
    # 所有响应都带（live.dynamic: false 只发静态文件时也一样）
    "Content-Security-Policy": ("default-src 'self'; script-src 'self'; style-src 'self' 'unsafe-inline'; "
                                "img-src 'self' data:; connect-src 'self'; frame-ancestors 'none'; "
                                "base-uri 'none'; form-action 'self'"),
    "Referrer-Policy": "same-origin",                  # 点链接去别的网站时不告诉对方从哪来（网站里本来也没有外链）
    "Cache-Control": "no-store",                       # 不缓存：退出登录后按“返回”看不到旧页面
    "X-Site-Server": serve_auth.SERVER_MARK,           # 自己人的记号：用来认出端口上跑的是不是这个服务器
}


class PasswordRequired(RuntimeError):
    """没有设置密码，又想在本机以外的地址上监听。"""


@dataclass
class ServerConfig:
    """服务器要用的设置。from_settings 从 settings.yaml 读；测试时直接造一个，指到临时文件夹。"""
    site_dir: Path
    status_file: Path
    state_dir: Path
    review_reports_dir: Path | None = None   # 周报、月报（data/review/reports/）；战绩页用 ../review/reports/xx.md 链接它们
    session_days: float = 30
    max_sessions: int = 50
    max_failures: int = 5
    lockout_minutes: float = 15
    max_login_body_bytes: int = 4096
    request_timeout_seconds: float = 20
    max_connections: int = 32
    # 现读现出用的完整 settings（live.dynamic 是 true 时 from_settings 放进来）；None = 只发静态文件（和以前一样，测试里直接造的也是这样）
    live_settings: dict | None = None

    @classmethod
    def from_settings(cls, settings: dict) -> "ServerConfig":
        ss = settings["site_server"]
        return cls(
            site_dir=resolve_path(settings["site"]["output_dir"]),
            status_file=resolve_path(settings["paths"]["reports"]) / ss["status_page"],
            state_dir=resolve_path(ss["state_dir"]),
            review_reports_dir=resolve_path(settings["site"]["review_dir"]) / "reports",
            session_days=ss["session_days"], max_sessions=ss["max_sessions"],
            max_failures=ss["max_failures"], lockout_minutes=ss["lockout_minutes"],
            max_login_body_bytes=ss["max_login_body_bytes"],
            request_timeout_seconds=ss["request_timeout_seconds"],
            max_connections=ss["max_connections"],
            live_settings=settings if (settings.get("live") or {}).get("dynamic", False) else None,
        )


def _token_hash(token: str) -> str:
    """会话文件里只存 cookie 的哈希：文件被人看到，也拼不出能用的 cookie。"""
    return hashlib.sha256(token.encode("ascii")).hexdigest()


@dataclass
class SiteApp:
    """服务器的“大脑”：密码、会话、输错锁定。几个请求同时来时用一把锁排队改这些东西。

    clock：现在几点（秒）。测试时换成假的时钟，不用真等 30 天也能测“会话过期”。
    """
    config: ServerConfig
    clock: Callable[[], float] = time.time
    _lock: threading.Lock = field(default_factory=threading.Lock, init=False)
    _login_lock: threading.Lock = field(default_factory=threading.Lock, init=False)
    _cred: dict | None = field(default=None, init=False)
    _cred_stamp: tuple | None = field(default=None, init=False)
    _sessions: dict = field(default_factory=dict, init=False)
    _failures: dict = field(default_factory=dict, init=False)   # IP → [连续输错几次, 锁到几点]
    live: object | None = field(default=None, init=False)        # 现读现出（src/web/live_site.LiveSite）；None = 只发静态文件
    # 操作按钮（src/web/actions.ActionRunner）；None = 没有按钮（只发静态文件，或 actions.enabled: false），按钮的地址都 404
    actions: object | None = field(default=None, init=False)
    # 防伪口令的密钥：服务器启动时随机生成，只在内存里（serve_auth.csrf_token）
    csrf_key: bytes = field(default_factory=serve_auth.new_csrf_key, init=False, repr=False)

    def __post_init__(self):
        self._load_sessions()
        if self.config.live_settings is not None:
            try:
                # 放在这里才导入：只发静态文件时不用加载 pandas 这些大库，启动快
                from src.web.live_site import LiveSite
                self.live = LiveSite(self.config.live_settings, self.config.site_dir, clock=self.clock)
            except Exception as err:   # noqa: BLE001 —— 故意的：现读现出起不来（设置写错、库有问题）也要能发静态文件
                self.live = None
                LOG.warning("现读现出起不来（%s），这次只发 data/site/ 的静态文件", type(err).__name__)
            try:
                # 按钮只在现读现出时有（需求文档第 8 节）；和现读现出分开起，那边起不来按钮照样能用
                self.actions = site_actions.runner_from_settings(self.config.live_settings)
            except Exception as err:   # noqa: BLE001 —— 故意的：按钮的设置写错了，网站照样能看，只是没有按钮
                self.actions = None
                LOG.warning("操作按钮起不来（%s），这次没有按钮", type(err).__name__)

    # ----- 密码 -----
    @property
    def cred_path(self) -> Path:
        return self.config.state_dir / serve_auth.CRED_FILE

    def credentials(self) -> dict | None:
        """当前的密码哈希。文件改了（双击“修改网站密码.bat”）会自动重新读，不用重启服务器。文件坏了按“没有密码”处理（谁都登不进）。"""
        try:
            st = self.cred_path.stat()
            stamp = (st.st_mtime_ns, st.st_size)
        except OSError:
            stamp = None
        with self._lock:
            if stamp != self._cred_stamp:
                try:
                    self._cred = serve_auth.load_credentials(self.cred_path) if stamp else None
                except serve_auth.CredentialsError as e:
                    LOG.error("密码文件有问题，暂时谁都登不进：%s", e)
                    self._cred = None
                self._cred_stamp = stamp
            return self._cred

    # ----- 登录和输错锁定 -----
    def locked_seconds(self, ip: str) -> int:
        """这个 IP 还要锁多少秒（0 = 没锁）。"""
        with self._lock:
            entry = self._failures.get(ip)
            if not entry or entry[1] <= self.clock():
                return 0
            return int(entry[1] - self.clock()) + 1

    def check_login(self, ip: str, username: str, password: str) -> tuple[str, int]:
        """核对账号密码。返回 (结果, 数字)：
        ("ok", 0)；("bad", 还能再试几次)；("locked", 还要锁几秒)；("no_password", 0) = 还没设置密码。

        同一时间只核对一个登录（_login_lock）：60 万次哈希要算零点几秒，一个个来，
        有人同时发很多次也绕不过“输错 5 次锁定”，电脑也不会被拖慢。
        """
        with self._login_lock:
            wait = self.locked_seconds(ip)
            if wait:
                return "locked", wait
            rec = self.credentials()
            if rec is None:
                return "no_password", 0
            # 账号错了也照样算一遍哈希：对的错的花的时间一样，别人分不出是账号错还是密码错
            pw_ok = serve_auth.verify_password(password[:256], rec)
            user_ok = secrets.compare_digest(username.strip().encode("utf-8"), rec["username"].encode("utf-8"))
            now = self.clock()
            with self._lock:
                if pw_ok and user_ok:
                    self._failures.pop(ip, None)
                    return "ok", 0
                entry = self._failures.get(ip)
                if entry is None or (entry[1] and entry[1] <= now):
                    entry = [0, 0.0]               # 第一次输错，或者上一次锁定已经过了：重新数
                entry[0] += 1
                if entry[0] >= self.config.max_failures:
                    entry[1] = now + self.config.lockout_minutes * 60
                self._failures[ip] = entry
                if len(self._failures) > 1000:     # 记得太多时清掉不再锁着的，防止越记越多
                    self._failures = {k: v for k, v in self._failures.items() if v[1] > now or k == ip}
                if entry[1]:
                    return "locked", int(entry[1] - now) + 1
                return "bad", self.config.max_failures - entry[0]

    # ----- 会话 -----
    @property
    def sessions_path(self) -> Path:
        return self.config.state_dir / serve_auth.SESSIONS_FILE

    def _load_sessions(self) -> None:
        """读会话文件（电脑重启后手机不用重新登录）。读不出来就当没有会话，大家重新登录一次。"""
        try:
            raw = json.loads(self.sessions_path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            return
        now = self.clock()
        if isinstance(raw, dict):
            self._sessions = {k: v for k, v in raw.items()
                              if isinstance(v, dict) and isinstance(v.get("expires"), (int, float))
                              and isinstance(v.get("version"), int) and v["expires"] > now}

    def _save_sessions(self) -> None:
        """在锁里面调用。写不进去（例如被杀毒软件占着）只记日志，不影响这次登录。"""
        try:
            serve_auth.write_atomic(self.sessions_path, json.dumps(self._sessions, indent=1))
        except OSError as e:
            LOG.warning("会话文件写不进去：%s", type(e).__name__)

    def new_session(self) -> str:
        """登录成功：生成一个随机会话，返回给浏览器的 cookie 值。只存它的哈希。"""
        rec = self.credentials()
        token = secrets.token_urlsafe(32)
        now = self.clock()
        with self._lock:
            self._sessions = {k: v for k, v in self._sessions.items() if v["expires"] > now}
            while len(self._sessions) >= self.config.max_sessions:   # 太多了先挤掉最早的
                oldest = min(self._sessions, key=lambda k: self._sessions[k].get("created", 0))
                del self._sessions[oldest]
            self._sessions[_token_hash(token)] = {
                "created": now, "expires": now + self.config.session_days * 86400,
                "version": rec["version"] if rec else 0,
            }
            self._save_sessions()
        return token

    def session_valid(self, token: str | None) -> bool:
        """这个 cookie 还能不能用：存在、没过期、登录后没改过密码。"""
        if not token:
            return False
        rec = self.credentials()
        key = _token_hash(token)
        with self._lock:
            s = self._sessions.get(key)
            if s is None:
                return False
            if s["expires"] <= self.clock() or rec is None or s["version"] != rec["version"]:
                del self._sessions[key]
                self._save_sessions()
                return False
            return True

    def end_session(self, token: str | None) -> None:
        """退出登录：删掉这个会话。"""
        if not token:
            return
        with self._lock:
            if self._sessions.pop(_token_hash(token), None) is not None:
                self._save_sessions()


# ---------- 网址检查 ----------

def _clean_for_log(path: str) -> str:
    """日志里记的路径：去掉 ? 后面的部分（以防万一有人把密码写在网址里），太长截断，怪字符换成 ?。"""
    path = path.split("?", 1)[0].split("#", 1)[0][:200]
    return "".join(c if c.isprintable() else "?" for c in path)


def safe_next(target: str | None) -> str:
    """登录后跳回哪一页。只接受本网站里的路径（以一个 / 开头），防止被人利用跳到别的网站。"""
    if (not target or not target.startswith("/") or target.startswith("//") or "\\" in target
            or any(ord(c) < 32 or ord(c) == 127 for c in target)):
        return "/"
    if target.split("?", 1)[0] in ("/login", "/logout"):
        return "/"
    return target


def _inside(path: str, root: str) -> bool:
    """真实路径 path 是不是在文件夹 root 里面（Windows 上不分大小写）。"""
    try:
        return os.path.commonpath([os.path.normcase(path), os.path.normcase(root)]) == os.path.normcase(root)
    except ValueError:            # 不在同一个盘
        return False


def resolve_static(root: Path, url_path: str) -> Path | None:
    """把网址里的路径换成网站文件夹里的文件。不安全或者没有这个文件都返回 None（服务器回 404）。

    挡住的写法：..、%2e%2e（解码以后就是 ..）、反斜杠 \\、冒号（C: 这种盘符、文件的“隐藏数据流”）、
    空字符、Windows 设备名（CON、NUL…）、以点或空格结尾的名字（Windows 会悄悄去掉，变成别的文件）。
    最后再查一次：把符号链接都展开以后的真实路径，必须还在网站文件夹里面。
    """
    try:
        decoded = unquote(url_path, encoding="utf-8", errors="strict")
    except UnicodeDecodeError:
        return None
    if "\x00" in decoded or "\\" in decoded or ":" in decoded or not decoded.startswith("/"):
        return None
    parts = decoded[1:].split("/")
    if parts[-1] == "":
        parts[-1] = "index.html"            # 以 / 结尾：找这个文件夹里的 index.html（不列目录）
    for seg in parts:
        if seg in ("", ".", "..") or seg != seg.rstrip(". ") or any(ord(c) < 32 for c in seg):
            return None
        if seg.split(".", 1)[0].lower() in WIN_DEVICES:
            return None
    real_root = os.path.realpath(root)
    real = os.path.realpath(os.path.join(real_root, *parts))
    if not _inside(real, real_root) or not os.path.isfile(real):
        return None
    return Path(real)


# ---------- 网页小块 ----------

BAR_CSS = """<style>
.lan-bar{display:flex;flex-wrap:wrap;gap:6px 14px;justify-content:flex-end;align-items:center;
 margin:-18px 0 14px;font-size:13px;font-family:"Segoe UI","PingFang SC","Microsoft YaHei",sans-serif}
.lan-bar a{color:var(--sub,#94a99d);text-decoration:none;padding:4px 10px;border:1px solid var(--line,#1d3027);
 border-radius:999px;background:rgba(16,28,22,.7)}
.lan-bar a:hover{color:var(--green,#34d399);border-color:var(--green,#34d399)}
</style>"""
BAR_HTML = ('<nav class="lan-bar" aria-label="网站">'
            '<a href="/">首页</a><a href="/status">数据状态</a><a href="/logout">退出登录</a></nav>')
_BODY_TAG = re.compile(rb"<body\b[^>]*>", re.I)


def inject_bar(page: bytes) -> bytes:
    """在网页 <body> 后面插一条“首页 · 数据状态 · 退出登录”。找不到 <body> 就原样发。"""
    m = _BODY_TAG.search(page)
    if not m:
        return page
    return page[:m.end()] + (BAR_CSS + BAR_HTML).encode("utf-8") + page[m.end():]


PAGE_CSS = """
:root{color-scheme:dark;--bg:#0a1310;--card:#101c16;--line:#1d3027;--text:#e2ebe5;--sub:#94a99d;
 --green:#34d399;--green-2:#10b981;--lime:#bef264;--amber:#fbbf24;--red:#f87171}
*{box-sizing:border-box}
html{-webkit-text-size-adjust:100%}
body{margin:0;min-height:100vh;display:flex;align-items:center;justify-content:center;padding:24px 16px;
 font-family:"Segoe UI","PingFang SC","Microsoft YaHei","Noto Sans SC",sans-serif;color:var(--text);
 background:radial-gradient(900px 480px at 0% -10%,rgba(16,185,129,.18),transparent 70%),var(--bg)}
.box{width:100%;max-width:380px;background:var(--card);border:1px solid var(--line);border-radius:18px;
 padding:30px 24px 24px;box-shadow:0 12px 40px rgba(0,0,0,.45)}
h1{margin:0 0 4px;font-size:22px;font-weight:650}
h1 span{color:var(--green)}
.sub{margin:0 0 22px;color:var(--sub);font-size:14px}
label{display:block;font-size:14px;color:var(--sub);margin:14px 0 6px}
input{width:100%;font-size:17px;padding:13px 14px;border-radius:12px;border:1px solid var(--line);
 background:#0c1712;color:var(--text);outline:none}
input:focus{border-color:var(--green);box-shadow:0 0 0 3px rgba(52,211,153,.18)}
button{width:100%;margin-top:22px;padding:14px;font-size:17px;font-weight:600;border:0;border-radius:12px;
 color:#06120c;background:linear-gradient(90deg,var(--lime),var(--green));cursor:pointer}
.err{margin:0 0 6px;padding:10px 12px;border-radius:10px;font-size:14px;color:var(--red);
 background:rgba(248,113,113,.08);border:1px solid rgba(248,113,113,.35)}
.note{margin:18px 0 0;color:var(--sub);font-size:12.5px;line-height:1.6}
a{color:var(--green)}
"""


def message_page(title: str, text: str, link: tuple[str, str] | None = ("/", "回首页")) -> bytes:
    """一个简单的提示页（找不到页面、网站正在更新等）。"""
    a = f'<p class="note"><a href="{html.escape(link[0])}">{html.escape(link[1])}</a></p>' if link else ""
    return (f'<!doctype html><html lang="zh-CN"><head><meta charset="utf-8">{VIEWPORT}'
            f'<title>{html.escape(title)}</title><style>{PAGE_CSS}</style></head><body><div class="box">'
            f'<h1>{html.escape(title)}</h1><p class="sub">{html.escape(text)}</p>{a}</div></body></html>').encode("utf-8")


def login_page(next_url: str, error: str = "", username: str = "", days: float = 30) -> bytes:
    """登录页：深色、绿色，和网站一个样子；手机上输入框和按钮都够大。所有放进页面的文字都转义。"""
    err = f'<p class="err" role="alert">{html.escape(error)}</p>' if error else ""
    return (
        f'<!doctype html><html lang="zh-CN"><head><meta charset="utf-8">{VIEWPORT}'
        f'<meta name="robots" content="noindex"><title>登录 · 足球预测</title><style>{PAGE_CSS}</style></head>'
        f'<body><main class="box"><h1>足球<span>预测</span></h1><p class="sub">请登录后查看</p>{err}'
        f'<form method="post" action="/login" autocomplete="on">'
        f'<input type="hidden" name="next" value="{html.escape(next_url)}">'
        f'<label for="u">账号</label>'
        f'<input id="u" name="username" value="{html.escape(username)}" autocomplete="username" '
        f'autocapitalize="none" autocorrect="off" spellcheck="false" required>'
        f'<label for="p">密码</label>'
        f'<input id="p" name="password" type="password" autocomplete="current-password" '
        f'autocapitalize="none" autocorrect="off" spellcheck="false" required>'
        f'<button type="submit">登录</button></form>'
        f'<p class="note">账号和密码在电脑上双击“查看网站地址和密码.bat”就能看到。登录一次 {days:g} 天内不用再登。</p>'
        f'</main></body></html>'
    ).encode("utf-8")


# ---------- 处理每一个请求 ----------

class Handler(BaseHTTPRequestHandler):
    """一个请求一个 Handler。self.server.app 是上面的 SiteApp。"""
    server_version = "site"                # 响应头里不写 Python 的版本号
    sys_version = ""

    @property
    def app(self) -> SiteApp:
        return self.server.app

    def setup(self):
        # 一个连接多少秒没动静就断开：防止有人只连不发，把服务器的线程占满
        self.timeout = self.server.app.config.request_timeout_seconds
        super().setup()

    def version_string(self) -> str:
        return self.server_version

    def end_headers(self):
        for k, v in SECURITY_HEADERS.items():
            self.send_header(k, v)
        super().end_headers()

    # ----- 日志：只记 IP、方法、路径（不带 ? 后面）、状态码，不记 cookie、密码 -----
    def log_request(self, code="-", size="-"):
        code = getattr(code, "value", code)
        LOG.info("%s %s %s %s", self.client_address[0], getattr(self, "command", None) or "-",
                 _clean_for_log(getattr(self, "path", "") or ""), code)

    def log_error(self, format, *args):
        # 默认会把整行请求原样写进去（可能带着网址里的东西），这里只记错误代码
        # 第一个参数通常是错误代码（如 400、408），超时时是一段说明；只留前 40 个字，免得带进网址里的东西
        LOG.warning("%s 请求有问题（%s）", self.client_address[0], str(args[0] if args else "")[:40])

    def log_message(self, format, *args):
        LOG.info("%s %s", self.client_address[0], _clean_for_log(format % args))

    # ----- 请求方法 -----
    def do_GET(self):
        self._dispatch()

    def do_HEAD(self):
        self._dispatch()

    def do_POST(self):
        route = self._route()
        if route == "/login":
            self._login_post()
        elif route is not None and route.startswith(ACTION_PREFIX) and self.app.config.live_settings is not None:
            # 操作按钮（只在现读现出时有这个地址；只发静态文件时和以前一样，下面回 405）
            self._action_post(route[len(ACTION_PREFIX):])
        else:
            self._not_allowed()

    def _not_allowed(self):
        """GET、HEAD 以外的请求方法一律 405；POST 只收 /login 和 /api/action/<名字>，别的地址 POST 也是 405。"""
        self._drain_body()
        self.send_response(405)
        self.send_header("Allow", ALLOW)
        self._send_body(message_page("不支持", "这个网站只能看，不能改。"), "text/html; charset=utf-8")

    do_PUT = do_DELETE = do_PATCH = do_OPTIONS = do_TRACE = do_CONNECT = _not_allowed

    def _drain_body(self, limit: int | None = None):
        """把对方发来的小段数据读掉再回复（不读就关连接，有的浏览器会显示“连接被重置”）。超过 limit 的不读，直接断开。"""
        limit = self.app.config.max_login_body_bytes if limit is None else limit
        try:
            n = int(self.headers.get("Content-Length") or 0)
        except ValueError:
            n = 0
        if 0 < n <= limit:
            try:
                self.rfile.read(n)
            except OSError:
                pass
        elif n:
            self.close_connection = True

    # ----- 小工具 -----
    def _route(self) -> str | None:
        """网址里 ? 前面的路径。不是以一个 / 开头的（例如 http://别的网站/…、*）返回 None。"""
        p = self.path
        if not p.startswith("/") or p.startswith("//"):
            return None
        return p.split("?", 1)[0].split("#", 1)[0]

    def _send_body(self, body: bytes, ctype: str, extra: dict | None = None):
        """在 send_response 之后调用：写剩下的响应头和内容。HEAD 请求只发响应头。"""
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        for k, v in (extra or {}).items():
            self.send_header(k, v)
        self.end_headers()
        if self.command != "HEAD":
            self.wfile.write(body)

    def _send(self, code: int, body: bytes, ctype: str = "text/html; charset=utf-8", extra: dict | None = None):
        self.send_response(code)
        self._send_body(body, ctype, extra)

    def _send_json(self, code: int, obj):
        """发 JSON（数据接口用）：中文原样写（ensure_ascii=False），不缓存（安全响应头里的 Cache-Control: no-store）。"""
        body = json.dumps(obj, ensure_ascii=False, allow_nan=False).encode("utf-8")
        self._send(code, body, JSON_TYPE)

    def _redirect(self, location: str, extra: dict | None = None):
        """跳到另一页（303：浏览器用 GET 打开新地址）。地址里的中文等先转成 %XX，响应头里只能放英文字符。"""
        self.send_response(303)
        self.send_header("Location", quote(location, safe="/?=&#%-._~!$()*+,;:@"))
        self._send_body(b"", "text/plain; charset=utf-8", extra)

    def _token(self) -> str | None:
        """从请求带来的 cookie 里找会话。格式不对的一律不认。"""
        raw = "; ".join(self.headers.get_all("Cookie") or [])
        for part in raw.split(";"):
            name, _, value = part.strip().partition("=")
            if name == COOKIE and re.fullmatch(r"[A-Za-z0-9_\-]{20,100}", value):
                return value
        return None

    def _cookie(self, token: str) -> str:
        age = int(self.app.config.session_days * 86400)
        return f"{COOKIE}={token}; Max-Age={age}; Path=/; HttpOnly; SameSite=Strict"

    def _logged_in(self) -> bool:
        return self.app.session_valid(self._token())

    # ----- 网址分到哪里 -----
    def _dispatch(self):
        route = self._route()
        if route is None:
            self._send(400, message_page("网址不对", "这个网址看不懂。"))
            return
        if route == "/login":
            query = parse_qs(urlsplit(self.path).query, max_num_fields=5)
            nxt = safe_next((query.get("next") or ["/"])[0])
            if self._logged_in():
                self._redirect(nxt)
            else:
                self._send(200, login_page(nxt, days=self.app.config.session_days))
            return
        if route == "/logout":
            self.app.end_session(self._token())
            self._redirect("/login", {"Set-Cookie": f"{COOKIE}=; Max-Age=0; Path=/; HttpOnly; SameSite=Strict"})
            return
        live = self.app.live
        if not self._logged_in():
            if (live is not None or self.app.actions is not None) and route.startswith(API_PREFIX):
                # 数据接口是网页里的脚本在问，不跳转，直接说“没登录”（401），脚本自己决定怎么办
                self._send_json(401, {"error": "not_logged_in"})
                return
            # 没登录：跳到登录页，记下原来想看哪一页，登录后回去
            self._redirect("/login?next=" + quote(safe_next(self.path), safe=""))
            return
        acts = self.app.actions
        if acts is not None and (route == ACTIONS_API or route.startswith(ACTION_PREFIX)):
            if route == ACTIONS_API:
                self._send_json(200, acts.status())          # 每个按钮的情况（只有写死的话和退出码，没有程序输出）
            else:
                # 按钮只能用 POST 点（GET 打开一个网址不能让电脑跑程序）；不管名字对不对都回 405，不透露有哪些按钮
                self.send_response(405)
                self.send_header("Allow", "POST")
                self._send_body(json.dumps({"error": "method_not_allowed"}).encode("utf-8"), JSON_TYPE)
            return
        if live is not None and route.startswith(API_PREFIX):
            self._api(route)
            return
        if (live is not None or acts is not None) and route.startswith(STATIC_PREFIX):
            # （按钮在时也发：数据状态页要用 site.js；现读现出起不来、按钮还在时也一样）
            # 本站自己的脚本、样式：只给 src/web/static/ 里的文件，挡路径穿越的办法和网站文件夹一样
            path = resolve_static(STATIC_DIR, route[len(STATIC_PREFIX) - 1:])
            if path is None:
                self._send(404, message_page("找不到这个文件", ""))
            else:
                self._serve_file(path)
            return
        if route == "/status":
            if acts is not None:
                self._status_with_actions()
            else:
                self._serve_file(self.app.config.status_file, missing=(
                    "数据状态页还没生成", "数据状态页每小时自动生成一次，过一会儿再来看。"))
            return
        reports = self.app.config.review_reports_dir
        if route.startswith(REPORTS_PREFIX) and reports is not None:
            # 战绩页里的周报、月报链接（../review/reports/xx.md）：只给这个文件夹里的 .md，按纯文本显示（手机上直接能看，不会变成下载）
            path = resolve_static(reports, route[len(REPORTS_PREFIX) - 1:])
            if path is None or path.suffix.lower() != ".md":
                self._send(404, message_page("找不到这份报告", "周报、月报每天复盘时重新生成。"))
            else:
                self._serve_file(path, ctype="text/plain; charset=utf-8")
            return
        if live is not None:
            got = live.page(route)          # 现读现出；None = 这一页不归它管，或者现在拿不到数据 → 下面发静态文件
            if got is not None:
                code, text, ctype = got
                if code == 404:
                    self._send(404, message_page("找不到这一页", "可能这场比赛已经不在列表里了。"))
                elif ctype.startswith("text/html"):
                    self._send(200, inject_bar(text.encode("utf-8")), ctype)
                else:
                    self._send(200, text.encode("utf-8"), ctype)
                return
        site = self.app.config.site_dir
        if not site.is_dir():
            # “足球预测_生成网站”换新网站的那一瞬间文件夹可能不在
            self._send(503, message_page("网站正在更新", "过几秒钟刷新一下就好。", link=None),
                       extra={"Retry-After": "5"})
            return
        path = resolve_static(site, route)
        if path is None:
            self._send(404, message_page("找不到这一页", "可能这场比赛已经不在列表里了。"))
            return
        self._serve_file(path)

    def _api(self, route: str):
        """数据接口（只读）。出了意外的错回 500 和一句 JSON，不把错误细节发出去（细节只记日志，也不带路径）。"""
        live = self.app.live
        try:
            if route == "/api/version":
                self._send_json(200, live.version_info())
            elif route == "/api/live":
                self._send_json(200, live.live_api())
            elif route == "/api/matches":
                body = live.matches_api()
                if body is None:
                    self._send_json(503, {"error": "data_unavailable"})
                else:
                    self._send_json(200, body)
            elif MATCH_API.fullmatch(route):
                code, body = live.match_api(int(MATCH_API.fullmatch(route).group(1)))
                self._send_json(code, body)
            else:
                self._send_json(404, {"error": "not_found"})
        except (ConnectionError, TimeoutError):
            raise
        except Exception as err:   # noqa: BLE001 —— 故意的：接口出错不能让服务器线程崩掉
            LOG.warning("数据接口 %s 出错（%s）", _clean_for_log(route), type(err).__name__)
            self._send_json(500, {"error": "server_error"})

    def _serve_file(self, path: Path, missing: tuple[str, str] = ("找不到这一页", ""), ctype: str | None = None):
        """发一个文件。整个读进来再发（网页都很小），读完马上关掉文件，不挡着“生成网站”换新文件夹。"""
        try:
            data = path.read_bytes()
        except OSError:
            self._send(404, message_page(*missing))
            return
        ctype = ctype or CONTENT_TYPES.get(path.suffix.lower(), "application/octet-stream")
        if ctype.startswith("text/html"):
            data = inject_bar(data)
        self._send(200, data, ctype)

    # ----- 操作按钮（需求文档第 8 节“网站上的操作按钮”、第 16 节“安全”） -----
    def _status_with_actions(self):
        """数据状态页 + 上面一块“操作”（文件本身不改，发出去时插）。状态页还没生成时，给一个只有按钮的简单页面。"""
        acts = self.app.actions
        try:
            data = self.app.config.status_file.read_bytes()
        except OSError:
            data = site_actions.missing_status_page()
        try:
            query = parse_qs(urlsplit(self.path).query, max_num_fields=5)
        except ValueError:
            query = {}
        # 跳回来时网址里的 done、action 只用来从固定的几句话里挑一句，本身不放进页面
        notice = site_actions.notice_text((query.get("done") or [None])[0], (query.get("action") or [None])[0], acts)
        csrf = serve_auth.csrf_token(self.app.csrf_key, self._token())
        page = site_actions.inject_panel(data, site_actions.panel_html(acts, csrf, notice))
        self._send(200, inject_bar(page), "text/html; charset=utf-8")

    def _same_origin_strict(self) -> bool:
        """按钮用的同源检查，比登录那里严：必须带 Origin 或 Referer，主机和端口要等于这次请求的 Host。
        Origin 是 "null"、是别的网站、两样都没带，一律不行；浏览器自己标了 Sec-Fetch-Site 是 cross-site / same-site 的也不行。
        （本网站自己的页面提交时，浏览器会带 Origin；响应头 Referrer-Policy: same-origin 也让同站的 Referer 照常带上。）"""
        host = (self.headers.get("Host") or "").strip().lower()
        if not host:
            return False
        if (self.headers.get("Sec-Fetch-Site") or "").strip().lower() in ("cross-site", "same-site"):
            return False
        origin = self.headers.get("Origin")
        if origin is not None:
            return origin != "null" and urlsplit(origin).netloc.lower() == host
        referer = self.headers.get("Referer")
        return bool(referer) and urlsplit(referer).netloc.lower() == host

    def _action_refused(self, code: int, error: str, title: str, text: str, wants_json: bool):
        """按钮请求被挡下：脚本发来的回 JSON（error 代号 + 一句话），表单发来的回一个简单的提示页。都没有程序输出、路径。"""
        if wants_json:
            self._send_json(code, {"error": error, "text": (title + "：" + text) if text else title})
        else:
            self._send(code, message_page(title, text, link=("/status", "回数据状态页")))

    def _read_small_form(self, wants_json: bool) -> dict | None:
        """读按钮表单：只收 application/x-www-form-urlencoded，最多 max_login_body_bytes（4 KB）。不对就回错、返回 None。"""
        ctype = (self.headers.get("Content-Type") or "").split(";", 1)[0].strip().lower()
        try:
            length = int(self.headers.get("Content-Length") or "")
        except ValueError:
            length = -1
        if length < 0 or length > self.app.config.max_login_body_bytes:
            self._drain_body(limit=65536)          # 不太大的先读掉，再大的不读、断开
            self.close_connection = True
            self._action_refused(413, "too_large", "提交的内容太长", "", wants_json)
            return None
        body = self.rfile.read(length) if length else b""
        if ctype != "application/x-www-form-urlencoded":
            self._action_refused(415, "bad_type", "提交的格式不对", "", wants_json)
            return None
        try:
            return parse_qs(body.decode("utf-8"), max_num_fields=5)
        except (UnicodeDecodeError, ValueError):
            return {}

    def _action_post(self, name: str):
        """点按钮：POST /api/action/<名字>。一道道检查，哪一道不过都不启动程序：
        总开关 → 登录 → 同源 → 名字在白名单里 → 表单格式和大小 → 防伪口令 → 有没有撞车（ActionRunner.start 里）。
        脚本发来的（Accept 里有 application/json）回 JSON；没有脚本时的表单回 303 跳回 /status?done=<结果代号>。"""
        acts = self.app.actions
        wants_json = "application/json" in (self.headers.get("Accept") or "").lower()
        if acts is None:                       # actions.enabled: false：当作没有这个地址
            self._drain_body()
            self._send_json(404, {"error": "not_found"})
            return
        if not self._logged_in():
            self._drain_body()
            if wants_json:
                self._send_json(401, {"error": "not_logged_in"})
            else:
                self._redirect("/login?next=" + quote("/status", safe=""))
            return
        if not self._same_origin_strict():
            self._drain_body()
            LOG.warning("%s 按钮请求不是从本网站发的，没理它", self.client_address[0])
            self._action_refused(403, "cross_origin", "不能这样点", "请在本网站的数据状态页上点按钮。", wants_json)
            return
        if not site_actions.NAME.fullmatch(name) or name not in acts.actions:
            self._drain_body()
            self._action_refused(404, "not_found", "没有这个按钮", "", wants_json)
            return
        form = self._read_small_form(wants_json)
        if form is None:
            return
        if not serve_auth.csrf_valid(self.app.csrf_key, self._token(), (form.get(serve_auth.CSRF_FIELD) or [None])[0]):
            LOG.warning("%s 按钮请求的防伪口令不对，没理它", self.client_address[0])
            self._action_refused(403, "bad_csrf", "页面过期了",
                                 "刷新一下数据状态页再点（网站服务器重启过，或者这一页是退出登录以前打开的）。", wants_json)
            return
        result = acts.start(name)
        LOG.info("%s 点了按钮 %s（%s）", self.client_address[0], name, result)
        if wants_json:
            code = {"started": 202, "running": 409, "busy": 409, "failed": 500}[result]
            self._send_json(code, {"result": result, "text": site_actions.RESULT_TEXT[result], **acts.status()})
        else:
            self._redirect(f"/status?done={result}&action={name}#actions")

    # ----- 登录 -----
    def _same_origin(self) -> bool:
        """登录表单是不是从本网站自己的页面提交的（防止别的网站偷偷帮你提交）。没带 Origin 的（老浏览器）放行。
        Origin 是 "null"（别的网页里藏一个沙盒框架自动提交时就是这样）一律拒绝：否则别人能故意输错把你锁在外面。
        本网站自己的登录页提交时，现在的浏览器都会带真实的 Origin。"""
        origin = self.headers.get("Origin")
        if origin is None:
            return True
        if origin == "null":
            return False
        return urlsplit(origin).netloc.lower() == (self.headers.get("Host") or "").lower()

    def _login_post(self):
        cfg = self.app.config
        if not self._same_origin():
            self._drain_body()
            self._send(403, message_page("不能这样登录", "请直接打开本网站的登录页登录。", link=("/login", "去登录")))
            return
        ctype = (self.headers.get("Content-Type") or "").split(";", 1)[0].strip().lower()
        try:
            length = int(self.headers.get("Content-Length") or "")
        except ValueError:
            length = -1
        if length < 0 or length > cfg.max_login_body_bytes:
            self._drain_body(limit=65536)          # 不太大的先读掉（直接断开时对方可能收不到这条回复），再大的不读
            self.close_connection = True
            self._send(413, message_page("提交的内容太长", "", link=("/login", "回登录页")))
            return
        body = self.rfile.read(length) if length else b""
        if ctype != "application/x-www-form-urlencoded":
            self._send(415, message_page("提交的格式不对", "", link=("/login", "回登录页")))
            return
        try:
            form = parse_qs(body.decode("utf-8"), max_num_fields=10)
        except (UnicodeDecodeError, ValueError):
            form = {}
        username = (form.get("username") or [""])[0]
        password = (form.get("password") or [""])[0]
        nxt = safe_next((form.get("next") or ["/"])[0])
        result, n = self.app.check_login(self.client_address[0], username, password)
        if result == "ok":
            LOG.info("%s 登录成功", self.client_address[0])
            self._redirect(nxt, {"Set-Cookie": self._cookie(self.app.new_session())})
            return
        LOG.warning("%s 登录失败（%s）", self.client_address[0], result)
        extra = None
        if result == "locked":
            msg = f"输错太多次，请 {max(1, (n + 59) // 60)} 分钟后再试。"
            code, extra = 429, {"Retry-After": str(n)}
        elif result == "no_password":
            msg, code = "还没有设置密码：在电脑上双击“查看网站地址和密码.bat”。", 503
        else:
            msg, code = f"账号或密码不对，还能再试 {n} 次。", 200
        self._send(code, login_page(nxt, msg, username[:64], days=cfg.session_days), extra=extra)


# ---------- 服务器 ----------

class SiteServer(ThreadingHTTPServer):
    """几个请求同时来时各开一个线程处理（手机、电脑同时看不用排队）。"""
    daemon_threads = True          # 关服务器时不等还没处理完的连接
    # Windows 上“地址重用”会让两个程序同时占着同一个端口，结果谁收到请求说不准。关掉它，并要求独占端口
    allow_reuse_address = False
    allow_reuse_port = False

    def __init__(self, address, app: SiteApp):
        self.app = app
        # 同时处理的连接数有上限：有人故意开几千个很慢的连接时，不至于把线程和内存用光
        self._slots = threading.BoundedSemaphore(app.config.max_connections)
        super().__init__(address, Handler)

    def process_request(self, request, client_address):
        if not self._slots.acquire(blocking=False):
            LOG.warning("%s 同时连接太多，这个连接直接关掉", client_address[0])
            self.shutdown_request(request)
            return
        try:
            super().process_request(request, client_address)    # 开一个线程去处理
        except Exception:
            self._slots.release()
            raise

    def process_request_thread(self, request, client_address):
        try:
            super().process_request_thread(request, client_address)
        finally:
            self._slots.release()                               # 这个连接处理完，空出一个位置

    def server_bind(self):
        if hasattr(socket, "SO_EXCLUSIVEADDRUSE"):          # 只有 Windows 有：别的程序不能再绑同一个端口
            self.socket.setsockopt(socket.SOL_SOCKET, socket.SO_EXCLUSIVEADDRUSE, 1)
        # 不调用 HTTPServer.server_bind：它会查电脑的完整域名，有的网络里要卡好几秒
        self.socket.bind(self.server_address)
        self.server_address = self.socket.getsockname()
        self.server_name, self.server_port = str(self.server_address[0]), self.server_address[1]

    def handle_error(self, request, client_address):
        # 默认会把错误打到屏幕上；没有黑窗口时打不出来，这里写进日志。手机切走网页导致的断开不算错
        exc = sys.exc_info()[1]
        if isinstance(exc, (ConnectionError, TimeoutError)):
            LOG.debug("%s 连接断开", client_address[0])
        else:
            LOG.exception("%s 处理请求出错", client_address[0])


def is_loopback(host: str) -> bool:
    """是不是“只有本机能连”的地址（127.0.0.1、::1、localhost）。看不懂的当作不是。"""
    if host.lower() == "localhost":
        return True
    try:
        return ipaddress.ip_address(host).is_loopback
    except ValueError:
        return False


def create_server(config: ServerConfig, host: str, port: int, clock: Callable[[], float] = time.time) -> SiteServer:
    """造好服务器（已经占上端口，还没开始接请求）。没有设置密码时，只允许在本机地址上监听。"""
    app = SiteApp(config, clock=clock)
    if not is_loopback(host) and app.credentials() is None:
        raise PasswordRequired(f"还没有设置网站密码，不能在 {host} 上对局域网开放")
    return SiteServer((host, port), app)


def setup_logging(settings: dict) -> None:
    """访问日志写进 logs/site_server.log（满了换新文件，旧的留几个）。有黑窗口时屏幕上也显示。"""
    ss = settings["site_server"]
    path = resolve_path(ss["log_file"])
    path.parent.mkdir(parents=True, exist_ok=True)
    LOG.setLevel(logging.INFO)
    LOG.propagate = False
    if not LOG.handlers:
        fh = logging.handlers.RotatingFileHandler(path, maxBytes=ss["log_max_bytes"],
                                                  backupCount=ss["log_backups"], encoding="utf-8")
        fh.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(message)s"))
        LOG.addHandler(fh)
        if sys.stderr is not None:
            LOG.addHandler(logging.StreamHandler())


def _port_busy(e: OSError) -> bool:
    """端口被占用（或者被 Windows 保留）的错误。"""
    return e.errno in (errno.EADDRINUSE, errno.EACCES) or getattr(e, "winerror", None) in (10048, 10013)


def start_live_poller(settings: dict, stop: threading.Event) -> threading.Thread | None:
    """settings.yaml 的 live.live_poller 是 true 时，在后台开一个线程跑即时比分程序（WD3 写的 src.collect.dongqiudi_live 的
    run_forever(settings, stop_event)）。它会联网请求懂球帝，把比分写进 scores.json（dongqiudi.live.output），
    并往 football.db 的 fetch_log 表写抓取记录。没有这个程序（导入失败）就记一行日志、不开。
    后台线程（daemon）：服务器关掉时它跟着结束；stop 被设上时它应该自己停。返回开起来的线程，没开是 None。"""
    if not (settings.get("live") or {}).get("live_poller", False):
        return None
    try:
        # 按完整名字到 sys.modules 里找（找不到才真去加载），这样测试换上的假程序一定会被用到；
        # 写成 from src.collect import dongqiudi_live 的话，真程序已经被加载过时会绕开测试的假程序。
        target = importlib.import_module("src.collect.dongqiudi_live").run_forever
    except Exception as err:   # noqa: BLE001 —— 故意的：即时比分程序有问题不能挡住网站启动
        LOG.warning("即时比分程序加载不了（%s），这次不开即时比分", type(err).__name__)
        return None

    def body():
        try:
            target(settings, stop)
        except Exception as err:   # noqa: BLE001
            LOG.warning("即时比分程序出错停了（%s）；网站照常", type(err).__name__)

    t = threading.Thread(target=body, name="live-poller", daemon=True)
    t.start()
    LOG.info("即时比分程序已在后台启动")
    return t


def stop_live_poller(thread: threading.Thread | None, stop: threading.Event, timeout: float) -> bool:
    """服务器关掉时叫即时比分线程停下：设上 stop，最多等 timeout 秒（live.poller_stop_seconds）。
    它正在联网请求时可能一下停不了；它是后台线程，不等它，程序照样退出。返回它是不是已经停了。"""
    stop.set()
    if thread is None:
        return True
    thread.join(timeout)
    if thread.is_alive():
        LOG.warning("即时比分程序 %s 秒内没停下，不等它了", timeout)
        return False
    return True


def run(settings: dict, host: str | None = None, port: int | None = None, ready: Callable | None = None) -> int:
    """启动服务器，一直运行到被关掉。返回值：0 = 正常（包括“已经有一个在运行”）；1 = 出错；2 = 端口被别的程序占了。

    ready：测试用，服务器开始接请求前调用一次，把服务器交出来（测试拿它关服务器）。
    """
    ss = settings["site_server"]
    host = host or ss["host"]
    port = ss["port"] if port is None else port
    setup_logging(settings)
    new_plain = None
    try:
        _rec, new_plain = serve_auth.ensure_credentials(settings)   # 第一次运行：自动生成账号密码
        if new_plain:
            # 马上写进 txt：后面万一端口被占、提前退出，密码也不会丢（哈希倒推不出密码）
            serve_auth.write_info_file(settings, plain=new_plain, port=port or None)
            LOG.info("第一次运行，已生成网站密码（写在 %s）", serve_auth.INFO_FILE)
    except (serve_auth.CredentialsError, OSError) as e:
        LOG.error("密码文件有问题：%s", e)      # 不覆盖；下面 create_server 会拒绝对局域网开放
    try:
        server = create_server(ServerConfig.from_settings(settings), host, port)
    except PasswordRequired as e:
        LOG.error("%s。双击“修改网站密码.bat”设置密码后再启动", e)
        return 1
    except OSError as e:
        if not _port_busy(e):
            LOG.error("启动失败：%s", e)
            return 1
        if serve_auth.probe_server(port):
            LOG.info("端口 %s 上已经有一个网站服务器在运行，这次不再开第二个", port)
            return 0
        LOG.error("端口 %s 被别的程序占用了（或被 Windows 保留）。请告诉 Claude 换一个端口（settings.yaml 的 site_server.port）", port)
        return 2
    real_port = server.server_address[1]
    try:
        serve_auth.write_info_file(settings, plain=new_plain, port=real_port)
    except (OSError, serve_auth.CredentialsError) as e:
        LOG.warning("地址和密码文件写不进去：%s", type(e).__name__)
    LOG.info("网站服务器已启动：http://%s:%s（地址和密码见 %s）", host, real_port, serve_auth.INFO_FILE)
    stop = threading.Event()
    poller = start_live_poller(settings, stop)
    if server.app.live is not None:
        # 现读现出：启动时先在后台读一次数据，第一个打开网页的人不用等好几秒
        threading.Thread(target=server.app.live.snapshot, name="live-warmup", daemon=True).start()
    if ready:
        ready(server)
    try:
        server.serve_forever(poll_interval=0.5)
    except KeyboardInterrupt:
        pass
    finally:
        # 告诉即时比分程序停下，最多等几秒
        stop_live_poller(poller, stop, float((settings.get("live") or {}).get("poller_stop_seconds", 5)))
        server.server_close()
        LOG.info("网站服务器已关闭")
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="局域网网站服务器（带登录）")
    parser.add_argument("--host", help="监听地址，默认用 settings.yaml 的 site_server.host")
    parser.add_argument("--port", type=int, help="端口，默认用 settings.yaml 的 site_server.port")
    args = parser.parse_args(argv)
    return run(load_settings(), args.host, args.port)


if __name__ == "__main__":
    sys.exit(main())
