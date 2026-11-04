"""网站上的操作按钮的自动检查（src/web/actions.py、src/web/serve.py，需求文档第 8 节“网站上的操作按钮”、第 16 节“安全”）。

不跑真的刷新数据、出预测、生成网站、推送，不联网、不碰真实数据库：
- 按钮的白名单换成假的：子程序跑的是这里临时写的小模块 fakejob（放在临时文件夹里，子程序就在那个文件夹里跑），
  它按参数睡几秒、打印几句、用指定的退出码结束；
- 数据库路径、锁文件、心跳文件、日志全部指到 pytest 的临时文件夹；
- 服务器在 127.0.0.1 的随机端口上起（tests/test_serve.py 的 Running），测完马上关掉。
"""
import copy
import importlib.util
import json
import os
import re
import subprocess
import sys
import threading
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path
from urllib.parse import urlencode

import pytest

from src.common.config import load_settings
from src.jobs import data_lock
from src.web import actions as A
from src.web import serve, serve_auth
from tests.test_serve import make_settings as serve_settings
from tests.test_serve import Running, make_config, tree  # noqa: F401（tree 是 pytest 的 fixture）

FAKE_TOKEN = "pp-FAKE-token-1234567890abcdef"        # 假口令（只在测试里）：看它会不会漏到网页、日志里

# 假的子程序：python -m fakejob <模式> <秒数> <退出码>
FAKE_JOB = r'''
import sys, time, os
mode, seconds, code = sys.argv[1], float(sys.argv[2]), int(sys.argv[3])
time.sleep(seconds)
if mode == "leak":
    print("Traceback: File " + os.getcwd() + "\\fakejob.py, line 3")
    print("token=" + os.environ.get("FAKE_TOKEN", "") + " password: hunter2hunter2")
    print("C:\\Users\\someone\\secret.txt https://example.invalid/x?token=abc123456")
elif mode == "skip":
    print("定时预测：跳过（上次运行还没结束）")
elif mode == "push_no_token":
    print("【没成功】还没设置口令：先双击“设置微信推送.bat”")
else:
    print("fake job ok")
sys.exit(code)
'''


def fake(name, mode="ok", seconds=0, code=0, **kw):
    kw.setdefault("judge", A.judge_predict if mode == "skip" else A.judge_push if mode.startswith("push") else A.judge_plain)
    return A.Action(name, f"假按钮{name}", "测试用", "fakejob", (mode, str(seconds), str(code)), **kw)


FAKES = {a.name: a for a in (
    fake("ok"),
    fake("slow", seconds=1.5),
    fake("db_a", seconds=1.5, writes_db=True),
    fake("db_b", writes_db=True),
    fake("hang", seconds=60),
    fake("leak", mode="leak", code=3),
    fake("skip", mode="skip"),
    fake("push", mode="push_no_token"),
    fake("site_job", heartbeat="site"),
)}


def make_settings(tmp):
    """按钮管理器要用的设置：全部指到临时文件夹。"""
    return {
        "paths": {"database": str(tmp / "data" / "football.db")},
        "data_refresh": {"lock_timeout_seconds": 0, "lock_poll_seconds": 0.01},
        "schedule": {"lock_file": str(tmp / "data" / "jobs" / "scheduled_predict.lock"), "lock_stale_minutes": 60},
        "jobs": {"state_dir": str(tmp / "data" / "jobs")},
        "actions": {"enabled": True, "log_file": str(tmp / "logs" / "site_actions.log"), "log_max_bytes": 100000,
                    "log_backups": 1, "log_tail_chars": 4000, "poll_seconds": 3, "busy_minutes": 15,
                    "timeout_minutes": {n: (0.02 if n == "hang" else 1) for n in FAKES}},
    }


class CountingPopen:
    """包一层真的 subprocess.Popen：数一数开了几个子程序，测完等它们都结束。"""
    def __init__(self):
        self.procs = []
        self.lock = threading.Lock()

    def __call__(self, *a, **kw):
        p = subprocess.Popen(*a, **kw)
        with self.lock:
            self.procs.append(p)
        return p


@pytest.fixture
def act_env(tree, tmp_path, monkeypatch):
    """临时文件夹里写好 fakejob.py；造一个用假白名单的按钮管理器。返回 (settings, runner, popen)。"""
    work = tmp_path / "work"
    work.mkdir()
    (work / "fakejob.py").write_text(FAKE_JOB, encoding="utf-8")
    monkeypatch.setenv("FAKE_TOKEN", FAKE_TOKEN)
    s = make_settings(tmp_path)
    popen = CountingPopen()
    runner = A.ActionRunner(s, actions=FAKES, popen=popen, cwd=work, secrets=lambda: {"pushplus_token": FAKE_TOKEN})
    yield s, runner, popen
    for p in popen.procs:                               # 别留下还在跑的子程序
        if p.poll() is None:
            p.kill()
        p.wait(10)


class ActRunning(Running):
    """起一个有按钮的服务器（live_settings 给一个只够按钮用的设置；现读现出这里不测，起不来也没关系）。"""
    def __init__(self, tree, runner, settings):
        super().__init__(make_config(tree, live_settings=settings))
        self.server.app.live = None
        self.server.app.actions = runner

    def origin(self):
        return {"Origin": f"http://127.0.0.1:{self.port}"}

    def page_csrf(self, h):
        """打开数据状态页，从按钮表单里拿防伪口令。"""
        r, body = self.req("GET", "/status", h)
        assert r.status == 200
        m = re.search(r'name="csrf" value="([0-9a-f]{64})"', body.decode("utf-8"))
        assert m, "页面上没有防伪口令"
        return m.group(1)

    def click(self, name, h, csrf, origin=True, json_accept=True, extra=None):
        headers = {**h, "Content-Type": "application/x-www-form-urlencoded"}
        if origin:
            headers.update(self.origin())
        if json_accept:
            headers["Accept"] = "application/json"
        headers.update(extra or {})
        r, body = self.req("POST", f"/api/action/{name}", headers, urlencode({"csrf": csrf} if csrf is not None else {}))
        return r, body


def wait_done(runner, name, seconds=20):
    end = time.monotonic() + seconds
    while time.monotonic() < end:
        if not runner.is_running(name):
            return
        time.sleep(0.05)
    raise AssertionError(f"{name} 一直没跑完")


def state_of(srv, h, name):
    r, body = srv.req("GET", "/api/actions", h)
    assert r.status == 200 and r.getheader("Content-Type") == "application/json; charset=utf-8"
    return next(a for a in json.loads(body)["actions"] if a["name"] == name)


@pytest.fixture
def asrv(tree, act_env):
    s, runner, popen = act_env
    with ActRunning(tree, runner, s) as srv:
        yield srv, runner, popen


# ---------- 登录、请求方法、防伪口令、同源、白名单 ----------

def test_not_logged_in_is_refused(asrv):
    srv, runner, popen = asrv
    r, body = srv.click("ok", {}, "0" * 64)
    assert r.status == 401 and json.loads(body) == {"error": "not_logged_in"}
    r, _ = srv.click("ok", {}, "0" * 64, json_accept=False)               # 没有脚本的表单：跳到登录页
    assert r.status == 303 and r.getheader("Location").startswith("/login?next=")
    r, body = srv.req("GET", "/api/actions", {})
    assert r.status == 401 and json.loads(body) == {"error": "not_logged_in"}
    r, _ = srv.req("GET", "/status", {})
    assert r.status == 303
    assert popen.procs == []


def test_get_and_other_methods_are_refused(asrv):
    srv, runner, popen = asrv
    h = srv.cookie()
    for name in ("ok", "nope"):
        r, _ = srv.req("GET", f"/api/action/{name}", h)
        assert r.status == 405 and r.getheader("Allow") == "POST", name
        r, _ = srv.req("HEAD", f"/api/action/{name}", h)
        assert r.status == 405
    for m in ("PUT", "DELETE", "PATCH"):
        r, _ = srv.req(m, "/api/action/ok", {**h, **srv.origin()})
        assert r.status == 405
    form = {**h, **srv.origin(), "Content-Type": "application/x-www-form-urlencoded"}
    r, _ = srv.req("POST", "/api/actions", form, "a=1")                    # 别的地址照旧不收 POST
    assert r.status == 405
    assert popen.procs == []


def test_missing_or_wrong_csrf_is_refused(asrv):
    srv, runner, popen = asrv
    h = srv.cookie()
    good = srv.page_csrf(h)
    flipped = good[:-1] + ("0" if good[-1] != "0" else "1")
    for bad in (None, "", "x", "0" * 64, flipped, good.upper(), good + "0"):
        r, body = srv.click("ok", h, bad)
        assert r.status == 403 and json.loads(body)["error"] == "bad_csrf", bad
    # 口令和会话绑定：另一次登录拿不到也用不了这一个
    h2 = srv.cookie()
    assert h2 != h and srv.page_csrf(h2) != good
    r, _ = srv.click("ok", h2, good)
    assert r.status == 403
    # 没有脚本的表单被挡下时给一个提示页，不跳转、不启动
    r, body = srv.click("ok", h, None, json_accept=False)
    assert r.status == 403 and "页面过期了" in body.decode("utf-8")
    assert popen.procs == []


def test_cross_site_origin_is_refused(asrv):
    srv, runner, popen = asrv
    h = srv.cookie()
    csrf = srv.page_csrf(h)
    me = f"http://127.0.0.1:{srv.port}"
    cases = [
        {"Origin": "http://evil.example"},
        {"Origin": "null"},
        {"Origin": f"http://127.0.0.1:{srv.port + 1}"},                     # 端口不一样也不行
        {"Referer": "http://evil.example/page"},                            # 没有 Origin、Referer 是别的网站
        {},                                                                 # 两样都没带
        {"Origin": me, "Sec-Fetch-Site": "cross-site"},
        {"Origin": me, "Sec-Fetch-Site": "same-site"},
    ]
    for extra in cases:
        r, body = srv.click("ok", h, csrf, origin=False, extra=extra)
        assert r.status == 403 and json.loads(body)["error"] == "cross_origin", extra
    assert popen.procs == []
    # 没有 Origin、Referer 是本站的：可以
    r, _ = srv.click("ok", h, csrf, origin=False, extra={"Referer": me + "/status"})
    assert r.status == 202
    wait_done(runner, "ok")


def test_unknown_names_are_refused(asrv):
    srv, runner, popen = asrv
    h = srv.cookie()
    csrf = srv.page_csrf(h)
    # refresh_data 是正式白名单里的名字，但这个服务器的白名单是假的那份：也不认
    for name in ("nope", "refresh_data", "OK", "ok%20", "..%2Fok", "ok/x", "a" * 40, "", "ok;x"):
        r, body = srv.click(name, h, csrf)
        assert r.status == 404, name
        assert json.loads(body)["error"] == "not_found", name
    assert popen.procs == []


def test_wrong_body_type_and_size(asrv):
    srv, runner, popen = asrv
    h = srv.cookie()
    csrf = srv.page_csrf(h)
    base = {**h, **srv.origin(), "Accept": "application/json"}
    r, _ = srv.req("POST", "/api/action/ok", {**base, "Content-Type": "application/json"}, json.dumps({"csrf": csrf}))
    assert r.status == 415
    r, _ = srv.req("POST", "/api/action/ok", {**base, "Content-Type": "application/x-www-form-urlencoded"},
                   "csrf=" + csrf + "&x=" + "a" * 5000)
    assert r.status == 413
    assert popen.procs == []


# ---------- 跑起来、撞车、超时 ----------

def hold_lock(settings, source=None):
    """另开一个进程拿着锁（和定时任务正在更新时一样），返回这个进程；往它的输入里写空就会放锁退出。"""
    call = "data_write_lock(s)" if source is None else f"source_refresh_lock(s, {source!r})"
    code = ("import sys\nfrom src.jobs.data_lock import data_write_lock, source_refresh_lock\n"
            f"s = {settings!r}\nwith {call}:\n    print('in', flush=True)\n    sys.stdin.read()\n")
    p = subprocess.Popen([sys.executable, "-X", "utf8", "-c", code], stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                         text=True)
    assert p.stdout.readline().strip() == "in"
    return p


def test_click_runs_fixed_command_and_reports(asrv):
    srv, runner, popen = asrv
    h = srv.cookie()
    csrf = srv.page_csrf(h)
    r, body = srv.click("ok", h, csrf)
    assert r.status == 202
    got = json.loads(body)
    assert got["result"] == "started" and got["text"] == A.RESULT_TEXT["started"]
    wait_done(runner, "ok")
    st = state_of(srv, h, "ok")
    assert st["state"] == "done" and st["exit_code"] == 0 and st["text"].startswith("跑完了")
    assert st["started_at"].endswith("+00:00") and st["finished_at"].endswith("+00:00")    # 接口里是 UTC
    # 子程序就是白名单里写死的那一条：同一个 python、-m 模块、写死的参数
    (p,) = popen.procs
    assert p.args == [sys.executable, "-m", "fakejob", "ok", "0", "0"]


def test_popen_arguments_are_fixed(act_env):
    """不真的开子程序：看传给 Popen 的东西（列表参数、不经过 shell、没有窗口、输入关掉）。"""
    s, _, _ = act_env
    calls = []

    class FakeProc:
        returncode = 0
        args = None

        def communicate(self, timeout=None):
            return b"fake job ok", b""

    def fake_popen(argv, **kw):
        calls.append((argv, kw))
        return FakeProc()

    runner = A.ActionRunner(s, actions=FAKES, popen=fake_popen, cwd=Path("."))
    assert runner.start("ok") == "started"
    wait_done(runner, "ok")
    (argv, kw) = calls[0]
    assert argv == [sys.executable, "-m", "fakejob", "ok", "0", "0"]
    assert kw["shell"] is False and kw["stdin"] == subprocess.DEVNULL and kw["stderr"] == subprocess.STDOUT
    assert kw["env"]["PYTHONUTF8"] == "1"
    if sys.platform == "win32":
        assert kw["creationflags"] == subprocess.CREATE_NO_WINDOW


def test_lock_held_shows_busy(asrv, tmp_path):
    srv, runner, popen = asrv
    h = srv.cookie()
    csrf = srv.page_csrf(h)
    holder = hold_lock(make_settings(tmp_path))
    try:
        r, body = srv.click("db_b", h, csrf)
        assert r.status == 409 and json.loads(body)["result"] == "busy"
        assert json.loads(body)["text"] == "正在更新，稍后再试。"
        r, _ = srv.click("db_b", h, csrf, json_accept=False)                 # 没有脚本：跳回去，上面写“正在更新”
        assert r.status == 303 and r.getheader("Location") == "/status?done=busy&action=db_b#actions"
        page = srv.req("GET", "/status?done=busy&action=db_b", h)[1].decode("utf-8")
        assert "正在更新，稍后再试" in page
        r, _ = srv.click("ok", h, csrf)                                     # 不写数据库的按钮不受这把锁影响
        assert r.status == 202
    finally:
        holder.communicate("", timeout=10)
    assert [p.args[3] for p in popen.procs] == ["ok"]                       # 没有排队、没有硬跑
    wait_done(runner, "ok")
    r, _ = srv.click("db_b", h, csrf)                                       # 锁放了以后就能点
    assert r.status == 202
    wait_done(runner, "db_b")


def test_other_busy_signals(act_env, tmp_path):
    s, runner, popen = act_env
    holder = hold_lock(s, "cups")                                           # 杯赛那一路正在采集
    try:
        assert runner.start("db_b") == "busy"
    finally:
        holder.communicate("", timeout=10)
    # 定时预测的锁文件在、没过期 = 定时预测正在跑
    lock = tmp_path / "data" / "jobs" / "scheduled_predict.lock"
    lock.parent.mkdir(parents=True, exist_ok=True)
    lock.write_text("{}", encoding="utf-8")
    assert runner.start("db_b") == "busy"
    old = time.time() - 61 * 60                                             # 过期的残留锁不算
    os.utime(lock, (old, old))
    assert runner.start("db_b") == "started"
    wait_done(runner, "db_b")
    # 生成网站的心跳：“结束”是空的、刚开始 = 正在跑；很久以前开始、一直没结束的不算
    hb = tmp_path / "data" / "jobs" / "site.json"
    now = datetime.now(timezone.utc)
    hb.write_text(json.dumps({"开始": now.strftime("%Y-%m-%d %H:%M:%S"), "结束": None}), encoding="utf-8")
    assert runner.start("site_job") == "busy"
    hb.write_text(json.dumps({"开始": (now - timedelta(hours=2)).strftime("%Y-%m-%d %H:%M:%S"), "结束": None}),
                  encoding="utf-8")
    assert runner.start("site_job") == "started"
    wait_done(runner, "site_job")
    assert [p.args[3] for p in popen.procs] == ["ok", "ok"]                 # 只真的开了两次（db_b、site_job）


def test_two_db_buttons_block_each_other(act_env):
    s, runner, popen = act_env
    assert runner.start("db_a") == "started"
    assert runner.start("db_b") == "busy"
    assert runner.start("ok") == "started"                                  # 不写数据库的照常
    wait_done(runner, "db_a")
    wait_done(runner, "ok")
    assert runner.start("db_b") == "started"
    wait_done(runner, "db_b")


def test_repeated_clicks_run_once(asrv):
    srv, runner, popen = asrv
    h = srv.cookie()
    csrf = srv.page_csrf(h)
    results = []

    def one():
        r, body = srv.click("slow", h, csrf)
        results.append((r.status, json.loads(body)["result"]))

    threads = [threading.Thread(target=one) for _ in range(6)]             # 6 个请求同时来
    for t in threads:
        t.start()
    for t in threads:
        t.join(20)
    assert sorted(results) == [(202, "started")] + [(409, "running")] * 5
    assert len(popen.procs) == 1                                            # 只开了一个子程序
    r, body = srv.click("slow", h, csrf)                                    # 还在跑：再点也是“正在跑”
    assert r.status == 409 and json.loads(body)["text"] == "正在跑，这次没有再启动一个。"
    page = srv.req("GET", "/status", h)[1].decode("utf-8")
    assert re.search(r'data-action="slow"><form[^>]*><input[^>]*><button type="submit" disabled>', page)
    wait_done(runner, "slow")
    assert len(popen.procs) == 1


def test_timeout_ends_the_process(act_env):
    s, runner, popen = act_env
    began = time.monotonic()
    assert runner.start("hang") == "started"                                # 要睡 60 秒，最长只给 1.2 秒
    wait_done(runner, "hang", seconds=30)
    assert time.monotonic() - began < 30
    (p,) = popen.procs
    assert p.poll() is not None                                             # 子程序已经结束了
    st = next(a for a in runner.status()["actions"] if a["name"] == "hang")
    assert st["state"] == "error" and "已经结束了它" in st["text"]
    assert "超时，已结束" in Path(s["actions"]["log_file"]).read_text(encoding="utf-8")


def test_error_text_has_no_secrets_or_paths(asrv, tmp_path):
    srv, runner, popen = asrv
    h = srv.cookie()
    csrf = srv.page_csrf(h)
    assert srv.click("leak", h, csrf)[0].status == 202
    wait_done(runner, "leak")
    st = state_of(srv, h, "leak")
    assert st["state"] == "error" and st["exit_code"] == 3
    assert st["text"].endswith("（退出码 3）")
    page = srv.req("GET", "/status", h)[1].decode("utf-8")
    api = srv.req("GET", "/api/actions", h)[1].decode("utf-8")
    log = Path(make_settings(tmp_path)["actions"]["log_file"]).read_text(encoding="utf-8")
    assert "Traceback" not in api and "Traceback" not in page                # 网页上没有程序输出
    for text in (page, api, log):
        assert FAKE_TOKEN not in text and "hunter2" not in text and "abc123456" not in text
        assert str(tmp_path) not in text and str(tmp_path).replace("\\", "/") not in text
        assert "someone" not in text and str(Path.home()) not in text
    assert "Traceback" in log and "token=***" in log and "?***" in log     # 日志里有输出，但脱敏了
    assert "<项目>" in log and "退出码 3" in log


def test_judges_use_fixed_sentences(act_env):
    s, runner, popen = act_env
    for name in ("skip", "push"):
        assert runner.start(name) == "started"
        wait_done(runner, name)
    st = {a["name"]: a for a in runner.status()["actions"]}
    assert st["skip"]["state"] == "done" and "这次跳过了" in st["skip"]["text"]
    assert st["push"]["state"] == "error" and "设置微信推送.bat" in st["push"]["text"]
    assert A.judge_push(0, "【成功】已发送")[0] == "done"
    assert A.judge_push(0, "什么都没写")[0] == "error"
    assert A.judge_plain(1, "x") == ("error", "程序出错了，详细原因写在 logs/site_actions.log。")


# ---------- 页面：没有内联脚本、没有脚本也能用、总开关、静态模式 ----------

STATUS_PAGE = ("<!doctype html><html><head><meta charset='utf-8'><title>数据状态</title></head>"
               "<body><div class='wrap'><h1>数据状态</h1><p>概况</p><h2>各项检查</h2><p>STATUSBODY</p></div></body></html>")


def test_status_page_has_no_inline_script(asrv, tree):
    (tree / "reports" / "status.html").write_text(STATUS_PAGE, encoding="utf-8")
    srv, runner, popen = asrv
    h = srv.cookie()
    r, body = srv.req("GET", "/status", h)
    page = body.decode("utf-8")
    assert r.status == 200 and r.getheader("Content-Security-Policy", "").find("script-src 'self'") >= 0
    scripts = re.findall(r"<script\b[^>]*>(.*?)</script>", page, flags=re.S | re.I)
    assert scripts == [""]                                                  # 只有一个、而且是空的（src=）
    assert re.findall(r"<script\b[^>]*>", page, flags=re.I) == ['<script src="/static/site.js" defer>']
    assert page.index('<script src="/static/site.js"') < page.index("</head>")
    assert not re.search(r"\son[a-z]+\s*=", page, flags=re.I)               # 没有 onclick= 这类
    assert "javascript:" not in page.lower()
    assert "推荐" not in page
    # “操作”一块插在第一个小标题前面，原来的内容都还在
    assert page.index('id="actions"') < page.index("各项检查") and "STATUSBODY" in page
    assert page.count('class="act-form"') == len(FAKES)
    assert page.count('<button type="submit"') == len(FAKES)
    r, js = srv.req("GET", "/static/site.js", h)                            # 现读现出没起来，脚本照样能拿到
    assert r.status == 200 and b"/api/actions" in js


def test_plain_form_post_works_without_js(asrv):
    srv, runner, popen = asrv
    h = srv.cookie()
    csrf = srv.page_csrf(h)
    r, body = srv.click("ok", h, csrf, json_accept=False)
    assert r.status == 303 and r.getheader("Location") == "/status?done=started&action=ok#actions"
    wait_done(runner, "ok")
    page = srv.req("GET", "/status?done=started&action=ok", h)[1].decode("utf-8")
    assert "“假按钮ok”：已开始" in page and 'class="act-notice"' in page
    # 网址里乱写的东西不会搬到页面上
    for q in ("done=%3Cb%3Ex&action=ok", "done=started&action=%3Cscript%3E", "done=started&action=nope"):
        page = srv.req("GET", "/status?" + q, h)[1].decode("utf-8")
        assert 'class="act-notice"' not in page and "<b>x" not in page and "<script>" not in page


def test_missing_status_file_still_shows_buttons(asrv, tree):
    (tree / "reports" / "status.html").unlink()
    srv, runner, popen = asrv
    h = srv.cookie()
    r, body = srv.req("GET", "/status", h)
    page = body.decode("utf-8")
    assert r.status == 200 and "数据状态页还没生成" in page and page.count('class="act-form"') == len(FAKES)


def test_enabled_false_means_no_buttons_and_404(tree):
    s = copy.deepcopy(load_settings())
    for off in ({**s["actions"], "enabled": False}, None):
        t = copy.deepcopy(s)
        if off is None:
            t.pop("actions")
        else:
            t["actions"] = off
        assert A.runner_from_settings(t) is None
    # 服务器按设置自己造：总开关关着 → app.actions 是 None
    t = serve_settings(tree)                                                # 路径都在临时文件夹里
    t["paths"]["database"] = str(tree / "data" / "football.db")
    t["actions"]["enabled"] = False
    t["live"] = {**t["live"], "dynamic": True, "live_poller": False}
    srv = Running(make_config(tree, live_settings=t))
    assert srv.server.app.actions is None
    srv.server.app.live = None                                              # 现读现出这里不测
    with srv:
        h = srv.cookie()
        r, body = srv.req("GET", "/status", h)
        assert r.status == 200 and b"act-panel" not in body and b'name="csrf"' not in body
        assert body.startswith(b"<html><body>")                            # 原样的状态页（加了顶上那条导航）
        r, _ = srv.req("GET", "/api/actions", h)
        assert r.status == 404
        form = {**h, "Origin": f"http://127.0.0.1:{srv.port}", "Content-Type": "application/x-www-form-urlencoded",
                "Accept": "application/json"}
        for name in ("refresh_data", "build_site", "test_push", "predict"):
            r, body = srv.req("POST", f"/api/action/{name}", form, "csrf=" + "0" * 64)
            assert r.status == 404 and json.loads(body) == {"error": "not_found"}, name


def test_static_mode_unchanged(tree):
    with Running(make_config(tree)) as srv:                                 # live_settings 是 None：只发静态文件
        assert srv.server.app.actions is None
        h = srv.cookie()
        r, body = srv.req("GET", "/status", h)
        assert r.status == 200 and b"act-panel" not in body and b"/static/site.js" not in body
        assert "数据状态STATUS".encode("utf-8") in body
        form = {**h, "Origin": f"http://127.0.0.1:{srv.port}", "Content-Type": "application/x-www-form-urlencoded"}
        r, body = srv.req("POST", "/api/action/refresh_data", form, "csrf=" + "0" * 64)
        assert r.status == 405 and "这个网站只能看，不能改。" in body.decode("utf-8")
        assert r.getheader("Allow") == serve.ALLOW
        r, _ = srv.req("GET", "/api/actions", h)
        assert r.status == 404
        r, _ = srv.req("GET", "/static/site.js", h)
        assert r.status == 404


# ---------- 白名单、设置、脱敏 ----------

def test_whitelist_is_the_four_fixed_commands():
    got = {n: (a.module, a.args, a.writes_db, a.heartbeat) for n, a in A.ACTIONS.items()}
    assert got == {
        "refresh_data": ("src.jobs.refresh_data", (), True, None),
        "predict": ("src.jobs.scheduled_predict", (), True, None),
        "build_site": ("src.jobs.build_site", (), False, "site"),
        "test_push": ("src.notify.send", ("test",), False, "notify"),
    }
    for a in A.ACTIONS.values():
        assert importlib.util.find_spec(a.module) is not None, a.module   # 模块真的在（不运行它）
        assert A.NAME.fullmatch(a.name)
        assert "推荐" not in a.label + a.note
    assert A.ACTIONS["test_push"].judge is A.judge_push and A.ACTIONS["predict"].judge is A.judge_predict
    assert "推荐" not in "".join(A.RESULT_TEXT.values())


def test_real_settings_have_timeouts_for_every_button():
    cfg = load_settings()["actions"]
    assert cfg["enabled"] is True
    assert set(cfg["timeout_minutes"]) == set(A.ACTIONS)
    assert all(float(v) > 0 for v in cfg["timeout_minutes"].values())
    s = make_settings(Path("."))
    s["actions"]["timeout_minutes"] = {"ok": 1}
    with pytest.raises(ValueError):
        A.ActionRunner(s, actions={"ok": FAKES["ok"], "slow": FAKES["slow"]})
    s["actions"]["timeout_minutes"] = {"ok": 1, "slow": 0}
    with pytest.raises(ValueError):
        A.ActionRunner(s, actions={"ok": FAKES["ok"], "slow": FAKES["slow"]})


def test_scrub_masks_secrets_and_paths():
    root = Path(r"D:\proj\football")
    home = Path(r"C:\Users\alice")
    text = ("口令 pp-REAL-value-xyz 出现了；token=abcd1234 password: hunter2hunter2 api_key=\"k-9999\"\n"
            r"File D:\proj\football\src\x.py, D:/proj/football/data, C:\Users\alice\a.txt, C:\Users\bob\b.txt" "\n"
            "https://www.pushplus.plus/send?token=abcdef123&x=1 结束")
    out = A.scrub(text, {"pp-REAL-value-xyz"}, root=root, home=home)
    for bad in ("pp-REAL-value-xyz", "abcd1234", "hunter2", "k-9999", "abcdef123", "alice", "bob", r"D:\proj", "D:/proj"):
        assert bad not in out, bad
    assert "token=***" in out and "password: ***" in out and "https://www.pushplus.plus/send?***" in out
    assert r"<项目>\src\x.py" in out and "<项目>/data" in out and r"~\a.txt" in out and r"C:\Users\<用户>" in out
    assert A.scrub("普通的话 key 没有值", set(), root=root, home=home) == "普通的话 key 没有值"


def test_no_recommend_word_in_new_code():
    root = Path(__file__).resolve().parents[1]
    for path in ("src/web/actions.py", "src/web/static/site.js"):
        assert "推荐" not in (root / path).read_text(encoding="utf-8"), path
