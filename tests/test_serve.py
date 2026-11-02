"""局域网网站服务器的自动检查（src/web/serve.py、src/web/serve_auth.py，需求文档第 8 节“局域网网站”、第 16 节“安全”）。

真的在 127.0.0.1 的随机端口上起服务器，用临时文件夹当网站，测完马上关掉，不留进程。
密码哈希的迭代次数在测试里调小（1000 次），只为了跑得快；正式设置是 60 万次。
"""
import copy
import http.client
import json
import logging
import os
import socket
import threading
from pathlib import Path
from urllib.parse import urlencode

import pytest

from src.common.config import PROJECT_ROOT, load_settings
from src.web import serve, serve_auth

PASSWORD = "abcd-efgh-jkmn-pqrs"          # 测试用的假密码（自动生成的格式）
SECRET = "SECRET-OUTSIDE-SITE-7f3a"       # 放在网站文件夹外面的文件内容：任何响应里都不能出现


class Clock:
    """假时钟：测试里想让时间过去多久就过去多久。"""
    def __init__(self):
        self.t = 1_800_000_000.0

    def __call__(self):
        return self.t


@pytest.fixture
def tree(tmp_path):
    """临时的网站文件夹、状态页、密码文件，以及一个放在网站外面的“机密”文件。"""
    site = tmp_path / "site"
    (site / "match").mkdir(parents=True)
    (site / "team").mkdir()
    (site / "index.html").write_text("<!doctype html><html><head></head><body class='x'><h1>首页INDEX</h1>"
                                     "<a href='match/1.html'>m</a></body></html>", encoding="utf-8")
    (site / "match" / "1.html").write_text("<html><body>单场MATCH1 <a href='../index.html'>回</a></body></html>",
                                           encoding="utf-8")
    (site / "team" / "index.html").write_text("<html><body>TEAMINDEX</body></html>", encoding="utf-8")
    (site / "style.css").write_text("body{color:red}", encoding="utf-8")
    (site / "data.json").write_text('{"a": 1}', encoding="utf-8")
    (tmp_path / "secret.txt").write_text(SECRET, encoding="utf-8")
    (tmp_path / "sitex").mkdir()                        # 名字以 site 开头的“邻居”文件夹，也不能被读到
    (tmp_path / "sitex" / "s.txt").write_text(SECRET, encoding="utf-8")
    reports = tmp_path / "reports"
    reports.mkdir()
    (reports / "status.html").write_text("<html><body>数据状态STATUS</body></html>", encoding="utf-8")
    state = tmp_path / "state"
    state.mkdir()
    serve_auth.save_credentials(state / serve_auth.CRED_FILE,
                                serve_auth.make_record("admin", PASSWORD, 1000, generated=True, version=1))
    return tmp_path


def make_config(tree: Path, **kw) -> serve.ServerConfig:
    base = dict(site_dir=tree / "site", status_file=tree / "reports" / "status.html", state_dir=tree / "state",
                session_days=30, max_failures=5, lockout_minutes=15, max_login_body_bytes=4096,
                request_timeout_seconds=5)
    base.update(kw)
    return serve.ServerConfig(**base)


def make_settings(tree: Path, iterations: int = 1000) -> dict:
    """正式设置拷一份，把所有路径改到临时文件夹（绝对路径原样用，不会碰到真的 data/）。"""
    s = copy.deepcopy(load_settings())
    s["site"]["output_dir"] = str(tree / "site")
    s["site"]["review_dir"] = str(tree / "review")
    s["paths"]["reports"] = str(tree / "reports")
    ss = s["site_server"]
    ss["state_dir"] = str(tree / "state")
    ss["log_file"] = str(tree / "logs" / "site_server.log")
    ss["pbkdf2_iterations"] = iterations
    return s


class Running:
    """起一个服务器（后台线程），with 结束时关掉。"""
    def __init__(self, config, clock=None, host="127.0.0.1"):
        self.clock = clock or Clock()
        self.server = serve.create_server(config, host, 0, clock=self.clock)
        self.port = self.server.server_address[1]
        self.thread = threading.Thread(target=self.server.serve_forever, kwargs={"poll_interval": 0.05}, daemon=True)

    def __enter__(self):
        self.thread.start()
        return self

    def __exit__(self, *exc):
        self.server.shutdown()
        self.server.server_close()
        self.thread.join(5)
        assert not self.thread.is_alive()

    def req(self, method, path, headers=None, body=None):
        c = http.client.HTTPConnection("127.0.0.1", self.port, timeout=10)
        try:
            c.request(method, path, body=body, headers=headers or {})
            r = c.getresponse()
            data = r.read()
            return r, data
        finally:
            c.close()

    def login(self, password=PASSWORD, username="admin", next_url=None, origin=True, ctype=None):
        form = {"username": username, "password": password}
        if next_url is not None:
            form["next"] = next_url
        h = {"Content-Type": ctype or "application/x-www-form-urlencoded"}
        if origin:
            h["Origin"] = f"http://127.0.0.1:{self.port}"
        return self.req("POST", "/login", h, urlencode(form))

    def cookie(self):
        """登录成功，返回请求头里要带的 Cookie。"""
        r, _ = self.login()
        assert r.status == 303
        return {"Cookie": r.getheader("Set-Cookie").split(";", 1)[0]}


@pytest.fixture
def srv(tree):
    with Running(make_config(tree)) as s:
        yield s


# ---------- 登录、会话 ----------

def test_not_logged_in_redirects_to_login_with_next(srv):
    for path in ("/", "/match/1.html", "/status", "/nope.html", "/../secret.txt"):
        r, body = srv.req("GET", path)
        assert r.status == 303, path
        assert r.getheader("Location").startswith("/login?next=")
        assert SECRET.encode() not in body
    r, _ = srv.req("GET", "/match/1.html?x=1")
    assert r.getheader("Location") == "/login?next=%2Fmatch%2F1.html%3Fx%3D1"
    r, body = srv.req("GET", "/login?next=%2Fmatch%2F1.html")
    assert r.status == 200
    text = body.decode("utf-8")
    assert "登录" in text and 'type="password"' in text and 'value="/match/1.html"' in text
    assert "viewport" in text                                   # 手机上按屏幕宽度排版
    assert r.getheader("Cache-Control") == "no-store"


def test_wrong_password_no_cookie(srv):
    r, body = srv.login("wrong-password")
    assert r.status == 200
    assert r.getheader("Set-Cookie") is None
    assert "账号或密码不对，还能再试 4 次" in body.decode("utf-8")
    r, _ = srv.login(PASSWORD, username="root")                # 账号错也不行
    assert r.status == 200 and r.getheader("Set-Cookie") is None


def test_right_password_gets_cookie_and_returns_to_page(srv):
    r, _ = srv.login(next_url="/match/1.html")
    assert r.status == 303
    assert r.getheader("Location") == "/match/1.html"
    sc = r.getheader("Set-Cookie")
    assert sc.startswith(serve.COOKIE + "=")
    for attr in ("HttpOnly", "SameSite=Strict", "Path=/", f"Max-Age={30 * 86400}"):
        assert attr in sc
    h = {"Cookie": sc.split(";", 1)[0]}
    r, body = srv.req("GET", "/match/1.html", h)
    assert r.status == 200 and "单场MATCH1".encode() in body
    assert b'href="/logout"' in body and b'href="/status"' in body        # 顶上插了“退出登录”“数据状态”
    assert b"href='../index.html'" in body                                 # 原来的相对链接不动
    r, body = srv.req("GET", "/", h)
    assert r.status == 200 and "首页INDEX".encode() in body
    assert body.index(b"lan-bar") > body.index(b"<body")                   # 插在 <body> 后面
    r, body = srv.req("GET", "/style.css", h)
    assert r.status == 200 and r.getheader("Content-Type") == "text/css; charset=utf-8" and body == b"body{color:red}"
    r, body = srv.req("GET", "/data.json", h)
    assert r.getheader("Content-Type").startswith("application/json")
    r, body = srv.req("GET", "/team/", h)                                  # 文件夹只给 index.html，不列目录
    assert r.status == 200 and b"TEAMINDEX" in body
    r, _ = srv.req("GET", "/login", h)                                     # 已经登录再开登录页：直接回首页
    assert r.status == 303 and r.getheader("Location") == "/"


def test_generated_password_ignores_case_and_dashes(srv):
    r, _ = srv.login(PASSWORD.replace("-", "").upper())
    assert r.status == 303
    r, _ = srv.login(" " + PASSWORD.replace("-", " ") + " ")
    assert r.status == 303


def test_status_page(srv, tree):
    h = srv.cookie()
    r, body = srv.req("GET", "/status", h)
    assert r.status == 200 and "数据状态STATUS".encode() in body and b"lan-bar" in body
    (tree / "reports" / "status.html").unlink()
    r, body = srv.req("GET", "/status", h)
    assert r.status == 404 and "数据状态页还没生成".encode() in body


def test_review_reports_served_as_text(tree):
    """战绩页的周报、月报链接（../review/reports/xx.md）：登录后能看，按纯文本发；只给这个文件夹里的 .md。"""
    rep = tree / "review" / "reports"
    rep.mkdir(parents=True)
    (rep / "周报_2026-W45.md").write_text("# 周报W45", encoding="utf-8")
    (rep / "x.html").write_text("<html><body>不该给</body></html>", encoding="utf-8")
    with Running(make_config(tree, review_reports_dir=rep)) as s:
        assert s.req("GET", "/review/reports/x.md")[0].status == 303      # 没登录先去登录
        h = s.cookie()
        r, body = s.req("GET", "/review/reports/%E5%91%A8%E6%8A%A5_2026-W45.md", h)
        assert r.status == 200 and r.getheader("Content-Type") == "text/plain; charset=utf-8"
        assert "# 周报W45".encode() in body and b"lan-bar" not in body
        for bad in ("/review/reports/x.html", "/review/reports/../../secret.txt",
                    "/review/reports/%2e%2e/%2e%2e/secret.txt", "/review/reports/nope.md"):
            r, body = s.req("GET", bad, h)
            assert r.status == 404 and SECRET.encode() not in body, bad


def test_logout(srv):
    h = srv.cookie()
    assert srv.req("GET", "/", h)[0].status == 200
    r, _ = srv.req("GET", "/logout", h)
    assert r.status == 303 and r.getheader("Location") == "/login"
    assert "Max-Age=0" in r.getheader("Set-Cookie")
    assert srv.req("GET", "/", h)[0].status == 303          # 旧 cookie 不能再用


def test_session_expires(srv):
    h = srv.cookie()
    srv.clock.t += 29 * 86400
    assert srv.req("GET", "/", h)[0].status == 200
    srv.clock.t += 2 * 86400                                 # 一共过了 31 天
    assert srv.req("GET", "/", h)[0].status == 303


def test_forged_or_garbage_cookie_rejected(srv):
    for c in (f"{serve.COOKIE}=" + "A" * 43, f"{serve.COOKIE}=../../x", f"{serve.COOKIE}=", "other=1"):
        assert srv.req("GET", "/", {"Cookie": c})[0].status == 303


def test_sessions_survive_restart_and_store_only_hashes(tree):
    cfg = make_config(tree)
    with Running(cfg) as s:
        h = s.cookie()
        clock = s.clock
    token = h["Cookie"].split("=", 1)[1]
    stored = (tree / "state" / serve_auth.SESSIONS_FILE).read_text(encoding="utf-8")
    assert token not in stored                               # 会话文件里只有哈希
    with Running(cfg, clock=clock) as s:                     # “电脑重启”后还是登录状态
        assert s.req("GET", "/", h)[0].status == 200


def test_password_change_logs_everyone_out(srv, tree):
    h = srv.cookie()
    settings = make_settings(tree)
    serve_auth.set_password(settings, "new-password-123", generated=False)
    rec = serve_auth.load_credentials(tree / "state" / serve_auth.CRED_FILE)
    assert rec["version"] == 2 and rec["generated"] is False
    os.utime(tree / "state" / serve_auth.CRED_FILE, ns=(1, 1))   # 确保服务器发现文件变了（不靠时间精度）
    assert srv.req("GET", "/", h)[0].status == 303
    assert srv.login(PASSWORD)[0].status == 200                     # 旧密码不行了
    assert srv.login("NEW-PASSWORD-123")[0].status == 200           # 自己设的密码要分大小写
    assert srv.login("new-password-123")[0].status == 303


# ---------- 输错锁定 ----------

def test_lockout_after_failures(srv):
    for i in range(4):
        r, _ = srv.login("bad")
        assert r.status == 200
    r, body = srv.login("bad")                     # 第 5 次：锁定
    assert r.status == 429 and int(r.getheader("Retry-After")) > 0
    assert "分钟后再试" in body.decode("utf-8")
    r, _ = srv.login(PASSWORD)                     # 锁着的时候密码对也不行
    assert r.status == 429 and r.getheader("Set-Cookie") is None
    srv.clock.t += 15 * 60 + 1
    r, _ = srv.login(PASSWORD)                     # 锁定时间过了就能登录
    assert r.status == 303


def test_lockout_is_per_ip_and_success_resets(tree):
    app = serve.SiteApp(make_config(tree, max_failures=3), clock=Clock())
    assert app.check_login("1.1.1.1", "admin", "x") == ("bad", 2)
    assert app.check_login("1.1.1.1", "admin", "x") == ("bad", 1)
    assert app.check_login("1.1.1.1", "admin", "x")[0] == "locked"
    assert app.check_login("2.2.2.2", "admin", PASSWORD) == ("ok", 0)     # 别的 IP 不受影响
    assert app.check_login("3.3.3.3", "admin", "x") == ("bad", 2)
    assert app.check_login("3.3.3.3", "admin", PASSWORD) == ("ok", 0)     # 登录成功清零
    assert app.check_login("3.3.3.3", "admin", "x") == ("bad", 2)


# ---------- 路径穿越 ----------

TRAVERSAL = [
    "/../secret.txt", "/match/../../secret.txt", "/%2e%2e/secret.txt", "/%2E%2E/secret.txt",
    "/%2e%2e%2fsecret.txt", "/match/%2e%2e%2f%2e%2e%2fsecret.txt", "/..%2fsecret.txt", "/..%5csecret.txt",
    "/%5c..%5csecret.txt", "/match/..%5c..%5csecret.txt", "/%252e%252e/secret.txt", "/../sitex/s.txt",
    "/%2e%2e/sitex/s.txt", "/./index.html", "/match//1.html", "/index.html%00.css", "/index.html.",
    "/index.html%20", "/index.html::$DATA", "/C:/Windows/win.ini", "/%c0%ae%c0%ae/secret.txt",
    "/CON", "/nul.html", "/match/", "/no_such.html", "/" + "%2e%2e/" * 10 + "secret.txt",
]


@pytest.mark.parametrize("path", TRAVERSAL)
def test_path_traversal_blocked(srv, path):
    r, body = srv.req("GET", path, srv.cookie())
    assert r.status in (400, 404), path
    assert SECRET.encode() not in body


def test_absolute_form_and_backslash_paths_blocked(srv):
    h = srv.cookie()
    for path in ("http://127.0.0.1/secret.txt", "//evil.example/x", "\\..\\secret.txt", "*"):
        r, body = srv.req("GET", path, h)
        assert r.status in (400, 404), path
        assert SECRET.encode() not in body


def test_symlink_pointing_outside_blocked(srv, tree):
    try:
        os.symlink(tree / "secret.txt", tree / "site" / "link.txt")
        os.symlink(tree / "sitex", tree / "site" / "linkdir", target_is_directory=True)
    except (OSError, NotImplementedError):
        pytest.skip("这台电脑不让普通用户建符号链接（Windows 要开发者模式）；resolve_static 的真实路径检查另有单元测试")
    h = srv.cookie()
    for path in ("/link.txt", "/linkdir/s.txt"):
        r, body = srv.req("GET", path, h)
        assert r.status == 404 and SECRET.encode() not in body


@pytest.mark.skipif(os.name != "nt", reason="目录联接（junction）只有 Windows 有")
def test_junction_pointing_outside_blocked(srv, tree):
    # Windows 的“目录联接”普通用户就能建，效果和指向文件夹的符号链接一样
    import subprocess
    link = tree / "site" / "junc"
    proc = subprocess.run(["cmd", "/c", "mklink", "/J", str(link), str(tree / "sitex")], capture_output=True)
    if proc.returncode != 0 or not (link / "s.txt").exists():
        pytest.skip("建不了目录联接")
    r, body = srv.req("GET", "/junc/s.txt", srv.cookie())
    assert r.status == 404 and SECRET.encode() not in body
    assert serve.resolve_static(tree / "site", "/junc/s.txt") is None


def test_resolve_static_unit(tree):
    site = tree / "site"
    assert serve.resolve_static(site, "/index.html") == Path(os.path.realpath(site / "index.html"))
    assert serve.resolve_static(site, "/") == Path(os.path.realpath(site / "index.html"))
    assert serve.resolve_static(site, "/match/1.html") is not None
    for bad in TRAVERSAL + ["/%ff.html", "/match", "/team"]:
        got = serve.resolve_static(site, bad)
        assert got is None or os.path.normcase(str(got)).startswith(os.path.normcase(os.path.realpath(site)) + os.sep), bad
    assert serve.resolve_static(site, "/../secret.txt") is None
    assert serve.resolve_static(site, "/match/") is None              # 没有 index.html 的文件夹：404，不列目录


def test_site_folder_missing_returns_503(srv, tree):
    h = srv.cookie()
    (tree / "site").rename(tree / "site_old")                        # “生成网站”正在换新文件夹
    r, body = srv.req("GET", "/", h)
    assert r.status == 503 and r.getheader("Retry-After") == "5"


# ---------- 请求方法、请求头 ----------

def test_methods(srv):
    h = srv.cookie()
    for m in ("PUT", "DELETE", "PATCH", "OPTIONS", "TRACE"):
        r, _ = srv.req(m, "/index.html", h)
        assert r.status == 405 and r.getheader("Allow") == serve.ALLOW, m
    r, _ = srv.req("POST", "/index.html", {**h, "Content-Type": "application/x-www-form-urlencoded"}, "a=1")
    assert r.status == 405
    r, _ = srv.req("FOO", "/", h)
    assert r.status == 501
    r, body = srv.req("HEAD", "/", h)
    assert r.status == 200 and body == b"" and int(r.getheader("Content-Length")) > 0


def test_security_headers_everywhere(srv):
    h = srv.cookie()
    for method, path, hh in (("GET", "/", h), ("GET", "/login", {}), ("GET", "/x.html", h), ("PUT", "/", h),
                             ("GET", "/", {})):
        r, _ = srv.req(method, path, hh)
        assert r.getheader("X-Content-Type-Options") == "nosniff"
        assert r.getheader("X-Frame-Options") == "DENY"
        assert "frame-ancestors 'none'" in r.getheader("Content-Security-Policy")
        assert r.getheader("Referrer-Policy")
        assert r.getheader("Cache-Control") == "no-store"
        assert "Python" not in (r.getheader("Server") or "")


def test_login_post_guards(srv):
    r, _ = srv.req("POST", "/login", {"Content-Type": "application/x-www-form-urlencoded",
                                       "Origin": "http://evil.example"}, urlencode({"username": "admin", "password": PASSWORD}))
    assert r.status == 403 and r.getheader("Set-Cookie") is None              # 别的网站帮你提交
    r, _ = srv.login(ctype="application/json")
    assert r.status == 415 and r.getheader("Set-Cookie") is None
    r, _ = srv.req("POST", "/login", {"Content-Type": "application/x-www-form-urlencoded"}, "x=" + "a" * 5000)
    assert r.status == 413
    r, _ = srv.login(origin=False)                                             # 没带 Origin（老浏览器）照样能登
    assert r.status == 303


@pytest.mark.parametrize("nxt", ["//evil.example/", "http://evil.example/", "/\\evil.example", "javascript:alert(1)",
                                 "/login", "/logout", "/x\r\nSet-Cookie: a=b"])
def test_login_next_cannot_leave_site(srv, nxt):
    r, _ = srv.login(next_url=nxt)
    assert r.status == 303 and r.getheader("Location") == "/"


def test_login_page_escapes_input(srv):
    r, body = srv.req("GET", "/login?next=" + "/%22%3E%3Cscript%3Ex")
    assert b"<script>" not in body
    r, body = srv.login("bad", username='<img src=x>')
    assert b"<img src=x>" not in body and b"&lt;img" in body


# ---------- 没有密码、端口占用 ----------

def test_no_password_refuses_non_loopback(tree):
    (tree / "state" / serve_auth.CRED_FILE).unlink()
    cfg = make_config(tree)
    for host in ("0.0.0.0", "192.168.1.10", "::"):
        with pytest.raises(serve.PasswordRequired):
            serve.create_server(cfg, host, 0)
    # 只在本机地址上可以起，但谁都登不进
    with Running(cfg) as s:
        r, body = s.login()
        assert r.status == 503 and r.getheader("Set-Cookie") is None
        assert s.req("GET", "/")[0].status == 303


def test_broken_credentials_not_overwritten(tree):
    p = tree / "state" / serve_auth.CRED_FILE
    p.write_text("{坏了", encoding="utf-8")
    with pytest.raises(serve_auth.CredentialsError):
        serve_auth.ensure_credentials(make_settings(tree))
    assert p.read_text(encoding="utf-8") == "{坏了"
    with pytest.raises(serve.PasswordRequired):
        serve.create_server(make_config(tree), "0.0.0.0", 0)


def test_is_loopback():
    assert serve.is_loopback("127.0.0.1") and serve.is_loopback("::1") and serve.is_loopback("localhost")
    assert not serve.is_loopback("0.0.0.0") and not serve.is_loopback("192.168.1.2") and not serve.is_loopback("pc.local")


def test_run_second_instance_exits_and_other_program_reported(tree):
    settings = make_settings(tree)
    holder = {}
    t = threading.Thread(target=lambda: holder.setdefault("rc", serve.run(settings, "127.0.0.1", 0,
                                                                          ready=lambda s: holder.setdefault("srv", s))),
                         daemon=True)
    t.start()
    try:
        for _ in range(100):
            if "srv" in holder:
                break
            threading.Event().wait(0.05)
        port = holder["srv"].server_address[1]
        assert serve_auth.probe_server(port)
        assert serve.run(settings, "127.0.0.1", port) == 0          # 自己已经在运行：不开第二个，正常退出
    finally:
        holder["srv"].shutdown()
        t.join(5)
    assert holder["rc"] == 0 and not t.is_alive()
    # 端口被别的程序占着：返回 2
    other = socket.socket()
    try:
        if hasattr(socket, "SO_EXCLUSIVEADDRUSE"):
            other.setsockopt(socket.SOL_SOCKET, socket.SO_EXCLUSIVEADDRUSE, 1)
        other.bind(("127.0.0.1", 0))
        other.listen(1)
        assert serve.run(settings, "127.0.0.1", other.getsockname()[1]) == 2
    finally:
        other.close()
        for h in list(serve.LOG.handlers):                          # 测试里加的日志文件关掉，免得占着临时文件夹
            serve.LOG.removeHandler(h)
            h.close()


def test_logs_have_no_password_or_cookie(tree):
    records = []
    handler = logging.Handler(logging.DEBUG)
    handler.emit = lambda rec: records.append(rec.getMessage())
    old_level = serve.LOG.level
    serve.LOG.addHandler(handler)
    serve.LOG.setLevel(logging.DEBUG)
    try:
        with Running(make_config(tree)) as s:
            s.login("wrong-guess-zzz")
            h = s.cookie()
            s.req("GET", "/match/1.html?password=" + PASSWORD, h)
            s.req("GET", "/%2e%2e/secret.txt", h)
    finally:
        serve.LOG.removeHandler(handler)
        serve.LOG.setLevel(old_level)
    text = "\n".join(records)
    token = h["Cookie"].split("=", 1)[1]
    assert "/match/1.html" in text                                  # 路径照记
    for leak in (PASSWORD, PASSWORD.replace("-", ""), "wrong-guess-zzz", token, serve.COOKIE + "="):
        assert leak not in text


# ---------- 密码哈希、自动生成、txt ----------

def test_hash_and_verify():
    rec = serve_auth.make_record("admin", "Secret-Pass-1", 2000, generated=False, version=1)
    assert "Secret-Pass-1" not in json.dumps(rec)
    assert rec["algorithm"] == "pbkdf2_sha256" and rec["iterations"] == 2000 and len(bytes.fromhex(rec["salt"])) == 16
    assert serve_auth.verify_password("Secret-Pass-1", rec)
    assert not serve_auth.verify_password("secret-pass-1", rec)
    assert not serve_auth.verify_password("", rec)
    rec2 = serve_auth.make_record("admin", "Secret-Pass-1", 2000, generated=False, version=1)
    assert rec2["salt"] != rec["salt"] and rec2["hash"] != rec["hash"]          # 加了盐：同一个密码哈希也不一样
    for broken in ({**rec, "hash": "zz"}, {**rec, "algorithm": "md5"}, {**rec, "iterations": 0}, {}):
        assert not serve_auth.verify_password("Secret-Pass-1", broken)
    import hashlib
    expect = hashlib.pbkdf2_hmac("sha256", b"Secret-Pass-1", bytes.fromhex(rec["salt"]), 2000).hex()
    assert rec["hash"] == expect                                                  # 和标准算法一致


def test_generate_password_format():
    pws = {serve_auth.generate_password(4, 4) for _ in range(200)}
    assert len(pws) == 200
    for pw in pws:
        groups = pw.split("-")
        assert len(groups) == 4 and all(len(g) == 4 for g in groups)
        assert set(pw.replace("-", "")) <= set(serve_auth.ALPHABET)
    assert not set("0o1li") & set(serve_auth.ALPHABET)


def test_settings_block_is_complete():
    ss = load_settings()["site_server"]
    assert ss["port"] == 8765 and ss["host"] == "0.0.0.0" and ss["pbkdf2_iterations"] >= 600000
    cfg = serve.ServerConfig.from_settings(load_settings())
    assert cfg.site_dir.name == "site" and cfg.status_file.name == "status.html"
    assert ss["state_dir"].startswith("data/")                     # 密码、会话都在 data/ 下（不进 Git）
    gitignore = (PROJECT_ROOT / ".gitignore").read_text(encoding="utf-8").splitlines()
    assert "data/" in gitignore and "logs/" in gitignore


def test_ensure_credentials_and_info_file(tree):
    settings = make_settings(tree)
    (tree / "state" / serve_auth.CRED_FILE).unlink()
    rec, plain = serve_auth.ensure_credentials(settings)
    assert plain and serve_auth.verify_password(plain, rec) and rec["username"] == "admin"
    assert plain not in (tree / "state" / serve_auth.CRED_FILE).read_text(encoding="utf-8")
    again, plain2 = serve_auth.ensure_credentials(settings)        # 第二次不重新生成
    assert plain2 is None and again["hash"] == rec["hash"]
    path = serve_auth.write_info_file(settings, plain=plain, port=8765, addresses=("192.168.1.23", ["10.0.0.5"]))
    raw = path.read_bytes()
    assert raw.startswith(b"\xef\xbb\xbf")                          # 记事本打开不乱码
    text = raw.decode("utf-8-sig")
    assert "http://192.168.1.23:8765" in text and "http://10.0.0.5:8765" in text
    assert "账号：admin" in text and f"密码：{plain}" in text and "公共 Wi-Fi" in text
    # 下次刷新地址（IP 变了）：密码从旧 txt 里接着用（对得上哈希）
    text2 = serve_auth.write_info_file(settings, port=8765, addresses=("192.168.1.99", [])).read_text(encoding="utf-8-sig")
    assert "http://192.168.1.99:8765" in text2 and f"密码：{plain}" in text2 and "192.168.1.23" not in text2
    # 有人把 txt 里的密码改错了：不显示错的密码
    path.write_text(text2.replace(plain, "aaaa-bbbb-cccc-dddd"), encoding="utf-8-sig")
    text3 = serve_auth.write_info_file(settings, port=8765, addresses=(None, [])).read_text(encoding="utf-8-sig")
    assert "aaaa-bbbb-cccc-dddd" not in text3 and "这里没有记录" in text3 and "没找到这台电脑的局域网地址" in text3


def test_cmd_change(tree):
    settings = make_settings(tree)
    cred = tree / "state" / serve_auth.CRED_FILE
    before = cred.read_text(encoding="utf-8")
    answers = iter(["short"])
    assert serve_auth.cmd_change(settings, ask=lambda _p: next(answers)) == 1      # 太短
    answers = iter(["long-enough-1", "long-enough-2"])
    assert serve_auth.cmd_change(settings, ask=lambda _p: next(answers)) == 1      # 两次不一样
    answers = iter([" padded-password "])
    assert serve_auth.cmd_change(settings, ask=lambda _p: next(answers)) == 1      # 前后有空格
    assert cred.read_text(encoding="utf-8") == before
    answers = iter(["My-New-Pass-9", "My-New-Pass-9"])
    assert serve_auth.cmd_change(settings, ask=lambda _p: next(answers)) == 0
    rec = serve_auth.load_credentials(cred)
    assert rec["version"] == 2 and serve_auth.verify_password("My-New-Pass-9", rec)
    assert "密码：My-New-Pass-9" in serve_auth.info_path(settings).read_text(encoding="utf-8-sig")
    assert serve_auth.cmd_change(settings, ask=lambda _p: "") == 0                 # 直接回车：随机生成
    rec = serve_auth.load_credentials(cred)
    assert rec["version"] == 3 and rec["generated"] is True
    m = serve_auth._PW_LINE.search(serve_auth.info_path(settings).read_text(encoding="utf-8-sig"))
    assert m and serve_auth.verify_password(m.group(1), rec)


def test_safe_next_and_inject_bar():
    assert serve.safe_next("/match/1.html?x=1") == "/match/1.html?x=1"
    for bad in (None, "", "x", "//a", "/\\a", "/a\nb", "/login?next=/"):
        assert serve.safe_next(bad) == "/"
    assert serve.inject_bar(b"no body tag") == b"no body tag"
    out = serve.inject_bar(b"<HTML><BODY class='a'>x</BODY>")
    assert out.startswith(b"<HTML><BODY class='a'><style>") and out.endswith(b"x</BODY>")


# ---------- 定时任务、防火墙脚本、双击文件（只看内容，不运行） ----------

def test_register_task_and_firewall_scripts():
    reg = (PROJECT_ROOT / "src/jobs/register_tasks.ps1").read_bytes()
    assert reg.startswith(b"\xef\xbb\xbf")                           # 带 BOM，否则中文任务名会乱码
    text = reg.decode("utf-8-sig")
    assert "足球预测_网站服务" in text and "-m src.web.serve" in text and "-AtLogOn" in text
    block = text[text.index("# 局域网网站服务"):]
    assert "$pythonw" in block and "IgnoreNew" in block and "-RestartCount" in block
    fw = (PROJECT_ROOT / "src/jobs/lan_access.ps1").read_bytes()
    assert fw.startswith(b"\xef\xbb\xbf")
    fw = fw.decode("utf-8-sig")
    assert "-Profile Private" in fw and "-Protocol TCP" in fw and "-LocalPort $port" in fw
    assert "RunAs" in fw and "Get-NetConnectionProfile" in fw
    assert "Set-NetConnectionProfile" not in fw                      # 不替用户改网络类型
    assert "Get-NetFirewallRule -DisplayName $rule" in fw            # 先找同名规则，重复运行不重复加
    for name, needle in (("开启局域网访问.bat", "lan_access.ps1"), ("关闭局域网访问.bat", "-Close"),
                         ("修改网站密码.bat", "serve_auth change"), ("查看网站地址和密码.bat", "serve_auth show")):
        raw = (PROJECT_ROOT / name).read_bytes()
        assert not raw.startswith(b"\xef\xbb\xbf") and b"\r\n" in raw, name
        assert "chcp 65001" in raw.decode("utf-8") and needle in raw.decode("utf-8"), name


def test_firewall_info_is_ascii_json(capsys):
    assert serve_auth.cmd_firewall_info(load_settings()) == 0
    out = capsys.readouterr().out
    assert out.isascii()
    info = json.loads(out)
    assert info == {"port": 8765, "rule": "足球预测_局域网网站", "task": "足球预测_网站服务"}
