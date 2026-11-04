"""网站“现读现出”和数据接口的自动检查（src/web/live_site.py、src/web/serve.py，需求文档第 8 节“现读现出”、第 16 节“安全”）。

用 test_site.py 造的临时数据库和临时 data/review/（不碰真实数据库），test_serve.py 的临时网站文件夹、密码文件，
在 127.0.0.1 的随机端口上起服务器，测完马上关掉。生成网页时的“现在”固定成 test_site.NOW（2026-11-06 12:00 UTC）。
"""
import json
import logging
import os
import sqlite3
import sys
import threading
import time
import types

import pytest

from src.web import live_site, serve
from tests.test_serve import Clock, Running, make_config, tree  # noqa: F401（tree 是 pytest 的 fixture）
from tests.test_site import NOW, env, write_review  # noqa: F401（env 是 pytest 的 fixture）

CSP = ("default-src 'self'; script-src 'self'; style-src 'self' 'unsafe-inline'; img-src 'self' data:; "
       "connect-src 'self'; frame-ancestors 'none'")


@pytest.fixture
def world(env, tree):
    """临时数据库（比赛 1 踢完 2–1、10 和 11 还没踢、20 是欧冠）+ 复盘文件 + 网站文件夹 + 密码。
    比赛 1、10 在 match_sources 里记上懂球帝编号 9001、9010（即时比分靠它对上）。"""
    db, s, tmp = env
    write_review(tmp)
    con = sqlite3.connect(db)
    con.executemany("insert into match_sources values (?, ?, ?, 'x.json', '2026-11-01 00:00:00')",
                    [(1, "dongqiudi", "9001"), (10, "dongqiudi", "9010")])
    con.commit()
    con.close()
    s["paths"]["database"] = str(db)
    s["live"] = {"dynamic": True, "cache_seconds": 60, "stale_seconds": 600, "retry_seconds": 15,
                 "scores_file": str(tmp / "live" / "scores.json"), "live_poller": False}
    return s, tmp


class CountingLoader:
    """包一层真的 load_from_db：数一数读了几次；fail = True 时故意出错（模拟数据库被锁）。"""
    def __init__(self):
        self.calls = 0
        self.fail = False

    def __call__(self, settings, db_path, now):
        self.calls += 1
        if self.fail:
            raise sqlite3.OperationalError("database is locked")
        return live_site.load_from_db(settings, db_path, now)


def make_live(s, tmp, clock, loader=None):
    return live_site.LiveSite(s, tmp / "site", clock=clock, now=lambda: NOW, loader=loader)


class LiveRunning(Running):
    """起一个现读现出的服务器：LiveSite 换成“现在”固定的那个（loader 可以换成会出错的）。"""
    def __init__(self, s, tmp, loader=None, clock=None):
        super().__init__(make_config(tmp, live_settings=s), clock=clock)
        self.live = make_live(s, tmp, self.clock, loader)
        self.server.app.live = self.live


def get_json(srv, path, h):
    r, body = srv.req("GET", path, h)
    assert r.getheader("Content-Type") == "application/json; charset=utf-8"
    assert r.getheader("Cache-Control") == "no-store"
    return r.status, json.loads(body.decode("utf-8"))


def write_scores(tmp, matches, status="ok", age=0):
    p = tmp / "live" / "scores.json"
    p.parent.mkdir(exist_ok=True)
    p.write_text(json.dumps({"updated_at": "2026-11-06T12:00:00+00:00", "status": status, "note": "",
                             "matches": matches}, ensure_ascii=False), encoding="utf-8")
    t = time.time() - age
    os.utime(p, (t, t))
    return p


PLAYING = {"dqd_id": 9010, "league": "E0", "home": "利物浦", "away": "曼城", "kickoff_utc": "2026-11-07T10:00:00+00:00",
           "state": "Playing", "minute": "63", "home_goals": 2, "away_goals": 1, "home_red": 0, "away_red": 1}


# ---------- 数据接口 ----------

def test_api_needs_login_401_json(world):
    s, tmp = world
    with LiveRunning(s, tmp) as srv:
        for path in ("/api/version", "/api/live", "/api/matches", "/api/match/1", "/api/nothing"):
            status, body = get_json(srv, path, {})
            assert status == 401 and body == {"error": "not_logged_in"}, path
        r, _ = srv.req("GET", "/static/site.js")
        assert r.status == 303                                   # 脚本文件也要先登录
        r, _ = srv.req("GET", "/match/1.html")
        assert r.status == 303 and r.getheader("Location").startswith("/login")   # 网页还是跳登录页


def test_api_version_and_matches(world):
    s, tmp = world
    with LiveRunning(s, tmp) as srv:
        h = srv.cookie()
        status, v = get_json(srv, "/api/version", h)
        assert status == 200 and v["mode"] == "live" and len(v["data_version"]) == 16
        assert v["generated_at"] == "2026-11-06T12:00:00+00:00"
        status, body = get_json(srv, "/api/matches", h)
        assert status == 200 and body["data_version"] == v["data_version"]
        up = {m["match_id"]: m for m in body["upcoming"]}
        rec = {m["match_id"]: m for m in body["recent"]}
        assert set(up) == {10, 11} and set(rec) == {1}            # 欧冠的 20 不在首页上，接口里也没有
        m10 = up[10]
        assert m10["home"] == "利物浦" and m10["away"] == "曼城" and m10["league"] == "E0" and m10["league_name"] == "英超"
        assert m10["kickoff_utc"] == "2026-11-07T10:00:00+00:00" and m10["page"] == "/match/10.html"
        assert m10["latest"]["kind"] == "T-24" and abs(sum(m10["latest"]["probs"]) - 1) < 1e-6
        assert up[11]["latest"] is None
        assert rec[1]["score"] == [2, 1]
        text = json.dumps(body, ensure_ascii=False)
        assert str(tmp) not in text and "\\\\" not in text      # 不带电脑上的路径


def test_api_match_detail(world):
    s, tmp = world
    with LiveRunning(s, tmp) as srv:
        h = srv.cookie()
        status, m = get_json(srv, "/api/match/1", h)
        assert status == 200 and m["match_id"] == 1 and m["score"] == [2, 1]
        kinds = [v["kind"] for v in m["versions"]]
        assert kinds == ["T-24", "T-6", "T-1", "preview"]                  # 预览版只留最新的那一行（和单场页一样）
        t24 = m["versions"][0]
        assert t24["top_scores"] and len(t24["top_scores"][0]["scores"]) == s["site"]["top_scores"]
        h0, a0, p0 = t24["top_scores"][0]["scores"][0]
        assert isinstance(h0, int) and isinstance(a0, int) and 0 < p0 < 1
        assert set(m["picks"]["pick"]) == {"1x2", "ah", "ou", "score"} == set(m["picks"]["model"])
        one = m["picks"]["pick"]["1x2"]
        assert one["text"] and one["outcome"] in ("win", "loss")        # 踢完了：写中没中
        assert m["latest"]["kind"] == "T-1"
        assert get_json(srv, "/api/match/999", h) == (404, {"error": "not_found"})
        assert get_json(srv, "/api/match/20", h)[0] == 404             # 欧冠这场没有单场页
        assert get_json(srv, "/api/matchx", h)[0] == 404


def test_api_live_missing_normal_stale_broken(world):
    s, tmp = world
    with LiveRunning(s, tmp) as srv:
        h = srv.cookie()
        assert get_json(srv, "/api/live", h) == (200, {"matches": [], "status": "off", "match_ids": {}})
        other = {"dqd_id": 5555, "league": "E0", "state": "Playing"}           # 对不上我们的比赛
        wrong = {**PLAYING, "dqd_id": 9001, "league": "SP1"}                   # 编号对上了，联赛对不上
        write_scores(tmp, [PLAYING, other, wrong])
        status, body = get_json(srv, "/api/live", h)
        assert status == 200 and body["status"] == "ok" and body["stale"] is False
        assert body["matches"] == [PLAYING, other, wrong]                      # 原样转发
        assert body["match_ids"] == {"9010": 10}
        assert body["updated_at"] == "2026-11-06T12:00:00+00:00" and body["age_seconds"] < 60
        write_scores(tmp, [PLAYING], age=1000)
        assert get_json(srv, "/api/live", h)[1]["stale"] is True             # 超过 stale_seconds（600）没更新
        p = tmp / "live" / "scores.json"
        p.write_text("{坏了", encoding="utf-8")
        status, body = get_json(srv, "/api/live", h)
        assert status == 200 and body["status"] == "error" and body["matches"] == [] and body["stale"] is True


def test_live_score_shown_on_pages(world):
    s, tmp = world
    with LiveRunning(s, tmp) as srv:
        h = srv.cookie()
        write_scores(tmp, [PLAYING, {**PLAYING, "dqd_id": 9001, "home_goals": 5}])
        _, page = srv.req("GET", "/", h)
        html = page.decode("utf-8")
        assert "进行中 63&#x27;" in html and "红牌 0–1" in html
        assert "5–1" not in html                       # 比赛 1 已经有正式赛果 2–1，不用即时比分
        assert "data-match-id='10' data-league='E0' data-kickoff='2026-11-07T10:00:00+00:00'" in html
        _, page = srv.req("GET", "/match/10.html", h)
        assert "进行中 63&#x27;" in page.decode("utf-8")
        write_scores(tmp, [PLAYING], status="off")
        _, page = srv.req("GET", "/", h)
        assert "进行中" not in page.decode("utf-8")     # WD3 说关了：不显示


# ---------- 动态网页 ----------

def test_dynamic_pages_csp_script_no_advice(world):
    s, tmp = world
    with LiveRunning(s, tmp) as srv:
        h = srv.cookie()
        snap = srv.live.snapshot()
        team = next(iter(snap.team_by_file))
        for path in ("/", "/index.html", "/match/1.html", "/match/10.html", f"/team/{team}", "/record.html",
                     "/lifecycle.html", "/bots.html"):
            r, body = srv.req("GET", path, h)
            html = body.decode("utf-8")
            assert r.status == 200, path
            assert r.getheader("Content-Security-Policy") == CSP, path
            assert '<script src="/static/site.js" defer></script>' in html, path
            assert html.lower().count("<script") == 1, path          # 只有这一个，没有内联脚本
            assert "http-equiv" not in html, path                   # 不自己整页重新加载
            assert "推荐" not in html, path
            assert "现场从数据库读出来生成的" in html and "退出登录" in html, path
            assert "首页INDEX" not in html                          # 不是静态文件
        r, body = srv.req("GET", "/lifecycle.json", h)
        assert r.status == 200 and r.getheader("Content-Type").startswith("application/json")
        json.loads(body)
        r, _ = srv.req("GET", "/match/999.html", h)
        assert r.status == 404
        r, body = srv.req("GET", "/style.css", h)                  # 不归现读现出管的：还发静态文件
        assert r.status == 200 and body == b"body{color:red}"
        r, _ = srv.req("GET", "/login")
        assert r.getheader("Content-Security-Policy") == CSP


def test_static_files_and_traversal(world):
    s, tmp = world
    with LiveRunning(s, tmp) as srv:
        h = srv.cookie()
        r, body = srv.req("GET", "/static/site.js", h)
        assert r.status == 200 and r.getheader("Content-Type") == "text/javascript; charset=utf-8"
        assert b"use strict" in body
        r, _ = srv.req("GET", "/static/site.css", h)
        assert r.status == 200 and r.getheader("Content-Type") == "text/css; charset=utf-8"
        for bad in ("/static/../serve.py", "/static/%2e%2e/serve.py", "/static/..%2fserve.py", "/static/..\\serve.py",
                    "/static/%5c..%5cserve.py", "/static/C:/Windows/win.ini", "/static/nul", "/static/site.js.",
                    "/static/../../config/secrets.yaml", "/static/", "/static/x.js"):
            r, body = srv.req("GET", bad, h)
            assert r.status in (400, 404), bad
            assert b"LiveSite" not in body and b"import" not in body, bad


# ---------- 内存里的数据：什么时候重新读 ----------

def test_cache_reload_rules(world):
    s, tmp = world
    clock, loader = Clock(), CountingLoader()
    live = make_live(s, tmp, clock, loader)
    first = live.snapshot()
    assert first is not None and loader.calls == 1
    clock.t += 30
    assert live.snapshot() is first and loader.calls == 1          # 60 秒以内、文件没变：不重新读
    clock.t += 31
    second = live.snapshot()
    assert second is not first and loader.calls == 2               # 超过 cache_seconds：重新读
    db = s["paths"]["database"]
    st = os.stat(db)
    os.utime(db, ns=(st.st_atime_ns, st.st_mtime_ns + 5_000_000_000))   # 数据库改了（修改时间变了）
    third = live.snapshot()
    assert loader.calls == 3 and third.version != second.version   # 不等 60 秒，马上重新读
    (tmp / "review" / "matches" / "10.json").write_text("{}", encoding="utf-8")   # 复盘文件夹多了一个文件
    assert live.snapshot().version != third.version and loader.calls == 4


def test_concurrent_requests_load_once(world):
    s, tmp = world
    gate, calls = threading.Event(), []

    def slow(settings, db_path, now):
        calls.append(1)
        gate.wait(5)
        return live_site.load_from_db(settings, db_path, now)

    live = make_live(s, tmp, Clock(), slow)
    got = []
    threads = [threading.Thread(target=lambda: got.append(live.snapshot())) for _ in range(5)]
    for t in threads:
        t.start()
    time.sleep(0.3)
    gate.set()
    for t in threads:
        t.join(10)
    assert len(calls) == 1 and len(got) == 5 and all(g is got[0] for g in got)


def test_failure_keeps_old_data_then_static(world, caplog):
    s, tmp = world
    clock, loader = Clock(), CountingLoader()
    live = make_live(s, tmp, clock, loader)
    first = live.snapshot()
    loader.fail = True
    clock.t += 61
    with caplog.at_level(logging.WARNING, logger="site_server"):
        assert live.snapshot() is first                          # 读失败：用上一次的
    assert "database is locked" in caplog.text and str(tmp) not in caplog.text
    assert live.version_info()["mode"] == "stale"
    calls = loader.calls
    clock.t += 5
    assert live.snapshot() is first and loader.calls == calls    # retry_seconds（15）以内不再试
    clock.t += 11
    live.snapshot()
    assert loader.calls == calls + 1                             # 过了 15 秒再试一次
    loader.fail = False
    clock.t += 16
    assert live.snapshot() is not first and live.version_info()["mode"] == "live"   # 好了就换新的


def test_no_data_at_all_serves_static_files(world):
    s, tmp = world
    loader = CountingLoader()
    loader.fail = True
    with LiveRunning(s, tmp, loader=loader) as srv:
        h = srv.cookie()
        r, body = srv.req("GET", "/", h)
        assert r.status == 200 and "首页INDEX".encode() in body          # 静态文件（和以前一样）
        r, body = srv.req("GET", "/match/1.html", h)
        assert r.status == 200 and "单场MATCH1".encode() in body
        status, v = get_json(srv, "/api/version", h)
        assert status == 200 and v["mode"] == "static" and v["data_version"].startswith("static-")
        assert get_json(srv, "/api/matches", h) == (503, {"error": "data_unavailable"})
        assert get_json(srv, "/api/match/1", h) == (503, {"error": "data_unavailable"})


def test_render_error_falls_back_to_static_page(world, monkeypatch, caplog):
    s, tmp = world
    with LiveRunning(s, tmp) as srv:
        h = srv.cookie()

        def boom(*a, **k):
            raise RuntimeError("坏了 C:\\Users\\someone\\x")

        monkeypatch.setattr(live_site.match, "render", boom)
        with caplog.at_level(logging.WARNING, logger="site_server"):
            r, body = srv.req("GET", "/match/1.html", h)
        assert r.status == 200 and "单场MATCH1".encode() in body
        assert "RuntimeError" in caplog.text and "someone" not in caplog.text   # 日志不带错误里的路径
        r, body = srv.req("GET", "/", h)
        assert "首页INDEX".encode() not in body                               # 别的页照常现读现出


# ---------- 开关 ----------

def test_dynamic_false_is_static_only(world):
    s, tmp = world
    s["live"]["dynamic"] = False
    cfg = serve.ServerConfig.from_settings({**s, "site": {**s["site"], "output_dir": str(tmp / "site")}})
    assert cfg.live_settings is None
    with Running(make_config(tmp)) as srv:
        assert srv.server.app.live is None
        r, _ = srv.req("GET", "/api/version")
        assert r.status == 303                                   # 和以前一样：没登录跳登录页
        h = srv.cookie()
        r, body = srv.req("GET", "/", h)
        assert "首页INDEX".encode() in body
        for path in ("/api/version", "/static/site.js"):
            r, _ = srv.req("GET", path, h)
            assert r.status == 404, path                         # 没有数据接口、没有 /static/


def test_from_settings_reads_live_block(world):
    s, _tmp = world
    assert serve.ServerConfig.from_settings(s).live_settings is s
    blk = live_site.settings_live({"live": {"cache_seconds": 5}})
    assert blk["cache_seconds"] == 5 and blk["stale_seconds"] == 600 and blk["dynamic"] is True


def test_live_poller_thread(monkeypatch, caplog):
    stop = threading.Event()
    assert serve.start_live_poller({"live": {"live_poller": False}}, stop) is None
    seen = []
    fake = types.ModuleType("src.collect.dongqiudi_live")
    fake.run_forever = lambda settings, ev: seen.append((settings["x"], ev))
    monkeypatch.setitem(sys.modules, "src.collect.dongqiudi_live", fake)
    t = serve.start_live_poller({"live": {"live_poller": True}, "x": 1}, stop)
    t.join(5)
    assert t.daemon and seen == [(1, stop)]
    monkeypatch.setitem(sys.modules, "src.collect.dongqiudi_live", None)   # 导入失败
    with caplog.at_level(logging.WARNING, logger="site_server"):
        assert serve.start_live_poller({"live": {"live_poller": True}}, stop) is None
    assert "即时比分程序加载不了" in caplog.text
