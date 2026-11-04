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
from datetime import timedelta
from pathlib import Path

import pytest

from src.web import live_site, serve
from tests.test_serve import Clock, Running, make_config, tree  # noqa: F401（tree 是 pytest 的 fixture）
from tests.test_site import NOW, env, write_review  # noqa: F401（env 是 pytest 的 fixture）

CSP = ("default-src 'self'; script-src 'self'; style-src 'self' 'unsafe-inline'; img-src 'self' data:; "
       "connect-src 'self'; frame-ancestors 'none'; base-uri 'none'; form-action 'self'")


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
    s["live"] = {"dynamic": True, "meta_refresh": True, "cache_seconds": 60, "min_reload_seconds": 60, "wait_seconds": 10,
                 "stale_seconds": 600, "retry_seconds": 15, "live_poller": False, "poller_stop_seconds": 5}
    dq = s.get("dongqiudi") or {}
    s["dongqiudi"] = {**dq, "live": {**(dq.get("live") or {}), "output": str(tmp / "live" / "scores.json")}}
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
        refresh = int(s["site"]["refresh_minutes"]) * 60
        for path in ("/", "/index.html", "/match/1.html", "/match/10.html", f"/team/{team}", "/record.html",
                     "/lifecycle.html", "/bots.html"):
            r, body = srv.req("GET", path, h)
            html = body.decode("utf-8")
            assert r.status == 200, path
            assert r.getheader("Content-Security-Policy") == CSP, path
            assert '<script src="/static/site.js" defer></script>' in html, path
            assert html.lower().count("<script") == 1, path          # 只有这一个，没有内联脚本
            # 这里的 world 故意写了 live.meta_refresh: true：另外按 site.refresh_minutes 整页重新加载
            #（settings.yaml 现在是 false，那种情况见 test_meta_refresh_switch_off 和 test_site_js.py）
            assert f'<meta http-equiv="refresh" content="{refresh}">' in html, path
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

def write_fetch_log(db):
    """模拟即时比分程序往数据库写一行抓取记录（数据库的修改时间变了，但网页用的内容没变）。"""
    con = sqlite3.connect(db)
    con.execute("insert into fetch_log (url, source, data_type, fetched_at_utc, status) values ('u', 'dongqiudi', 'live', "
                "'2026-11-06 12:00:00', 'saved')")
    con.commit()
    con.close()


def test_cache_reload_rules(world):
    s, tmp = world
    clock, loader = Clock(), CountingLoader()
    s["live"]["cache_seconds"], s["live"]["min_reload_seconds"] = 120, 30
    live = make_live(s, tmp, clock, loader)
    db = s["paths"]["database"]
    first = live.snapshot()
    assert first is not None and loader.calls == 1
    write_fetch_log(db)
    clock.t += 10
    assert live.snapshot() is first and loader.calls == 1          # 文件变了，但离上次读不到 min_reload_seconds（30）：不读
    clock.t += 21
    second = live.snapshot()
    assert second is not first and loader.calls == 2               # 过了 30 秒、文件变过：重新读
    assert second.version == first.version                         # 只多了抓取记录，内容没变：数据版本不变
    clock.t += 31
    assert live.snapshot() is second and loader.calls == 2         # 文件没再变、没到 cache_seconds（120）：不读
    con = sqlite3.connect(db)
    con.execute("update teams set name_zh = '阿森纳队' where team_id = 'Arsenal'")
    con.commit()
    con.close()
    third = live.snapshot()
    assert loader.calls == 3 and third.version != second.version   # 内容真的变了：数据版本跟着变
    clock.t += 31
    (tmp / "review" / "matches" / "10.json").write_text("{}", encoding="utf-8")   # 复盘文件夹多了一个文件
    assert live.snapshot() is not third and loader.calls == 4
    clock.t += 121
    fourth = live.snapshot()
    assert loader.calls == 5 and fourth.stamp == live.fingerprint()   # 超过 cache_seconds：文件没变也重新读一次


def test_data_version_is_content_hash(world):
    """数据版本只看内容：同一个数据库读两次（“现在”差一分钟）一样；和文件的修改时间无关。"""
    s, tmp = world
    times = iter([NOW, NOW + timedelta(minutes=1)])
    live = live_site.LiveSite(s, tmp / "site", clock=Clock(), now=lambda: next(times))
    a = live._build(live.fingerprint())
    write_fetch_log(s["paths"]["database"])
    b = live._build(live.fingerprint())
    assert a.version == b.version and len(a.version) == 16 and a.stamp != b.stamp


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


def test_stop_live_poller_waits_with_timeout(monkeypatch, caplog):
    """服务器关掉时：听话的线程马上停；正在联网、不理 stop 的线程最多等 timeout 秒，不挡着退出。"""
    stop = threading.Event()
    assert serve.stop_live_poller(None, stop, 5) is True and stop.is_set()
    good_stop = threading.Event()
    fake = types.ModuleType("src.collect.dongqiudi_live")
    fake.run_forever = lambda settings, ev: ev.wait(30)
    monkeypatch.setitem(sys.modules, "src.collect.dongqiudi_live", fake)
    t = serve.start_live_poller({"live": {"live_poller": True}}, good_stop)
    started = time.monotonic()
    assert serve.stop_live_poller(t, good_stop, 5) is True and not t.is_alive()
    assert time.monotonic() - started < 3
    release = threading.Event()
    fake.run_forever = lambda settings, ev: release.wait(30)          # 不理 stop（比如正在等懂球帝回话）
    bad_stop = threading.Event()
    t = serve.start_live_poller({"live": {"live_poller": True}}, bad_stop)
    started = time.monotonic()
    with caplog.at_level(logging.WARNING, logger="site_server"):
        assert serve.stop_live_poller(t, bad_stop, 0.3) is False
    assert 0.25 <= time.monotonic() - started < 3 and t.daemon and "没停下" in caplog.text
    release.set()
    t.join(5)


# ---------- 网页自动刷新 ----------

def test_static_and_dynamic_pages_keep_meta_refresh(world):
    """静态页照旧带 meta refresh、没有脚本；现读现出的页在 live.meta_refresh: true 时也带，refresh_minutes 是 0 时不带。"""
    s, tmp = world
    from src.web.site.common import LiveMinutes, page
    minutes = int(s["site"]["refresh_minutes"])
    tag = f'<meta http-equiv="refresh" content="{minutes * 60}">'
    static = page("t", "<p>x</p>", "", NOW, minutes)
    assert tag in static and "<script" not in static.lower() and "/static/site.css" not in static
    live = page("t", "<p>x</p>", "", NOW, LiveMinutes(minutes))
    assert tag in live and live.lower().count("<script") == 1 and "现场从数据库读出来生成的" in live
    off = page("t", "<p>x</p>", "", NOW, LiveMinutes(0))
    assert "http-equiv" not in off and '<script src="/static/site.js" defer></script>' in off
    # 真的生成一次静态首页（和“生成网站”同一个函数）：有 meta refresh、没有脚本、带 data-*
    data = live_site.load_from_db(s, Path(s["paths"]["database"]), NOW)[0]
    from src.web.site import pages
    from src.web.site.common import TeamFiles
    teams = TeamFiles({data.matches[m][k] for m in data.page_ids for k in ("home_team", "away_team")})
    html = pages.render_index(data, teams)
    assert tag in html and "<script" not in html.lower() and "data-match-id='10'" in html


def test_meta_refresh_switch_off(world):
    """live.meta_refresh: false（settings.yaml 现在的设置，site.js 自己更新）：现读现出的页面不带 meta refresh，静态页照旧。"""
    s, tmp = world
    s["live"]["meta_refresh"] = False
    with LiveRunning(s, tmp) as srv:
        h = srv.cookie()
        for path in ("/", "/match/10.html", "/record.html", "/bots.html", "/lifecycle.html"):
            html = srv.req("GET", path, h)[1].decode("utf-8")
            assert "http-equiv" not in html and '<script src="/static/site.js" defer></script>' in html, path


# ---------- 刚启动、出错 ----------

def test_first_load_wait_has_timeout(world):
    """一份数据都没有、正在读的那个很慢：别的请求最多等 wait_seconds 秒，然后网页发静态文件、接口回 503。"""
    s, tmp = world
    s["live"]["wait_seconds"] = 0.3
    gate = threading.Event()

    def slow(settings, db_path, now):
        gate.wait(10)
        return live_site.load_from_db(settings, db_path, now)

    with LiveRunning(s, tmp, loader=slow) as srv:
        warm = threading.Thread(target=srv.live.snapshot, daemon=True)
        warm.start()                                            # 像服务器启动时那样先在后台读
        time.sleep(0.2)
        h = srv.cookie()
        started = time.monotonic()
        r, body = srv.req("GET", "/", h)
        assert r.status == 200 and "首页INDEX".encode() in body and time.monotonic() - started < 3
        assert get_json(srv, "/api/matches", h) == (503, {"error": "data_unavailable"})
        assert get_json(srv, "/api/match/1", h) == (503, {"error": "data_unavailable"})
        assert get_json(srv, "/api/version", h)[1]["mode"] == "static"
        gate.set()
        warm.join(10)
        r, body = srv.req("GET", "/", h)
        assert "首页INDEX".encode() not in body and "现场从数据库读出来生成的" in body.decode("utf-8")


def test_live_site_init_error_falls_back_to_static(world, monkeypatch, caplog):
    """建 LiveSite 时出错（例如设置写错）：服务器照样起来，只发静态文件，日志记一行。"""
    s, tmp = world

    class Broken:
        def __init__(self, *a, **k):
            raise KeyError("site")

    monkeypatch.setattr(live_site, "LiveSite", Broken)
    with caplog.at_level(logging.WARNING, logger="site_server"):
        with Running(make_config(tmp, live_settings=s)) as srv:
            assert srv.server.app.live is None
            h = srv.cookie()
            r, body = srv.req("GET", "/", h)
            assert r.status == 200 and "首页INDEX".encode() in body
            assert srv.req("GET", "/api/version", h)[0].status == 404
    assert "现读现出起不来（KeyError）" in caplog.text


@pytest.mark.parametrize("text", [
    "[" * 100_000 + "]" * 100_000,                                                   # 套得太深（json 自己会 RecursionError）
    '{"status": "ok", "matches": ' + "[" * 50 + "]" * 50 + "}",                     # 套得比约定深
    '{"status": "ok", "matches": [{"dqd_id": 9010, "minute": 1e999}]}',             # 无穷大
    '{"status": "ok", "matches": [{"dqd_id": 9010, "minute": NaN}]}',
    '{"status": "ok", "matches": [{"dqd_id": 9010, "minute": -Infinity}]}',
    chr(0xFEFF) * 2 + "{}",                                                         # 两个 BOM：不是 JSON
])
def test_broken_scores_file_is_error_not_static(world, text):
    """scores.json 怎么坏都只算“读不了”（status error），网页照样现读现出，不退回静态文件。"""
    s, tmp = world
    p = tmp / "live" / "scores.json"
    p.parent.mkdir(exist_ok=True)
    p.write_text(text, encoding="utf-8")
    with LiveRunning(s, tmp) as srv:
        h = srv.cookie()
        status, body = get_json(srv, "/api/live", h)
        assert status == 200 and body["status"] == "error" and body["matches"] == [] and body["match_ids"] == {}
        r, page = srv.req("GET", "/", h)
        assert r.status == 200 and "首页INDEX".encode() not in page and "现场从数据库读出来生成的" in page.decode("utf-8")


def test_scores_file_location_from_dongqiudi_live_output(world):
    s, tmp = world
    assert live_site.scores_path(s) == tmp / "live" / "scores.json"
    assert live_site.scores_path({}).parts[-3:] == ("data", "live", "scores.json")       # 没写时的默认位置
    assert live_site.scores_path({"dongqiudi": {"live": {}}}).parts[-3:] == ("data", "live", "scores.json")
    assert "scores_file" not in live_site.DEFAULTS


def test_csp_on_static_only_server_too(world):
    """live.dynamic: false 只发静态文件时，CSP 也是同一条。"""
    _s, tmp = world
    with Running(make_config(tmp)) as srv:
        r, _ = srv.req("GET", "/login")
        assert r.getheader("Content-Security-Policy") == CSP
        r, _ = srv.req("GET", "/", srv.cookie())
        assert r.getheader("Content-Security-Policy") == CSP
