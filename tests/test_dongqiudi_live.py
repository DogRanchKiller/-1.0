"""实时比分（src/collect/dongqiudi_live.py）的自动检查。全部用临时文件夹里造的数据库和假网站，不联网、不碰真数据库。"""
import json
import sqlite3
import threading
from datetime import datetime, timedelta, timezone

import pytest

from src.collect import dongqiudi_live as live
from src.collect.dongqiudi import Blocked
from src.collect.dongqiudi_live import FetchFailed, LiveFetcher, parse_live, tick, write_scores
from src.common.db import connect

UTC = timezone.utc
T0 = datetime(2026, 11, 8, 15, 0, tzinfo=UTC)   # 一个周六下午（标准时间）开球


# ---------- 造数据 ----------
def make_db(path, games):
    """造一个只有 matches、match_sources 两张表（只放用得到的列）的数据库。
    games：[(懂球帝编号, 联赛, 开球时间, 状态, 开球时间精确到)]。"""
    con = sqlite3.connect(path)
    con.execute("create table matches (match_id integer primary key, competition text, kickoff_utc text,"
                " status text, date_precision text)")
    con.execute("create table match_sources (match_id integer, source text, source_key text)")
    for i, (dqd, lg, ko, status, precision) in enumerate(games, 1):
        con.execute("insert into matches values (?,?,?,?,?)", (i, lg, ko.strftime("%Y-%m-%d %H:%M"), status, precision))
        con.execute("insert into match_sources values (?, 'dongqiudi', ?)", (i, str(dqd)))
    con.commit()
    con.close()


def game(dqd, lg="E0", ko=T0, status="scheduled", precision="day"):
    return (dqd, lg, ko, status, precision)


def settings_for(tmp_path, games, **live_cfg):
    make_db(tmp_path / "t.db", games)
    return {"paths": {"database": str(tmp_path / "t.db"), "raw": str(tmp_path / "raw")},
            "dongqiudi": {"min_interval_seconds": 0, "league_ids": {"E0": 4, "SP1": 3},
                          "live": {"before_minutes": 5, "after_minutes": 150, "every_seconds": 120, "pause_minutes": 60,
                                   "show_played_minutes": 300, "idle_check_minutes": 10,
                                   "output": str(tmp_path / "live" / "scores.json"), **live_cfg}}}


def item(mid, ko=T0, status="Playing", score=("", ""), minute="", extra="", period="", red=("", ""),
         home="阿森纳", away="切尔西", events=(None, None)):
    """懂球帝赛程接口里的一场比赛（只放用得到的字段，和原始层里真实的字段名一样）。"""
    return {"match_id": str(mid), "start_play": ko.strftime("%Y-%m-%d %H:%M:%S"), "status": status,
            "team_A_name": home, "team_B_name": away, "fs_A": score[0], "fs_B": score[1], "gameweek": "11",
            "minute": minute, "minute_extra": extra, "minute_period": period, "rc_A": red[0], "rc_B": red[1],
            "team_A_events": events[0], "team_B_events": events[1]}


def ev(code, title="某球员 50'"):
    """事件列表里的一项（真实例子：{"title": "马特塔 13'", "code": "RC", ...}）。"""
    return {"title": title, "logo": "", "color": "", "code": code, "sort": 50000}


class FakeFetcher:
    """假的请求程序：pages 里每个联赛给一份 JSON（或者一个要抛出的错误）。记下每次请求，自己记“上次请求”和“被拒绝”。"""
    def __init__(self, pages=None):
        self.pages = pages or {}
        self.calls, self.blocks = [], []

    def last_request(self, league):
        times = [t for lg, t in self.calls if lg == league]
        return max(times) if times else None

    def last_block(self, since):
        hits = [b for b in self.blocks if b[0] >= since]
        return max(hits) if hits else None

    def get(self, league, now):
        self.calls.append((league, now))
        r = self.pages.get(league, {"list": []})
        if isinstance(r, Exception):
            if isinstance(r, Blocked):
                self.blocks.append((now, str(r)))
            raise r
        return r

    def close(self):
        pass


def read_out(s):
    with open(s["dongqiudi"]["live"]["output"], encoding="utf-8") as f:
        return json.load(f)


# ---------- 什么时候请求 ----------
def test_no_match_in_window_no_request(tmp_path):
    """没有比赛在“开球前 5 分钟到开球后 150 分钟”里：一次都不请求，status=off，文件照写。"""
    s = settings_for(tmp_path, [game(1, ko=T0), game(2, "SP1", ko=T0 + timedelta(days=1)),
                                game(3, ko=T0 - timedelta(hours=1), status="finished"),          # 数据库里已经踢完
                                game(4, ko=T0 - timedelta(hours=1), precision="round")])         # 开球时间是占位的
    f = FakeFetcher()
    for now in (T0 - timedelta(minutes=6), T0 + timedelta(minutes=151)):
        r = tick(s, now, f)
        assert r["requests"] == 0 and r["status"] == "off"
    assert f.calls == []
    out = read_out(s)
    assert out["status"] == "off" and out["matches"] == [] and "不请求" in out["note"]
    # 下一次该看的时间：下一场开球前 5 分钟，最多隔 10 分钟
    r = tick(s, T0 - timedelta(minutes=30), f)
    assert r["next_check_utc"] == T0 - timedelta(minutes=20) and f.calls == []
    r = tick(s, T0 - timedelta(minutes=8), f)
    assert r["next_check_utc"] == T0 - timedelta(minutes=5)


def test_window_edges(tmp_path):
    """开球前 5 分钟、开球后 150 分钟都算在窗口里。"""
    s = settings_for(tmp_path, [game(1, ko=T0)])
    f = FakeFetcher({"E0": {"list": [item(1, status="Fixture")]}})
    assert tick(s, T0 - timedelta(minutes=5), f)["requests"] == 1
    f2 = FakeFetcher({"E0": {"list": [item(1, minute="90", extra="5", period="2H")]}})
    assert tick(s, T0 + timedelta(minutes=150), f2)["requests"] == 1


def test_throttle_per_league(tmp_path):
    """两个联赛都有比赛在踢：每个联赛每 120 秒最多请求一次；没请求的那一轮接着显示上一份。"""
    s = settings_for(tmp_path, [game(1, ko=T0), game(2, "SP1", ko=T0), game(3, "SP1", ko=T0 + timedelta(minutes=10))])
    f = FakeFetcher({"E0": {"list": [item(1, score=("1", "0"), minute="20", period="1H")]},
                     "SP1": {"list": [item(2, score=("0", "0"), minute="20", period="1H", home="皇马", away="巴萨"),
                                      item(3, ko=T0 + timedelta(minutes=10), status="Fixture", home="塞维利亚", away="贝蒂斯")]}})
    now = T0 + timedelta(minutes=20)
    r = tick(s, now, f)
    assert r["requests"] == 2 and r["requested"] == ["E0", "SP1"] and r["next_check_utc"] == now + timedelta(seconds=120)
    r = tick(s, now + timedelta(seconds=119), f)
    assert r["requests"] == 0 and r["waiting"] == ["E0", "SP1"] and r["status"] == "ok"
    assert r["next_check_utc"] == now + timedelta(seconds=120)
    assert [g["dqd_id"] for g in read_out(s)["matches"]] == [1, 2, 3]   # 没请求也接着显示上一份
    r = tick(s, now + timedelta(seconds=120), f)
    assert r["requests"] == 2
    assert [lg for lg, _ in f.calls] == ["E0", "SP1", "E0", "SP1"]


def test_real_fetcher_throttle_uses_fetch_log(tmp_path):
    """真的 LiveFetcher（假网站）：上一次请求时间从 fetch_log 读，按联赛分开；不存原始文件；data_type 是 fixtures_live。"""
    from tests.test_dongqiudi import FakeSite
    s = settings_for(tmp_path, [game(1, ko=T0), game(2, "SP1", ko=T0)])
    site, clock = FakeSite(), [T0 + timedelta(minutes=30)]
    for lid, mid in ((4, 1), (3, 2)):
        site.pages[live.live_url(lid, clock[0], live.live_cfg(s))] = (200, {"list": [item(mid, score=("2", "1"), minute="30")]})

    def make():
        return LiveFetcher(s, log_db=tmp_path / "log.db", session=site, sleep=lambda x: None, now=lambda: clock[0])
    assert tick(s, clock[0], make())["requests"] == 2
    clock[0] += timedelta(seconds=60)
    assert tick(s, clock[0], make())["requests"] == 0   # 新的 LiveFetcher（像重新启动的程序）也记得上一次
    con = sqlite3.connect(tmp_path / "log.db")
    rows = con.execute("select source, data_type, status, http_status, file_path, url from fetch_log").fetchall()
    assert [(r[0], r[1], r[2], r[3], r[4]) for r in rows] == [("dongqiudi", "fixtures_live", "saved", 200, None)] * 2
    assert "start=2026-11-08%2000%3A00%3A00" in rows[0][5]   # 开球后 150 分钟以前那天的 0 点：同一个赛程接口
    assert not (tmp_path / "raw").exists()                     # 没存原始文件
    assert len(site.requests) == 2


def test_live_url_previous_day_after_midnight():
    """标准时间凌晨 1 点：start 取前一天 0 点，昨天深夜开球、还在踢的比赛也在这一页。"""
    url = live.live_url(4, datetime(2026, 11, 9, 1, 0, tzinfo=UTC), live.DEFAULTS)
    assert url.startswith("https://api.dongqiudi.com/data/tab/league/new/4?start=2026-11-08%2000%3A00%3A00")


# ---------- 读懂球帝的返回 ----------
def test_parse_states_score_minute_red():
    data = {"list": [
        # 红牌在事件列表里（rc_A / rc_B 在真实数据里都是空的）：RC 直接红牌、Y2C 两黄变一红都算，进球不算
        item(1, score=("2", "1"), minute="63", period="2H", events=([ev("G"), ev("G")], [ev("OG"), ev("Y2C")])),
        item(2, score=("0", "0"), minute="45", period="HT"),
        item(3, score=("1", "1"), minute="45", extra="2", period="1H", events=([ev("RC"), ev("G")], None)),
        item(4, status="Played", score=("3", "0"), minute="90", period="FT", red=("2", "0")),   # 以后 rc 字段有数也认
        item(5, status="Fixture", ko=T0 + timedelta(hours=2)),
        item(6, status="Postponed"),                       # 延期：不显示，但状态表里有
        {"match_id": "坏的", "status": "Playing"},           # 编号不对：跳过，不报错
    ]}
    games, states = parse_live(data, "E0")
    by = {g["dqd_id"]: g for g in games}
    assert set(by) == {1, 2, 3, 4, 5} and states[6] == "Postponed"
    assert by[1] == {"dqd_id": 1, "league": "E0", "home": "阿森纳", "away": "切尔西", "kickoff_utc": "2026-11-08T15:00:00+00:00",
                     "state": "Playing", "minute": "63", "home_goals": 2, "away_goals": 1, "home_red": 0, "away_red": 1}
    assert by[2]["minute"] == "中场" and by[3]["minute"] == "45+2" and by[3]["home_red"] == 1
    assert (by[4]["state"], by[4]["minute"], by[4]["home_goals"], by[4]["away_goals"], by[4]["home_red"]) == ("Played", "", 3, 0, 2)
    assert (by[5]["state"], by[5]["minute"], by[5]["home_goals"], by[5]["away_goals"]) == ("Fixture", "", None, None)
    assert by[5]["kickoff_utc"] == "2026-11-08T17:00:00+00:00"


def test_played_stops_requests_and_stays_shown(tmp_path):
    """懂球帝写踢完以后，这个联赛不再请求；踢完的比赛开球后 300 分钟内还显示，之后就不显示了。"""
    s = settings_for(tmp_path, [game(1, ko=T0), game(2, ko=T0, status="scheduled")])
    f = FakeFetcher({"E0": {"list": [item(1, status="Played", score=("2", "2"), period="FT"),
                                     item(2, status="Postponed")]}})
    now = T0 + timedelta(minutes=115)
    assert tick(s, now, f)["requests"] == 1
    r = tick(s, now + timedelta(minutes=5), f)            # 新的一轮（内存不带过来）也从文件里记得 1 踢完；2 延期不在文件里
    assert r["requests"] == 1                              # 2 还要再确认一次（只在内存里记“延期”）
    memory = {}
    tick(s, now + timedelta(minutes=10), f, memory)
    assert tick(s, now + timedelta(minutes=13), f, memory)["requests"] == 0   # 同一个后台线程：都记住了
    out = read_out(s)
    assert out["status"] == "ok" and [(g["dqd_id"], g["state"]) for g in out["matches"]] == [(1, "Played")]
    r = tick(s, T0 + timedelta(minutes=301), f, memory)
    assert r["status"] == "off" and read_out(s)["matches"] == []


def test_fetch_failed_keeps_previous(tmp_path):
    """一个联赛这一轮没拿到（不是被拒绝）：只跳过它，接着显示上一份，下一轮再试；别的联赛照常。"""
    s = settings_for(tmp_path, [game(1, ko=T0), game(2, "SP1", ko=T0)])
    now = T0 + timedelta(minutes=30)
    good = {"E0": {"list": [item(1, score=("1", "0"), minute="30")]},
            "SP1": {"list": [item(2, score=("0", "1"), minute="30")]}}
    tick(s, now, FakeFetcher(good))
    r = tick(s, now + timedelta(minutes=3), FakeFetcher({"E0": FetchFailed("HTTP 500"), "SP1": good["SP1"]}))
    assert r["failed"] == ["E0"] and r["requested"] == ["SP1"] and r["status"] == "ok"
    assert [g["dqd_id"] for g in read_out(s)["matches"]] == [1, 2] and "没拿到" in read_out(s)["note"]


# ---------- 被拒绝 ----------
@pytest.mark.parametrize("code, body, kind", [(403, b"", "failed"), (429, b"", "failed"),
                                              (200, "<html>请完成验证</html>".encode(), "suspicious")])
def test_blocked_pauses(tmp_path, code, body, kind):
    """403、429、拦截页：整个停下（后面的联赛也不请求），写 paused；60 分钟内一次都不请求；过了 60 分钟才再看。"""
    from tests.test_dongqiudi import FakeSite
    s = settings_for(tmp_path, [game(1, ko=T0), game(2, "SP1", ko=T0)])
    site, clock = FakeSite(), [T0 + timedelta(minutes=10)]
    cfg = live.live_cfg(s)
    site.pages[live.live_url(4, clock[0], cfg)] = (code, body)
    site.pages[live.live_url(3, clock[0], cfg)] = (200, {"list": [item(2, minute="10")]})

    def make():
        return LiveFetcher(s, log_db=tmp_path / "log.db", session=site, sleep=lambda x: None, now=lambda: clock[0])
    r = tick(s, clock[0], make())
    assert r["status"] == "paused" and r["requests"] == 1 and len(site.requests) == 1   # SP1 也没请求
    out = read_out(s)
    assert out["status"] == "paused" and out["matches"] == [] and "暂停" in out["note"] and "不绕过" in out["note"]
    row = sqlite3.connect(tmp_path / "log.db").execute("select data_type, status, http_status, file_path from fetch_log").fetchone()
    assert row[:2] == ("fixtures_live", kind) and row[3] is None
    assert row[2] == code
    for minutes in (1, 30, 59):                    # 60 分钟内：一次都不请求（重新启动也一样，看的是 fetch_log）
        clock[0] = T0 + timedelta(minutes=10 + minutes)
        r = tick(s, clock[0], make())
        assert r["status"] == "paused" and r["requests"] == 0
    assert r["next_check_utc"] == T0 + timedelta(minutes=70)
    assert len(site.requests) == 1
    clock[0] = T0 + timedelta(minutes=71)          # 过了 60 分钟：再看（网站这次正常）
    for lid, mid in ((4, 1), (3, 2)):
        site.pages[live.live_url(lid, clock[0], cfg)] = (200, {"list": [item(mid, minute="71")]})
    assert tick(s, clock[0], make())["requests"] == 2 and read_out(s)["status"] == "ok"


def test_hourly_task_block_also_pauses(tmp_path):
    """每小时任务（或回补）被拒绝过（fetch_log 里有 403）：实时比分也停 60 分钟。赔率接口“返回了另一场比赛”不算。"""
    s = settings_for(tmp_path, [game(1, ko=T0)])
    con = connect(tmp_path / "log.db")
    ins = ("insert into fetch_log (url, source, data_type, fetched_at_utc, status, http_status, note)"
           " values ('u', 'dongqiudi', ?, ?, ?, ?, ?)")
    con.execute(ins, ("odds_live", (T0 - timedelta(minutes=5)).isoformat(timespec="seconds"), "suspicious", 200, "返回了另一场比赛 1"))
    con.commit()
    fetcher = LiveFetcher(s, log_db=tmp_path / "log.db")
    assert fetcher.last_block(T0 - timedelta(hours=1)) is None
    con.execute(ins, ("odds_live", (T0 - timedelta(minutes=50)).isoformat(timespec="seconds"), "failed", 429, "网站拒绝访问"))
    con.commit()
    con.close()
    t, why = fetcher.last_block(T0 - timedelta(hours=1))
    assert t == T0 - timedelta(minutes=50) and "429" in why
    f = FakeFetcher()
    f.last_block = fetcher.last_block              # 用真的 fetch_log 判断，请求用假的
    assert tick(s, T0 + timedelta(minutes=5), f)["status"] == "paused" and f.calls == []
    assert tick(s, T0 + timedelta(minutes=11), f)["requests"] == 1   # 拒绝已经过去 61 分钟


# ---------- 输出文件 ----------
def test_output_format(tmp_path):
    """scores.json 的字段名、类型和网站服务器约好的一样。"""
    s = settings_for(tmp_path, [game(1, ko=T0)])
    tick(s, T0 + timedelta(minutes=63), FakeFetcher({"E0": {"list": [item(1, score=("2", "1"), minute="63",
                                                                      events=(None, [ev("RC")]))]}}))
    out = read_out(s)
    assert list(out) == ["updated_at", "status", "note", "matches"]
    assert out["updated_at"] == "2026-11-08T16:03:00+00:00" and out["status"] in ("ok", "paused", "off")
    g = out["matches"][0]
    assert list(g) == ["dqd_id", "league", "home", "away", "kickoff_utc", "state", "minute",
                       "home_goals", "away_goals", "home_red", "away_red"]
    assert (type(g["dqd_id"]), type(g["minute"]), type(g["home_goals"]), type(g["away_red"])) == (int, str, int, int)
    assert (g["home_goals"], g["away_goals"], g["home_red"], g["away_red"], g["minute"]) == (2, 1, 0, 1, "63")


def test_atomic_write(tmp_path, monkeypatch):
    """先写临时文件再改名：改名失败时旧文件原样留着，临时文件也删掉。"""
    path = tmp_path / "live" / "scores.json"
    write_scores(path, {"status": "ok", "n": 1})
    assert json.loads(path.read_text(encoding="utf-8")) == {"status": "ok", "n": 1}

    def boom(src, dst):
        raise OSError("磁盘满了")
    monkeypatch.setattr(live.os, "replace", boom)
    with pytest.raises(OSError):
        write_scores(path, {"status": "ok", "n": 2})
    assert json.loads(path.read_text(encoding="utf-8")) == {"status": "ok", "n": 1}
    assert [p.name for p in path.parent.iterdir()] == ["scores.json"]


# ---------- 出错也不停 ----------
class CountingStop:
    """假的 stop_event：被叫去“睡”n 次以后就当作要停了。记下每次睡几秒。"""
    def __init__(self, n):
        self.n, self.waits = n, []

    def is_set(self):
        return len(self.waits) >= self.n

    def wait(self, seconds):
        self.waits.append(seconds)


def test_errors_do_not_kill_run_forever(tmp_path):
    """每一轮都出意外（请求程序坏了、造请求程序也坏了）：run_forever 不退出，每轮写 status，睡一会儿再来。"""
    s = settings_for(tmp_path, [game(1, ko=T0)])

    class Broken(FakeFetcher):
        def last_block(self, since):
            raise RuntimeError("数据库坏了")
    made = []

    def make():
        made.append(1)
        if len(made) == 2:
            raise ValueError("造不出来")
        return Broken()
    stop = CountingStop(3)
    live.run_forever(s, stop, make_fetcher=make, clock=lambda: T0 + timedelta(minutes=30))
    assert len(made) == 3 and len(stop.waits) == 3          # 三轮都跑了，没有退出
    assert all(live.MIN_SLEEP_SECONDS <= w <= 600 for w in stop.waits)
    out = read_out(s)
    assert out["status"] == "off" and "出错" in out["note"] and "RuntimeError" in out["note"]


def test_run_forever_thread_stops_on_event(tmp_path):
    """真的线程 + threading.Event：没有比赛时睡着，stop_event.set() 以后马上停。"""
    s = settings_for(tmp_path, [])
    stop, f = threading.Event(), FakeFetcher()
    th = threading.Thread(target=live.run_forever, args=(s, stop), kwargs={"make_fetcher": lambda: f}, daemon=True)
    th.start()
    for _ in range(100):
        if (tmp_path / "live" / "scores.json").exists():
            break
        threading.Event().wait(0.05)
    stop.set()
    th.join(timeout=5)
    assert not th.is_alive() and f.calls == [] and read_out(s)["status"] == "off"


def test_bad_inputs_do_not_raise(tmp_path):
    """数据库文件不在、上一份 scores.json 坏了：tick 不抛错，写 off 或照常。"""
    s = settings_for(tmp_path, [game(1, ko=T0)])
    out = tmp_path / "live" / "scores.json"
    out.parent.mkdir(parents=True)
    out.write_text("{坏了", encoding="utf-8")
    f = FakeFetcher({"E0": {"list": [item(1, minute="30")]}})
    r = tick(s, T0 + timedelta(minutes=30), f)
    assert r["status"] == "ok" and read_out(s)["matches"][0]["minute"] == "30"
    s["paths"]["database"] = str(tmp_path / "没有.db")
    r = tick(s, T0 + timedelta(minutes=40), f)
    assert r["status"] == "off" and "FileNotFoundError" in r["note"] and len(f.calls) == 1
    assert read_out(s)["status"] == "off"
