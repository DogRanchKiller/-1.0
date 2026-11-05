"""国际赛（第 11 阶段，工作树 INB）懂球帝赔率下载、导入的自动检查。用假网站、合成的原始文件和合成的数据库，
不联网，不碰真的数据库，也不用真的 martj42 数据。

查的事：两次请求隔 10 秒、遇到 403/429/拦截页马上停、考试赛季锁住（日期范围和每场开球时间两道锁）、
赛季页编号核对、导入（主客对调的赔率对调、比分不一样的不导、对不上 / 对上两场的不导、多家平均、
开球后的价格不要）、重跑一样、原始层少了文件报错不动数据库、中文名只填空着的、只动自己的来源。
"""
import copy
import gzip
import json
from datetime import datetime, timedelta, timezone

import pytest
import yaml

from src.clean import dongqiudi_intl_import as imp
from src.clean.dongqiudi_reader import parse_season_state
from src.clean.schema import create_tables
from src.collect.dongqiudi import ODDS_URL, SEASON_URL, Blocked
from src.collect.dongqiudi_intl import IntlCollector, in_window, kickoff_span, window
from src.collect.dongqiudi_kor import Enough
from src.common.config import load_settings
from src.common.db import connect

UTC = timezone.utc


# ---------- 合成的设置、页面、文件 ----------
@pytest.fixture
def settings():
    """真的 settings 拷一份，换成两个小赛事（INT 在 leagues 的 intl 组里，INA 合并进来的）。"""
    s = copy.deepcopy(load_settings())
    assert s["leagues"]["intl"] == {"INT": "国际赛"}
    s["dongqiudi_intl"] = dict(s["dongqiudi_intl"], competitions={
        "INT_WC": {"cid": 61, "seasons": {"2022": 501}},
        "INT_Q": {"cid": 207, "seasons": {"2026": 502}}})
    return s


def g(mid, t, home, hid, away, aid, score="1-0", st="Played"):
    """赛季页数据里的一场（字段名和懂球帝一样：a 主队、b 客队、t 北京时间、s 比分、st 状态）。"""
    return {"id": str(mid), "t": t, "a": home, "aid": hid, "b": away, "bid": aid, "s": score, "ps": "", "st": st}


def state_html(sid, cid, name, rounds):
    """赛季页：有战报链接，页面数据藏在 window.__INITIAL_STATE__ 里。rounds = [(轮次名字, [比赛])]。"""
    state = {"zhanbaoStore": {"pages": {f"/zhanbao/saiji/{sid}": {
        "season": {"sid": str(sid), "name": name, "cid": str(cid), "comp": "测试赛事", "cur": False},
        "rounds": [{"name": n, "lc": f"/zhanbao/lunci/{sid}-{i}", "m": ms} for i, (n, ms) in enumerate(rounds, 1)]}}}}
    return ('<html><a href="/zhanbao/lunci/1">轮次</a><script>window.__INITIAL_STATE__='
            + json.dumps(state, ensure_ascii=False) + ';</script></html>')


# 球队：懂球帝编号 → (中文名, martj42 英文名)
TEAMS = {"11": ("甲国", "Alpha"), "12": ("乙国", "Beta"), "13": ("丙国", "Gamma"), "14": ("丁国", "Delta"),
         "15": ("戊国", "Epsilon")}
# 世界杯 2022（北京时间）
WC = [("小组赛 第1轮", [
    g(1001, "2022-11-21 18:00", "甲国", "11", "乙国", "12", "2-1"),   # 正常对上
    g(1002, "2022-11-22 03:00", "丙国", "13", "丁国", "14", "0-1"),   # 数据库里是 丁国 - 丙国（主客对调）
    g(1003, "2022-11-25 21:00", "甲国", "11", "丙国", "13", "1-1"),   # 比分和 martj42 不一样（martj42 记 2-1）
    g(1004, "2022-11-26 21:00", "乙国", "12", "丁国", "14", "0-0"),   # martj42 没有这场：对不上
    g(1005, "2022-11-27 21:00", "甲国", "11", "己国", "99", "3-0"),   # 己国不在队名对照：不导
    g(1006, "2022-11-28 21:00", "乙国", "12", "丙国", "13", "", st="Fixture")])]   # 没踢
# 预选赛 2026 周期：一个赛季页里跨 2023、2024 年
Q = [("第一阶段", [
    g(2001, "2023-10-12 20:00", "甲国", "11", "戊国", "15", "1-0"),   # 范围里
    g(2002, "2024-01-01 07:00", "戊国", "15", "甲国", "11", "0-0"),   # 北京 1 月 1 日 = UTC 2023-12-31 23:00：范围里
    g(2003, "2024-01-01 09:00", "乙国", "12", "戊国", "15", "2-2"),   # UTC 2024-01-01 01:00：考试赛季，不下不导
    g(2004, "2024-01-01", "丙国", "13", "戊国", "15", "1-1"),         # 只有日期：北京这一天跨进 2024 年，保守不下
    g(2005, "2024-03-21 20:00", "丁国", "14", "戊国", "15", "0-3")])] # 考试赛季


def book(name, cid, home, draw, away, ts):
    return {"name": name, "cid": cid, "begin": {"homeWin": home, "draw": draw, "awayWin": away, "ts": ts},
            "now": {"homeWin": home, "draw": draw, "awayWin": away, "ts": ts}}


ODDS = {
    # 1001 开球：北京 11-21 18:00
    1001: [book("Bet3**", "281", "2.00", "3.40", "3.60", "11-21 17:30"),
           book("威廉希*", "115", "2.10", "3.30", "3.40", "11-21 17:55"),
           book("平*", "47", "2.05", "3.50", "3.70", "11-21 17:59"),
           book("澳*", "1", "1.50", "4.00", "6.00", "11-21 18:30"),           # 开球以后：滚球价，不要
           book("竞彩官方", "1129", "1.90", "3.20", "3.50", "11-21 10:00")],   # 竞彩：不算公司
    # 1002 懂球帝主队 = 丙国；数据库主队 = 丁国：主胜、客胜要对调
    1002: [book("Bet3**", "281", "3.00", "3.20", "2.40", "11-21 20:00"),
           book("威廉希*", "115", "3.10", "3.30", "2.30", "11-21 20:00")],
    1003: [book("Bet3**", "281", "2.00", "3.40", "3.60", "11-25 20:00")],   # 比分不一样：不导
    2001: [book("Bet3**", "281", "1.50", "4.00", "6.00", "10-12 19:00")],   # 只有一家：没有多家平均
    2002: [book("Bet3**", "281", "4.00", "3.50", "1.80", "12-31 22:00"),
           book("威廉希*", "115", "4.20", "3.40", "1.75", "12-31 22:30")],
}


def write_raw(root, odds=None):
    """把赛季页和赔率文件按下载程序的放法写到 root/dongqiudi 下。"""
    for code, label, sid, cid, rounds in (("INT_WC", "2022", 501, 61, WC), ("INT_Q", "2026", 502, 207, Q)):
        p = root / "dongqiudi" / "season" / code / f"{label}.html.gz"
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_bytes(gzip.compress(state_html(sid, cid, label, rounds).encode("utf-8")))
    folder = {1001: ("INT_WC", "2022"), 1002: ("INT_WC", "2022"), 1003: ("INT_WC", "2022"),
              2001: ("INT_Q", "2026"), 2002: ("INT_Q", "2026")}
    for mid, books in (ODDS if odds is None else odds).items():
        code, label = folder[mid]
        p = root / "dongqiudi" / "odds_final" / code / label / f"{mid}.json.gz"
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_bytes(gzip.compress(json.dumps({"matchId": mid, "euro": books, "asia": [], "size": []},
                                               ensure_ascii=False).encode("utf-8")))
    return root


# ---------- 下载 ----------
def serve(site, wc=None, q=None):
    site.pages[SEASON_URL.format(season_id=501)] = (200, state_html(501, 61, "2022", WC if wc is None else wc).encode())
    site.pages[SEASON_URL.format(season_id=502)] = (200, state_html(502, 207, "2026", Q if q is None else q).encode())
    for mid in (1001, 1002, 1003, 1004, 1005, 2001, 2002, 2003, 2004, 2005):
        site.pages[ODDS_URL.format(match_id=mid)] = (200, {"matchId": mid, "euro": [], "asia": [], "size": []})


@pytest.fixture
def kenv(tmp_path, settings):
    from tests.test_dongqiudi import Clock, FakeSite
    con = connect(tmp_path / "test.db")
    site, clock = FakeSite(), Clock(datetime(2026, 11, 5, 8, 0, tzinfo=UTC))
    cfg = dict(settings["dongqiudi_intl"], min_interval_seconds=0)

    def make(max_requests=None, **over):
        return IntlCollector(con, tmp_path / "raw", dict(cfg, **over), settings, max_requests=max_requests,
                             session=site, sleep=lambda x: None, now=lambda: clock.t)
    yield con, site, make, tmp_path / "raw" / "dongqiudi"
    con.close()


def odds_requested(site):
    return sorted(int(u.split("/index/")[1].split("?")[0]) for u in site.requests if "odds" in u)


def test_collector_downloads_window_only_and_resumes(kenv):
    """只下踢完的、开球在 2021-01-01 到 2023-12-31（UTC）的比赛；2024 年（考试赛季）的一场都不请求。再跑不重复请求。"""
    con, site, make, out = kenv
    serve(site)
    make().run()
    assert odds_requested(site) == [1001, 1002, 1003, 1004, 1005, 2001, 2002]
    assert (out / "season/INT_Q/2026.html.gz").exists() and (out / "odds_final/INT_Q/2026/2002.json.gz").exists()
    n = len(site.requests)
    c = make()
    c.run()
    assert len(site.requests) == n and c.requests_made() == 0


def test_collector_waits_10_seconds_between_requests(tmp_path, settings):
    """settings 里写的 10 秒：每两次请求之间都等够 10 秒。"""
    from tests.test_dongqiudi import FakeSite
    assert settings["dongqiudi_intl"]["min_interval_seconds"] == 10
    site, sleeps, t = FakeSite(), [], [0.0]
    base = datetime(2026, 11, 5, 8, 0, tzinfo=UTC)

    def sleep(s):
        sleeps.append(s)
        t[0] += s
    c = IntlCollector(connect(tmp_path / "t.db"), tmp_path / "raw", settings["dongqiudi_intl"], settings, session=site,
                      sleep=sleep, clock=lambda: t[0], now=lambda: base + timedelta(seconds=t[0]))
    serve(site)
    c.run(only=["INT_WC"], limit=3)
    assert len(site.requests) == 4                       # 赛季页 1 + 赔率 3（小试 limit）
    assert sleeps == [pytest.approx(10)] * 3


def test_collector_budget_stops(kenv):
    con, site, make, out = kenv
    serve(site)
    with pytest.raises(Enough):
        make(max_requests=3).run()
    assert len(site.requests) == 3


@pytest.mark.parametrize("code", [403, 429])
def test_collector_stops_on_403_429(kenv, code):
    """赛季页或赔率遇到 403/429：马上停（Blocked），不重试，后面的一个也不请求。"""
    con, site, make, out = kenv
    serve(site)
    site.pages[SEASON_URL.format(season_id=501)] = (code, b"")
    with pytest.raises(Blocked):
        make().run()
    assert site.requests == [SEASON_URL.format(season_id=501)]
    serve(site)
    site.requests.clear()
    site.pages[ODDS_URL.format(match_id=1005)] = (code, b"")   # 世界杯从近往远下，1005 是第一场
    with pytest.raises(Blocked):
        make().run()
    assert odds_requested(site) == [1005]


@pytest.mark.parametrize("which", ["season", "odds"])
def test_collector_stops_on_block_pages(kenv, which):
    """赛季页是拦截页、赔率接口不是 JSON：马上停。"""
    con, site, make, out = kenv
    serve(site)
    if which == "season":
        site.pages[SEASON_URL.format(season_id=502)] = (200, b"<html>please verify</html>")
        with pytest.raises(Blocked):
            make().run()
        assert odds_requested(site) == []
    else:
        site.pages[ODDS_URL.format(match_id=1005)] = (200, b"<html>blocked</html>")
        with pytest.raises(Blocked, match="不是 JSON"):
            make().run()
        assert odds_requested(site) == [1005]


def test_collector_checks_season_ids(kenv):
    """赛季页里的赛事编号和 settings 不一样：报错，不下赔率。"""
    con, site, make, out = kenv
    serve(site)
    site.pages[SEASON_URL.format(season_id=501)] = (200, state_html(501, 999, "2022", WC).encode())
    with pytest.raises(ValueError, match="编号写错了"):
        make().run()
    assert odds_requested(site) == []


def test_collector_refuses_exam_seasons(kenv):
    """日期范围碰到 2024 年（考试赛季）：一建好就报错，一次也不请求；单场开球在 2024 年的，直接调用也报错。"""
    con, site, make, out = kenv
    serve(site)
    with pytest.raises(ValueError, match="考试赛季锁住"):
        make(kickoff_to="2024-03-31")
    assert site.requests == []
    c = make()
    _info, games = parse_season_state(state_html(502, 207, "2026", Q))
    for mid in (2003, 2004, 2005):
        with pytest.raises(ValueError, match="考试赛季锁住"):
            c.odds("INT_Q", "2026", next(x for x in games if x.match_id == mid), 0)
    assert site.requests == []


def test_window_edges(settings):
    """范围按 UTC：北京 2024-01-01 07:00（UTC 前一天 23:00）在范围里，09:00 不在；只有日期的按北京这一天头尾算。"""
    start, end = window(settings["dongqiudi_intl"], settings)
    assert (start, end) == (datetime(2021, 1, 1, tzinfo=UTC), datetime(2024, 1, 1, tzinfo=UTC))
    _info, games = parse_season_state(state_html(502, 207, "2026", Q))
    by = {x.match_id: x for x in games}
    assert [m for m in by if in_window(by[m], start, end)] == [2001, 2002]
    assert kickoff_span(by[2004]) == (datetime(2023, 12, 31, 16, 0, tzinfo=UTC),
                                      datetime(2024, 1, 1, 15, 59, 59, tzinfo=UTC))


# ---------- 导入（合成的数据库，形状和 INA 导入 martj42 以后一样） ----------
def build_db(con):
    """建表、放几场 INT 比赛（INA 的写法：英文名 + NT、round = 日期、leg 0、kickoff 空、'day_world'、比分来源 martj42）。"""
    create_tables(con)                                   # match_venue 表也在 schema 里（INA）
    teams = [("Alpha NT", None), ("Beta NT", "老写法"), ("Gamma NT", ""), ("Delta NT", None), ("Epsilon NT", None)]
    con.executemany("insert into teams (team_id, name_zh, country) values (?, ?, ?)",
                    [(t, zh, t[:-3]) for t, zh in teams])
    con.executemany("insert into team_names (source, name, team_id) values ('martj42', ?, ?)",
                    [(t[:-3], t) for t, _ in teams])
    matches = [  # (编号, 日期, 主, 客, 主进球, 客进球)
        (1, "2022-11-21", "Alpha NT", "Beta NT", 2, 1),     # ↔ 懂球帝 1001
        (2, "2022-11-21", "Delta NT", "Gamma NT", 1, 0),    # ↔ 懂球帝 1002（主客对调；日期差 1 天）
        (3, "2022-11-25", "Alpha NT", "Gamma NT", 2, 1),    # ↔ 懂球帝 1003（比分不一样）
        (4, "2023-10-12", "Alpha NT", "Epsilon NT", 1, 0),  # ↔ 懂球帝 2001
        (5, "2023-12-31", "Epsilon NT", "Alpha NT", 0, 0),  # ↔ 懂球帝 2002
        (6, "2024-01-01", "Beta NT", "Epsilon NT", 2, 2)]   # 考试赛季（真的 INA 不导，这里放着看会不会被碰）
    for mid, d, h, a, hg, ag in matches:
        con.execute("insert into matches (match_id, competition, season, home_team, away_team, match_date, kickoff_utc, "
                    "status, round, leg, date_precision) values (?, 'INT', ?, ?, ?, ?, null, 'finished', ?, 0, 'day_world')",
                    (mid, int(d[:4]), h, a, d, d))
        con.execute("insert into match_sources values (?, 'martj42', ?, 'data/raw/martj42/x/results.csv', '2026-11-05 00:00:00')",
                    (mid, f"{d}|{h}|{a}"))
        con.execute("insert into match_stats (match_id, source, home_goals, away_goals, result) values (?, 'martj42', ?, ?, ?)",
                    (mid, hg, ag, "H" if hg > ag else "A" if hg < ag else "D"))
        con.execute("insert into match_venue values (?, 'martj42', 1, 'X', 'Y', 'FIFA World Cup')", (mid,))
    con.execute("insert into odds (source, match_id, bookmaker, market, timing, home, draw, away) "
                "values ('football-data', 1, 'Bet365', '1x2', 'close', 9.0, 9.0, 9.0)")   # 别的来源：不能被碰
    con.commit()


def write_teams(path, teams=None):
    data = {"teams": {tid: {"zh": zh, "martj42": en} for tid, (zh, en) in (TEAMS if teams is None else teams).items()}}
    path.write_text(yaml.safe_dump(data, allow_unicode=True), encoding="utf-8")
    return path


@pytest.fixture
def ienv(tmp_path):
    con = connect(tmp_path / "imp.db")
    build_db(con)
    yield con, write_raw(tmp_path / "raw"), write_teams(tmp_path / "teams_intl.yaml")
    con.close()


def intl_odds(con):
    return {(mid, b): (h, d, a) for mid, b, h, d, a in con.execute(
        "select match_id, bookmaker, home, draw, away from odds where source = 'dongqiudi-intl'")}


def others(con):
    """别的来源、比赛表、比分：导入前后要一模一样。"""
    return [con.execute(q).fetchall() for q in (
        "select * from matches order by match_id", "select * from match_stats order by match_id, source",
        "select * from match_sources where source != 'dongqiudi-intl' order by match_id",
        "select * from odds where source != 'dongqiudi-intl' order by match_id",
        "select * from team_names where source != 'dongqiudi-intl' order by name",
        "select * from match_venue order by match_id")]


def test_import_links_swaps_and_odds(settings, ienv):
    con, raw, teams = ienv
    before = others(con)
    rep = imp.run(con, settings, raw, teams)
    keys = con.execute("select match_id, source_key from match_sources where source = 'dongqiudi-intl' "
                       "order by match_id").fetchall()
    assert keys == [(1, "1001"), (2, "1002"), (4, "2001"), (5, "2002")]   # 1003 比分不一样、1004 对不上、1005 没对照
    o = intl_odds(con)
    # 1001：Bet365、威廉希尔、平博（澳门开球后的不要，竞彩不算），三家平均
    assert {b for m, b in o if m == 1} == {"Bet365", "William Hill", "平博 Pinnacle", "多家平均"}
    assert o[1, "多家平均"] == pytest.approx((2.05, 3.4, round((3.6 + 3.4 + 3.7) / 3, 4)))
    # 1002 主客对调：懂球帝主胜 3.00 是数据库客队（丙国 Gamma）赢
    assert o[2, "Bet365"] == (2.40, 3.20, 3.00) and o[2, "William Hill"] == (2.30, 3.30, 3.10)
    assert o[2, "多家平均"] == pytest.approx((2.35, 3.25, 3.05))
    assert not any(m == 3 for m, _ in o)                                  # 比分不一样：不导赔率
    assert {b for m, b in o if m == 4} == {"Bet365"}                      # 只有一家：没有多家平均
    assert o[5, "多家平均"] == pytest.approx((4.1, 3.45, 1.775))
    assert not any(m == 6 for m, _ in o)                                  # 考试赛季：不读
    assert con.execute("select distinct market, timing from odds where source = 'dongqiudi-intl'").fetchall() == \
        [("1x2", "close")]
    assert others(con) == before                                          # 比赛、比分、别的来源一行没动
    # 报告
    assert [x[2] for x in rep["swapped"]] == [1002]
    assert [x[2] for x in rep["score_diff"]] == [1003] and [x[2] for x in rep["unmatched"]] == [1004]
    assert dict(rep["unmapped_teams"]) == {("99", "己国"): 1}
    wc, q = rep["by"][(2022, "INT_WC")], rep["by"][(2023, "INT_Q")]
    assert (wc["比赛"], wc["对上"], wc["主客对调"], wc["比分不一样"], wc["对不上"], wc["队名对照里没有"], wc["导入"]) == \
        (5, 3, 1, 1, 1, 1, 2)
    assert (wc["有多家平均"], wc["有平博胜平负"], wc["去掉的开球后价格"]) == (2, 1, 1)
    assert (q["比赛"], q["导入"], q["有多家平均"]) == (2, 2, 1)
    assert not any(y >= 2024 for y, _ in rep["by"])
    assert rep["odds"]["竞彩、北单（不算公司）"] == 1
    # 中文名只填空着的；队名对照两种写法
    zh = dict(con.execute("select team_id, name_zh from teams").fetchall())
    assert zh == {"Alpha NT": "甲国", "Beta NT": "老写法", "Gamma NT": "丙国", "Delta NT": "丁国", "Epsilon NT": "戊国"}
    names = dict(con.execute("select name, team_id from team_names where source = 'dongqiudi-intl'").fetchall())
    assert names["懂球帝编号 13"] == "Gamma NT" and names["丙国"] == "Gamma NT" and len(names) == 10
    text = imp.report_markdown(rep, settings["dongqiudi_intl"], ["x", "x"])
    assert "主客对调" in text and "1003" in text and "指纹一样" in text and "推荐" not in text


def test_import_rerun_identical(settings, ienv):
    con, raw, teams = ienv
    imp.run(con, settings, raw, teams)
    first = imp.content_digest(con, settings["dongqiudi_intl"])
    rest = others(con)
    imp.run(con, settings, raw, teams)
    assert imp.content_digest(con, settings["dongqiudi_intl"]) == first
    assert others(con) == rest


def test_import_missing_raw_file_stops_and_keeps_db(settings, ienv):
    """导过一次以后，原始层少了一场的赔率文件、或者少了一个赛季页：报错，数据库一行都不动。"""
    con, raw, teams = ienv
    imp.run(con, settings, raw, teams)
    before = imp.content_digest(con, settings["dongqiudi_intl"]), others(con)
    odds_file = raw / "dongqiudi" / "odds_final" / "INT_Q" / "2026" / "2002.json.gz"
    keep = odds_file.read_bytes()
    odds_file.unlink()
    with pytest.raises(ValueError, match="缺赔率文件"):
        imp.run(con, settings, raw, teams)
    assert (imp.content_digest(con, settings["dongqiudi_intl"]), others(con)) == before
    odds_file.write_bytes(keep)
    (raw / "dongqiudi" / "season" / "INT_Q" / "2026.html.gz").unlink()
    with pytest.raises(ValueError, match="缺赛季页"):
        imp.run(con, settings, raw, teams)
    assert (imp.content_digest(con, settings["dongqiudi_intl"]), others(con)) == before


def test_import_two_candidates_not_imported(settings, ienv):
    """日期差 ≤ 1 天里同一对队有两场：不猜，不导。"""
    con, raw, teams = ienv
    con.execute("insert into matches (match_id, competition, season, home_team, away_team, match_date, status, round, "
                "leg, date_precision) values (7, 'INT', 2022, 'Alpha NT', 'Beta NT', '2022-11-22', 'finished', "
                "'2022-11-22', 0, 'day_world')")
    con.commit()
    rep = imp.run(con, settings, raw, teams)
    assert [(x[2], x[8]) for x in rep["multi"]] == [(1001, [1, 7])]
    assert not any(m in (1, 7) for m, _ in intl_odds(con))


def test_import_needs_int_matches_and_registration(settings, ienv, tmp_path):
    con, raw, teams = ienv
    bad = copy.deepcopy(settings)
    bad["leagues"].pop("intl")
    with pytest.raises(ValueError, match="没有登记"):
        imp.run(con, bad, raw, teams)
    empty = connect(tmp_path / "empty.db")
    with pytest.raises(ValueError, match="还没有 INT 的比赛"):
        imp.run(empty, settings, raw, teams)
    empty.close()


def test_team_map_rejects_two_ids_one_name(tmp_path):
    p = write_teams(tmp_path / "t.yaml", {"1": ("甲国", "Alpha"), "2": ("甲国二", "Alpha")})
    with pytest.raises(ValueError, match="同一个 martj42 名字"):
        imp.load_team_map(p)
    p = write_teams(tmp_path / "t2.yaml", {"1": ("", "Alpha")})
    with pytest.raises(ValueError, match="要有 zh 和 martj42"):
        imp.load_team_map(p)


def test_real_team_map_loads(settings):
    """仓库里的 config/teams_intl.yaml 读得进来（编号唯一、两个名字都有、没有两个编号对到同一个英文名）。"""
    from src.common.config import resolve_path
    m = imp.load_team_map(resolve_path(settings["dongqiudi_intl"]["teams_file"]))
    assert len(m) > 100 and m["50000344"] == {"zh": "中国", "martj42": "China"}
