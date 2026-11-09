"""韩K（第 11 阶段）下载、导入的自动检查。用假网站、假原始文件，不联网，不碰真的数据库。

查的事：一年一个赛季的写法、轮次名字分类（附加赛不算联赛）、遇到 403/429/拦截页马上停、请求数到上限停、
考试赛季锁住、导入（附加赛排除、同一对队不同轮次、开球后的价格不要、多家平均、重跑一样、只动自己的来源）、
杯赛检查不把韩K 当杯赛。
"""
import copy
import gzip
import json
from datetime import datetime, timezone

import pytest

from src.clean import dongqiudi_kor_import as imp
from src.clean.dongqiudi_reader import parse_history, parse_season_state, round_kind
from src.clean.schema import create_tables
from src.clean.quality import all_league_codes, cup_checks, league_codes
from src.collect.dongqiudi import HISTORY_URL, ODDS_URL, SEASON_URL, Blocked
from src.collect.dongqiudi_kor import Enough, KorCollector, check_season
from src.common.config import load_settings
from src.common.db import connect

UTC = timezone.utc


@pytest.fixture(scope="module")
def settings():
    return load_settings()


# ---------- 造假的页面和文件 ----------
def g(mid, t, home, hid, away, aid, score="1-0", st="Played"):
    """赛季页数据里的一场（字段名和懂球帝一样：a 主队、b 客队、t 北京时间、s 比分、st 状态）。"""
    return {"id": mid, "t": t, "a": home, "aid": hid, "b": away, "bid": aid, "s": score, "st": st}


def state_html(year, rounds, cur=False):
    """赛季页：有战报链接，页面数据藏在 window.__INITIAL_STATE__ 里。rounds = [(轮次名字, [比赛])]。"""
    state = {"zhanbaoStore": {"pages": {f"/zhanbao/saiji/{year}": {
        "season": {"sid": year, "name": str(year), "cur": cur},
        "rounds": [{"name": n, "lc": f"/zhanbao/lunci/{year}-{i}", "m": ms} for i, (n, ms) in enumerate(rounds, 1)]}}}}
    return ('<html><a href="/zhanbao/lunci/1">轮次</a><script>window.__INITIAL_STATE__='
            + json.dumps(state, ensure_ascii=False) + ';</script></html>')


HISTORY_KOR = ('<a href="/zhanbao/saiji/30026">2026</a><a href="/zhanbao/saiji/20741">2023</a>'
               '<a href="/zhanbao/saiji/20000">2022</a><a href="/zhanbao/saiji/21198">2023/24</a>')

# 2021：四个队各踢 2 场（A 队那时叫“老A”）
S2021 = [("联赛 第1轮", [g(201, "2021-03-01 18:00", "老A", "1", "B", "2", "2-0"),
                         g(202, "2021-03-01 18:00", "C", "3", "D", "4", "0-0")]),
         ("联赛 第2轮", [g(203, "2021-03-08 18:00", "B", "2", "C", "3", "1-1"),
                         g(204, "2021-03-08 18:00", "D", "4", "老A", "1", "0-3")])]
# 2022：A 改名“新A”；新A-B 在“联赛 第1轮”和“争冠组 第1轮”各碰一次；
# 一场没踢完（不导入）；一场升降级附加赛（D 对 K2 的 E 队，不导入）
S2022 = [("联赛 第1轮", [g(101, "2022-03-01 18:00", "新A", "1", "B", "2", "2-1"),
                         g(102, "2022-03-01 18:00", "C", "3", "D", "4", "0-0")]),
         ("联赛 第2轮", [g(103, "2022-03-08 18:00", "B", "2", "C", "3", "1-2"),
                         g(104, "2022-03-08 18:00", "D", "4", "新A", "1", "1-1")]),
         ("争冠组 第1轮", [g(105, "2022-10-01 18:00", "新A", "1", "B", "2", "0-1"),
                           g(106, "2022-10-01 18:00", "C", "3", "D", "4", "", st="Fixture")]),
         ("保级组 第1轮", [g(107, "2022-10-02 18:00", "C", "3", "D", "4", "3-0")]),
         ("升降级附加赛", [g(108, "2022-12-01 18:00", "D", "4", "E", "5", "1-0")])]
# 2024（考试赛季）：四个队各踢 2 场
S2024 = [("联赛 第1轮", [g(301, "2024-03-01 18:00", "新A", "1", "B", "2", "1-0"),
                         g(302, "2024-03-01 18:00", "C", "3", "D", "4", "2-2")]),
         ("联赛 第2轮", [g(303, "2024-03-08 18:00", "B", "2", "C", "3", "0-1"),
                         g(304, "2024-03-08 18:00", "D", "4", "新A", "1", "1-3")])]


def kfx(mid, home, hid, away, aid, start, gw, status="Fixture", score=("", "")):
    """赛程接口里的一场（字段名和懂球帝一样）。start 是标准时间。"""
    return {"match_id": str(mid), "team_A_name": home, "team_A_id": hid, "team_B_name": away, "team_B_id": aid,
            "start_play": start, "status": status, "fs_A": score[0], "fs_B": score[1], "gameweek": str(gw)}


def write_fixtures(root, name, rows):
    p = root / "dongqiudi" / "fixtures" / "KOR" / name
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_bytes(gzip.compress(json.dumps({"list": rows}, ensure_ascii=False).encode("utf-8")))


def write_live(root):
    """2026（当前赛季）的两次赛程抓取：后一次覆盖前一次（403 踢完了，404 改了时间）。
    405 和 401 是同一对队（第 34 轮，分组赛）；406 时间待定；407 不是数字轮次（附加赛，不导入）。"""
    write_fixtures(root, "20260301T0000Z_1.json.gz", [
        kfx(401, "新A", "1", "B", "2", "2026-03-01 05:00:00", 1, "Played", ("2", "0")),
        kfx(402, "C", "3", "D", "4", "2026-03-01 05:00:00", 1, "Played", ("0", "0")),
        kfx(403, "B", "2", "C", "3", "2026-03-08 05:00:00", 2),
        kfx(404, "D", "4", "新A", "1", "2026-03-08 05:00:00", 2)])
    write_fixtures(root, "20261109T0100Z_1.json.gz", [
        kfx(403, "B", "2", "C", "3", "2026-03-08 05:00:00", 2, "Played", ("1", "1")),
        kfx(404, "D", "4", "新A", "1", "2026-11-21 05:00:00", 2),
        kfx(405, "新A", "1", "B", "2", "2026-11-22 05:00:00", 34),
        kfx(406, "C", "3", "D", "4", "2026-11-29 05:00:00", 35, "Uncertain"),
        kfx(407, "D", "4", "E", "5", "2026-12-10 05:00:00", "")])
    return root


def book(name, cid, home, draw, away, ts):
    return {"name": name, "cid": cid, "begin": {"homeWin": home, "draw": draw, "awayWin": away, "ts": ts},
            "now": {"homeWin": home, "draw": draw, "awayWin": away, "ts": ts}}


# 101 开球：北京时间 03-01 18:00
ODDS = {
    101: [book("Bet3**", "281", "2.00", "3.40", "3.60", "03-01 17:30"),
          book("威廉希*", "115", "2.10", "3.30", "3.40", "03-01 17:55"),
          book("平*", "47", "2.05", "3.50", "3.70", "03-01 17:59"),
          book("澳*", "1", "1.50", "4.00", "6.00", "03-01 18:30"),          # 开球以后变的：滚球价，不要
          book("竞彩官方", "1129", "1.90", "3.20", "3.50", "03-01 10:00")],  # 竞彩：不算公司
    102: [book("Bet3**", "281", "2.50", "3.10", "2.90", "03-01 12:00"),
          book("立*", "82", "1.50", "2.00", "2.50", "03-01 12:00")],        # 抽水 1.57：留这家，不算平均
    103: [book("Bet3**", "281", "2.40", "3.20", "2.90", "03-08 19:00")],     # 只有开球以后的价
}


def write_raw(root, seasons=None, odds=None):
    """把赛季页和赔率文件按下载程序的放法写到 root/dongqiudi 下。"""
    seasons = {2021: S2021, 2022: S2022} if seasons is None else seasons
    odds = ODDS if odds is None else odds
    for y, rounds in seasons.items():
        p = root / "dongqiudi" / "season" / "KOR" / f"{y}.html.gz"
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_bytes(gzip.compress(state_html(y, rounds).encode("utf-8")))
    for mid, books in odds.items():
        p = root / "dongqiudi" / "odds_final" / "KOR" / "2022" / f"{mid}.json.gz"
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_bytes(gzip.compress(json.dumps({"matchId": mid, "euro": books, "asia": [], "size": []},
                                               ensure_ascii=False).encode("utf-8")))
    return root


def kor_cfg(settings):
    return settings["dongqiudi_kor"]


# ---------- 读取 ----------
def test_history_calendar_year_only():
    """韩K 的历届赛季页只写一个年份：只认“2023”；“2023/24”这样的不认（反过来五大联赛的读法也不认“2023”）。"""
    assert parse_history(HISTORY_KOR, calendar_year=True) == {2026: 30026, 2023: 20741, 2022: 20000}
    assert parse_history(HISTORY_KOR) == {2023: 21198}


def test_parse_season_state_reads_rounds_and_times():
    info, games = parse_season_state(state_html(2022, S2022))
    assert info["name"] == "2022" and len(games) == 8
    first = games[0]
    assert (first.match_id, first.round_no, first.round_name, first.home, first.home_id, first.away_id) == \
        (101, 1, "联赛 第1轮", "新A", "1", "2")
    assert first.kickoff_bj.astimezone(UTC) == datetime(2022, 3, 1, 10, 0, tzinfo=UTC)   # 北京时间减 8 小时
    assert (first.home_goals, first.away_goals, first.status) == (2, 1, "Played")
    unplayed = next(x for x in games if x.match_id == 106)
    assert unplayed.home_goals is None and unplayed.status == "Fixture"
    assert games[-1].round_no == 5 and games[-1].round_name == "升降级附加赛"
    assert parse_season_state("<html>没有数据</html>") is None


def test_round_kind_rules(settings):
    c = kor_cfg(settings)
    p, n = c["league_round_prefixes"], c["non_league_rounds"]
    for name in ("联赛 第1轮", "争冠组 第5轮", "保级组 第38轮", "冠军附加 第2轮", "降级附加 第3轮"):
        assert round_kind(name, p, n) == "league"
    for name in ("升降级附加赛", "附加赛 第2轮", "附加赛第一轮", "半决赛", "总决赛", "决赛"):
        assert round_kind(name, p, n) == "other"
    for name in ("第1轮", "小组赛 第1轮", "季后赛"):          # 认不出：报错，不猜
        with pytest.raises(ValueError, match="认不出的轮次名字"):
            round_kind(name, p, n)


def test_check_season_flags_block_page():
    assert check_season(state_html(2022, S2022).encode()) is None
    assert check_season(b"<html>please verify</html>") is not None
    assert check_season(b'<a href="/zhanbao/1">x</a>') is not None     # 有链接但没有页面数据


# ---------- 下载 ----------
@pytest.fixture
def kenv(tmp_path, settings):
    from tests.test_dongqiudi import Clock, FakeSite
    con = connect(tmp_path / "test.db")
    site, clock = FakeSite(), Clock(datetime(2026, 11, 3, 8, 0, tzinfo=UTC))
    cfg = dict(kor_cfg(settings), min_interval_seconds=0)

    def make(max_requests=None):
        return KorCollector(con, tmp_path / "raw", cfg, settings, max_requests=max_requests, session=site,
                            sleep=lambda s: None, now=lambda: clock.t)
    yield con, site, make, tmp_path / "raw" / "dongqiudi"
    con.close()


def serve_kor(site):
    site.pages[HISTORY_URL.format(league_id=117)] = (200, HISTORY_KOR.encode())
    site.pages[SEASON_URL.format(season_id=20000)] = (200, state_html(2022, S2022).encode())
    site.pages[SEASON_URL.format(season_id=20741)] = (200, state_html(2023, S2021).encode())
    for mid in (101, 102, 103, 104, 105, 107, 201, 202, 203, 204):
        site.pages[ODDS_URL.format(match_id=mid)] = (200, {"matchId": mid, "euro": [], "asia": [], "size": []})


def test_collector_downloads_league_games_only_and_resumes(kenv):
    """只下联赛比赛、踢完的赔率（附加赛 108、没踢完的 106 不下）；再跑一次不重复请求。"""
    con, site, make, out = kenv
    serve_kor(site)
    make().run(2022, 2022, 2023)
    odds = sorted(u for u in site.requests if "odds" in u)
    assert odds == sorted(ODDS_URL.format(match_id=m) for m in (101, 102, 103, 104, 105, 107, 201, 202, 203, 204))
    assert (out / "season/KOR/2022.html.gz").exists() and (out / "odds_final/KOR/2023/201.json.gz").exists()
    n = len(site.requests)
    c = make()
    c.run(2022, 2022, 2023)
    assert len(site.requests) == n and c.requests_made() == 0           # 都已有：一次也不请求


def test_collector_budget_stops(kenv):
    con, site, make, out = kenv
    serve_kor(site)
    c = make(max_requests=4)
    with pytest.raises(Enough):
        c.run(2022, 2022, 2023)
    assert len(site.requests) == 4


@pytest.mark.parametrize("code", [403, 429])
def test_collector_stops_on_403_429(kenv, code):
    """赛季页或赔率遇到 403/429：马上停（Blocked），不重试。"""
    con, site, make, out = kenv
    serve_kor(site)
    site.pages[SEASON_URL.format(season_id=20741)] = (code, b"")
    with pytest.raises(Blocked):
        make().run(2022, 2022, 2023)
    assert site.requests.count(SEASON_URL.format(season_id=20741)) == 1
    assert not any("odds" in u for u in site.requests)
    serve_kor(site)
    site.pages[ODDS_URL.format(match_id=201)] = (code, b"")
    with pytest.raises(Blocked):
        make().run(2022, 2022, 2023)
    assert site.requests.count(ODDS_URL.format(match_id=201)) == 1
    assert ODDS_URL.format(match_id=202) not in site.requests          # 停了就不往下请求


@pytest.mark.parametrize("which", ["season", "odds"])
def test_collector_stops_on_block_pages(kenv, which):
    """赛季页是拦截页、赔率接口不是 JSON：马上停。"""
    con, site, make, out = kenv
    serve_kor(site)
    if which == "season":
        site.pages[SEASON_URL.format(season_id=20741)] = (200, b"<html>please verify</html>")
        with pytest.raises(Blocked):
            make().run(2022, 2022, 2023)
        assert not any("odds" in u for u in site.requests)
    else:
        site.pages[ODDS_URL.format(match_id=201)] = (200, b"<html>blocked</html>")
        with pytest.raises(Blocked, match="不是 JSON"):
            make().run(2022, 2022, 2023)
        assert ODDS_URL.format(match_id=202) not in site.requests


def test_collector_exam_season_pages_only_odds_locked_live_refused(kenv):
    """考试赛季：赛季页可以下（比分当热身历史），赔率锁住；当前赛季（上线后）连赛季页也不下。年份不对时一次也不请求。"""
    con, site, make, out = kenv
    serve_kor(site)
    with pytest.raises(ValueError, match="考试赛季锁住"):
        make().run(2022, 2022, 2024)               # 没另写赔率到哪年 = 到 2024：考试赛季的赔率锁住
    with pytest.raises(ValueError, match="上线后"):
        make().run(2022, 2022, 2026, 2023)         # 赛季页不能有当前赛季
    assert site.requests == []                                         # 一次也没请求
    site.pages[HISTORY_URL.format(league_id=117)] = (200, (HISTORY_KOR + '<a href="/zhanbao/saiji/22122">2024</a>').encode())
    site.pages[SEASON_URL.format(season_id=22122)] = (200, state_html(2024, S2024).encode())
    make().run(2022, 2022, 2024, 2023)
    assert (out / "season/KOR/2024.html.gz").exists()
    assert not (out / "odds_final/KOR/2024").exists()
    assert not any(ODDS_URL.format(match_id=m) in site.requests for m in (301, 302, 303, 304))


# ---------- 导入 ----------
@pytest.fixture
def ienv(tmp_path):
    con = connect(tmp_path / "imp.db")
    create_tables(con)
    yield con, write_raw(tmp_path / "raw")
    con.close()


def kor_odds(con):
    return {(sid, b): (h, d, a) for sid, b, h, d, a in con.execute(
        "select s.source_key, o.bookmaker, o.home, o.draw, o.away from odds o "
        "join match_sources s on s.match_id = o.match_id and s.source = o.source where o.source = 'dongqiudi-kor'")}


def test_import_league_games_odds_and_report(settings, ienv):
    con, raw = ienv
    rep = imp.run(con, settings, raw)
    # 2021 四场 + 2022 五场（101–105、107；106 没踢完、108 是附加赛，都不导入）
    rows = con.execute("select m.season, m.home_team, m.away_team, m.round, m.leg, m.kickoff_utc, m.match_date, "
                       "s.source_key from matches m join match_sources s using (match_id) "
                       "where m.competition = 'KOR' order by s.source_key").fetchall()
    assert [r[7] for r in rows] == ["101", "102", "103", "104", "105", "107", "201", "202", "203", "204"]
    by = {r[7]: r for r in rows}
    assert by["101"][1:7] == ("新A", "B", "1", 0, "2022-03-01 10:00", "2022-03-01")
    assert by["105"][1:4] == ("新A", "B", "3")                 # 同一对队，第 3 个轮次：两场都在
    assert by["201"][1] == "新A"                               # 旧写法“老A”换成这个编号最近一季的写法
    assert "E" not in {t for r in rows for t in r[1:3]}
    assert rep["teams_per_season"] == {2021: 4, 2022: 4}
    assert [(y, rn) for y, _d, rn, *_ in rep["playoffs"]] == [(2022, "升降级附加赛")]
    assert [x[0] for x in rep["not_played"]] == [2022]
    names = dict(con.execute("select name, team_id from team_names where source = 'dongqiudi-kor'").fetchall())
    assert names["老A"] == "新A" and names["懂球帝编号 1"] == "新A"
    assert con.execute("select country from teams where team_id = '新A'").fetchone()[0] == "韩国"
    # 赔率：101 有 Bet365、William Hill、平博（澳门开球后的不要，竞彩不算），三家平均
    o = kor_odds(con)
    assert {b for (m, b) in o if m == "101"} == {"Bet365", "William Hill", "平博 Pinnacle", "多家平均"}
    assert o["101", "多家平均"] == pytest.approx((2.05, 3.4, round((3.6 + 3.4 + 3.7) / 3, 4)))
    assert o["101", "平博 Pinnacle"] == (2.05, 3.5, 3.7)
    # 102：立博抽水太大，留着这家但不算平均 → 只剩一家合格，没有多家平均
    assert {b for (m, b) in o if m == "102"} == {"Bet365", "Ladbrokes"}
    assert not any(m == "103" for m, _ in o)                   # 只有开球后的价：整场没有赔率
    assert rep["odds"]["去掉：最新价格在开球以后（滚球价）"] == 2
    assert rep["odds"]["竞彩、北单（不算公司）"] == 1
    tab = {r["赛季"]: r for r in rep["season_table"]}
    assert (tab[2022]["场数"], tab[2022]["有多家平均"], tab[2022]["有平博胜平负"]) == (6, 1, 1)
    text = imp.report_markdown(rep, kor_cfg(settings), ["x", "x"])
    assert "升降级附加赛：1 场" in text and "指纹一样" in text and "推荐" not in text


def test_import_rerun_identical_and_other_sources_untouched(settings, ienv):
    con, raw = ienv
    con.execute("insert into teams (team_id, name_zh, country) values ('Arsenal', '阿森纳', '英格兰')")
    con.execute("insert into team_names (source, name, team_id) values ('dongqiudi', '阿森纳', 'Arsenal')")
    con.commit()
    imp.run(con, settings, raw)
    first = imp.content_digest(con, kor_cfg(settings))
    ids = con.execute("select match_id from matches where competition = 'KOR' order by match_id").fetchall()
    imp.run(con, settings, raw)
    assert imp.content_digest(con, kor_cfg(settings)) == first
    assert con.execute("select match_id from matches where competition = 'KOR' order by match_id").fetchall() == ids
    assert con.execute("select team_id from team_names where source = 'dongqiudi'").fetchall() == [("Arsenal",)]


def test_import_missing_season_or_odds_file_stops_and_keeps_db(settings, ienv):
    """导过一次以后，原始层少了一季的赛季页、或者少了一场的赔率文件：报错，数据库一行都不动。"""
    con, raw = ienv
    imp.run(con, settings, raw)
    before = imp.content_digest(con, kor_cfg(settings))
    season_2021 = raw / "dongqiudi" / "season" / "KOR" / "2021.html.gz"
    keep = season_2021.read_bytes()
    season_2021.unlink()
    with pytest.raises(ValueError, match=r"缺赛季页.*\[2021\]|\[2021\].*缺赛季页"):
        imp.run(con, settings, raw)
    assert imp.content_digest(con, kor_cfg(settings)) == before
    season_2021.write_bytes(keep)
    (raw / "dongqiudi" / "odds_final" / "KOR" / "2022" / "101.json.gz").unlink()
    with pytest.raises(ValueError, match="缺赔率文件"):
        imp.run(con, settings, raw)
    assert imp.content_digest(con, kor_cfg(settings)) == before


def test_import_unknown_round_name_stops(settings, tmp_path):
    bad = copy.deepcopy(S2022)
    bad[-1] = ("季后赛", bad[-1][1])
    con = connect(tmp_path / "x.db")
    with pytest.raises(ValueError, match="认不出的轮次名字"):
        imp.run(con, settings, write_raw(tmp_path / "raw", {2022: bad}, {}))
    assert con.execute("select count(*) from matches where competition = 'KOR'").fetchone()[0] == 0
    con.close()


def test_import_playoff_left_in_is_caught(settings, tmp_path):
    """附加赛要是被当成联赛比赛（名字规则漏了）：E 队只踢 1 场，出场太少，报错。"""
    s = copy.deepcopy(settings)
    s["dongqiudi_kor"]["league_round_prefixes"] = [*s["dongqiudi_kor"]["league_round_prefixes"], "附加赛"]
    bad = copy.deepcopy(S2022)
    bad[-1] = ("附加赛 第1轮", bad[-1][1])
    s["dongqiudi_kor"]["non_league_rounds"] = [n for n in s["dongqiudi_kor"]["non_league_rounds"]
                                               if n != "附加赛 第2轮"]
    con = connect(tmp_path / "x.db")
    with pytest.raises(ValueError, match="出场太少"):
        imp.run(con, s, write_raw(tmp_path / "raw", {2022: bad}, {}))
    con.close()


def test_import_date_mismatch_stops(settings, tmp_path):
    """北京时间 06:00 开球 = 英国前一天：日期对不上，报错（不猜哪天算比赛日）。"""
    bad = copy.deepcopy(S2021)
    bad[0][1][0]["t"] = "2021-03-01 06:00"
    con = connect(tmp_path / "x.db")
    with pytest.raises(ValueError, match="日期不一样"):
        imp.run(con, settings, write_raw(tmp_path / "raw", {2021: bad}, {}))
    con.close()


def test_import_exam_season_scores_only_and_live_season_page_refused(settings, tmp_path):
    """考试赛季 2024：比赛、比分照导（热身历史），赔率文件就算有也不导；last_season 写到当前赛季：报错，一行都不写。"""
    raw = write_raw(tmp_path / "raw", {2022: S2022, 2024: S2024})
    p = raw / "dongqiudi" / "odds_final" / "KOR" / "2024" / "301.json.gz"
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_bytes(gzip.compress(json.dumps({"matchId": 301, "euro": ODDS[101], "asia": [], "size": []}).encode("utf-8")))
    con = connect(tmp_path / "x.db")
    imp.run(con, settings, raw)
    n = dict(con.execute("select season, count(*) from matches where competition = 'KOR' group by season").fetchall())
    assert n == {2022: 6, 2024: 4}
    assert con.execute("select count(*) from match_stats s join matches m using (match_id) "
                       "where m.season = 2024 and s.source = 'dongqiudi-kor'").fetchone()[0] == 4
    assert not any(m == "301" for m, _ in kor_odds(con))
    s = copy.deepcopy(settings)
    s["dongqiudi_kor"]["last_season"] = 2026
    con2 = connect(tmp_path / "y.db")
    with pytest.raises(ValueError, match="上线后"):
        imp.run(con2, s, raw)
    assert con2.execute("select count(*) from sqlite_master where name = 'matches'").fetchone()[0] == 0   # 连表都没建
    con.close()
    con2.close()


def live_rows(con):
    return {k: r for k, *r in con.execute(
        "select s.source_key, m.season, m.round, m.home_team, m.away_team, m.match_date, m.kickoff_utc, m.status, "
        "m.date_precision from matches m join match_sources s on s.match_id = m.match_id and s.source = 'dongqiudi-kor' "
        "where m.season = 2026")}


def test_import_live_season_from_fixtures(settings, ienv):
    """当前赛季 2026：从赛程文件读，最后一次看到的为准；没踢的也进比赛表（scheduled），每日预测才看得到。
    round = 赛程里的第几轮（同一对队第 1 轮、第 34 轮两场都在）；时间待定的不写开球时间、精度 round；
    不是数字轮次的（附加赛）不导入；当前赛季不导赔率（赔率由懂球帝导入按编号补）。"""
    con, raw = ienv
    write_live(raw)
    rep = imp.run(con, settings, raw)
    rows = live_rows(con)
    assert set(rows) == {"401", "402", "403", "404", "405", "406"}
    assert rows["401"] == [2026, "1", "新A", "B", "2026-03-01", "2026-03-01 05:00", "finished", "day"]
    assert rows["403"][6] == "finished"                                  # 后一次抓取：踢完了
    assert rows["404"][4:7] == ["2026-11-21", "2026-11-21 05:00", "scheduled"]   # 后一次抓取：改了时间
    assert rows["405"][:4] == [2026, "34", "新A", "B"]
    assert rows["406"][4:] == ["2026-11-26", None, "scheduled", "round"]
    assert [(y, rn) for y, _d, rn, *_ in rep["playoffs"]][-1] == (2026, "赛程“第  轮”")
    assert rep["live"] == {(2026, "finished"): 3, (2026, "scheduled"): 3}
    assert rep["teams_per_season"][2026] == 4
    stats = {k: (h, a) for k, h, a in con.execute(
        "select s.source_key, t.home_goals, t.away_goals from match_stats t join match_sources s "
        "on s.match_id = t.match_id and s.source = t.source join matches m on m.match_id = t.match_id where m.season = 2026")}
    assert stats == {"401": (2, 0), "402": (0, 0), "403": (1, 1)}
    assert not any(int(m) > 400 for m, _ in kor_odds(con))
    assert "E" not in {t for (t,) in con.execute("select team_id from teams")}
    # 下一小时：404 踢完了。比赛编号不变，状态、比分跟着变
    ids = dict(con.execute("select s.source_key, s.match_id from match_sources s where s.source = 'dongqiudi-kor'"))
    write_fixtures(raw, "20261121T0900Z_1.json.gz",
                   [kfx(404, "D", "4", "新A", "1", "2026-11-21 05:00:00", 2, "Played", ("0", "1"))])
    imp.run(con, settings, raw)
    assert dict(con.execute("select s.source_key, s.match_id from match_sources s where s.source = 'dongqiudi-kor'")) == ids
    assert live_rows(con)["404"][6] == "finished"
    assert con.execute("select home_goals, away_goals from match_stats where match_id = ? and source = 'dongqiudi-kor'",
                       (ids["404"],)).fetchone() == (0, 1)


def test_hourly_import_links_kor_by_match_id(settings, ienv):
    """懂球帝导入（每小时）：韩K 按懂球帝比赛编号对上韩K 导入写的比赛，不投票定队名、不新增比赛；
    赛前快照进变盘记录；韩K 往届的赔率文件不读。之后再跑韩K 导入，懂球帝那边的对照还在。"""
    from src.clean import dongqiudi_import as dq
    con, raw = ienv
    write_live(raw)
    p = raw / "dongqiudi" / "odds_live" / "KOR" / "404" / "20261109T0200Z.json.gz"
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_bytes(gzip.compress(json.dumps({"matchId": 404, "euro": [
        book("Bet3**", "281", "2.50", "3.20", "2.80", "11-09 09:00")], "asia": [], "size": []}).encode("utf-8")))
    imp.run(con, settings, raw)
    n_matches = con.execute("select count(*) from matches").fetchone()[0]
    s = copy.deepcopy(settings)
    s["dongqiudi"]["league_ids"] = {"KOR": 117}
    out = dq.run(con, raw, raw.parent / "reports", s, now=datetime(2026, 11, 9, 12, 0, tzinfo=UTC))
    res = {ln.cand.dqd_id: ln.result for ln in out["links"]}
    assert res == {401: "对上", 402: "对上", 403: "对上", 404: "对上", 405: "对上", 406: "对上", 407: "数据库里没有这场"}
    assert out["mapping"] == {}
    assert con.execute("select count(*) from matches").fetchone()[0] == n_matches
    linked = dict(con.execute("select source_key, match_id from match_sources where source = 'dongqiudi'"))
    kor = dict(con.execute("select source_key, match_id from match_sources where source = 'dongqiudi-kor'"))
    assert linked == {k: kor[k] for k in ("401", "402", "403", "404", "405", "406")}
    assert con.execute("select count(*) from odds_changes where source = 'dongqiudi' and match_id = ?",
                       (kor["404"],)).fetchone()[0] == 1                 # 初盘和最新一样：去重后一行
    assert out["stats"]["文件：比赛没对上，跳过"] == 0                    # 2022 的韩K 赔率文件没读
    assert con.execute("select count(*) from match_stats where source = 'dongqiudi'").fetchone()[0] == 0
    imp.run(con, settings, raw)
    assert dict(con.execute("select source_key, match_id from match_sources where source = 'dongqiudi'")) == linked


def test_cup_checks_skip_kor(settings, ienv):
    """韩K 的 round 记第几个轮次（不是空的）：杯赛检查要跳过全部联赛，不然会把韩K 当成杯赛淘汰赛。"""
    con, raw = ienv
    imp.run(con, settings, raw)
    assert "KOR" in all_league_codes(settings) and "KOR" not in league_codes(settings)
    assert all(not rows for _n, _e, rows in cup_checks(con, all_league_codes(settings)))
    assert any(rows for _n, _e, rows in cup_checks(con, league_codes(settings)))   # 只跳过五大、次级就会报
