"""football-data 清洗进数据库、数据质量检查的自动检查。用手工造的几行数据和内存里的临时数据库，不碰真实数据库。"""
import sqlite3
from collections import Counter
from datetime import date

import pandas as pd
import pytest

from src.clean.football_data_import import (kickoff_to_utc, load_teams, odds_slots, parse_dates, parse_file,
                                            parse_handicap, write_database)
from src.clean.football_data_reader import keep_league_rows
from src.clean.quality import current_season, run_checks
from src.clean.schema import create_tables
from src.common.config import load_football_data_columns, load_settings


@pytest.fixture(scope="module")
def settings():
    return load_settings()


@pytest.fixture(scope="module")
def slots(settings):
    return odds_slots(load_football_data_columns(settings))


def find(slots, bookmaker, market, timing):
    return next(s for s in slots if (s["bookmaker"], s["market"], s["timing"]) == (bookmaker, market, timing))


# ---------- 换算 ----------
def test_parse_dates_both_formats_and_century():
    d = parse_dates(pd.Series(["25/08/07", "25/08/2007", "14/08/93", "05/12/25", "坏日期", None]))
    assert d.dt.strftime("%Y-%m-%d").tolist()[:4] == ["2007-08-25", "2007-08-25", "1993-08-14", "2025-12-05"]
    assert d[4:].isna().all()


def test_kickoff_to_utc_follows_summer_time():
    """英国夏天比标准时间早 1 小时，冬天一样。"""
    dates = parse_dates(pd.Series(["16/08/2025", "17/01/2026", "16/08/2025"]))
    out = kickoff_to_utc(dates, pd.Series(["15:00", "15:00", None]), "Europe/London")
    assert out.tolist()[:2] == ["2025-08-16 14:00", "2026-01-17 15:00"]
    assert pd.isna(out[2])                               # 没有开球时间 → 空（写进数据库是 NULL）


def test_parse_handicap():
    assert parse_handicap("-0.5") == -0.5
    assert parse_handicap("-0.5,-1") == -0.75          # 半一
    assert parse_handicap(None) is None
    assert parse_handicap(float("nan")) is None
    assert parse_handicap("abc") is None


# ---------- 数据字典 → 赔率分组 ----------
def test_closing_pinnacle_slot(slots):
    s = find(slots, "平博 Pinnacle", "1x2", "close")
    assert s["fields"]["home"] == ["PSCH"] and s["line"] is None


def test_renamed_columns_in_one_slot(slots):
    s = find(slots, "多家平均", "1x2", "pre")
    assert {"BbAvH", "AvgH"} <= set(s["fields"]["home"])


def test_handicap_line_columns(slots):
    """Bet365 早年有自己的盘口列，先用它，再用多家通用的；收盘用收盘盘口。"""
    pre = find(slots, "Bet365", "ah", "pre")["line"]
    assert pre[0] == "B365AH" and {"BbAHh", "AHh"} <= set(pre)
    assert "AHCh" in find(slots, "Bet365", "ah", "close")["line"]
    assert find(slots, "Bet365", "ou", "pre")["line"] == 2.5


# ---------- 队名 ----------
def test_teams_aliases(settings):
    teams, names = load_teams(settings)
    assert names["M'Gladbach"] == names["M'gladbach"] == "M'gladbach"
    assert names["Villareal"] == "Villarreal"
    assert len({t["英文"] for t in teams}) == len(teams)


def test_keep_league_rows():
    issues = []
    t = keep_league_rows(pd.DataFrame({"Div": ["SP2", "SP1", "SP2"]}), "SP2", issues)
    assert len(t) == 2 and len(issues) == 1 and "SP1" in issues[0]


# ---------- 一个文件 → 数据库 ----------
def small_table(rows=None):
    t = pd.DataFrame(rows or [
        {"Div": "E0", "Date": "16/08/2025", "Time": "15:00", "HomeTeam": "Arsenal", "AwayTeam": "Chelsea",
         "FTHG": "2", "FTAG": "1", "FTR": "H", "B365H": "2.0", "B365D": "3.5", "B365A": "0",
         "AHh": "-0.5,-1", "B365AHH": "1.9", "B365AHA": "2.0"},
        {"Div": "E0", "Date": "17/08/2025", "Time": "", "HomeTeam": "Chelsea", "AwayTeam": "Mystery FC",
         "FTHG": "0", "FTAG": "0", "FTR": "D", "B365H": "2.1", "B365D": "3.4", "B365A": "3.6",
         "AHh": "", "B365AHH": "1.9", "B365AHA": "2.0"},
    ])
    return t.assign(row=range(len(t)))


def test_parse_file(settings, slots):
    names = {"Arsenal": "Arsenal", "Chelsea": "Chelsea"}
    issues, unknown = [], Counter()
    m, s, o, dropped = parse_file(small_table(), "E0", 2025, names, slots, settings, issues, unknown)
    assert len(m) == 1 and unknown == Counter({("E0", "2025/26", "Mystery FC"): 1})   # 认不出的队不导入
    assert m.iloc[0]["kickoff_utc"] == "2025-08-16 14:00"
    assert s.iloc[0]["home_goals"] == 2
    b365 = o[(o.bookmaker == "Bet365") & (o.market == "1x2")].iloc[0]
    assert b365["home"] == 2.0 and pd.isna(b365["away"]) and dropped == 1               # 赔率 0 不要
    ah = o[(o.bookmaker == "Bet365") & (o.market == "ah")].iloc[0]
    assert ah["line"] == -0.75


def import_into(con, settings, slots, table):
    names = {"Arsenal": "Arsenal", "Chelsea": "Chelsea", "Mystery FC": "Mystery FC"}
    teams = [{"英文": n, "国家": "英格兰"} for n in names]
    m, s, o, _ = parse_file(table, "E0", 2025, names, slots, settings, [], Counter())
    m = m.assign(raw_file="data/raw/test.csv", fetched_at_utc="2026-10-30T00:00:00+00:00")
    write_database(con, teams, names, m, s, o)


def test_reimport_keeps_match_id(settings, slots):
    """同一份数据导两次：比赛编号不变，赔率不会重复。"""
    con = sqlite3.connect(":memory:")
    import_into(con, settings, slots, small_table())
    first = con.execute("select home_team, match_id from matches order by 1").fetchall()
    n_odds = con.execute("select count(*) from odds").fetchone()[0]
    import_into(con, settings, slots, small_table())
    assert con.execute("select home_team, match_id from matches order by 1").fetchall() == first
    assert con.execute("select count(*) from odds").fetchone()[0] == n_odds
    assert len(first) == 2



def test_replaces_dongqiudi_stand_in_score(settings, slots):
    """懂球帝先顶上的比分：football-data 有了这场，就删掉懂球帝那份；它没有的场次不动。"""
    con = sqlite3.connect(":memory:")
    import_into(con, settings, slots, small_table())
    mid = con.execute("select match_id from matches where home_team = 'Arsenal'").fetchone()[0]
    con.execute("insert into matches (competition, season, home_team, away_team, match_date, status, round, leg)"
                " values ('E0', 2025, 'Chelsea', 'Arsenal', '2025-12-01', 'finished', '', 0)")
    other = con.execute("select match_id from matches where home_team = 'Chelsea' and away_team = 'Arsenal'").fetchone()[0]
    con.execute("insert into match_sources values (?, 'dongqiudi', '1', 'x', 'y')", (other,))
    con.executemany("insert into match_stats (match_id, source, home_goals, away_goals, result)"
                    " values (?, 'dongqiudi', 1, 1, 'D')", [(mid,), (other,)])
    con.commit()
    import_into(con, settings, slots, small_table())
    assert con.execute("select match_id, source from match_stats where source = 'dongqiudi'").fetchall() == [(other, "dongqiudi")]
    assert con.execute("select source from match_stats where match_id = ?", (mid,)).fetchall() == [("football-data",)]

# ---------- 质量检查 ----------
def test_current_season():
    assert current_season(date(2026, 10, 30)) == 2026
    assert current_season(date(2026, 3, 1)) == 2025


def test_quality_checks_find_problems(settings):
    con = sqlite3.connect(":memory:")
    create_tables(con)
    con.executemany("insert into matches (match_id, competition, season, home_team, away_team, match_date, status) "
                    "values (?, 'E0', 2024, ?, ?, ?, 'finished')",
                    [(1, "A", "B", "2024-08-17"), (2, "B", "A", "2024-08-17"), (3, "A", "A", "2026-01-01")])
    con.executemany("insert into match_stats (match_id, source, home_goals, away_goals, result) values (?, 's', ?, ?, ?)",
                    [(1, 2, 1, "A"), (2, None, None, None), (3, 0, 0, "D")])
    con.executemany("insert into odds (match_id, source, bookmaker, market, timing, home, draw, away) "
                    "values (?, 's', ?, '1x2', 'pre', ?, ?, ?)",
                    [(1, "Bet365", 3.0, 3.0, 3.0),          # 倒数之和 1.0：正常
                     (2, "Bet365", 4.0, 4.0, 4.0),          # 0.75：普通公司不可能，要报
                     (1, "多家最高", 3.3, 3.3, 3.3),         # 0.91：多家最高正常
                     (2, "Betfair 交易所", 2.0, 2.0, 2.0)])  # 1.5：交易所也太高，要报
    found = {c["名称"]: c["rows"] for c in run_checks(con, settings, today=date(2026, 10, 30))}
    assert len(found["比分缺失"]) == 1
    assert len(found["赛果和比分对不上"]) == 1               # 2-1 却写客胜
    assert len(found["主客队是同一支"]) == 1
    # A、B 都在 8 月 17 日踢了两场；“A 对 A”那场 A 也算出现两次
    assert len(found["同一天同一队踢两场"]) == 3
    assert len(found["日期不在赛季范围内"]) == 1             # 2024/25 赛季不会有 2026 年的比赛
    flagged = " ".join(found["胜平负赔率抽水异常"])
    assert "Bet365" in flagged and "Betfair 交易所" in flagged and "多家最高" not in flagged
