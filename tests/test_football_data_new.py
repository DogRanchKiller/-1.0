"""日职、中超“新联赛”文件：下载（假网站，不联网）、导入（小样本 CSV）、附加赛排除、幂等、老联赛导入不删它们。

全部用临时文件夹和临时数据库，不碰真实的 data/football.db，也不联网。
小样本：日职 2023 赛季 4 个队双循环（12 场）+ 1 场附加赛（第 5 个队只踢 1 场）+ 1 场重复的主客队（冠军赛）；
        2026/2027 赛季 1 场（秋春制写法，换成开始年份 2026）。
"""
import copy
import sqlite3
from collections import Counter
from datetime import date

import pandas as pd
import pytest

from src.clean import football_data_new_import as imp
from src.clean.football_data_import import load_teams, odds_slots, parse_file, write_database
from src.clean.schema import create_tables
from src.collect import football_data_new as dl
from src.common.config import load_football_data_columns, load_settings
from src.common.db import connect

HEADER = ("Country,League,Season,Date,Time,Home,Away,HG,AG,Res,PSCH,PSCD,PSCA,MaxCH,MaxCD,MaxCA,AvgCH,AvgCD,AvgCA,"
          "BFECH,BFECD,BFECA,B365CH,B365CD,B365CA,,,")
LEAGUE = ["Kashima Antlers", "Urawa Reds", "Cerezo Osaka", "Kofu"]
PLAYOFF_TEAM = "Gamba Osaka"            # teams.yaml 里有的队，只踢 1 场 → 当作附加赛球队


def csv_rows() -> list[str]:
    rows, day = [], pd.Timestamp("2023-03-04")      # 3 月 4 日：英国冬令时（UTC = 英国时间）
    pairs = [(h, a) for h in LEAGUE for a in LEAGUE if h != a]
    for i, (h, a) in enumerate(pairs):
        d = (day + pd.Timedelta(days=7 * (i // 2))).strftime("%d/%m/%Y")
        rows.append(f"Japan, J1 League,2023,{d},06:00,{h},{a},2,1,H,2.0,3.4,4.0,2.1,3.6,4.2,1.95,3.3,3.9,,,,,,,,,")
    # 附加赛：第 5 个队只出场 1 次（少于中位数 × 0.5），不导入
    rows.append(f"Japan, J1 League,2023,10/12/2023,06:00,{LEAGUE[0]},{PLAYOFF_TEAM},1,1,D,2.0,3.4,4.0,,,,,,,,,,,,,,,")
    # 同一赛季同一对主客队第二次出现（冠军赛）：只留日期最早的那场
    rows.append(f"Japan, J1 League,2023,17/12/2023,06:00,{LEAGUE[0]},{LEAGUE[1]},0,0,D,2.5,3.2,3.0,,,,,,,,,,,,,,,")
    # 秋春制赛季：2026/2027 → 2026；8 月是英国夏令时（UTC = 英国时间 − 1 小时）
    rows.append("Japan,J1 League,2026/2027,08/08/2026,11:00,Kofu,Urawa Reds,0,3,A,,,,,,,2.8,3.3,2.5,,,,,,,,,")
    return rows


CSV = ("﻿" + HEADER + "\n" + "\n".join(csv_rows()) + "\n").encode("utf-8")
NOTES = b"Notes for Football Data\n\nAll data is in csv format\n"


@pytest.fixture
def settings(tmp_path):
    s = copy.deepcopy(load_settings())
    s["paths"]["raw"] = str(tmp_path / "raw")        # 原始层放临时文件夹
    s["football_data_new"]["leagues"] = ["JPN"]
    return s


# ---------- 下载（假网站） ----------
class FakeResponse:
    def __init__(self, status_code, content=b""):
        self.status_code, self.content = status_code, content


class FakeSite:
    """假网站：pages 里写每个网址返回什么（状态码, 内容）；没写的返回 404。记下每次请求。"""
    def __init__(self, pages):
        self.pages, self.requests, self.headers = pages, [], {}

    def get(self, url, timeout):
        self.requests.append(url)
        code, content = self.pages.get(url, (404, b""))
        return FakeResponse(code, content)

    def close(self):
        pass


def downloader(con, raw, pages):
    site = FakeSite(pages)
    return site, dl.NewLeagueDownloader(con, raw, min_interval=0, session=site, sleep=lambda s: None)


def jpn_url(settings):
    return dl.new_league_url(settings["football_data_new"], "JPN")


def fetch_rows(con):
    return con.execute("select url, data_type, status, http_status from fetch_log order by id").fetchall()


def test_looks_like_new_csv():
    assert dl.looks_like_new_csv(CSV)                           # 带 BOM 也认
    assert not dl.looks_like_new_csv(b"Div,Date,HomeTeam\n")    # 老联赛文件不是
    assert not dl.looks_like_new_csv(b"<html>error</html>")


def test_download_saves_then_unchanged_then_new_snapshot(settings, tmp_path):
    con = connect(tmp_path / "test.db")              # 临时数据库文件（带 fetch_log 表）
    raw, url = tmp_path / "raw", jpn_url(settings)
    folder = raw / "football-data" / "new" / "JPN"
    site, d = downloader(con, raw, {url: (200, CSV), dl.NOTES_URL: (200, NOTES)})
    d.run_new(settings["football_data_new"], date(2026, 11, 3))
    assert site.requests == [dl.NOTES_URL, url]                 # 只请求列说明和这一个联赛
    assert (folder / "2026-11-03.csv").read_bytes() == CSV      # 原样保存
    _, d = downloader(con, raw, {url: (200, CSV)})
    assert d.run_new(settings["football_data_new"], date(2026, 11, 4), with_notes=False) == {"unchanged": 1}
    assert not (folder / "2026-11-04.csv").exists()             # 没变化不存新文件
    _, d = downloader(con, raw, {url: (200, CSV + b"Japan,J1 League,2026/2027,15/08/2026,11:00,Kofu,Cerezo Osaka,1,0,H\n")})
    d.run_new(settings["football_data_new"], date(2026, 11, 10), with_notes=False)
    assert sorted(p.name for p in folder.glob("*.csv")) == ["2026-11-03.csv", "2026-11-10.csv"]
    assert [r[2] for r in fetch_rows(con)] == ["saved", "saved", "unchanged", "saved"]
    assert {r[1] for r in fetch_rows(con) if r[0] == url} == {dl.DATA_TYPE}   # 和老联赛的 league_csv 分开记


@pytest.mark.parametrize("code", [403, 429])
def test_download_stops_on_403_429(settings, tmp_path, code):
    con = connect(tmp_path / "test.db")              # 临时数据库文件（带 fetch_log 表）
    site, d = downloader(con, tmp_path / "raw", {dl.NOTES_URL: (code, b"")})
    with pytest.raises(dl.Stopped):
        d.run_new(settings["football_data_new"], date(2026, 11, 3))
    assert site.requests == [dl.NOTES_URL]                      # 停下后不再请求联赛文件，也不重试
    assert fetch_rows(con) == [(dl.NOTES_URL, "notes", "failed", code)]


def test_download_missing_and_suspicious(settings, tmp_path):
    con = connect(tmp_path / "test.db")              # 临时数据库文件（带 fetch_log 表）
    raw = tmp_path / "raw"
    settings["football_data_new"]["leagues"] = ["JPN", "CHN"]
    chn = dl.new_league_url(settings["football_data_new"], "CHN")
    _, d = downloader(con, raw, {jpn_url(settings): (200, b"<html>Sorry</html>")})   # CHN 没写 → 404
    counts = d.run_new(settings["football_data_new"], date(2026, 11, 3), with_notes=False)
    assert counts == {"suspicious": 1, "not_found": 1}
    jpn = raw / "football-data" / "new" / "JPN"
    assert [p.name for p in jpn.iterdir()] == ["2026-11-03.suspicious.html"]          # 可疑内容另存，不当 csv
    assert imp.latest_snapshot(raw, "new", "JPN") is None                            # 导入时看不到它
    assert not (raw / "football-data" / "new" / "CHN").exists()
    assert fetch_rows(con)[1] == (chn, dl.DATA_TYPE, "not_found", 404)


# ---------- 导入（小样本 CSV） ----------
@pytest.mark.parametrize("text, year", [("2014", 2014), (" 2023 ", 2023), ("2026/2027", 2026)])
def test_season_from_text(text, year):
    assert imp.season_from_text(text) == year


@pytest.mark.parametrize("text", ["2026/2028", "26/27", "2014/15", ""])
def test_season_from_text_rejects_unknown(text):
    with pytest.raises(ValueError):
        imp.season_from_text(text)


def downloaded(settings, tmp_path, content=CSV):
    """用假网站“下载”一次：原始层有快照，fetch_log 有对得上的记录（导入要求的）。返回数据库连接。"""
    con = connect(tmp_path / "test.db")              # 临时数据库文件（带 fetch_log 表）
    _, d = downloader(con, tmp_path / "raw", {jpn_url(settings): (200, content)})
    d.run_new(settings["football_data_new"], date(2026, 11, 3), with_notes=False)
    return con


def table(con, sql):
    return con.execute(sql).fetchall()


def test_import_small_csv(settings, tmp_path):
    con = downloaded(settings, tmp_path)
    out = imp.run(settings, con)
    rep = out["leagues"]["JPN"]
    assert out["matches"] == 13                                  # 12 场联赛 + 2026 赛季 1 场
    assert [p[1:4] for p in rep["playoffs"]] == [["10/12/2023", "Kashima Antlers", "Gamba Osaka"]]   # 附加赛不导入，列进报告
    assert rep["repeats"] == ["JPN|2023|17/12/2023|Kashima Antlers|Urawa Reds"]      # 重复的主客队只留最早一场
    assert rep["teams_per_season"] == {2023: 4, 2026: 2}
    assert table(con, "select count(*) from matches where home_team = 'Gamba Osaka' or away_team = 'Gamba Osaka'") == [(0,)]
    # 赛季用开始年份；开球时间换成 UTC：英国夏令时（2023-03-26 起、8 月）减 1 小时，冬令时（3 月 4 日，下面那场）不变
    assert table(con, "select season, kickoff_utc from matches where home_team = 'Kofu' and away_team = 'Urawa Reds' "
                      "order by season") == [(2023, "2023-04-08 05:00"), (2026, "2026-08-08 10:00")]
    first = table(con, "select m.kickoff_utc, s.home_goals, s.away_goals, s.result from matches m join match_stats s "
                       "using (match_id) where home_team = 'Kashima Antlers' and away_team = 'Urawa Reds'")
    assert first == [("2023-03-04 06:00", 2, 1, "H")]           # 留下的是 3 月那场，不是 12 月冠军赛
    books = dict(table(con, "select bookmaker, count(*) from odds where market = '1x2' and timing = 'close' group by 1"))
    assert books == {"平博 Pinnacle": 12, "多家最高": 12, "多家平均": 13}   # 2026 那场只有平均收盘
    assert table(con, "select distinct source from match_sources") == [("football-data-new",)]
    assert table(con, "select count(*) from team_names where source = 'football-data-new'") == [(4,)]


def test_import_twice_same_ids(settings, tmp_path):
    """幂等：同一份快照导两次，比赛编号、场数、比分、赔率都一样，不会重复。"""
    con = downloaded(settings, tmp_path)
    imp.run(settings, con)
    queries = ["select match_id, competition, season, home_team, away_team, kickoff_utc from matches order by 1",
               "select * from match_stats order by match_id", "select count(*) from odds",
               "select * from match_sources order by match_id"]
    before = [table(con, q) for q in queries]
    imp.run(settings, con)
    assert [table(con, q) for q in queries] == before


def test_import_refuses_unverified_snapshot(settings, tmp_path):
    """原始层的文件和下载记录对不上（被改过、或者不是程序下载的）：不导入。"""
    con = downloaded(settings, tmp_path)
    path = imp.latest_snapshot(tmp_path / "raw", "new", "JPN")
    path.write_bytes(CSV + b"Japan,J1 League,2023,24/12/2023,06:00,Kofu,Cerezo Osaka,9,0,H\n")
    with pytest.raises(ValueError, match="下载记录"):
        imp.run(settings, con)
    assert table(con, "select count(*) from matches") == [(0,)]


def test_import_rejects_unknown_team(settings, tmp_path):
    con = downloaded(settings, tmp_path, CSV + b"Japan,J1 League,2023,24/12/2023,06:00,Kofu,Nowhere FC,1,0,H\n")
    with pytest.raises(ValueError, match="Nowhere FC"):
        imp.run(settings, con)


def test_league_teams_threshold():
    t = pd.DataFrame({"season": [2020] * 7,
                      "home_team": ["A", "B", "C", "A", "B", "C", "A"],
                      "away_team": ["B", "C", "A", "C", "A", "B", "X"]})
    assert imp.league_teams(t, 0.5) == {2020: {"A", "B", "C"}}   # X 只踢 1 场，中位数 4 × 0.5 = 2
    assert imp.league_teams(t, 0.0) == {2020: {"A", "B", "C", "X"}}


def test_old_league_import_keeps_new_league_rows(settings, tmp_path):
    """老联赛（football-data）整批重导只删自己的行：日职、中超（football-data-new）的比赛、比分、赔率都还在。"""
    con = downloaded(settings, tmp_path)
    imp.run(settings, con)
    before = [table(con, f"select count(*) from {t} where source = 'football-data-new'")
              for t in ("match_sources", "match_stats", "odds", "team_names")]
    old = pd.DataFrame([{"Div": "E0", "Date": "16/08/2025", "Time": "15:00", "HomeTeam": "Arsenal",
                         "AwayTeam": "Chelsea", "FTHG": "2", "FTAG": "1", "FTR": "H"}]).assign(row=0)
    names = {"Arsenal": "Arsenal", "Chelsea": "Chelsea"}
    slots = odds_slots(load_football_data_columns(settings))
    m, s, o, _ = parse_file(old, "E0", 2025, names, slots, settings, [], Counter())
    m = m.assign(raw_file="data/raw/test.csv", fetched_at_utc="2026-10-30T00:00:00+00:00")
    write_database(con, [{"英文": n, "国家": "英格兰"} for n in names], names, m, s, o)
    after = [table(con, f"select count(*) from {t} where source = 'football-data-new'")
             for t in ("match_sources", "match_stats", "odds", "team_names")]
    assert after == before and before[0] == [(13,)]
    assert table(con, "select count(*) from matches where competition = 'E0'") == [(1,)]


def test_teams_yaml_covers_real_aliases():
    """改过名的中超球队合并成一个编号（别名），日职、中超的队都登记了国家。"""
    teams, names = load_teams(load_settings())
    assert names["Shanghai SIPG"] == names["Shanghai Port"] == "Shanghai Port"
    assert names["Shandong Luneng"] == "Shandong Taishan"
    assert names["Guangzhou Evergrande"] == "Guangzhou FC"
    countries = {t["英文"]: t["国家"] for t in teams}
    assert countries["Kashima Antlers"] == "日本" and countries["Beijing Guoan"] == "中国"


# ---------- 回测取目标的日期窗口（一年一个赛季的联赛） ----------
def window_store(settings):
    """内存数据库：日职 2022、2023 赛季（开发）各 3 场，2024（考试）1 场；中超 2021 赛季最后一场踢到 2022-01-12；英超 1 场。"""
    from src.features import FeatureStore
    rows = [(1, "JPN", 2022, "A", "B", "2022-02-19"), (2, "JPN", 2022, "B", "C", "2022-07-02"),
            (3, "JPN", 2022, "C", "A", "2022-12-03"), (4, "JPN", 2023, "A", "C", "2023-02-17"),
            (5, "JPN", 2023, "B", "A", "2023-12-02"), (6, "JPN", 2023, "C", "B", "2023-06-10"),
            (7, "JPN", 2024, "A", "B", "2024-02-23"), (8, "CHN", 2021, "D", "E", "2022-01-12"),
            (9, "CHN", 2021, "E", "D", "2021-04-20"), (10, "E0", 2023, "F", "G", "2023-08-12")]
    c = sqlite3.connect(":memory:")
    create_tables(c)
    c.executemany("insert into teams (team_id, country) values (?, 'x')", [(t,) for t in "ABCDEFG"])
    c.executemany("insert into matches (match_id, competition, season, home_team, away_team, match_date, kickoff_utc,"
                  " status) values (?, ?, ?, ?, ?, ?, ? || ' 10:00', 'finished')", [(*r, r[-1]) for r in rows])
    c.commit()
    return FeatureStore(c, "开发", settings)


def test_calendar_window_takes_whole_seasons(settings):
    from src.evaluate.backtest import season_targets
    fs = window_store(settings)
    t = season_targets(fs, ["JPN"], 2022, 2023)
    assert sorted(t["match_id"]) == [1, 2, 3, 4, 5, 6]          # 2–6 月的比赛也在（跨年窗口 6 月 15 日起会漏）
    t = season_targets(fs, ["CHN"], 2021, 2021)
    assert sorted(t["match_id"]) == [8, 9]                      # 中超 2021 踢到 2022-01-12，也算进 2021 赛季


def test_calendar_window_never_touches_exam_season(settings):
    """最后一个开发赛季 2023：窗口停在 2024-02-01，2024 赛季（考试）2 月 23 日开赛，碰不到；
    赛季范围写到 2024，取数层直接报错。"""
    from src.evaluate.backtest import season_targets
    from src.features import SeasonLockedError
    fs = window_store(settings)
    assert 7 not in set(season_targets(fs, ["JPN"], 2023, 2023)["match_id"])
    with pytest.raises(SeasonLockedError):
        season_targets(fs, ["JPN"], 2023, 2024)


def test_calendar_and_cross_year_leagues_cannot_mix(settings):
    from src.evaluate.backtest import season_targets
    with pytest.raises(ValueError, match="不能放在同一次回测"):
        season_targets(window_store(settings), ["JPN", "E0"], 2022, 2023)


def test_asia_leagues_do_not_change_model_fingerprints():
    """加了 leagues.asia 以后，所有模型的设置（模型版本指纹）和没有 asia 时完全一样：上线的 gb_l7 不会变成新版本。"""
    from src.models.registry import available, model_config
    from src.predict.preview import model_version
    s = load_settings()
    without = copy.deepcopy(s)
    without["leagues"].pop("asia")
    for name in available(s):
        assert model_config(name, s) == model_config(name, without), name
    cfg = model_config("gb_l7", s)
    assert "JPN" not in cfg["leagues"] and "CHN" not in cfg["leagues"]
    assert model_version("gb_l7", cfg) == model_version("gb_l7", model_config("gb_l7", without))


def test_rest_day_schedule_skips_other_season_system(settings):
    """回测五大联赛 2023/24 时截止时间到 2024 年 5 月：取全部赛程会碰到 2 月开赛的日职 2024（考试赛季）报错；
    排除另一种赛季制度的联赛以后不报错，五大联赛的赛程照样有。"""
    from src.evaluate.backtest import other_season_system
    from src.features import SeasonLockedError
    fs = window_store(settings)
    with pytest.raises(SeasonLockedError):
        fs.schedule("2024-05-01 00:00:00", "休息天数和欧战后标签")
    skip = other_season_system(settings, ["E0"])
    assert "JPN" in skip and "CHN" in skip and "E0" not in skip
    got = fs.schedule("2024-05-01 00:00:00", "休息天数和欧战后标签", exclude_competitions=skip)
    assert set(got["competition"]) == {"E0"}
    other = other_season_system(settings, ["JPN"])               # 日职回测：排除跨年赛季的联赛，不排除自己
    assert "E0" in other and "JPN" not in other and "CHN" not in other


def test_after_euro_labels_skip_other_season_system(settings):
    """复盘标签的“欧战后”：数据库里有日职 2024（考试赛季）时，算英超 2023/24 的标签（截止时间到 2024 年 5 月）不报错。
    以前取赛程不排除日职、中超，整批重算标签会在第一个联赛的 2023 赛季报 SeasonLockedError。"""
    from src.review.labels import after_euro_flags
    fs = window_store(settings)
    sched = pd.DataFrame([{"match_id": 10, "competition": "E0", "season": 2023, "home_team": "F", "away_team": "G",
                           "match_date": "2023-08-12", "kickoff_earliest_utc": "2023-08-12 10:00:00",
                           "cutoff_utc": "2024-05-01 00:00:00"}])
    assert after_euro_flags(fs, sched, settings) == {10: (False, False)}
