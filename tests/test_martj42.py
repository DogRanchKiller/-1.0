"""国际赛 martj42（第 11 阶段 INA）下载、导入、取数层的自动检查。用假网站、假原始文件，不联网，不碰真的数据库。

查的事：下载（robots.txt、只访问 raw.githubusercontent.com、403/429 马上停、同一天不重下）；
导入（赛季 = 年份、round = 日期、球队加“ NT”、点球大战进 cup_details、比分不是平局的点球不写、同一天同一对队两场报错、
登记过的跳过、编号撞了报错、考试赛季锁住、原始层少了年份报错、重跑一样、只动自己的来源、场地表不留孤儿）；
可知时间 'day_world'（'day'、'round' 一点不变）；取数层 match_venues（用途、运行模式、没有表时返回空表）；
数据检查（同一天同一队两场也查 'day_world'，INT 不当杯赛）；模型版本指纹加不加 INT 都一样。
"""
import copy
import sqlite3

import pandas as pd
import pytest

from src.clean import martj42_import as imp
from src.clean.quality import all_league_codes, cup_checks, league_codes, run_checks
from src.clean.schema import create_tables
from src.collect.martj42 import Martj42Downloader, Stopped, file_url
from src.common.config import load_settings
from src.common.db import connect
from src.features import FeatureStore, SeasonLockedError, UsageNotAllowedError
from src.features.known_time import _local_midnight_utc, compute_known_times

HEADER = "date,home_team,away_team,home_score,away_score,tournament,city,country,neutral"
RESULTS = [
    "2010-06-11,South Africa,Mexico,1,1,FIFA World Cup,Johannesburg,South Africa,FALSE",
    "2010-07-02,Uruguay,Ghana,1,1,FIFA World Cup,Johannesburg,South Africa,TRUE",       # 点球大战（平局）
    "2011-03-29,England,Ghana,1,1,Friendly,London,England,FALSE",
    "2011-11-15,Senegal,Egypt,1,0,FIFA World Cup qualification,Dakar,Senegal,FALSE",    # 点球大战，但比分不是平局
    "2023-12-10,Japan,Korea Republic,2,0,Friendly,Tokyo,Japan,FALSE",
    "2024-01-20,Japan,Iraq,1,2,AFC Asian Cup,Doha,Qatar,TRUE",                         # 考试赛季：不导入
    "2026-11-20,Japan,Iraq,,,Friendly,Tokyo,Japan,FALSE",                              # 还没踢
]
SHOOTOUTS = ["date,home_team,away_team,winner,first_shooter",
             "2010-07-02,Uruguay,Ghana,Uruguay,",
             "2011-11-15,Senegal,Egypt,Egypt,",
             "2012-05-05,Nowhere,Somewhere,Nowhere,",                                     # results.csv 里没有
             "2024-01-25,Japan,Iraq,Japan,"]


@pytest.fixture(scope="module")
def settings():
    return load_settings()


def write_raw(folder, results=RESULTS, shootouts=SHOOTOUTS):
    folder.mkdir(parents=True, exist_ok=True)
    (folder / "results.csv").write_text("\n".join([HEADER, *results]) + "\n", encoding="utf-8")
    (folder / "shootouts.csv").write_text("\n".join(shootouts) + "\n", encoding="utf-8")
    return folder


@pytest.fixture
def env(tmp_path, settings):
    folder = write_raw(tmp_path / "raw" / "martj42" / "2026-11-05")
    con = connect(tmp_path / "t.db")
    return con, folder


def rows(con, sql, *p):
    return con.execute(sql, p).fetchall()


# ---------- 下载 ----------
class FakeResponse:
    def __init__(self, status_code, content=b""):
        self.status_code, self.content = status_code, content
        self.text = content.decode("utf-8", "replace")


class FakeSite:
    """pages：网址 → (状态码, 内容)；没有的网址返回 404。"""
    def __init__(self, pages):
        self.pages, self.requests, self.headers = pages, [], {}

    def get(self, url, timeout):
        self.requests.append(url)
        return FakeResponse(*self.pages.get(url, (404, b"")))


def downloader(tmp_path, pages):
    con = connect(tmp_path / "fetch.db")
    site = FakeSite(pages)
    return con, site, Martj42Downloader(con, tmp_path / "raw", min_interval=0, session=site, sleep=lambda s: None)


def test_download_two_files_after_robots(tmp_path, settings):
    """先看 robots.txt（404 = 没有限制），再下两个文件，存到 raw/martj42/<日期>/；同一天再跑不再请求。"""
    from datetime import date
    cfg = settings["martj42"]
    res, sho = file_url(cfg, "results"), file_url(cfg, "shootouts")
    body = (HEADER + "\n").encode()
    con, site, d = downloader(tmp_path, {res: (200, body), sho: (200, b"date,home_team,away_team,winner\n")})
    d.run_martj42(cfg, date(2026, 11, 5))
    assert site.requests == [cfg["robots_url"], res, sho]
    out = tmp_path / "raw" / "martj42" / "2026-11-05"
    assert (out / "results.csv").read_bytes() == body and (out / "shootouts.csv").exists()
    assert {r[0] for r in rows(con, "select status from fetch_log where source = 'martj42'")} == {"not_found", "saved"}
    d.run_martj42(cfg, date(2026, 11, 5))
    assert len(site.requests) == 3


def test_download_stops_on_robots_disallow_and_429(tmp_path, settings):
    from datetime import date
    cfg = settings["martj42"]
    res = file_url(cfg, "results")
    _c, site, d = downloader(tmp_path, {cfg["robots_url"]: (200, b"User-agent: *\nDisallow: /\n")})
    with pytest.raises(Stopped):
        d.run_martj42(cfg, date(2026, 11, 5))
    assert site.requests == [cfg["robots_url"]]                  # robots.txt 不让下：一个文件都不请求
    _c, site, d = downloader(tmp_path / "b", {res: (429, b"")})
    with pytest.raises(Stopped):
        d.run_martj42(cfg, date(2026, 11, 5))
    assert site.requests[-1] == res and len(site.requests) == 2  # 429 马上停，不再下第二个


def test_only_raw_githubusercontent(settings):
    cfg = dict(settings["martj42"], base_url="https://example.com/x")
    with pytest.raises(ValueError):
        file_url(cfg, "results")


# ---------- 导入 ----------
def test_import_basic(env, settings):
    """只导开发赛季、比分不空的；赛季 = 年份；round = 日期；球队加 NT；比分进 match_stats；场地进 match_venue。"""
    con, folder = env
    rep = imp.run(con, settings, folder)
    assert rep["matches"] == 5 and rep["after_last_date"] == 2 and rep["no_score"] == 0
    got = rows(con, "select competition, season, home_team, away_team, match_date, kickoff_utc, status, round, leg, "
                    "date_precision from matches order by match_date")
    assert got[1] == ("INT", 2010, "Uruguay NT", "Ghana NT", "2010-07-02", None, "finished", "2010-07-02", 0, "day_world")
    assert {r[1] for r in got} == {2010, 2011, 2023}
    assert rows(con, "select name_zh, country from teams where team_id = 'Korea Republic NT'") == [(None, "Korea Republic")]
    assert ("martj42", "Ghana", "Ghana NT") in rows(con, "select * from team_names")
    assert rows(con, "select home_goals, away_goals, result from match_stats s join matches m using (match_id) "
                     "where m.home_team = 'Japan NT'") == [(2, 0, "H")]
    venue = rows(con, "select v.neutral, v.city, v.country, v.tournament from match_venue v join matches m using (match_id) "
                      "where m.home_team = 'Uruguay NT'")
    assert venue == [(1, "Johannesburg", "South Africa", "FIFA World Cup")]
    assert rows(con, "select count(*) from match_venue") == [(5,)]
    keys = {r[0] for r in rows(con, "select source_key from match_sources where source = 'martj42'")}
    assert "2010-07-02|Uruguay|Ghana" in keys


def test_shootouts(env, settings):
    """平局踢点球的进 cup_details（加时 = 1、加时比分 = 比分、点球比分空、note 写胜者）；
    比分不是平局的、对不上的、考试赛季的不写，列进报告。"""
    con, folder = env
    rep = imp.run(con, settings, folder)
    cup = rows(con, "select m.home_team, c.source, c.extra_time, c.aet_home, c.aet_away, c.pen_home, c.pen_away, c.note "
                    "from cup_details c join matches m using (match_id)")
    assert cup == [("Uruguay NT", "martj42", 1, 1, 1, None, None, "点球大战胜者 Uruguay NT")]
    assert rows(con, "select result from match_stats s join matches m using (match_id) "
                     "where m.home_team = 'Uruguay NT'") == [("D",)]       # 点球大战胜平负按平局
    assert [r[:3] for r in rep["shootouts_not_draw"]] == [("2011-11-15", "Senegal", "Egypt")]
    assert rep["shootouts_unmatched"] == [("2012-05-05", "Nowhere", "Somewhere")]
    assert rep["shootouts_after_last_date"] == 1


def test_rerun_same_and_other_sources_untouched(env, settings):
    """同一批原始文件导两次，内容指纹一样、比赛编号不变；别的来源的比赛一行都不动。"""
    con, folder = env
    create_tables(con)
    with con:
        con.execute("insert into teams values ('Arsenal', null, 'England')")
        con.execute("insert into teams values ('Chelsea', null, 'England')")
        con.execute("insert into matches (competition, season, home_team, away_team, match_date, status) "
                    "values ('E0', 2010, 'Arsenal', 'Chelsea', '2010-08-14', 'finished')")
        con.execute("insert into match_sources values (1, 'football-data', 'k', null, null)")
    before = rows(con, "select * from matches where competition = 'E0'")
    imp.run(con, settings, folder)
    d1, ids1 = imp.content_digest(con, settings["martj42"]), rows(con, "select match_id, round from matches order by 1")
    imp.run(con, settings, folder)
    assert imp.content_digest(con, settings["martj42"]) == d1
    assert rows(con, "select match_id, round from matches order by 1") == ids1
    assert rows(con, "select * from matches where competition = 'E0'") == before


def test_same_day_pair_raises_unless_registered(tmp_path, settings):
    """同一天同一对队（主客对调也算）两场：报错；settings 登记过的不报错，这一天这一对都不导入。"""
    extra = ["2015-06-01,Peru,Chile,1,0,Friendly,Lima,Peru,FALSE", "2015-06-01,Chile,Peru,2,2,Friendly,Santiago,Chile,FALSE"]
    folder = write_raw(tmp_path / "r", RESULTS + extra)
    con = connect(tmp_path / "t.db")
    create_tables(con)
    with pytest.raises(ValueError, match="同一天同一对队"):
        imp.run(con, settings, folder)
    assert rows(con, "select count(*) from matches") == [(0,)]
    s = copy.deepcopy(settings)
    s["martj42"]["same_day_pairs"] = [["2015-06-01", "Chile", "Peru"]]
    rep = imp.run(con, s, folder)
    assert rep["matches"] == 5 and len(rep["same_day_skipped"]) == 2
    assert rows(con, "select count(*) from matches where match_date = '2015-06-01'") == [(0,)]


def test_team_id_clash_raises(env, settings):
    """“England NT”已经有了、又不是 martj42 建的：报错，一行都不写。"""
    con, folder = env
    create_tables(con)
    with con:
        con.execute("insert into teams values ('England NT', null, 'England')")
    with pytest.raises(ValueError, match="不是 martj42 建的"):
        imp.run(con, settings, folder)
    assert rows(con, "select count(*) from matches") == [(0,)]


def test_only_dev_seasons(env, settings):
    """last_date 设错（放进了 2024）时，程序按 season_role 再查一遍，考试赛季报错。"""
    con, folder = env
    s = copy.deepcopy(settings)
    s["martj42"]["last_date"] = "2024-12-31"
    create_tables(con)
    with pytest.raises(ValueError, match="不是开发赛季"):
        imp.run(con, s, folder)
    assert rows(con, "select count(*) from matches") == [(0,)]


def test_lost_seasons_raise_and_orphans_cleaned(env, tmp_path, settings):
    """原始层少了数据库里已有的年份：报错、一行都不动。只少一场（年份还在）：这场连同比分、场地一起删掉，不留孤儿。"""
    con, folder = env
    imp.run(con, settings, folder)
    before = imp.content_digest(con, settings["martj42"])
    no2011 = write_raw(tmp_path / "a", [r for r in RESULTS if not r.startswith("2011")])
    with pytest.raises(ValueError, match="没有"):
        imp.run(con, settings, no2011)
    assert imp.content_digest(con, settings["martj42"]) == before
    one_less = write_raw(tmp_path / "b", [r for r in RESULTS if not r.startswith("2011-03-29")])
    imp.run(con, settings, one_less)
    assert rows(con, "select count(*) from matches where home_team = 'England NT'") == [(0,)]
    assert rows(con, "select count(*) from match_venue") == [(4,)]
    assert rows(con, "select count(*) from match_venue where match_id not in (select match_id from matches)") == [(0,)]
    assert rows(con, "select count(*) from match_stats where match_id not in (select match_id from matches)") == [(0,)]


def test_int_registered_outside_top_second(settings):
    """INT 登记在 leagues.intl，不在 top / second；导入程序发现它被放进 top 就报错。"""
    assert settings["leagues"]["intl"] == {"INT": "国际赛"}
    assert "INT" not in league_codes(settings) and "INT" in all_league_codes(settings)
    s = copy.deepcopy(settings)
    s["leagues"]["top"]["INT"] = "国际赛"
    with pytest.raises(ValueError, match="top / second"):
        imp.run(sqlite3.connect(":memory:"), s)


# ---------- 可知时间 'day_world' ----------
def old_compute_known_times(matches, fs):
    """加 'day_world' 以前的算法（原样抄下来），用来证明 'day'、'round' 的结果一点没变。"""
    tz = fs["date_only_timezone"]
    buffer = pd.Timedelta(hours=float(fs["result_available_after_kickoff_hours"]))
    kick = pd.to_datetime(matches["kickoff_utc"], errors="coerce")
    day_start = _local_midnight_utc(matches["match_date"], tz, ambiguous_early=True)
    next_day = (pd.to_datetime(matches["match_date"], format="%Y-%m-%d") + pd.Timedelta(days=1)).dt.strftime("%Y-%m-%d")
    day_end = _local_midnight_utc(next_day, tz, ambiguous_early=False)
    if "date_precision" in matches.columns:
        extra_days = float(fs.get("round_date_window_days", 7)) - 1
        is_round = (matches["date_precision"] == "round").to_numpy()
        day_end = day_end.where(~is_round, day_end + pd.Timedelta(days=extra_days))
    earliest, latest = kick.fillna(day_start), kick.fillna(day_end)
    f = lambda s: s.dt.strftime("%Y-%m-%d %H:%M:%S").to_numpy()
    return pd.DataFrame({"match_id": matches["match_id"].to_numpy(), "kickoff_earliest_utc": f(earliest),
                         "kickoff_latest_utc": f(latest), "result_known_utc": f(latest + buffer),
                         "close_known_utc": f(latest)})


def test_day_world_times(settings):
    """'day_world'：最早开球 = 前一天 10:00 UTC（这一天 00:00 UTC+14），最晚开球 = 第二天 12:00 UTC（这一天 24:00 UTC−12），
    比分 = 最晚开球 + 3 小时，收盘 = 最晚开球。有开球时刻的照样用开球时刻。"""
    m = pd.DataFrame({"match_id": [1, 2], "match_date": ["2022-12-18", "2022-12-18"],
                      "kickoff_utc": [None, "2022-12-18 15:00"], "date_precision": ["day_world", "day_world"]})
    kt = compute_known_times(m, settings["features"]).set_index("match_id")
    assert kt.loc[1].to_dict() == {"kickoff_earliest_utc": "2022-12-17 10:00:00", "kickoff_latest_utc": "2022-12-19 12:00:00",
                                   "result_known_utc": "2022-12-19 15:00:00", "close_known_utc": "2022-12-19 12:00:00"}
    assert kt.loc[2, "kickoff_earliest_utc"] == kt.loc[2, "kickoff_latest_utc"] == "2022-12-18 15:00:00"


def test_day_and_round_unchanged(settings):
    """同一批 'day'、'round' 比赛（含夏令时切换那几天、有开球时刻的），加 'day_world' 前后算出来一模一样；
    和 'day_world' 混在一起算时，'day'、'round' 的行也一样；没有 date_precision 列时也一样。"""
    dates = ["2023-03-25", "2023-03-26", "2023-10-28", "2023-10-29", "2024-01-06", "2010-08-14", "1993-08-14"]
    n = len(dates)
    m = pd.DataFrame({"match_id": range(1, 2 * n + 2), "match_date": dates * 2 + ["2024-01-06"],
                      "kickoff_utc": [None] * (2 * n) + ["2024-01-06 15:00"],
                      "date_precision": ["day"] * n + ["round"] * n + ["day"]})
    fs = settings["features"]
    assert compute_known_times(m, fs).equals(old_compute_known_times(m, fs))
    mixed = pd.concat([m, pd.DataFrame({"match_id": [99], "match_date": ["2023-03-26"], "kickoff_utc": [None],
                                        "date_precision": ["day_world"]})], ignore_index=True)
    assert compute_known_times(mixed, fs).iloc[:len(m)].equals(old_compute_known_times(m, fs))
    plain = m.drop(columns="date_precision")
    assert compute_known_times(plain, fs).equals(old_compute_known_times(plain, fs))


# ---------- 取数层 match_venues ----------
def test_match_venues(env, settings):
    """“比赛场地”按数据字典查用途；赛前就知道，截止时间之前、之后的都给；可以按赛事、since 筛。"""
    con, folder = env
    imp.run(con, settings, folder)
    fs = FeatureStore(con, "开发", settings)
    v = fs.match_venues("2000-01-01", "研究实验", competition="INT")
    assert list(v.columns) == ["match_id", "source", "neutral", "city", "country", "tournament"] and len(v) == 5
    assert len(fs.match_venues("2000-01-01", "A版输入", since="2011-01-01")) == 3
    assert fs.match_venues("2000-01-01", "研究实验", competition="E0").empty
    with pytest.raises(UsageNotAllowedError):
        fs.match_venues("2000-01-01", "研究评估")             # 字典里“比赛场地”没写这个用途
    with pytest.raises(ValueError):
        fs.match_venues(None, "研究实验")                     # 截止时间必须给


def test_match_venues_exam_season_locked(env, settings):
    """开发模式取到期末考试赛季的场地：报错（不悄悄删掉）。"""
    con, folder = env
    imp.run(con, settings, folder)
    with con:
        con.execute("insert into matches (competition, season, home_team, away_team, match_date, status, round, "
                    "date_precision) values ('INT', 2024, 'Japan NT', 'Ghana NT', '2024-03-01', 'finished', "
                    "'2024-03-01', 'day_world')")
        con.execute("insert into match_venue values (last_insert_rowid(), 'martj42', 0, 'Tokyo', 'Japan', 'Friendly')")
    fs = FeatureStore(con, "开发", settings)
    with pytest.raises(SeasonLockedError):
        fs.match_venues("2000-01-01", "研究实验")


def test_match_venues_without_table(tmp_path, settings):
    """旧库还没有 match_venue 表：返回同样几列的空表，不报错、不建表。"""
    con = connect(tmp_path / "old.db")
    create_tables(con)
    con.execute("drop table match_venue")
    fs = FeatureStore(con, "开发", settings)
    v = fs.match_venues("2020-01-01", "研究实验")
    assert v.empty and list(v.columns) == ["match_id", "source", "neutral", "city", "country", "tournament"]
    assert not rows(con, "select 1 from sqlite_master where name = 'match_venue'")


# ---------- 数据检查 ----------
def test_quality_same_day_checks_day_world_and_int_not_cup(env, settings):
    """“同一天同一队踢两场”也查 'day_world'；INT 在 leagues 里，round 不空也不当杯赛查。"""
    from datetime import date
    con, folder = env
    extra = ["2011-03-29,Ghana,Togo,0,0,Friendly,Accra,Ghana,FALSE"]          # 加纳同一天又踢一场（对手不同）
    imp.run(con, settings, write_raw(folder.parent / "x", RESULTS + extra))
    found = {c["名称"]: c["rows"] for c in run_checks(con, settings, today=date(2026, 11, 5))}
    assert found["同一天同一队踢两场"] == ["2011-03-29 Ghana NT：2 场"]
    assert all(not r for _n, _e, r in cup_checks(con, all_league_codes(settings)))


# ---------- 五大联赛不受影响 ----------
def test_int_does_not_change_model_fingerprints():
    """加了 leagues.intl 以后，所有模型的设置（模型版本指纹）和没有 intl 时完全一样：上线的 gb_l7 不会变成新版本。"""
    from src.models.registry import available, model_config
    from src.predict.preview import model_version
    s = load_settings()
    without = copy.deepcopy(s)
    without["leagues"].pop("intl")
    for name in available(s):
        assert model_config(name, s) == model_config(name, without), name
        assert model_version(name, model_config(name, s)) == model_version(name, model_config(name, without)), name
    assert "INT" not in model_config("gb_l7", s)["leagues"]
