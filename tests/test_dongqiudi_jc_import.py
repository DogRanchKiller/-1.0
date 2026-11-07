"""竞彩联赛导入（src/clean/dongqiudi_jc_import.py）的自动检查。

用临时文件夹里的假原始文件 + 临时数据库，不碰真实数据。
造的数据（懂球帝赛程，中文队名 + 懂球帝球队编号）：
  荷甲 N1（7 月起算新赛季）
    7001 阿贾克斯(11) 对 埃因霍温(12)  2026-11-01 已踢完 2:1
    7002 埃因霍温(12) 对 费耶诺德(13)  2026-11-08 13:30 开球，有赛前快照（平博亚盘、竞彩胜平负）
    7003 阿贾克斯(11) 对 费耶诺德(13)  2026-05-10 已踢完（2025/26 赛季，是期末考试赛季，不导入）
  巴甲 BRA（一年一个赛季）
    8001 弗拉门戈(21) 对 帕尔梅拉斯(22)  2026-03-01 已踢完 1:1（按开球那年算 2026 赛季，导入）
"""
import copy
import gzip
import json
from datetime import datetime, timezone

import pytest

from src.clean.dongqiudi_jc_import import run, season_of
from src.clean.schema import create_tables
from src.common.config import load_settings
from src.common.db import connect
from src.features import FeatureStore, LookAheadError

UTC = timezone.utc
SOURCE = "dongqiudi-jc"


def gz(path, obj):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(gzip.compress(json.dumps(obj, ensure_ascii=False).encode("utf-8")))


def fx(mid, home, away, start, status="Fixture", score=("", ""), week="12"):
    """赛程接口里的一场。home、away 是 (懂球帝球队编号, 中文名)。"""
    return {"match_id": str(mid), "team_A_id": home[0], "team_A_name": home[1], "team_B_id": away[0],
            "team_B_name": away[1], "start_play": start, "status": status, "fs_A": score[0], "fs_B": score[1],
            "gameweek": week}


AJAX, PSV, FEY = ("11", "阿贾克斯"), ("12", "埃因霍温"), ("13", "费耶诺德")
FLA, PAL = ("21", "弗拉门戈"), ("22", "帕尔梅拉斯")


def book(name, cid, begin, now, market="euro"):
    """一家公司的赔率。begin、now 是 (主, 平或盘口, 客, 北京时间)。"""
    def one(v):
        h, mid, a, ts = v
        if market == "euro":
            return {"homeWin": h, "draw": mid, "awayWin": a, "ts": ts}
        return {"homeWin": h, "awayWin": a, "draw": mid, "draw_value": mid, "ts": ts}
    return {"name": name, "cid": cid, "fixed_odds": 0, "begin": one(begin), "now": one(now)}


def fixtures_n1(raw, stamp="20261105T0000Z", ajax=AJAX):
    gz(raw / "dongqiudi" / "fixtures" / "N1" / f"{stamp}_1.json.gz", {"list": [
        fx(7001, ajax, PSV, "2026-11-01 13:30:00", "Played", ("2", "1")),
        fx(7002, PSV, FEY, "2026-11-08 13:30:00"),
        fx(7003, ajax, FEY, "2026-05-10 12:00:00", "Played", ("3", "0"), week="34")]})


@pytest.fixture
def env(tmp_path):
    s = copy.deepcopy(load_settings())
    s["dongqiudi_jc"]["collect"]["league_ids"] = {"N1": 1, "BRA": 19}
    con = connect(tmp_path / "t.db")   # 带抓取记录表 fetch_log
    create_tables(con)
    raw = tmp_path / "raw"
    fixtures_n1(raw)
    gz(raw / "dongqiudi" / "fixtures" / "BRA" / "20261105T0000Z_1.json.gz", {"list": [
        fx(8001, FLA, PAL, "2026-03-01 22:00:00", "Played", ("1", "1"), week="1")]})
    lot = {"name": "竞彩官方", "cid": "1129", "fixed_odds": 0,
           "begin": {"homeWin": "1.70", "draw": "3.90", "awayWin": "3.80"},
           "now": {"homeWin": "1.70", "draw": "3.90", "awayWin": "3.80"}}
    gz(raw / "dongqiudi" / "odds_live" / "N1" / "7002" / "20261106T0100Z.json.gz",
       {"matchId": 7002, "euro": [book("Bet3**", "281", ("1.80", "3.80", "4.20", "11-05 10:00"),
                                       ("1.75", "3.90", "4.40", "11-06 08:30")), lot],
        "asia": [book("平*", "47", ("0.90", "0.75", "0.95", "11-05 11:00"),
                      ("0.88", "0.75", "0.97", "11-06 09:00"), "asia")],
        "size": []})
    return con, raw, tmp_path / "reports", s


def go(env):
    con, raw, reports, s = env
    return run(con, raw, reports, s)


def mid_of(con, dqd_id):
    return con.execute("select match_id from match_sources where source = ? and source_key = ?",
                       (SOURCE, str(dqd_id))).fetchone()[0]


def test_season_of():
    assert season_of("N1", datetime(2026, 7, 1, tzinfo=UTC), ["BRA"]) == 2026
    assert season_of("N1", datetime(2026, 6, 30, tzinfo=UTC), ["BRA"]) == 2025
    assert season_of("BRA", datetime(2026, 3, 1, tzinfo=UTC), ["BRA"]) == 2026


def test_matches_teams_and_scores(env):
    con = env[0]
    out = go(env)
    rows = {r[0]: r[1:] for r in con.execute(
        "select s.source_key, m.competition, m.season, m.home_team, m.away_team, m.kickoff_utc, m.status, m.round"
        " from matches m join match_sources s on s.match_id = m.match_id and s.source = ?", (SOURCE,))}
    assert rows["7001"] == ("N1", 2026, "阿贾克斯", "埃因霍温", "2026-11-01 13:30", "finished", "12")
    assert rows["7002"] == ("N1", 2026, "埃因霍温", "费耶诺德", "2026-11-08 13:30", "scheduled", "12")
    assert rows["8001"][:2] == ("BRA", 2026)          # 巴甲 3 月开球：按开球那年算 2026 赛季
    assert "7003" not in rows                          # 2025/26 不是上线赛季：不导入
    assert out["counts"]["比赛"] == 3 and out["counts"]["球队"] == 5
    assert dict(con.execute("select team_id, country from teams")) == {
        "阿贾克斯": "荷兰", "埃因霍温": "荷兰", "费耶诺德": "荷兰", "弗拉门戈": "巴西", "帕尔梅拉斯": "巴西"}
    assert con.execute("select count(*) from team_names where source = ?", (SOURCE,)).fetchone()[0] == 5
    # 比分：只写踢完的两场
    assert sorted(con.execute("select home_goals, away_goals, result from match_stats where source = ?", (SOURCE,))) \
        == [(1, 1, "D"), (2, 1, "H")]
    assert (env[2] / env[3]["dongqiudi_jc"]["report_file"]).exists()


def test_odds_and_lottery_use_own_source(env):
    con = env[0]
    go(env)
    mid = mid_of(con, 7002)
    rows = con.execute("select source, bookmaker, market, line, home from odds_changes where match_id = ?"
                       " order by market, changed_at_utc", (mid,)).fetchall()
    assert {r[0] for r in rows} == {SOURCE}
    assert [r[1:] for r in rows if r[2] == "ah"][0] == ("平博 Pinnacle", "ah", -0.75, 1.9)
    assert [r[4] for r in rows if r[2] == "1x2"] == [1.8, 1.75]
    assert con.execute("select source, product, home from lottery_odds where match_id = ?", (mid,)).fetchall() \
        == [(SOURCE, "竞彩胜平负", 1.7)]
    assert con.execute("select count(*) from odds_changes where source = 'dongqiudi'").fetchone()[0] == 0


def test_rerun_keeps_match_ids(env):
    con = env[0]
    first = go(env)["ids"]
    snap = lambda: (con.execute("select * from matches order by match_id").fetchall(),
                    con.execute("select count(*) from odds_changes").fetchone(),
                    con.execute("select count(*) from lottery_odds").fetchone(),
                    con.execute("select count(*) from match_stats").fetchone())
    before = snap()
    assert go(env)["ids"] == first
    assert snap() == before


def test_team_rename_keeps_match_id(env):
    """懂球帝把“阿贾克斯”改叫“阿贾克斯队”：球队编号换成新写法，比赛编号不变。
    （每场比赛只看最后一次的样子，三场都换了新写法，旧写法就不再出现在 team_names 里。）"""
    con, raw = env[0], env[1]
    old = go(env)["ids"]
    fixtures_n1(raw, stamp="20261106T0000Z", ajax=("11", "阿贾克斯队"))
    new = go(env)["ids"]
    assert new == old
    assert con.execute("select home_team from matches where match_id = ?", (old[7001],)).fetchone()[0] == "阿贾克斯队"
    names = dict(con.execute("select name, team_id from team_names where source = ?", (SOURCE,)))
    assert names["阿贾克斯队"] == "阿贾克斯队" and "阿贾克斯" not in names
    assert con.execute("select count(*) from odds_changes where match_id = ?", (old[7002],)).fetchone()[0] > 0


def test_two_dqd_teams_with_same_name_refused(env):
    con, raw = env[0], env[1]
    gz(raw / "dongqiudi" / "fixtures" / "BRA" / "20261105T0000Z_1.json.gz", {"list": [
        fx(8001, FLA, ("99", "埃因霍温"), "2026-03-01 22:00:00", "Played", ("1", "1"), week="1")]})
    with pytest.raises(ValueError, match="不合并"):
        go(env)
    assert con.execute("select count(*) from matches").fetchone()[0] == 0


def test_team_used_by_other_source_refused(env):
    con = env[0]
    con.execute("insert into teams (team_id, country) values ('费耶诺德', '荷兰')")
    con.execute("insert into team_names (source, name, team_id) values ('football-data', 'Feyenoord', '费耶诺德')")
    con.commit()
    with pytest.raises(ValueError, match="别的来源已经在用"):
        go(env)
    assert con.execute("select count(*) from matches").fetchone()[0] == 0


def test_missing_fixture_files_do_not_delete_odds(env):
    """原始层少了荷甲的赛程文件：数据库里已经有赔率的比赛对不上 → 报错，一行不动。"""
    con, raw = env[0], env[1]
    go(env)
    before = con.execute("select count(*) from odds_changes where source = ?", (SOURCE,)).fetchone()[0]
    assert before > 0
    for f in (raw / "dongqiudi" / "fixtures" / "N1").glob("*.json.gz"):
        f.unlink()
    with pytest.raises(ValueError, match="原始层里没有了"):
        go(env)
    assert con.execute("select count(*) from odds_changes where source = ?", (SOURCE,)).fetchone()[0] == before


def test_feature_store_reads_jc_league(env):
    """取数层：竞彩联赛（leagues.jc）在实时模式下能拿到竞彩价格和变盘记录，只给决定时间以前的。"""
    con, s = env[0], env[3]
    go(env)
    mid = mid_of(con, 7002)
    fs = FeatureStore(con, "实时", s)
    lot = fs.lottery_odds(mid, "2026-11-06 12:00", "机器人下注", product="竞彩胜平负")
    assert lot["home"].tolist() == [1.7]
    assert fs.lottery_odds(mid, "2026-11-06 00:30", "机器人下注", product="竞彩胜平负").empty   # 还没抓到
    ah = fs.odds_changes(mid, "2026-11-06 12:00", "机器人下注", market="ah")
    assert ah["bookmaker"].unique().tolist() == ["平博 Pinnacle"]
    with pytest.raises(LookAheadError):
        fs.odds_changes(mid, "2026-11-08 13:30", "机器人下注")   # 开球那一刻已经不是“赛前”
