"""懂球帝导入数据库（src/clean/dongqiudi_import.py）和取数层变盘记录、彩票赔率的自动检查。

用临时文件夹里的假原始文件 + 临时数据库，不碰真实数据。
造的数据：4 支英超球队。
  2023/24：双循环 12 场，football-data 有比分；懂球帝赛季页也有这 12 场（中文队名），用来投票定队名。
  2026/27（当前赛季）：懂球帝赛程里 4 场
    901 曼城 对 切尔西  已踢完 2:0（football-data 还没更新）
    902 利物浦 对 阿森纳  2026-11-07 15:00 开球，有赛前快照（两个快照，价格一部分重复）和彩票赔率
    903 阿森纳 对 曼城  时间待定（Uncertain），占位时间 2026-11-29 16:00
    904 曼城 对 某某队  对不上的队名 → 不导入
"""
import copy
import gzip
import json
from datetime import date, datetime, timedelta, timezone

import pytest

from src.clean.dongqiudi_import import fill_teams_yaml, run
from src.clean.schema import create_tables
from src.common.config import load_settings
from src.common.db import connect
from src.features import FeatureStore, LookAheadError

UTC = timezone.utc
TEAMS = [("曼城", "Man City"), ("切尔西", "Chelsea"), ("利物浦", "Liverpool"), ("阿森纳", "Arsenal")]
NOW = datetime(2026, 11, 6, 12, 0, tzinfo=UTC)


def gz(path, obj):
    path.parent.mkdir(parents=True, exist_ok=True)
    data = obj.encode("utf-8") if isinstance(obj, str) else json.dumps(obj, ensure_ascii=False).encode("utf-8")
    path.write_bytes(gzip.compress(data))


def fx(mid, home, away, start, status="Fixture", score=("", "")):
    return {"match_id": str(mid), "team_A_name": home, "team_B_name": away, "start_play": start, "status": status,
            "fs_A": score[0], "fs_B": score[1], "gameweek": "10"}


def book(name, cid, begin, now, market="euro"):
    """一家公司的赔率。begin、now 是 (主, 平或盘口, 客, 北京时间)。"""
    def one(v):
        h, mid, a, ts = v
        if market == "euro":
            return {"homeWin": h, "draw": mid, "awayWin": a, "ts": ts}
        return {"homeWin": h, "awayWin": a, "draw": mid, "draw_value": mid, "ts": ts}
    return {"name": name, "cid": cid, "fixed_odds": 0, "begin": one(begin), "now": one(now)}


@pytest.fixture
def env(tmp_path):
    s = copy.deepcopy(load_settings())
    s["dongqiudi"]["league_ids"] = {"E0": 4}
    con = connect(tmp_path / "t.db")   # 带抓取记录表 fetch_log
    create_tables(con)
    con.executemany("insert into teams (team_id, country) values (?, '英格兰')", [(en,) for _zh, en in TEAMS])
    raw = tmp_path / "raw"
    # 2023/24：双循环，football-data 有比分；懂球帝赛季页有同样的比赛
    html, k = [], 0
    for i, (zh_h, en_h) in enumerate(TEAMS):
        for j, (zh_a, en_a) in enumerate(TEAMS):
            if i == j:
                continue
            k += 1
            d = date(2023, 8, 12) + timedelta(days=7 * k)
            ko = f"{d.isoformat()} 14:00"
            con.execute("insert into matches (match_id, competition, season, home_team, away_team, match_date, kickoff_utc,"
                        " status) values (?, 'E0', 2023, ?, ?, ?, ?, 'finished')", (k, en_h, en_a, d.isoformat(), ko))
            con.execute("insert into match_sources values (?, 'football-data', ?, 'x.csv', '2024-06-01 00:00:00')",
                        (k, f"E0|{d}|{en_h}|{en_a}"))
            con.execute("insert into match_stats (match_id, source, home_goals, away_goals) values (?, 'football-data', ?, ?)",
                        (k, k % 5, k // 5))
            html.append(f'<li><span class="d">{d.month:02d}-{d.day:02d}</span><a href="/zhanbao/{1000 + k}">'
                        f'{zh_h} <b>{k % 5}-{k // 5}</b> {zh_a}<!----></a></li>')
    con.commit()
    gz(raw / "dongqiudi" / "season" / "E0" / "2023.html.gz",
       '<h2><a href="/zhanbao/lunci/1-1">第1轮</a></h2>' + "".join(html))
    # 2023/24 第 1 场（match_id 1，主队曼城 2023-08-19 14:00 UTC = 北京 22:00）的最终赔率
    gz(raw / "dongqiudi" / "odds_final" / "E0" / "2023" / "1001.json.gz",
       {"matchId": 1001, "euro": [book("Bet3**", "281", ("1.50", "4.00", "6.00", "08-15 10:00"),
                                       ("1.40", "4.50", "7.00", "08-19 21:50"))], "asia": [], "size": []})
    # 2026/27 赛程（两次抓取，后一次覆盖前一次：903 先是有时间的，后来变成时间待定）
    gz(raw / "dongqiudi" / "fixtures" / "E0" / "20261101T0000Z_1.json.gz",
       {"list": [fx(903, "阿森纳", "曼城", "2026-11-28 15:00:00")]})
    gz(raw / "dongqiudi" / "fixtures" / "E0" / "20261105T0000Z_1.json.gz", {"list": [
        fx(901, "曼城", "切尔西", "2026-11-01 15:00:00", "Played", ("2", "0")),
        fx(902, "利物浦", "阿森纳", "2026-11-07 15:00:00"),
        fx(903, "阿森纳", "曼城", "2026-11-29 16:00:00", "Uncertain"),
        fx(904, "曼城", "某某队", "2026-11-08 15:00:00")]})
    # 902 的两个赛前快照：Bet3** 的初盘两次都一样（只留一行），最新变了；还有彩票、没对照的公司
    lot = {"name": "竞彩官方", "cid": "1129", "fixed_odds": 0,
           "begin": {"homeWin": "1.87", "draw": "3.45", "awayWin": "3.20"}, "now": {"homeWin": "1.87", "draw": "3.45", "awayWin": "3.20"}}
    bd = {"name": "北单SP", "cid": "9998", "fixed_odds": 0,
          "begin": {"homeWin": "0.00", "draw": "0.00", "awayWin": "0.00"}, "now": {"homeWin": "2.58", "draw": "3.77", "awayWin": "2.86"}}
    gz(raw / "dongqiudi" / "odds_live" / "E0" / "902" / "20261105T0100Z.json.gz",
       {"matchId": 902, "euro": [book("Bet3**", "281", ("2.00", "3.50", "3.80", "11-04 10:00"), ("1.95", "3.60", "3.90", "11-05 08:30")),
                                 book("Odds**", "77", ("2.10", "3.40", "3.60", "11-04 10:00"), ("2.10", "3.40", "3.60", "11-04 10:00")),
                                 lot, bd],
        "asia": [book("平*", "47", ("0.90", "0.50", "0.95", "11-04 11:00"), ("0.88", "0.50", "0.97", "11-05 09:00"), "asia")],
        "size": []})
    lot2 = dict(lot, now={"homeWin": "1.80", "draw": "3.50", "awayWin": "3.40"})
    gz(raw / "dongqiudi" / "odds_live" / "E0" / "902" / "20261106T0100Z.json.gz",
       {"matchId": 902, "euro": [book("Bet3**", "281", ("2.00", "3.50", "3.80", "11-04 10:00"), ("1.90", "3.60", "4.00", "11-06 08:00")),
                                 lot2], "asia": [], "size": []})
    gz(raw / "dongqiudi" / "odds_live" / "E0" / "902" / "20261106T0200Z.suspicious.json.gz", {"matchId": 5})
    return con, raw, tmp_path / "reports", s


def go(env, **kw):
    con, raw, reports, s = env
    return run(con, raw, reports, s, now=NOW, **kw)


def test_teams_and_matches(env):
    con = env[0]
    out = go(env)
    assert out["mapping"] == dict(TEAMS)
    assert dict(con.execute("select name, team_id from team_names where source = 'dongqiudi'")) == dict(TEAMS)
    assert con.execute("select name_zh from teams where team_id = 'Man City'").fetchone()[0] == "曼城"
    # 以前的赛季只对照：12 场都对上，比赛总数不变（12 场 + 当前赛季新增 3 场）
    assert con.execute("select count(*) from match_sources where source = 'dongqiudi' and match_id <= 12").fetchone()[0] == 12
    assert con.execute("select count(*) from matches").fetchone()[0] == 15
    new = {r[0]: r[1:] for r in con.execute(
        "select home_team || '-' || away_team, match_date, kickoff_utc, status, date_precision from matches where season = 2026")}
    assert new["Man City-Chelsea"] == ("2026-11-01", "2026-11-01 15:00", "scheduled", "day")   # 踢完了也记未赛
    assert new["Liverpool-Arsenal"] == ("2026-11-07", "2026-11-07 15:00", "scheduled", "day")
    # 时间待定：不写开球时间，日期往前挪 3 天，精度记“这一轮”
    assert new["Arsenal-Man City"] == ("2026-11-26", None, "scheduled", "round")
    assert "Man City-某某队" not in new
    assert {ln.result for ln in out["links"] if ln.cand.dqd_id == 904} == {"队名没对上"}


def test_odds_changes_dedup_and_bookmakers(env):
    con = env[0]
    go(env)
    mid = con.execute("select match_id from match_sources where source = 'dongqiudi' and source_key = '902'").fetchone()[0]
    rows = con.execute("select bookmaker, bookmaker_raw, market, line, home, away, changed_at_utc, raw_file from odds_changes"
                       " where match_id = ? order by bookmaker, changed_at_utc", (mid,)).fetchall()
    bet365 = [r for r in rows if r[0] == "Bet365"]
    # 初盘在两个快照里都有，只留一行；最新两次不一样，各一行
    assert [(r[4], r[6]) for r in bet365] == [(2.0, "2026-11-04 02:00:00"), (1.95, "2026-11-05 00:30:00"),
                                             (1.9, "2026-11-06 00:00:00")]
    assert bet365[0][7].endswith("20261105T0100Z.json.gz")      # 第一次看到它的文件
    odds_raw = [r for r in rows if r[1] == "Odds**"]
    assert len(odds_raw) == 1 and odds_raw[0][0] == "Odds**"     # 没对照的公司照原样；初盘最新一样只留一行
    ah = [r for r in rows if r[2] == "ah"]
    assert ah[0][0] == "平博 Pinnacle" and ah[0][3] == -0.5 and ah[0][4] == 1.9   # 香港盘 +1，盘口正负号换过来
    assert not [r for r in rows if r[1] in ("竞彩官方", "北单SP")]   # 彩票不进变盘记录


def test_lottery_and_placeholders(env):
    con = env[0]
    go(env)
    rows = con.execute("select product, handicap, home, fetched_at_utc from lottery_odds order by product, fetched_at_utc").fetchall()
    # 竞彩：第一个快照的初盘（最新和初盘一样，不重复），第二个快照价格变了多一行；北单的初盘是占位的 0，不要
    assert rows == [("北单", 0, 2.58, "2026-11-05 01:00:00"),
                    ("竞彩胜平负", 0, 1.87, "2026-11-05 01:00:00"), ("竞彩胜平负", 0, 1.8, "2026-11-06 01:00:00")]


def test_rerun_same_result(env):
    con = env[0]
    go(env)
    snap = lambda: (con.execute("select match_id, competition, season, home_team, away_team, match_date, kickoff_utc,"
                                " date_precision from matches order by match_id").fetchall(),
                    con.execute("select count(*) from odds_changes").fetchone(),
                    con.execute("select count(*) from lottery_odds").fetchone())
    first = snap()
    go(env)
    assert snap() == first   # 比赛编号、赔率行数都不变


def test_feature_store(env, tmp_path):
    con = env[0]
    go(env)
    s = copy.deepcopy(env[3])
    s["leagues"] = {"top": {"E0": "英超"}, "second": {}}
    fs = FeatureStore(con, "实时", s)
    mid = con.execute("select match_id from match_sources where source = 'dongqiudi' and source_key = '902'").fetchone()[0]
    got = fs.odds_changes(mid, "2026-11-05 12:00", "B版输入", bookmaker="Bet365")
    assert got["home"].tolist() == [2.0, 1.95] and (got["known_at_utc"] <= "2026-11-05 12:00:00").all()
    latest = fs.latest_odds(mid, "2026-11-06 12:00", "B版输入", market="1x2")
    assert dict(zip(latest["bookmaker"], latest["home"])) == {"Bet365": 1.9, "Odds**": 2.1}
    with pytest.raises(LookAheadError):
        fs.odds_changes(mid, "2026-11-07 15:00", "B版输入")    # 开球那一刻已经不是“赛前”
    lot = fs.lottery_odds(mid, "2026-11-05 12:00", "显示", product="竞彩胜平负")
    assert lot["home"].tolist() == [1.87]
    # 以前的比赛：只给开球前的价格（08-19 21:50 北京时间 = 13:50 UTC，开球 14:00，给；赛后看也一样）
    old = fs.odds_changes(1, "2024-06-01", "显示")
    assert old["home"].tolist() == [1.5, 1.4]


def test_fill_teams_yaml(tmp_path):
    p = tmp_path / "teams.yaml"
    p.write_text("球队:\n  - {英文: \"Man City\", 中文: '', 国家: 英格兰}  # 注释\n"
                 "  - {英文: \"Chelsea\", 中文: '切尔西', 国家: 英格兰}\n", encoding="utf-8")
    assert fill_teams_yaml(p, {"Man City": "曼城", "Chelsea": "别的名字"}) == 1
    text = p.read_text(encoding="utf-8")
    assert "{英文: \"Man City\", 中文: '曼城', 国家: 英格兰}  # 注释" in text and "'切尔西'" in text
