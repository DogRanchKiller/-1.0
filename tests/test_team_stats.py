"""球队页四块（src/web/site/team_stats.py：近期状态、历史交锋、Elo 评分变化、积分榜）的自动检查。

内存里的临时数据库，只有英超 E0，4 支队 A、B、C、D，不碰真实数据。现在 = 2026-09-10 12:00 UTC。
  1  E0 2025/26（期末考试赛季）A 对 B  2026-03-01 15:00  2:0
  2  E0 2025/26               C 对 A  2026-04-01 15:00  1:1
  3  E0 2025/26               B 对 A  2026-05-01 15:00  1:0
  4  E0 2026/27（上线后）       A 对 C  2026-08-20 15:00  3:1
  5  E0 2026/27               B 对 D  2026-08-20 15:00  0:0
  6  E0 2026/27               D 对 A  2026-08-27 15:00  0:2
  7  E0 2026/27               C 对 B  2026-08-27 15:00  2:2
  8  FAC（足总杯）2026/27       A 对 D  2026-09-03 19:45  0:5   ← 杯赛，球队页四块都不算它
  9  E0 2026/27 A 对 B 2026-09-15 15:00 还没踢（A、B 接下来的联赛）
  10 E0 2026/27 C 对 D 2026-10-20 15:00 还没踢（超过 14 天，C、D 接下来没有联赛）
"""
import copy
import sqlite3
from datetime import datetime, timezone
from types import SimpleNamespace

import pytest

from src.clean.schema import create_tables
from src.common.config import load_settings
from src.features import FeatureStore, SeasonLockedError
from src.web.site import team_stats
from src.web.site.common import TeamFiles

NOW = datetime(2026, 9, 10, 12, 0, tzinfo=timezone.utc)
MATCHES = [  # match_id, 联赛, 赛季, 主, 客, 开球（UTC）, 比分（没踢 = None）
    (1, "E0", 2025, "A", "B", "2026-03-01 15:00", (2, 0, "H")),
    (2, "E0", 2025, "C", "A", "2026-04-01 15:00", (1, 1, "D")),
    (3, "E0", 2025, "B", "A", "2026-05-01 15:00", (1, 0, "H")),
    (4, "E0", 2026, "A", "C", "2026-08-20 15:00", (3, 1, "H")),
    (5, "E0", 2026, "B", "D", "2026-08-20 15:00", (0, 0, "D")),
    (6, "E0", 2026, "D", "A", "2026-08-27 15:00", (0, 2, "A")),
    (7, "E0", 2026, "C", "B", "2026-08-27 15:00", (2, 2, "D")),
    (8, "FAC", 2026, "A", "D", "2026-09-03 19:45", (0, 5, "A")),
    (9, "E0", 2026, "A", "B", "2026-09-15 15:00", None),
    (10, "E0", 2026, "C", "D", "2026-10-20 15:00", None),
]


@pytest.fixture(scope="module")
def settings():
    s = copy.deepcopy(load_settings())
    s["leagues"] = {"top": {"E0": "英超"}, "second": {}, "asia": {}}
    return s


@pytest.fixture
def con():
    c = sqlite3.connect(":memory:")
    create_tables(c)
    c.executemany("insert into teams (team_id, country) values (?, 'England')", [(t,) for t in "ABCD"])
    for mid, comp, season, h, a, ko, score in MATCHES:
        c.execute("insert into matches (match_id, competition, season, home_team, away_team, match_date, kickoff_utc, status)"
                  " values (?, ?, ?, ?, ?, ?, ?, ?)", (mid, comp, season, h, a, ko[:10], ko, "finished" if score else "scheduled"))
        if score:
            c.execute("insert into match_stats (match_id, source, home_goals, away_goals, result) values (?, 'football-data', ?, ?, ?)",
                      (mid, *score))
    c.commit()
    return c


def make_data(con, settings, page_ids=(4, 9)):
    """render、collect 用到的 SiteData 那几项（只造这几项）。"""
    up = {mid: {"match_id": mid, "competition": comp, "season": season, "home_team": h, "away_team": a,
                "kickoff_earliest_utc": f"{ko}:00", "status": "scheduled"}
          for mid, comp, season, h, a, ko, score in MATCHES if comp == "E0"}
    data = SimpleNamespace(now=NOW, settings=settings, matches=up, page_ids=set(page_ids), team_stats={}, team_stats_error=None,
                           zh=lambda t: t, comp=lambda c: {"E0": "英超"}.get(c, c))
    fs = FeatureStore(con, "实时", settings)
    data.team_stats = team_stats.collect(fs, data, set("ABCD"))
    return data


def plain(html: str) -> str:
    return html.replace("&#x27;", "'").replace("&amp;", "&").replace("&quot;", '"').replace("&lt;", "<").replace("&gt;", ">")


# ---------- 算 ----------
def test_live_mode_exam_season_can_be_shown_not_reviewed(con, settings):
    """实时模式：期末考试赛季（2025/26）的比分可以显示在球队页上，但仍不能拿来复盘。"""
    live = FeatureStore(con, "实时", settings)
    assert 1 in live.historical_matches(NOW, "显示")["match_id"].tolist()
    with pytest.raises(SeasonLockedError):
        live.historical_matches(NOW, "复盘")


def test_form_and_summary(con, settings):
    a = make_data(con, settings).team_stats["teams"]["A"]
    assert [r["match_id"] for r in a["form"]] == [6, 4, 3, 2, 1]          # 从近到远；8 号是杯赛，不算
    assert [r["result"] for r in a["form"]] == ["W", "W", "L", "D", "W"]
    assert a["form_summary"] == {"n": 5, "w": 3, "d": 1, "l": 1, "points": 10, "gf": 8, "ga": 3}
    settings2 = copy.deepcopy(settings)
    settings2["site"]["team_form_matches"] = 2                            # 只看最近 2 场
    a2 = make_data(con, settings2).team_stats["teams"]["A"]
    assert [r["match_id"] for r in a2["form"]] == [6, 4] and a2["form_summary"]["points"] == 6


def test_opponent_and_h2h(con, settings):
    t = make_data(con, settings).team_stats["teams"]
    assert t["A"]["opponent"] == "B" and t["B"]["opponent"] == "A"       # 9 号：5 天后
    assert [r["match_id"] for r in t["A"]["h2h"]] == [3, 1]
    assert t["A"]["h2h_summary"] == {"n": 2, "w": 1, "d": 0, "l": 1, "points": 3, "gf": 2, "ga": 1}
    assert t["B"]["h2h_summary"]["w"] == 1 and t["B"]["h2h_summary"]["l"] == 1
    assert t["C"]["opponent"] is None and t["C"]["h2h"] == []            # 10 号在 14 天以后
    assert t["C"]["where"] == ["E0", 2026]                               # 没有接下来的联赛：用最近踢完那场的联赛赛季


def test_elo_change_is_this_match_only(con, settings):
    """“这场变了” = 赛后 − 赛前；赛前已经换过赛季（往平均拉回），换赛季那一下不算在这场里。"""
    a = make_data(con, settings).team_stats["teams"]["A"]
    track = a["elo"]
    assert [t["match_id"] for t in track] == [1, 2, 3, 4, 6]
    for prev, cur in zip(track, track[1:]):
        if prev["season"] == cur["season"]:
            assert cur["before"] == prev["after"]                         # 同一赛季：上一场赛后 = 这一场赛前
    assert track[3]["before"] != track[2]["after"]                       # 换赛季：往平均拉回了
    assert abs(track[3]["before"]) < abs(track[2]["after"]) + 1e-9
    assert track[0]["after"] > track[0]["before"] and track[2]["after"] < track[2]["before"]   # 赢了加分、输了减分
    c = make_data(con, settings).team_stats["teams"]["C"]
    a4, c4 = track[3], next(t for t in c["elo"] if t["match_id"] == 4)
    assert abs((a4["after"] - a4["before"]) + (c4["after"] - c4["before"])) < 0.15   # 一场里两队一加一减
    assert set(a["elo_form"]) == {6, 4, 3, 2, 1} and a["elo_other_league"] is False


def test_table(con, settings):
    st = make_data(con, settings).team_stats
    tb = st["tables"]["E0|2026"]
    assert tb["note"] is None
    assert [(r["team"], r["points"], r["played"]) for r in tb["rows"]] == [("A", 6, 2), ("B", 2, 2), ("C", 1, 2), ("D", 1, 2)]
    assert set(st["tables"]) == {"E0|2026"}                              # 四队同一个联赛赛季，只算一次


def test_skip_league_groups(con, settings):
    """site.team_skip_league_groups 里的联赛组（国际赛 intl）不读：A 多踢一场国际赛，球队页四块都不算它。"""
    con.execute("insert into matches (match_id, competition, season, home_team, away_team, match_date, kickoff_utc, status)"
                " values (11, 'INT', 2026, 'A', 'B', '2026-09-05', '2026-09-05 15:00', 'finished')")
    con.execute("insert into match_stats (match_id, source, home_goals, away_goals, result) values (11, 'football-data', 4, 0, 'H')")
    s2 = copy.deepcopy(settings)
    s2["leagues"]["intl"] = {"INT": "国际赛"}
    s2["site"]["team_skip_league_groups"] = ["intl"]
    a = make_data(con, s2).team_stats["teams"]["A"]
    assert [r["match_id"] for r in a["form"]] == [6, 4, 3, 2, 1] and a["elo_other_league"] is False
    s2["site"]["team_skip_league_groups"] = []                           # 不跳过：11 号进了近期状态（对照，证明上面真是跳过的作用）
    assert make_data(con, s2).team_stats["teams"]["A"]["form"][0]["match_id"] == 11


def test_league_without_rules_has_note(con, settings):
    s = copy.deepcopy(settings)
    s["standings"]["league_groups"] = ["second"]                         # E0 不在能算积分榜的联赛里
    tb = make_data(con, s).team_stats["tables"]["E0|2026"]
    assert tb == {"rows": None, "note": "这个联赛还没有积分规则，不算积分榜"}


# ---------- 画 ----------
def test_render_team_blocks(con, settings):
    data = make_data(con, settings)
    teams = TeamFiles("ABCD")
    html = plain(team_stats.render(data, "A", teams))
    assert "近期状态（最近 5 场联赛）" in html and "3 胜 1 平 1 负，拿了 10 分，进 8 球、丢 3 球" in html
    assert "href='../match/4.html'" in html and "0–5" not in html       # 有页面的比赛链过去；杯赛不列
    assert "历史交锋：对 <a href='../team/B.html'>B</a>" in html and "1 胜 0 平 1 负" in html
    assert "<svg class='chart'" in html and "2025/26" in html and "2026/27" in html
    assert "这场变了" in html
    assert "积分榜：英超 2026/27" in html and "<tr class='mine'>" in html
    assert html.count("<tr class='mine'>") == 1 and "href='../team/A.html'" not in html   # 自己那一行不链自己
    c = plain(team_stats.render(data, "C", teams))
    assert "接下来 14 天没有联赛，不列交锋" in c and "积分榜：英超 2026/27" in c


def test_render_error_and_missing_team(con, settings):
    data = make_data(con, settings)
    teams = TeamFiles("ABCD")
    assert "球队数据暂时算不出来" in team_stats.render(data, "没有这队", teams)
    data.team_stats_error = "ValueError: 坏了"
    html = team_stats.render(data, "A", teams)
    assert "球队数据暂时算不出来" in html and "近期状态" not in html


def test_render_escapes_names(con, settings):
    data = make_data(con, settings)
    data.zh = lambda t: "<b>坏</b>" if t == "B" else t
    html = team_stats.render(data, "A", TeamFiles("ABCD"))
    assert "<b>" not in html and "&lt;b&gt;坏&lt;/b&gt;" in html


def test_elo_svg_single_point():
    """只有一场时也能画（点放在中间，不除以 0）。"""
    svg = team_stats.elo_svg([{"match_id": 1, "season": 2026, "before": 0.0, "after": 12.5}], str, "x")
    assert svg.startswith("<svg") and "polyline" in svg
