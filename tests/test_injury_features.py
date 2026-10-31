"""步骤 7 伤停研究：src/clean/injury_features.py 的自动检查（不碰真实数据，用造的球员）。

检查的东西：
  - 球队对照：五个数都一样才投票；同一季两队五个数一样的不投；一个编号对到两个队就报错；对照文件和投票不一致就报错；
  - 在队名单：转会当天还不算在新队（严格早于那天），走的那天以后不算；同一天两次转会，按文件“新的在前”的顺序判断先后；
  - 身价：用那天之前最后一次估价，当天的估价不用；
  - 伤病：开始那天不算、结束那天之后不算；没有结束日期的不用；
  - 一队的四个数和手算一样；对不上编号的队是空值；
  - 时间点：最早开球前 24 小时，英国日期（只有日期的比赛按当天 00:00 英国时间开球算）。
"""
import numpy as np
import pandas as pd
import pytest

from src.clean.injury_features import (PlayerData, build, check_aliases, decide, load_matches, vote, _day_numbers)
from src.clean.schema import create_tables
from src.common.config import load_settings

D = lambda s: int(_day_numbers(pd.Series([s]))[0])


# ======================= 球队对照 =======================
def table(rows, team_col):
    return pd.DataFrame(rows, columns=["competition", "season", team_col, "w", "d", "l", "gf", "ga"])


def test_vote_and_decide():
    ours = table([("E0", 2020, "Arsenal", 20, 8, 10, 60, 40), ("E0", 2021, "Arsenal", 22, 3, 13, 61, 48),
                  ("E0", 2020, "Chelsea", 19, 10, 9, 58, 36), ("E0", 2020, "X", 5, 5, 28, 20, 80),
                  ("E0", 2020, "Y", 5, 5, 28, 20, 80)], "team")
    tm = table([("E0", 2020, 11, 20, 8, 10, 60, 40), ("E0", 2021, 11, 22, 3, 13, 61, 48),
                ("E0", 2020, 631, 19, 10, 9, 58, 36), ("E0", 2020, 7, 5, 5, 28, 20, 80),
                ("E0", 2020, 8, 5, 5, 28, 20, 80)], "club_id").assign(tm_name="n")
    v = vote(ours, tm)
    assert set(zip(v["club_id"], v["team"], v["votes"])) == {(11, "Arsenal", 2), (631, "Chelsea", 1)}   # X、Y 分不清，不投
    assert decide(v) == {11: "Arsenal", 631: "Chelsea"}
    bad = pd.concat([v, pd.DataFrame([{"club_id": 999, "team": "Arsenal", "tm_name": "n", "votes": 1, "seasons": ""}])])
    with pytest.raises(ValueError, match="冲突"):
        decide(bad)
    check_aliases({11: "Arsenal", 631: "Chelsea"}, decide(v))
    with pytest.raises(ValueError, match="不一致"):
        check_aliases({11: "Chelsea", 631: "Arsenal"}, decide(v))
    with pytest.raises(ValueError, match="投票里没有"):
        check_aliases({11: "Arsenal", 631: "Chelsea", 5: "X"}, decide(v))


# ======================= 球员 =======================
def players():
    """球队 1、2。文件和 salimt 一样：每个球员新的转会在前。
      球员 10：2019-07-01 到 1 队，2021-01-10 转去 2 队；
      球员 11：2020-07-01 同一天先到 2 队（文件里靠后 = 早），又租借到 1 队（文件里靠前 = 晚）；
      球员 12：2020-01-01 到 1 队，一直在。"""
    transfers = pd.DataFrame([
        (10, "2021-01-10", 2), (10, "2019-07-01", 1),
        (11, "2020-07-01", 1), (11, "2020-07-01", 2),
        (12, "2020-01-01", 1),
    ], columns=["player_id", "transfer_date", "to_team_id"])
    values = pd.DataFrame([(10, "2020-01-01", 100.0), (10, "2020-10-01", 200.0), (11, "2020-06-01", 50.0),
                           (12, "2020-01-01", 0.0)], columns=["player_id", "date_unix", "value"])
    injuries = pd.DataFrame([(10, "2020-10-05", "2020-10-20"), (11, "2020-09-01", None)],
                            columns=["player_id", "from_date", "end_date"])
    return PlayerData(transfers, values, injuries, clubs=[1, 2], top_n=1)


def test_squad_strictly_before_the_day_and_file_order():
    p = players()
    assert set(p.squad(1, D("2020-07-01"))) == {10, 12}          # 11 号当天才来，不算
    assert set(p.squad(1, D("2020-07-02"))) == {10, 11, 12}      # 同一天先到 2 队再租到 1 队 → 在 1 队
    assert set(p.squad(2, D("2020-07-02"))) == set()
    assert set(p.squad(1, D("2021-01-10"))) == {10, 11, 12}      # 走的那天还算在 1 队（当天的转会还不知道）
    assert set(p.squad(1, D("2021-01-11"))) == {11, 12} and set(p.squad(2, D("2021-01-11"))) == {10}
    assert len(p.squad(999, D("2021-01-11"))) == 0


def test_value_and_injury_before_the_day():
    p = players()
    assert p.value(10, D("2020-10-01")) == 100.0 and p.value(10, D("2020-10-02")) == 200.0   # 当天的估价不用
    assert p.value(10, D("2019-12-31")) == 0.0 and p.value(99, D("2020-10-02")) == 0.0
    assert not p.injured(10, D("2020-10-05")) and p.injured(10, D("2020-10-06"))             # 受伤当天不算
    assert p.injured(10, D("2020-10-20")) and not p.injured(10, D("2020-10-21"))             # 结束那天还算伤着（当天复出当天才知道），第二天起不算
    assert not p.injured(11, D("2020-12-01"))                                                # 没有结束日期的不用


def test_side_by_hand():
    p = players()
    day = D("2020-10-10")                    # 1 队：10 号（200，伤）、11 号（50）、12 号（0）
    n, share, loss, top = p.side(1, day)
    assert n == 1 and share == pytest.approx(200 / 250)
    assert loss == pytest.approx(np.log(200) - np.log(50)) and top == pytest.approx(np.log(200))   # top_n = 1
    assert all(np.isnan(p.side(None, day))) and all(np.isnan(p.side(2, D("2020-01-01"))))


# ======================= 每场比赛的时间点 =======================
def test_as_of_is_24h_before_earliest_kickoff_uk_date():
    import sqlite3
    s = load_settings()
    con = sqlite3.connect(":memory:")
    create_tables(con)
    con.executemany("insert into teams (team_id, country) values (?, 'England')", [(t,) for t in "AB"])
    con.executemany("insert into matches (match_id, competition, season, home_team, away_team, match_date, kickoff_utc,"
                    " status) values (?, 'E0', 2020, ?, ?, ?, ?, 'finished')",
                    [(1, "A", "B", "2020-10-10", "2020-10-10 23:30"),        # 开球前 24 小时 = 10-09 23:30 UTC = 10-10 00:30 英国（夏令时）
                     (2, "B", "A", "2020-10-17", None)])                     # 只有日期：当天 00:00 英国 = 10-16 23:00 UTC
    m = load_matches(con, ["E0"], 2020, 2020, s).set_index("match_id")
    assert m.loc[1, "as_of_utc"] == "2020-10-09 23:30:00" and m.loc[1, "day"] == D("2020-10-10")
    assert m.loc[2, "as_of_utc"] == "2020-10-15 23:00:00" and m.loc[2, "day"] == D("2020-10-16")
    out = build(m.reset_index(), players(), {"A": 1}).set_index("match_id")
    assert out.loc[1, "home_inj_n"] == 1 and np.isnan(out.loc[1, "away_inj_n"])             # B 对不上编号 → 空值
    assert list(out.columns[:3]) == ["competition", "season", "as_of_utc"]
