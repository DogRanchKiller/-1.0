"""竞彩价格偏离虚拟实盘（src/evaluate/lottery_value.py）的自动检查。临时数据库、造的数据，不碰真实数据。

造的数据（英超 2026/27，上线后赛季，实时模式；“现在” = 2026-11-10 12:00 UTC）：
  1  2026-11-08 15:00 UTC 开球（北京 23:00），决定时间 = 开球前 6 小时 = 北京 17:00 = 09:00 UTC。利物浦 2-0 阿森纳。
     竞彩胜平负：08:00 抓到 2.10 / 3.30 / 3.20（要用这组）；10:00 抓到 5.00 / 3.30 / 1.50（决定时间以后，不能用）
     竞彩让球 -1：08:00 抓到 3.60 / 3.50 / 1.80
     平博亚盘 -0.75、大小球 2.5：决定时间前变、前存到（要用）；亚盘 -1.5：08:30 变、但 09:30 才存到（不能用）
     收盘（开球前最后一组）：亚盘 -1.0、大小球 2.75
     Bet365 胜平负（算平局概率用）
  2  2026-11-08 15:00 开球，有竞彩、没有平博 → 跳过
  3  2026-11-11 15:00 开球，还没到决定时间 → 不算
"""
import copy

import numpy as np
import pandas as pd
import pytest

from src.clean.schema import create_tables
from src.common.config import load_settings
from src.common.db import connect
from src.common.markets import asian_handicap, over_under, result_probs
from src.common.odds import devig
from src.evaluate import lottery_value as lv
from src.features import FeatureStore
from src.features.errors import UsageNotAllowedError

NOW = "2026-11-10 12:00:00"


@pytest.fixture
def env(tmp_path):
    s = copy.deepcopy(load_settings())
    s["leagues"] = {"top": {"E0": "英超"}, "second": {}}
    s["lottery_value"]["output_dir"] = str(tmp_path / "lv")
    s["lottery_value"]["count_from_utc"] = "2026-11-01 00:00:00"
    con = connect(tmp_path / "t.db")
    create_tables(con)
    con.executemany("insert into teams (team_id, country, name_zh) values (?, '英格兰', ?)",
                    [("Liverpool", "利物浦"), ("Arsenal", "阿森纳"), ("Man City", "曼城"), ("Chelsea", "切尔西")])
    m = [(1, "Liverpool", "Arsenal", "2026-11-08", "2026-11-08 15:00", "finished"),
         (2, "Man City", "Chelsea", "2026-11-08", "2026-11-08 15:00", "finished"),
         (3, "Arsenal", "Man City", "2026-11-11", "2026-11-11 15:00", "scheduled")]
    for mid, h, a, d, ko, st in m:
        con.execute("insert into matches (match_id, competition, season, home_team, away_team, match_date, kickoff_utc,"
                    " status, date_precision, round) values (?, 'E0', 2026, ?, ?, ?, ?, ?, 'day', ?)",
                    (mid, h, a, d, ko, st, str(mid)))
        con.execute("insert into match_sources values (?, 'football-data', ?, 'x.csv', '2026-10-01 00:00:00')",
                    (mid, f"k{mid}"))
    con.executemany("insert into match_stats (match_id, source, home_goals, away_goals) values (?, 'football-data', ?, ?)",
                    [(1, 2, 0), (2, 1, 1)])
    # (比赛, 玩法, 让球数, 主, 平, 客, 抓到时间)
    lot = [(1, "竞彩胜平负", 0, 2.10, 3.30, 3.20, "2026-11-08 08:00:00"),
           (1, "竞彩胜平负", 0, 5.00, 3.30, 1.50, "2026-11-08 10:00:00"),
           (1, "竞彩让球", -1, 3.60, 3.50, 1.80, "2026-11-08 08:00:00"),
           (2, "竞彩胜平负", 0, 2.00, 3.40, 3.50, "2026-11-08 08:00:00"),
           (3, "竞彩胜平负", 0, 2.00, 3.40, 3.50, "2026-11-10 08:00:00")]
    con.executemany("insert into lottery_odds values (?, 'dongqiudi', ?, ?, ?, ?, ?, ?, 'x.json.gz')", lot)
    # (公司, 玩法, 线, 主, 平, 客, 大, 小, 变盘时间, 存到时间)
    oc = [("平博 Pinnacle", "ah", -0.75, 2.00, None, 1.90, None, None, "2026-11-08 07:00:00", "2026-11-08 07:10:00"),
          ("平博 Pinnacle", "ah", -1.5, 2.40, None, 1.60, None, None, "2026-11-08 08:30:00", "2026-11-08 09:30:00"),
          ("平博 Pinnacle", "ou", 2.5, None, None, None, 1.85, 2.05, "2026-11-08 07:00:00", "2026-11-08 07:10:00"),
          ("平博 Pinnacle", "ah", -1.0, 2.10, None, 1.80, None, None, "2026-11-08 14:00:00", "2026-11-08 14:10:00"),
          ("平博 Pinnacle", "ou", 2.75, None, None, None, 1.95, 1.95, "2026-11-08 14:00:00", "2026-11-08 14:10:00"),
          ("Bet365", "1x2", None, 1.80, 3.80, 4.50, None, None, "2026-11-08 07:00:00", "2026-11-08 07:10:00")]
    con.executemany("insert into odds_changes (match_id, source, bookmaker, bookmaker_raw, market, line, home, draw, away,"
                    " over, under, changed_at_utc, fetched_at_utc, raw_file) values (1, 'dongqiudi', ?, ?, ?, ?, ?, ?, ?,"
                    " ?, ?, ?, ?, 'x.json.gz')", [(r[0], r[0], *r[1:]) for r in oc])
    con.execute("insert into odds_changes (match_id, source, bookmaker, bookmaker_raw, market, line, home, draw, away,"
                " over, under, changed_at_utc, fetched_at_utc, raw_file) values (2, 'dongqiudi', 'Bet365', 'Bet365',"
                " '1x2', null, 2.0, 3.4, 3.6, null, null, '2026-11-08 07:00:00', '2026-11-08 07:10:00', 'x.json.gz')")
    con.commit()
    return con, s


def run(env, now=NOW):
    con, s = env
    return lv.build(FeatureStore(con, "实时", s), s, now)


def test_decision_time_window():
    w = ["09:00", "21:30"]
    assert lv.decision_time("2026-11-08 19:00", 6, w) == "2026-11-08 13:00:00"   # 北京 03:00 开球 → 21:00，不改
    assert lv.decision_time("2026-11-08 20:00", 6, w) == "2026-11-08 13:30:00"   # 北京 04:00 → 22:00 → 改成 21:30
    assert lv.decision_time("2026-11-08 05:30", 6, w) == "2026-11-07 13:30:00"   # 北京 13:30 → 07:30 → 前一天 21:30
    assert lv.decision_time("2026-11-08 15:00", 6, w) == "2026-11-08 09:00:00"   # 北京 23:00 → 17:00，不改


def test_uses_only_prices_known_at_decision(env):
    cand, skipped = run(env)
    assert sorted(cand["match_id"].unique()) == [1]                      # 3 还没到决定时间，2 跳过
    c = cand.set_index(["market", "side"])
    assert c.loc[("result", "home"), "price"] == 2.10                     # 10:00 抓到的 5.00 不能用
    assert c.loc[("handicap", "home"), "handicap"] == -1
    # 公道价：平博亚盘 -0.75（-1.5 是决定时间以后才存到的）+ 大小球 2.5 + Bet365 的平局概率
    fair = lv.pinnacle_fair(FeatureStore(env[0], "实时", env[1]), 1, "2026-11-08 09:00:00", lv.BET_USAGE, env[1])
    assert fair["ah_line"] == -0.75 and fair["ou_line"] == 2.5
    m = fair["matrix"]
    assert asian_handicap(m, -0.75, "home")["win_share"] == pytest.approx(devig([2.00, 1.90], method="proportional")[0], abs=1e-6)
    assert over_under(m, 2.5, "over")["win_share"] == pytest.approx(devig([1.85, 2.05], method="proportional")[0], abs=1e-6)
    assert result_probs(m)[1] == pytest.approx(devig([1.80, 3.80, 4.50], method="proportional")[1], abs=1e-6)
    assert c.loc[("result", "home"), "p_fair"] == pytest.approx(result_probs(m)[0])
    assert c.loc[("result", "home"), "ev"] == pytest.approx(2.10 * result_probs(m)[0] - 1)
    # 让球 -1 主胜 = 主队赢 2 球以上
    i, j = np.indices(m.shape)
    assert c.loc[("handicap", "home"), "p_fair"] == pytest.approx(m[(i - j) >= 2].sum())
    assert skipped.set_index("match_id").loc[2, "原因"] == "决定时间：没有平博亚盘或大小球"


def test_close_and_settlement(env):
    cand, _ = run(env)
    c = cand.set_index(["market", "side"])
    close = lv.pinnacle_fair(FeatureStore(env[0], "实时", env[1]), 1, "2026-11-08 15:00:00", lv.CLOSE_USAGE, env[1])
    assert close["ah_line"] == -1.0 and close["ou_line"] == 2.75           # 收盘用开球前最后一组
    assert c.loc[("result", "home"), "clv"] == pytest.approx(2.10 * result_probs(close["matrix"])[0] - 1)
    # 2-0：主胜中、让球 -1 主胜也中（2 − 0 − 1 > 0）
    assert c.loc[("result", "home"), "profit"] == pytest.approx(1.10)
    assert c.loc[("result", "draw"), "profit"] == pytest.approx(-1.0)
    assert c.loc[("handicap", "home"), "profit"] == pytest.approx(2.60)
    assert (cand["period"] == lv.PERIOD_COUNTED).all()


def test_before_kickoff_no_close_no_settle(env):
    cand, _ = run(env, now="2026-11-08 12:00:00")                          # 过了决定时间、还没开球
    assert sorted(cand["match_id"].unique()) == [1]
    assert cand["clv"].isna().all() and (~cand["settled"]).all()
    p = lv.pending(cand, env[1], "2026-11-08 12:00:00")
    assert set(p["match_id"]) <= {1}


def test_best_per_match_one_bet():
    c = pd.DataFrame({"match_id": [1, 1, 1, 2, 2], "decision_utc": ["a"] * 5, "ev": [0.01, 0.08, -0.1, -0.2, 0.03]})
    b = lv.best_per_match(c, 0.05)
    assert list(b["ev"]) == [0.08, 0.03] and list(b["bet"]) == [True, False]


def test_verdict_rules():
    s = load_settings()
    assert lv.verdict(50, -0.2, -0.1, -0.1, s).startswith("注数不够")
    assert lv.verdict(120, -0.08, -0.01, -0.1, s).startswith("没看到")
    assert lv.verdict(320, 0.005, 0.04, 0.02, s).startswith("通过")
    assert lv.verdict(320, 0.005, 0.04, -0.02, s) == "继续攒"            # CLV 好但 ROI 不是正的
    assert lv.verdict(650, -0.01, 0.03, 0.01, s).startswith("没看到")


def test_log_append_only_and_mismatch(env):
    cand, skipped = run(env)
    stats, summary = lv.write(cand, skipped, env[1], NOW)
    assert stats["决定过的场"] == 1 and stats["和记录不一样的场"] == 0
    log1 = pd.read_csv(f"{env[1]['lottery_value']['output_dir']}/decisions_log.csv", encoding="utf-8-sig")
    # 数据被改过（例如重导）：决定时间以前的竞彩价格变了 → 记录不改，数出不一样的场
    env[0].execute("update lottery_odds set home = 9.0 where match_id = 1 and fetched_at_utc = '2026-11-08 08:00:00'")
    env[0].commit()
    cand2, skipped2 = run(env)
    stats2, _ = lv.write(cand2, skipped2, env[1], NOW)
    log2 = pd.read_csv(f"{env[1]['lottery_value']['output_dir']}/decisions_log.csv", encoding="utf-8-sig")
    assert len(log2) == len(log1) == 1 and log2.equals(log1)
    assert stats2["和记录不一样的场"] == 1
    main = summary[summary["主规则"]]
    assert len(main) == 1 and main.iloc[0]["结论"].startswith("注数不够")
    assert len(lv.short_text(summary)) <= 40


def test_empty_run(env, tmp_path):
    cand, skipped = run(env, now="2026-11-01 00:00:00")                    # 一场都还没到决定时间
    assert cand.empty
    stats, summary = lv.write(cand, skipped, env[1], "2026-11-01 00:00:00")
    assert stats["决定过的场"] == 0 and lv.short_text(summary) == "正式计数 0 注"


def test_lottery_usage_rules(env):
    fs = FeatureStore(env[0], "实时", env[1])
    assert len(fs.lottery_odds(1, "2026-11-08 09:00:00", "机器人下注")) == 2     # 决定时间以前的两条
    with pytest.raises(UsageNotAllowedError):
        fs.lottery_odds(1, "2026-11-08 09:00:00", "B版输入")                     # 竞彩价格不能当模型输入


def test_leagues_can_be_several_groups(env):
    # leagues 写几个组名：几个组里的联赛都算（2026-11-07 加次级联赛）
    con, s = env
    s["leagues"] = {"top": {"SP1": "西甲"}, "second": {"E0": "英超"}}
    s["lottery_value"]["leagues"] = "top"
    assert run(env)[0].empty                                               # 只看 top：这里没有 E0 的比赛
    s["lottery_value"]["leagues"] = ["top", "second"]
    assert sorted(run(env)[0]["match_id"].unique()) == [1]
