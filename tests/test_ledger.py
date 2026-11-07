"""四本账（src/evaluate/ledger.py）的自动检查。临时数据库、造的数据，不碰真实数据。

造的数据（英超 2026/27，上线后赛季，实时模式）：
  1  2026-10-10 14:00 利物浦 2-1 阿森纳。T-24 两条（取早的那条）、T-6 一条。
     平博亚盘：-0.5（早）→ -0.75（T-24 之前变、之前存到，要用这组）→ -1.0（T-24 之前变、但之后才存到，不能用）
     平博大小球：2.5。Bet365 也有一组亚盘（不是平博，不用）
  2  2026-10-17 14:00 曼城 0-0 利物浦。没有平博价格；融合状态是「只有模型」 → 亚盘、大小球、比分都缺市场价
  3  2026-11-20 15:00 还没踢 → 不出行
"""
import copy
import json

import numpy as np
import pandas as pd
import pytest

from src.clean.schema import create_tables
from src.common.config import load_settings
from src.common.db import connect
from src.common.markets import asian_handicap, over_under
from src.common.odds import devig
from src.evaluate.ledger import build, line_hit, line_score, summarize
from src.features import FeatureStore


def _matrix(lam_h: float, lam_a: float, n: int = 10) -> list:
    """两个独立泊松拼的比分矩阵，超过 n 球的并进最后一行 / 列（够测试用）。"""
    from scipy.stats import poisson
    ph = poisson.pmf(np.arange(n + 1), lam_h)
    pa = poisson.pmf(np.arange(n + 1), lam_a)
    ph[-1] += 1 - ph.sum()
    pa[-1] += 1 - pa.sum()
    return np.outer(ph, pa).tolist()


MODEL_M = _matrix(1.8, 0.9)
FUSED_M = _matrix(1.5, 1.1)


@pytest.fixture
def env(tmp_path):
    s = copy.deepcopy(load_settings())
    s["leagues"] = {"top": {"E0": "英超"}, "second": {}, "jc": {}}   # 竞彩联赛这组空着
    s["ledger"]["min_matches"] = 1000
    con = connect(tmp_path / "t.db")
    create_tables(con)
    con.executemany("insert into teams (team_id, country, name_zh) values (?, '英格兰', ?)",
                    [("Liverpool", "利物浦"), ("Arsenal", "阿森纳"), ("Man City", "曼城")])
    m = [(1, "Liverpool", "Arsenal", "2026-10-10", "2026-10-10 14:00", "finished"),
         (2, "Man City", "Liverpool", "2026-10-17", "2026-10-17 14:00", "finished"),
         (3, "Arsenal", "Man City", "2026-11-20", "2026-11-20 15:00", "scheduled")]
    for mid, h, a, d, ko, st in m:
        con.execute("insert into matches (match_id, competition, season, home_team, away_team, match_date, kickoff_utc,"
                    " status, date_precision, round) values (?, 'E0', 2026, ?, ?, ?, ?, ?, 'day', ?)",
                    (mid, h, a, d, ko, st, str(mid)))
        con.execute("insert into match_sources values (?, 'football-data', ?, 'x.csv', '2026-10-01 00:00:00')",
                    (mid, f"k{mid}"))
    con.executemany("insert into match_stats (match_id, source, home_goals, away_goals) values (?, 'football-data', ?, ?)",
                    [(1, 2, 1), (2, 0, 0)])
    # (公司, 玩法, 线, 主, 客, 大, 小, 变盘时间, 我们存到的时间)
    oc = [("平博 Pinnacle", "ah", -0.5, 1.90, 2.00, None, None, "2026-10-08 10:00:00", "2026-10-08 11:00:00"),
          ("平博 Pinnacle", "ah", -0.75, 2.05, 1.85, None, None, "2026-10-09 09:00:00", "2026-10-09 10:00:00"),
          ("平博 Pinnacle", "ah", -1.0, 2.30, 1.65, None, None, "2026-10-09 11:00:00", "2026-10-09 20:00:00"),
          ("平博 Pinnacle", "ou", 2.5, None, None, 1.80, 2.10, "2026-10-09 08:00:00", "2026-10-09 10:00:00"),
          ("Bet365", "ah", -1.5, 2.50, 1.55, None, None, "2026-10-09 09:30:00", "2026-10-09 10:00:00")]
    con.executemany("insert into odds_changes (match_id, source, bookmaker, bookmaker_raw, market, line, home, draw, away,"
                    " over, under, changed_at_utc, fetched_at_utc, raw_file) values (1, 'dongqiudi', ?, ?, ?, ?, ?, null, ?,"
                    " ?, ?, ?, ?, 'x.json.gz')", [(r[0], r[0], *r[1:]) for r in oc])
    fused = {"赔率概率": [0.5, 0.3, 0.2], "赔率公司数": 10, "模型比分矩阵": MODEL_M, "融合比分矩阵": FUSED_M,
             "融合": {"状态": "已融合", "w": 1.0}}
    only_model = {"赔率概率": [0.45, 0.3, 0.25], "模型比分矩阵": MODEL_M, "融合比分矩阵": MODEL_M, "融合": {"状态": "只有模型", "w": 1.0}}
    # (编号, 比赛, 时点, 截止, 出预测, 主胜, 平, 客胜, detail)
    p = [(1, 1, "T-24", "2026-10-09 14:00:00", "2026-10-09 15:00:00", 0.6, 0.25, 0.15, fused),
         (2, 1, "T-24", "2026-10-09 16:00:00", "2026-10-09 21:00:00", 0.1, 0.1, 0.8, fused),   # 晚出的那条，不用
         (3, 1, "T-6", "2026-10-10 08:00:00", "2026-10-10 08:10:00", 0.55, 0.27, 0.18, fused),
         (4, 2, "T-24", "2026-10-16 14:00:00", "2026-10-16 14:05:00", 0.4, 0.3, 0.3, only_model),
         (5, 3, "T-24", "2026-11-19 15:00:00", "2026-11-19 15:05:00", 0.4, 0.3, 0.3, fused)]
    for pid, mid, kind, cut, cr, ph, pd_, pa, det in p:
        h, a, d, ko = {1: ("Liverpool", "Arsenal", "2026-10-10", "2026-10-10 14:00"),
                       2: ("Man City", "Liverpool", "2026-10-17", "2026-10-17 14:00"),
                       3: ("Arsenal", "Man City", "2026-11-20", "2026-11-20 15:00")}[mid]
        # 模型名跟着 settings 的 ledger.model（2026-11-07 起上线的是 gb_xg）
        con.execute("insert into predictions values (?, ?, 'E0', 2026, ?, ?, ?, ?, ?, ?, 'v', ?, ?, ?, ?, ?, ?)",
                    (pid, mid, h, a, ko, d, kind, s["ledger"]["model"], cut, cr, ph, pd_, pa, json.dumps(det)))
    con.commit()
    return con, s


def run(env):
    con, s = env
    return build(FeatureStore(con, "实时", s), con, s)


def test_ledger_rows(env):
    rows, _ = run(env)
    assert sorted(rows["prediction_id"].unique()) == [1, 3, 4]      # 每场每个时点取最早的；没踢完的不出
    r = rows[rows["prediction_id"] == 1].set_index("通道")
    assert r.loc["胜平负", "结算"] == "主胜"
    assert r.loc["胜平负", "模型分数"] == pytest.approx(-np.log(0.6))
    assert r.loc["胜平负", "市场分数"] == pytest.approx(-np.log(0.5))
    # 亚盘：用 -0.75 那组（-1.0 那组变盘时间够早，但存到得晚）；2-1 在 -0.75 上主队赢半
    ah = r.loc["亚盘让球"]
    assert ah["线"] == -0.75 and ah["结算"] == "half_win"
    q_mkt = devig([2.05, 1.85], method="proportional")[0]
    q_mod = asian_handicap(np.array(MODEL_M), -0.75, "home")["win_share"]
    assert ah["市场概率"] == pytest.approx(q_mkt)
    assert ah["模型概率"] == pytest.approx(q_mod)
    assert ah["市场分数"] == pytest.approx(0.5 * -np.log(q_mkt))
    assert ah["差"] == pytest.approx(0.5 * (np.log(q_mkt) - np.log(q_mod)))
    # 大小球：2.5 上 3 球，大赢
    ou = r.loc["亚盘大小球"]
    assert ou["线"] == 2.5 and ou["结算"] == "win"
    assert ou["模型概率"] == pytest.approx(over_under(np.array(MODEL_M), 2.5, "over")["win_share"])
    assert ou["市场分数"] == pytest.approx(-np.log(devig([1.80, 2.10], method="proportional")[0]))
    # 比分：实际 2-1 那一格
    sc = r.loc["比分"]
    assert sc["模型分数"] == pytest.approx(-np.log(MODEL_M[2][1]))
    assert sc["市场分数"] == pytest.approx(-np.log(FUSED_M[2][1]))


def test_ledger_missing_market(env):
    rows, summary = run(env)
    r = rows[rows["prediction_id"] == 4].set_index("通道")
    assert r.loc["亚盘让球", "缺什么"] == "这条预测出之前没有平博的价格"
    assert r.loc["亚盘大小球", "缺什么"] == "这条预测出之前没有平博的价格"
    assert "只有模型" in r.loc["比分", "缺什么"] and pd.isna(r.loc["比分", "差"])
    assert pd.notna(r.loc["胜平负", "差"])
    assert r.loc["胜平负", "结算"] == "平"                  # 0-0
    t24 = summary[summary["时点"] == "T-24"].set_index("通道")
    assert t24.loc["胜平负", "对账的场数"] == 2 and t24.loc["胜平负", "缺市场价的场数"] == 0
    assert t24.loc["亚盘让球", "对账的场数"] == 1 and t24.loc["亚盘让球", "缺市场价的场数"] == 1
    assert t24.loc["亚盘让球", "结论"].startswith("场数不够")
    assert summary[summary["时点"] == "T-1"]["结论"].eq("还没有能对账的场").all()


def test_line_score_and_hit():
    eps = 1e-15
    assert line_score(0.6, "win", eps) == pytest.approx(-np.log(0.6))
    assert line_score(0.6, "loss", eps) == pytest.approx(-np.log(0.4))
    assert line_score(0.6, "half_win", eps) == pytest.approx(-0.5 * np.log(0.6))
    assert line_score(0.6, "half_loss", eps) == pytest.approx(-0.5 * np.log(0.4))
    assert line_score(0.6, "push", eps) == 0.0
    assert line_hit(0.6, "half_win") == (0.5, 0.5)
    assert line_hit(0.4, "half_win") == (0.0, 0.5)       # 看好另一边，这边赢半 = 没中
    assert line_hit(0.4, "loss") == (1.0, 1.0)
    assert line_hit(0.5, "win") == (0.0, 0.0)            # 正好 0.5 没看好哪边，不算
    assert line_hit(0.7, "push") == (0.0, 0.0)


def test_summary_verdict_only_with_enough_matches():
    s = copy.deepcopy(load_settings())
    s["ledger"]["min_matches"] = 50
    rng = np.random.default_rng(0)
    n = 60
    dates = pd.date_range("2026-08-01", periods=n, freq="D").strftime("%Y-%m-%d")
    rows = pd.DataFrame({"时点": "T-24", "通道": "胜平负", "比赛日期": dates, "开球前几小时": 20.0,
                         "模型分数": 1.0 + rng.normal(0, 0.01, n), "市场分数": 1.1 + rng.normal(0, 0.01, n),
                         "模型命中": 1.0, "模型份数": 1.0, "市场命中": 0.0, "市场份数": 1.0})
    rows["差"] = rows["模型分数"] - rows["市场分数"]
    t6 = rows.assign(时点="T-6")
    out = summarize(pd.concat([rows, t6, rows.head(10).assign(通道="比分")]), s).set_index(["时点", "通道"])
    assert out.loc[("T-24", "胜平负"), "结论"] == "更准"
    assert out.loc[("T-24", "胜平负"), "区间上限"] < 0
    assert out.loc[("T-6", "胜平负"), "结论"] == "只报告"      # 只有 T-24 下结论
    assert out.loc[("T-24", "比分"), "结论"].startswith("场数不够（10 场")
    assert out.loc[("T-24", "胜平负"), "模型命中率"] == 1.0 and out.loc[("T-24", "胜平负"), "市场命中率"] == 0.0
