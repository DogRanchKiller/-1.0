"""三个检验和市场比分矩阵 src/evaluate/market_checks.py 的检查。

用小的假数据核对：反推 λ 能还原、跟随比例 β 的含义对、分组对、赔率按顺序选、收盘只用同一家、考试赛季拿不到。
"""
import copy
import sqlite3

import numpy as np
import pandas as pd
import pytest

from src.clean.schema import create_tables
from src.common.config import load_settings
from src.evaluate import market_checks as mc
from src.features import FeatureStore


@pytest.fixture
def settings():
    s = copy.deepcopy(load_settings())
    s["compare"]["n_boot"] = 300
    return s


# ---------------- 市场比分矩阵 ----------------
def test_fit_recovers_lambdas():
    """用已知的 λ 算出四个概率，再反推回去，应该得到原来的 λ，误差几乎为 0。"""
    rng = np.random.default_rng(3)
    lh, la = rng.uniform(0.4, 3.2, 500), rng.uniform(0.3, 2.5, 500)
    target = mc.matrix_probs(lh, la)
    fh, fa = mc.fit_market_lambdas(target)
    assert np.allclose(fh, lh, atol=1e-5) and np.allclose(fa, la, atol=1e-5)
    assert np.abs(mc.matrix_probs(fh, fa) - target).max() < 1e-8


def test_fit_never_worse_than_start_on_inconsistent_odds():
    """赔率本身和泊松矩阵对不上（大 2.5 球概率故意改高）：拟合结果的误差要比出发点小，而且 λ 是正常的数。"""
    target = mc.matrix_probs([1.5, 1.2], [1.1, 0.9])
    target[:, 3] += [0.08, -0.08]
    fh, fa = mc.fit_market_lambdas(target)
    start = ((mc.matrix_probs([1.45, 1.45], [1.15, 1.15]) - target) ** 2).sum(axis=1)
    end = ((mc.matrix_probs(fh, fa) - target) ** 2).sum(axis=1)
    assert (end < start).all() and np.isfinite(fh).all() and (fh > 0).all() and (fa > 0).all()
    assert fh[0] + fa[0] > 2.6 > fh[1] + fa[1]               # 大球概率高的那场，反推的总进球更多


# ---------------- 跟随比例和分组 ----------------
def _frame(n=400, seed=0):
    rng = np.random.default_rng(seed)
    q = rng.dirichlet([9, 6, 6], n)
    p = np.clip(q + rng.normal(0, 0.04, (n, 3)), 0.02, None)
    p /= p.sum(axis=1, keepdims=True)
    df = pd.DataFrame(p, columns=mc.P).join(pd.DataFrame(q, columns=mc.Q))
    df["match_date"] = pd.date_range("2015-08-01", periods=n, freq="D").strftime("%Y-%m-%d")
    df["result"] = rng.choice(["H", "D", "A"], n)
    df["competition"], df["q_source"] = "E0", "平博 Pinnacle"
    from src.evaluate import metrics
    df["log_loss"] = metrics.log_loss_per_match(p, metrics.encode_results(df["result"]))
    return df


def test_check3_beta(settings):
    """收盘正好走到模型 → β = 1；收盘走一半 → 0.5；收盘的变化和模型无关 → β 接近 0。
    （95% 区间每 20 次约有 1 次碰巧不含 0，所以第三种只查 β 很小，不查结论。）"""
    for frac in [1.0, 0.5]:
        df = _frame()
        df[mc.C] = df[mc.Q].to_numpy() + frac * (df[mc.P].to_numpy() - df[mc.Q].to_numpy())
        out = mc.check3(mc.add_groups(df, settings), settings)
        row = out[out["分组方式"] == "全部"].iloc[0]
        assert row["跟随比例β"] == pytest.approx(frac)
        assert row["结论"] == "赔率往模型那边走" and row["往模型方向走的比例"] == pytest.approx(1.0)
    df = _frame()
    rng = np.random.default_rng(2)
    noise = rng.normal(0, 0.01, (len(df), 3))
    df[mc.C] = df[mc.Q].to_numpy() + noise - noise.mean(axis=1, keepdims=True)   # 和模型无关的随机变化
    row = mc.check3(mc.add_groups(df, settings), settings).iloc[0]
    assert abs(row["跟随比例β"]) < 0.02 and row["ci_low"] < 0 < row["ci_high"]


def test_ratio_ci_matches_plain_ratio(settings):
    num, den = np.arange(1, 101, dtype=float), np.full(100, 2.0)
    dates = pd.Series(pd.date_range("2016-01-01", periods=100, freq="D").strftime("%Y-%m-%d"))
    r, lo, hi = mc.ratio_ci(num, den, dates, settings)
    assert r == pytest.approx(num.sum() / den.sum()) and lo < r < hi


def test_add_groups(settings):
    df = pd.DataFrame({"p_home": [0.50, 0.45, 0.30], "p_draw": [0.25, 0.25, 0.30], "p_away": [0.25, 0.30, 0.40],
                       "q_home": [0.49, 0.40, 0.65], "q_draw": [0.26, 0.28, 0.20], "q_away": [0.25, 0.32, 0.15],
                       "result": ["H", "D", "A"], "match_date": ["2015-08-01"] * 3})
    for c in mc.C:
        df[c] = np.nan
    g = mc.add_groups(df, settings)
    assert list(g["gap_group"].astype(str)) == ["3% 以内", "3%–6%", "6% 以上"]
    assert list(g["strength"]) == ["势均力敌", "势均力敌", "强弱分明"]
    assert list(g["side"]) == ["模型更看好主队", "模型更看好主队", "模型更看好客队"]
    assert g["ll_c"].isna().all()                               # 没有收盘就不算收盘分数


# ---------------- 取赔率：按顺序选、收盘只用同一家、考试赛季拿不到 ----------------
def _db(settings):
    c = sqlite3.connect(":memory:")
    create_tables(c)
    c.executemany("insert into teams (team_id, country) values (?, 'England')", [("A",), ("B",)])
    rows = [(1, 2015, "A", "B", "2015-08-10"), (2, 2015, "B", "A", "2015-08-17"), (3, 2016, "A", "B", "2016-08-15"),
            (4, 2024, "A", "B", "2024-08-17")]
    c.executemany("insert into matches (match_id, competition, season, home_team, away_team, match_date, status)"
                  " values (?, 'E0', ?, ?, ?, ?, 'finished')", rows)
    c.executemany("insert into match_stats (match_id, source, home_goals, away_goals, result)"
                  " values (?, 'football-data', 1, 0, 'H')", [(r[0],) for r in rows])
    ins = ("insert into odds (match_id, source, bookmaker, market, timing, line, home, draw, away, over, under)"
           " values (?, 'football-data', ?, ?, ?, ?, ?, ?, ?, ?, ?)")
    odds = [
        (1, "平博 Pinnacle", "1x2", "pre", None, 2.0, 3.5, 4.0, None, None),     # 第 1 场：平博赛前、没有平博收盘
        (1, "多家平均", "1x2", "pre", None, 1.9, 3.4, 3.9, None, None),
        (1, "多家平均", "1x2", "close", None, 1.8, 3.5, 4.2, None, None),        # 有多家平均收盘也不能用
        (1, "多家平均", "ou", "pre", 2.5, None, None, None, 1.9, 1.9),
        (2, "多家平均", "1x2", "pre", None, 2.5, 3.2, 2.9, None, None),          # 第 2 场：只有多家平均
        (2, "多家平均", "1x2", "close", None, 2.4, 3.2, 3.0, None, None),
        (3, "平博 Pinnacle", "1x2", "pre", None, 1.5, 4.0, 7.0, None, None),     # 第 3 场：平博赛前 + 平博收盘
        (3, "平博 Pinnacle", "1x2", "close", None, 1.45, 4.2, 7.5, None, None),
        (4, "平博 Pinnacle", "1x2", "pre", None, 2.0, 3.5, 4.0, None, None),     # 考试赛季
    ]
    c.executemany(ins, odds)
    c.commit()
    return FeatureStore(c, "开发", settings)


def test_load_market_priority_and_same_book_close(settings):
    m = mc.load_market(_db(settings), settings, ["E0"]).set_index("match_id")
    assert set(m.index) == {1, 2, 3}                            # 第 4 场是考试赛季，拿不到
    assert m.loc[1, "q_source"] == "平博 Pinnacle" and pd.isna(m.loc[1, "c_home"])
    assert m.loc[2, "q_source"] == "多家平均" and m.loc[2, "c_home"] > 0
    assert m.loc[3, "q_source"] == "平博 Pinnacle" and m.loc[3, "c_home"] > m.loc[3, "q_home"]
    assert m.loc[1, "q_home"] == pytest.approx((1 / 2.0) / (1 / 2.0 + 1 / 3.5 + 1 / 4.0))   # 按比例去水
    assert m.loc[1, "m_over25"] == pytest.approx(0.5) and pd.isna(m.loc[2, "m_over25"])   # 第 2 场没有大小球
    assert np.allclose(m[mc.Q].sum(axis=1), 1)


def test_run_end_to_end(settings, tmp_path):
    """假的回测结果（3 场开发赛季 + 1 场考试赛季 + 别的模型、别的时点）→ run 只用 dc 的 T-24 开发赛季预测，各表都有。"""
    fs = _db(settings)
    base = {"competition": "E0", "result": "H", "home_goals": 1, "away_goals": 0, "p_over25": 0.5, "p_btts": 0.45,
            "brier_over25": 0.25, "brier_btts": 0.2, "brier": 0.3, "rps": 0.1}
    rows = []
    for mid, season, date in [(1, 2015, "2015-08-10"), (2, 2015, "2015-08-17"), (3, 2016, "2016-08-15"),
                              (4, 2024, "2024-08-17")]:
        for model, hz in [("dc", "T-24"), ("dc", "T-1"), ("poisson", "T-24")]:
            rows.append({**base, "match_id": mid, "season": season, "match_date": date, "model": model,
                         "horizon": hz, "p_home": 0.5, "p_draw": 0.27, "p_away": 0.23, "log_loss": -np.log(0.5)})
    run_dir = tmp_path / "fake"
    run_dir.mkdir()
    pd.DataFrame(rows).to_csv(run_dir / "predictions.csv", index=False, encoding="utf-8-sig")
    sc = pd.DataFrame(rows)[["match_id", "model", "home_goals", "away_goals"]].drop_duplicates(["match_id", "model"])
    sc.assign(score_logloss=2.0).to_csv(run_dir / "scores_matrix.csv", index=False, encoding="utf-8-sig")
    res = mc.run(fs, settings, run_dir, "dc")
    assert res["info"]["预测场数"] == 3 and res["info"]["有赛前赔率的场数"] == 3
    assert res["info"]["有同一家收盘的场数"] == 2 and res["info"]["市场矩阵场数"] == 1
    c1 = res["check1"]
    assert c1.loc[(c1["分组方式"] == "全部") & (c1["和谁比"] == "赛前") & (c1["metric"] == "log_loss"), "n"].item() == 3
    assert res["check3"].iloc[0]["n"] == 2
    mm = res["market_matrix"].set_index("metric")
    assert mm.loc["比分 log loss", "左边"] == pytest.approx(2.0) and mm.loc["比分 log loss", "n"] == 1
    assert res["market_matrix_recon"]["最大差"].max() < 0.05
