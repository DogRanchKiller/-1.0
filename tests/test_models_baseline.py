"""第 0 步模型（src/models）的自动检查：频率基线、频率泊松、Elo（手算例子）、有序 logit、模型登记表。

这些检查不需要数据库：直接把“截止前的历史比赛”表交给模型（回测框架交给模型的也是这样一张表）。
表里的列和取数层 historical_matches 返回的一样（只用到模型真正读的那几列）。
"""
import numpy as np
import pandas as pd
import pytest

from src.common.config import load_settings
from src.common.markets import poisson_score_matrix
from src.models import ordered_logit
from src.models.elo import EloModel, expected_home, goal_diff_multiplier
from src.models.frequency import FrequencyModel, FrequencyPoissonModel
from src.models.registry import available, build_model


def history(rows) -> pd.DataFrame:
    """造一张“历史比赛”表。rows：(match_id, 联赛, 赛季, 主队, 客队, 开球时间UTC, 主队进球, 客队进球)。"""
    out = []
    for mid, comp, season, home, away, ko, hg, ag in rows:
        res = "H" if hg > ag else ("D" if hg == ag else "A")
        out.append({"match_id": mid, "competition": comp, "season": season, "home_team": home, "away_team": away,
                    "kickoff_earliest_utc": ko, "home_goals": hg, "away_goals": ag, "result": res})
    return pd.DataFrame(out)


def targets(rows) -> pd.DataFrame:
    """造一张“要预测的比赛”表（没有比分）。rows：(match_id, 联赛, 赛季, 主队, 客队)。"""
    return pd.DataFrame(rows, columns=["match_id", "competition", "season", "home_team", "away_team"])


# ======================= 频率基线 =======================
def test_frequency_counts_shares_and_window():
    """E0 窗口里 5 场：3 主胜、1 平、1 客胜 → 每场都预测 [0.6, 0.2, 0.2]。
    窗口外（截止前 400 天以前）的 2 场客胜不算；SP1 只有 1 场，少于 min_matches，不出预测（None）。"""
    h = history([
        (1, "E0", 2023, "A", "B", "2023-09-01 14:00:00", 2, 0),
        (2, "E0", 2023, "C", "D", "2023-09-02 14:00:00", 1, 0),
        (3, "E0", 2023, "A", "C", "2023-10-01 14:00:00", 3, 1),
        (4, "E0", 2023, "B", "D", "2023-10-02 14:00:00", 1, 1),
        (5, "E0", 2023, "D", "A", "2023-11-01 14:00:00", 0, 2),
        (6, "E0", 2021, "A", "B", "2021-09-01 14:00:00", 0, 1),   # 窗口外
        (7, "E0", 2021, "C", "D", "2021-09-02 14:00:00", 0, 3),   # 窗口外
        (8, "SP1", 2023, "X", "Y", "2023-10-01 14:00:00", 1, 0),
    ])
    m = FrequencyModel(window_days=400, min_matches=3)
    m.fit(h, "2023-12-01 00:00:00")
    p = m.predict(targets([(9, "E0", 2023, "A", "D"), (10, "E0", 2023, "D", "A"), (11, "SP1", 2023, "X", "Y")]),
                  "2023-12-01 00:00:00")
    np.testing.assert_allclose(p[0], [0.6, 0.2, 0.2])
    np.testing.assert_allclose(p[1], p[0])          # 不管谁对谁，同一联赛每场都一样
    assert p[2] is None
    assert m.params()["E0"]["场数"] == 5


def test_frequency_ignores_rows_after_cutoff_window_start_only():
    """窗口只往前截（截止 − window_days），截止之后的比赛本来就不会出现在 history 里（取数层保证、回测再查一遍）。
    这里检查：开球时间在窗口起点之前 1 秒的比赛不算，正好在起点上的算。"""
    h = history([(1, "E0", 2023, "A", "B", "2023-01-01 00:00:00", 1, 0),     # 正好在起点：算
                 (2, "E0", 2022, "A", "B", "2022-12-31 23:59:59", 0, 1)])    # 早 1 秒：不算
    m = FrequencyModel(window_days=10, min_matches=1)
    m.fit(h, "2023-01-11 00:00:00")
    np.testing.assert_allclose(m.predict(targets([(3, "E0", 2023, "A", "B")]), "")[0], [1.0, 0.0, 0.0])


def test_frequency_poisson_uses_league_mean_goals():
    """E0 主队进球 2、1、3（平均 2.0），客队 0、1、1（平均 2/3）→ 泊松矩阵（主 2.0，客 0.667）。"""
    h = history([(1, "E0", 2023, "A", "B", "2023-09-01 14:00:00", 2, 0),
                 (2, "E0", 2023, "C", "D", "2023-09-02 14:00:00", 1, 1),
                 (3, "E0", 2023, "A", "C", "2023-10-01 14:00:00", 3, 1)])
    m = FrequencyPoissonModel(window_days=400, min_matches=3)
    m.fit(h, "2023-12-01 00:00:00")
    p = m.predict(targets([(9, "E0", 2023, "A", "D"), (10, "SP1", 2023, "X", "Y")]), "2023-12-01 00:00:00")
    np.testing.assert_allclose(p[0], poisson_score_matrix(2.0, 2 / 3))
    assert p[0].shape[0] == p[0].shape[1] and abs(p[0].sum() - 1) < 1e-6
    assert p[1] is None


# ======================= Elo：手算例子 =======================
# 公式（src/models/elo.py 开头）：
#   主队预期得分 E = 1 ÷ (1 + 10^(−(主队评分 + 主场优势 − 客队评分) ÷ 400))
#   赛后：主队评分 += K × (实际得分 − E)，客队评分减同样多（实际得分：赢 1、平 0.5、输 0）
# 例子：两队都是 1500（联赛第一个赛季，所有队从 1500 开始），主场优势 H = 65，K = 20。
#   10^(−65 ÷ 400) = 10^(−0.1625) ≈ 0.68787
#   E = 1 ÷ (1 + 0.68787) ≈ 0.59246        （主场优势 65 分 ≈ 主队预期得分 59%）
#   主队赢：+20 × (1 − 0.59246)   = +8.1508  → 主 1508.15，客 1491.85
#   打平：  +20 × (0.5 − 0.59246) = −1.8492  → 主 1498.15，客 1501.85（主场打平算“没达到预期”，要扣分）
#   主队输：+20 × (0 − 0.59246)   = −11.849
#   3 比 0 且用净胜球系数：净胜 3 球系数 = (11 + 3) ÷ 8 = 1.75 → +8.1508 × 1.75 = +14.264
E_HAND = 1 / (1 + 10 ** (-65 / 400))


def test_elo_expected_score_by_hand():
    assert E_HAND == pytest.approx(0.59246, abs=1e-5)
    assert expected_home(1500, 1500, 65) == pytest.approx(E_HAND)
    assert expected_home(1500, 1500, 0) == 0.5                       # 没有主场优势、一样强 → 0.5
    assert expected_home(1900, 1500, 0) == pytest.approx(10 / 11)    # 差 400 分：10 比 1


@pytest.mark.parametrize("hg, ag, gd, change", [
    (1, 0, False, 20 * (1 - E_HAND)),           # +8.1508
    (1, 1, False, 20 * (0.5 - E_HAND)),         # −1.8492
    (0, 2, False, 20 * (0 - E_HAND)),           # −11.849
    (3, 0, True, 20 * 1.75 * (1 - E_HAND)),     # +14.264（净胜 3 球 ×1.75）
    (3, 0, False, 20 * (1 - E_HAND)),           # 不用净胜球系数：3 比 0 和 1 比 0 一样
])
def test_elo_one_match_update_by_hand(hg, ag, gd, change):
    m = EloModel(k=20, home_advantage=65, goal_diff=gd)
    m.update(history([(1, "E0", 2023, "A", "B", "2023-08-12 14:00:00", hg, ag)]), "2023-08-13 00:00:00")
    assert m.ratings[("E0", "A")] == pytest.approx(1500 + change)
    assert m.ratings[("E0", "B")] == pytest.approx(1500 - change)   # 一个加多少，另一个减多少（总分不变）


def test_elo_goal_diff_multiplier():
    assert [goal_diff_multiplier(h, a) for h, a in [(0, 0), (1, 0), (2, 0), (0, 3), (5, 1)]] == \
           [1.0, 1.0, 1.5, 1.75, 1.875]                              # 净胜 4 球：(11 + 4) ÷ 8 = 1.875


def test_elo_second_match_uses_updated_ratings():
    """第二场接着第一场的评分算：A 1 比 0 赢 B 后 A = 1508.15、B = 1491.85；第二场 B 主场 0 比 0 平 A：
    B 主队预期得分 = 1 ÷ (1 + 10^(−(1491.85 + 65 − 1508.15) ÷ 400)) = 1 ÷ (1 + 10^(−0.12175)) = 1 ÷ (1 + 0.75553) ≈ 0.56963
    B 得分 0.5，变化 = 20 × (0.5 − 0.56963) ≈ −1.3926 → B ≈ 1490.46，A ≈ 1509.54。"""
    m = EloModel(k=20, home_advantage=65)
    m.update(history([(1, "E0", 2023, "A", "B", "2023-08-12 14:00:00", 1, 0),
                      (2, "E0", 2023, "B", "A", "2023-08-19 14:00:00", 0, 0)]), "2023-08-20 00:00:00")
    a1 = 1500 + 20 * (1 - E_HAND)
    b1 = 1500 - 20 * (1 - E_HAND)
    e2 = 1 / (1 + 10 ** (-(b1 + 65 - a1) / 400))
    assert e2 == pytest.approx(0.56963, abs=1e-5)
    assert m.ratings[("E0", "B")] == pytest.approx(b1 + 20 * (0.5 - e2))
    assert m.ratings[("E0", "A")] == pytest.approx(a1 - 20 * (0.5 - e2))
    # 同一场比赛再交一次（增量取数有重叠时会发生）不会重复加减分
    m.update(history([(2, "E0", 2023, "B", "A", "2023-08-19 14:00:00", 0, 0)]), "2023-08-21 00:00:00")
    assert m.ratings[("E0", "B")] == pytest.approx(b1 + 20 * (0.5 - e2))


def test_elo_new_season_regression_and_promoted():
    """新赛季、升班马（需求文档第 4 节）。拉回比例 r = 0.33，升班马默认比联赛平均低 100。
    2022/23（这个联赛第一个赛季）：A、B、C 都从 1500 开始；为了好算，把赛季末评分直接设成 A 1560、B 1440、C 1500
      → 赛季末联赛平均 = 1500。
    2023/24：
      A 留队：1560 − (1560 − 1500) × 0.33 = 1540.2
      B 留队：1440 − (1440 − 1500) × 0.33 = 1459.8
      D 升班马：以前没有升班马的记录 → 1500 + (−100) = 1400
      A 主场对 D 的评分差 = 1540.2 + 65 − 1400 = 205.2
    再把 2023/24 赛季末设成 A 1550、B 1470、D 1450 → 联赛平均 1490，D（升班马）比平均差 1450 − 1490 = −40。
    2024/25：
      升班马 E：1490 + (−40) = 1450（用以前升班马的真实表现，不再用默认 −100）
      C 2022/23 在、2023/24 不在（降级又升回来）→ 也按升班马：1450
      A：1550 − (1550 − 1490) × 0.33 = 1530.2
      E 主场对 A 的评分差 = 1450 + 65 − 1530.2 = −15.2
    """
    m = EloModel(k=20, home_advantage=65, season_regression=0.33, promoted_default=-100)
    m.update(history([(1, "E0", 2022, "A", "B", "2022-08-06 14:00:00", 1, 0),
                      (2, "E0", 2022, "B", "C", "2022-08-13 14:00:00", 0, 0)]), "2022-08-14 00:00:00")
    m.ratings[("E0", "A")], m.ratings[("E0", "B")], m.ratings[("E0", "C")] = 1560.0, 1440.0, 1500.0

    assert m.rating_diff("E0", 2023, "A", "D") == pytest.approx(205.2)
    assert m.ratings[("E0", "A")] == pytest.approx(1540.2)
    assert m.ratings[("E0", "D")] == pytest.approx(1400.0)
    m.rating_diff("E0", 2023, "B", "A")
    assert m.ratings[("E0", "B")] == pytest.approx(1459.8)
    assert m.ratings[("E0", "A")] == pytest.approx(1540.2)            # 同一赛季第二次碰到 A，不再拉回

    m.ratings[("E0", "A")], m.ratings[("E0", "B")], m.ratings[("E0", "D")] = 1550.0, 1470.0, 1450.0
    assert m.rating_diff("E0", 2024, "E", "A") == pytest.approx(-15.2)
    assert m.ratings[("E0", "E")] == pytest.approx(1450.0)
    m.rating_diff("E0", 2024, "C", "B")
    assert m.ratings[("E0", "C")] == pytest.approx(1450.0)
    assert m.promoted_offsets["E0"] == [(2023, pytest.approx(-40.0))]


def test_elo_predict_needs_fitted_logit_and_gives_probabilities():
    """有序 logit 拟合前（场数不够）不出预测；拟合后三个概率加起来是 1，强队主场打弱队主胜概率更大。"""
    rng = np.random.default_rng(1)
    strength = {"A": 1.0, "B": 0.5, "C": 0.0, "D": -0.5}          # 造数据用的“真实强弱”
    rows, mid = [], 0
    for week in range(60):
        for home, away in [("A", "D"), ("B", "C"), ("C", "A"), ("D", "B")][week % 2 * 2: week % 2 * 2 + 2]:
            mid += 1
            hg = rng.poisson(np.exp(0.3 + 0.4 * (strength[home] - strength[away])))
            ag = rng.poisson(np.exp(0.1 - 0.4 * (strength[home] - strength[away])))
            ko = (pd.Timestamp("2022-08-06 14:00") + pd.Timedelta(days=7 * week)).strftime("%Y-%m-%d %H:%M:%S")
            rows.append((mid, "E0", 2022, home, away, ko, hg, ag))
    h = history(rows)
    t = targets([(999, "E0", 2022, "A", "D"), (1000, "E0", 2022, "D", "A")])
    m = EloModel(min_matches=500, burn_in_seasons=0)
    m.fit(h, "2024-01-01 00:00:00")
    assert m.predict(t, "2024-01-01 00:00:00") == [None, None]      # 只有 120 场 < 500
    m = EloModel(min_matches=50, burn_in_seasons=0)
    m.fit(h, "2024-01-01 00:00:00")
    p = m.predict(t, "2024-01-01 00:00:00")
    assert all(abs(sum(x) - 1) < 1e-9 for x in p)
    assert p[0][0] > p[1][0] and p[0][2] < p[1][2]                   # A 主场打 D 比 D 主场打 A 主胜多、客胜少
    assert m.ratings[("E0", "A")] > m.ratings[("E0", "D")]


def test_elo_burn_in_seasons_not_used_for_logit():
    """头 burn_in_seasons 个赛季（评分还没稳定）不进有序 logit 的拟合数据。"""
    h = history([(1, "E0", 2020, "A", "B", "2020-09-01 14:00:00", 1, 0),
                 (2, "E0", 2021, "A", "B", "2021-09-01 14:00:00", 1, 0),
                 (3, "E0", 2022, "A", "B", "2022-09-01 14:00:00", 1, 0)])
    m = EloModel(burn_in_seasons=2)
    m.update(h, "2023-01-01 00:00:00")
    assert len(m.records["E0"]) == 1 and m.records["E0"][0][0] == "2022-09-01 14:00:00"


# ======================= 有序 logit =======================
def test_ordered_logit_recovers_parameters():
    """用已知参数 b = 1.2、c1 = −0.7、c2 = 0.45 造 2 万场比赛，拟合出来的参数应该很接近。"""
    rng = np.random.default_rng(7)
    diff = rng.uniform(-400, 400, 20000)
    true = np.array([1.2, -0.7, np.log(0.45 + 0.7)])               # 内部参数：c2 = c1 + e^g
    p = ordered_logit.result_probs(true, diff)                     # [主胜, 平, 客胜]
    u = rng.random(len(diff))
    outcome = np.where(u < p[:, 0], 0, np.where(u < p[:, 0] + p[:, 1], 1, 2))
    d = ordered_logit.describe(ordered_logit.fit_ordered_logit(diff, outcome))
    assert d["斜率b"] == pytest.approx(1.2, abs=0.08)
    assert d["切点c1"] == pytest.approx(-0.7, abs=0.06)
    assert d["切点c2"] == pytest.approx(0.45, abs=0.06)


def test_ordered_logit_probabilities_shape():
    """概率加起来是 1、都在 0–1 之间；评分差越大主胜越多、客胜越少。"""
    theta = np.array([1.0, -0.6, np.log(1.1)])
    p = ordered_logit.result_probs(theta, np.linspace(-800, 800, 81))
    np.testing.assert_allclose(p.sum(axis=1), 1.0)
    assert (p > 0).all() and (p < 1).all()
    assert (np.diff(p[:, 0]) > 0).all() and (np.diff(p[:, 2]) < 0).all()


# ======================= 模型登记表 =======================
def test_registry_builds_every_model_from_settings():
    s = load_settings()
    names = available(s)
    assert {"frequency", "frequency_poisson", "elo"} <= set(names)
    for n in names:
        assert build_model(n, s).name == n
    for n, diff in (s["models"].get("elo_variants") or {}).items():   # 变体只改写明的参数，其他和默认 elo 一样
        m, base = build_model(n, s), build_model("elo", s)
        for key in ("k", "home_advantage", "goal_diff", "season_regression", "promoted_default"):
            assert getattr(m, key) == diff.get(key, getattr(base, key))
    with pytest.raises(ValueError):
        build_model("没有这个模型", s)
