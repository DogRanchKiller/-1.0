"""步骤 8 B 版（赔率作为输入）的自动检查（需求文档第 4 节“步骤 8 怎么做”）。

一、src/common/markets.py 的独立泊松矩阵和 src/models/goals.py 的一样（取数层用前者，不能引用模型）。
二、赔率特征（src/features/market_odds.py）：平博 → 多家平均按顺序选、按比例去水、大 2.5 球、市场预期进球、别的来源不用。
三、回测框架接赔率：用 test_backtest 的小联赛，加上赛前赔率。
    历史比赛的赔率特征只给比分已经能知道的比赛；要预测的比赛只在截止时间不早于“开球前 24 小时”时有赔率；
    A版输入的模型拿不到赔率（取数层报错）。
四、market_pre 出的就是赛前赔率去水；梯度提升树的赔率特征“这队 / 对手”方向对、训练表和出预测时一样、A 版不能加赔率。
"""
import numpy as np
import pandas as pd
import pytest

from src.common import markets
from src.common.odds import devig_proportional
from src.evaluate.backtest import run_backtest
from src.features import UsageNotAllowedError
from src.features.market_odds import ODDS_COLS, odds_features
from src.models.base import Model
from src.models.goals import score_matrices
from src.models.market import MarketModel
from src.models.registry import build_model, model_config
from tests.test_adaptive import promo_history
from tests.test_backtest import CONST, MATCHES, fs, settings  # noqa: F401（pytest 的 fixture）
from tests.test_gbdt import gb

CFG = {"source": "football-data", "result_books": ["平博 Pinnacle", "多家平均"], "ou_book": "多家平均", "ou_line": 2.5,
       "devig_method": "proportional", "max_goals": 10}
COLS = ["match_id", "source", "bookmaker", "market", "line", "home", "draw", "away", "over", "under"]


def odds_rows(rows):
    return pd.DataFrame(rows, columns=COLS)


# ======================= 一、独立泊松矩阵 =======================
def test_poisson_matrices_same_as_models():
    lh, la = np.array([0.4, 1.3, 2.9]), np.array([1.1, 0.7, 3.4])
    np.testing.assert_allclose(markets.poisson_matrices(lh, la, 10), score_matrices(lh, la, 0.0, 0.0, 10), atol=1e-14)
    t = markets.matrix_probs([1.6, 0.9], [1.0, 1.4])
    fh, fa = markets.fit_market_lambdas(t)
    np.testing.assert_allclose([fh, fa], [[1.6, 0.9], [1.0, 1.4]], atol=1e-6)
    assert markets.fit_market_lambdas(np.zeros((0, 4)))[0].shape == (0,)


# ======================= 二、赔率特征 =======================
def test_odds_features_source_order_devig_and_lambdas():
    o = odds_rows([
        (1, "football-data", "平博 Pinnacle", "1x2", None, 2.0, 3.5, 4.0, None, None),
        (1, "football-data", "多家平均", "1x2", None, 1.9, 3.4, 3.9, None, None),       # 1 号有平博，不用平均
        (1, "football-data", "多家平均", "ou", 2.5, None, None, None, 1.9, 1.9),
        (1, "football-data", "多家平均", "ou", 3.5, None, None, None, 3.0, 1.35),       # 别的盘口不用
        (2, "football-data", "平博 Pinnacle", "1x2", None, 2.1, None, 3.0, None, None),  # 缺一个 → 用平均
        (2, "football-data", "多家平均", "1x2", None, 2.2, 3.3, 3.4, None, None),
        (3, "football-data", "多家平均", "1x2", None, 0.9, 3.3, 3.4, None, None),        # 不大于 1 → 没有
        (3, "football-data", "多家平均", "ou", 2.5, None, None, None, 2.1, 1.75),
        (4, "dongqiudi", "平博 Pinnacle", "1x2", None, 1.5, 4.0, 6.0, None, None),       # 别的来源不用
    ])
    f = odds_features(o, CFG).set_index("match_id")
    assert list(f.columns) == ODDS_COLS and sorted(f.index) == [1, 2, 3]
    np.testing.assert_allclose(f.loc[1, ["mo_p_home", "mo_p_draw", "mo_p_away"]], devig_proportional([2.0, 3.5, 4.0]))
    np.testing.assert_allclose(f.loc[2, ["mo_p_home", "mo_p_draw", "mo_p_away"]], devig_proportional([2.2, 3.3, 3.4]))
    assert f.loc[1, "mo_pinnacle"] == 1.0 and f.loc[2, "mo_pinnacle"] == 0.0
    assert f.loc[1, "mo_over25"] == pytest.approx(0.5) and np.isnan(f.loc[2, "mo_over25"])
    assert f.loc[3, ["mo_p_home", "mo_p_draw", "mo_p_away", "mo_pinnacle"]].isna().all()
    assert f.loc[3, "mo_over25"] == pytest.approx(devig_proportional([2.1, 1.75])[0])
    # 市场预期进球：只有 1 号四个数都有；反推出来的矩阵还原赔率
    assert f[["mo_loglam_home", "mo_loglam_away"]].notna().all(axis=1).tolist() == [True, False, False]
    # 独立泊松还原不了每一组赔率（平局常常少给一两个百分点，第 5 节“市场比分矩阵”），所以只核对“就是拟合的结果”
    target = np.array([[*devig_proportional([2.0, 3.5, 4.0]), 0.5]])
    lh, la = markets.fit_market_lambdas(target)
    np.testing.assert_allclose([f.loc[1, "mo_loglam_home"], f.loc[1, "mo_loglam_away"]], np.log([lh[0], la[0]]))
    assert f.loc[1, "mo_loglam_home"] > f.loc[1, "mo_loglam_away"]
    assert odds_features(odds_rows([]), CFG).empty


# ======================= 三、回测框架接赔率 =======================
PRE_PIN = [m[0] for m in MATCHES if m[2] == 2022 and m[0] % 10 == 1]       # 2022/23 每轮第 2 场有平博赛前
ALL_LEAGUE = [m[0] for m in MATCHES if m[2] in (2021, 2022)]


@pytest.fixture
def fs_odds(fs):
    c = fs.conn
    c.executemany("insert into odds (match_id, source, bookmaker, market, timing, line, home, draw, away, over, under)"
                  " values (?, 'football-data', '多家平均', '1x2', 'pre', null, 2.4, 3.3, 3.0, null, null)",
                  [(m,) for m in ALL_LEAGUE])
    c.executemany("insert into odds (match_id, source, bookmaker, market, timing, line, home, draw, away, over, under)"
                  " values (?, 'football-data', '多家平均', 'ou', 'pre', 2.5, null, null, null, 2.0, 1.8)",
                  [(m,) for m in ALL_LEAGUE])
    c.executemany("insert into odds (match_id, source, bookmaker, market, timing, line, home, draw, away, over, under)"
                  " values (?, 'football-data', '平博 Pinnacle', '1x2', 'pre', null, 1.7, 3.8, 5.5, null, null)",
                  [(m,) for m in PRE_PIN])
    c.commit()
    fs.refresh()
    return fs


class OddsSpy(Model):
    """要赔率的“偷看检查员”：记下每次收到的历史和目标。"""

    name = "odds_spy"
    usage = "B版输入"
    uses_pre_odds = True

    def __init__(self):
        self.log = []

    def fit(self, history, cutoff):
        self.log.append(("fit", cutoff, history.copy()))

    def update(self, new_rows, cutoff):
        self.log.append(("update", cutoff, new_rows.copy()))

    def predict(self, targets, cutoff):
        self.log.append(("predict", cutoff, targets.copy()))
        return [CONST] * len(targets)


def test_backtest_gives_odds_features_without_peeking(fs_odds):
    spy = OddsSpy()
    run_backtest(fs_odds, "E0", 2022, 2022, [spy], horizons=[48, 24])
    pin = devig_proportional([1.7, 3.8, 5.5])
    avg = devig_proportional([2.4, 3.3, 3.0])
    seen_hist = 0
    for kind, cutoff, df in spy.log:
        if df.empty:
            continue
        if kind in ("fit", "update"):
            assert (df["known_at_utc"] <= cutoff).all()
            if kind == "update":                         # 每一批新比赛都带着自己的赔率特征
                assert df["mo_p_home"].notna().all() and df["mo_over25"].notna().all()
                seen_hist += len(df)
            continue
        # 要预测的比赛：T-48 赔率还不能知道（开球前 24 小时才算能知道）→ 空值；T-24 有
        ko = df["kickoff_earliest_utc"].map(pd.Timestamp)
        early = (ko - pd.Timestamp(cutoff)) > pd.Timedelta(hours=24)
        assert df.loc[early, ODDS_COLS].isna().all().all()
        late = df[~early]
        for _, r in late.iterrows():
            want = pin if r["match_id"] in PRE_PIN else avg
            np.testing.assert_allclose(r[["mo_p_home", "mo_p_draw", "mo_p_away"]].to_numpy(float), want)
            assert r["mo_pinnacle"] == float(r["match_id"] in PRE_PIN)
            assert r["mo_over25"] == pytest.approx(devig_proportional([2.0, 1.8])[0])
    assert seen_hist > 0
    assert any(not df.empty and df[ODDS_COLS[0]].notna().any() for k, _, df in spy.log if k == "predict")


def test_a_version_cannot_get_odds(fs_odds):
    spy = OddsSpy()
    spy.usage = "A版输入"
    with pytest.raises(UsageNotAllowedError):
        run_backtest(fs_odds, "E0", 2022, 2022, [spy], horizons=[24])


# ======================= 四、market_pre 和梯度提升树 =======================
def test_market_pre_is_devigged_pre_odds(fs_odds, settings):
    m = build_model("market_pre", settings)
    assert m.usage == "B版输入" and m.uses_pre_odds and model_config("market_pre", settings)["market_odds"]
    res = run_backtest(fs_odds, "E0", 2022, 2022, [m], horizons=[48, 24])
    p = res.predictions[res.predictions["model"] == "market_pre"].set_index(["match_id", "horizon"])
    some = PRE_PIN[0]
    np.testing.assert_allclose(p.loc[(some, "T-24"), ["p_home", "p_draw", "p_away"]].to_numpy(float),
                               devig_proportional([1.7, 3.8, 5.5]))
    assert p.xs("T-48", level="horizon")["p_home"].isna().all()       # T-48 还没有赔率 → 不出预测
    with pytest.raises(ValueError):
        MarketModel("x", usage="A版输入")


ODDS_VALS = {"mo_p_home": 0.5, "mo_p_draw": 0.27, "mo_p_away": 0.23, "mo_over25": 0.55,
             "mo_loglam_home": 0.4, "mo_loglam_away": 0.05, "mo_pinnacle": 1.0}


def test_gbdt_odds_features_own_opp_and_training_matches_predict():
    h = promo_history()
    g, base = gb(extra=["odds"], usage="B版输入"), gb()
    assert g.uses_pre_odds and not getattr(base, "uses_pre_odds", False) and g.data == ("比分", "射门、xG")
    assert g.names[len(base.names):] == ["mkt_p_win", "mkt_p_draw", "mkt_p_loss", "mkt_over25", "mkt_loglam_for",
                                         "mkt_loglam_against", "mkt_pinnacle"]
    row = h[h["season"] == 2020].sort_values(["kickoff_earliest_utc", "match_id"]).head(1)
    before = h[h["kickoff_earliest_utc"] < row["kickoff_earliest_utc"].iloc[0]]
    for m in (g, base):
        m.update(before.assign(**ODDS_VALS), "2020-08-01 00:00:00")
    r = row.iloc[0]
    rest = (7.0,) * 2 + (0.0,) * 2
    odds = tuple(ODDS_VALS[c] for c in ODDS_COLS)
    f = g._features("E0", 2020, r["home_team"], r["away_team"], r["kickoff_earliest_utc"], rest, None, odds)
    hv, av = dict(zip(g.names, f[0])), dict(zip(g.names, f[1]))
    assert [hv["mkt_p_win"], hv["mkt_p_loss"], av["mkt_p_win"], av["mkt_p_loss"]] == [0.5, 0.23, 0.23, 0.5]
    assert hv["mkt_p_draw"] == av["mkt_p_draw"] == 0.27 and hv["mkt_over25"] == av["mkt_over25"] == 0.55
    assert [hv["mkt_loglam_for"], hv["mkt_loglam_against"], av["mkt_loglam_for"], av["mkt_loglam_against"]] == \
        [0.4, 0.05, 0.05, 0.4]
    fb = base._features("E0", 2020, r["home_team"], r["away_team"], r["kickoff_earliest_utc"], rest, None, odds)
    assert np.array_equal(f[0][:len(base.names)], fb[0], equal_nan=True) and len(fb[0]) == len(base.names)
    # 训练表里存的和出预测时算的一样
    g.update(row.assign(**ODDS_VALS, home_rest_days=7.0, away_rest_days=7.0, home_after_euro=0.0,
                        away_after_euro=0.0), "2020-08-01 00:00:00")
    assert np.array_equal(g.rows[-2], f[0], equal_nan=True) and np.array_equal(g.rows[-1], f[1], equal_nan=True)
    # 没有赔率的比赛（2005 年以前）→ 空值
    g2 = gb(extra=["odds"], usage="B版输入")
    g2.update(before, "2020-08-01 00:00:00")
    assert np.isnan(np.vstack(g2.rows)[:, -7:]).all()


def test_gbdt_odds_rules_and_registry():
    with pytest.raises(ValueError):
        gb(extra=["odds"])                               # 默认用途 A版输入，不能加赔率
    with pytest.raises(ValueError):
        gb(features="mktonly", usage="B版输入")           # mktonly 要有赔率特征
    m = gb(extra=["odds"], usage="B版输入", features="mktonly")
    assert [m.names[i] for i in m.use] == ["league", "is_home", "mkt_p_win", "mkt_p_draw", "mkt_p_loss", "mkt_over25",
                                            "mkt_loglam_for", "mkt_loglam_against", "mkt_pinnacle"]
    base, b = model_config("gb_l7"), model_config("gb_l7_b")
    assert {k: v for k, v in b.items() if k != "usage"} == base and b["usage"] == "B版输入"
    for name, feats in (("gb_b", "all"), ("gb_mkt", "mktonly")):
        mm = build_model(name)
        assert mm.extra == ("odds",) and mm.usage == "B版输入" and mm.features == feats and mm.uses_pre_odds
    assert not getattr(build_model("gb_l7_b"), "uses_pre_odds", False)


def test_b_study_gap_groups():
    """分歧组：两边三个概率里差得最多的那个，按 3% / 6% 分三份；一边没有预测的比赛不分。"""
    from src.evaluate.b_study import gap_groups

    rows = []
    for mid, pa, pb in [(1, [0.50, 0.27, 0.23], [0.49, 0.28, 0.23]),     # 差 1%
                        (2, [0.50, 0.27, 0.23], [0.46, 0.29, 0.25]),     # 差 4%
                        (3, [0.50, 0.27, 0.23], [0.40, 0.30, 0.30]),     # 差 10%
                        (4, [0.50, 0.27, 0.23], [np.nan] * 3)]:          # 一边没有
        rows += [{"match_id": mid, "horizon": "T-24", "model": "gb_b", **dict(zip(["p_home", "p_draw", "p_away"], pa))},
                 {"match_id": mid, "horizon": "T-24", "model": "market_pre",
                  **dict(zip(["p_home", "p_draw", "p_away"], pb))}]
    g = gap_groups(pd.DataFrame(rows), "gb_b", "market_pre", [0.03, 0.06])
    assert g.to_dict() == {1: "分歧 3% 以内", 2: "分歧 3%–6%", 3: "分歧 6% 以上"}
