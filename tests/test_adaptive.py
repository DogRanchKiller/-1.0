"""第 4 节步骤 5 补充：每队自己的变化速度（src/models/adaptive.py）的自动检查（不需要数据库，用模拟比赛）。

检查的东西：
  - 只有一个候选、或几个候选的 q 都一样时，ada 的预测和 dyn 完全一样（有升班马、跨赛季也一样）；
  - 权重更新和手算的“先开 α 次方、再乘泊松概率、再归一”一样；
  - 一队实力中途突然变好：这队的权重偏向 q 大的候选，实力不变的队偏向 q 小的；
  - 升班马第一场之前权重回到 1/K；对照组（per_team = false）各队用联赛那一套；
  - 出预测不改状态；登记表能按名字造出来，比分矩阵加起来是 1。
"""
import numpy as np
import pandas as pd
import pytest

from src.models.adaptive import LEAGUE_KEY, AdaptiveDynamicModel
from src.models.dynamic import DynamicModel
from src.models.registry import available, build_model, model_config
from tests.test_dynamic import jump_history
from tests.test_goals_models import history, targets


def promo_history(seed=0, seasons=(2017, 2018, 2019, 2020, 2021)):
    """8 队双循环，每个赛季换掉一个队（新队升上来；2020 年让 2017 年降级的队再升回来）。比分是泊松。"""
    rng = np.random.default_rng(seed)
    strength = {f"T{k}": (rng.normal(0, 0.3), rng.normal(0, 0.3)) for k in range(14)}
    teams = [f"T{k}" for k in range(8)]
    rows, mid, nxt = [], 0, 8
    for s in seasons:
        if s > seasons[0]:
            teams = teams[1:] + (["T0"] if s == 2020 else [f"T{nxt}"])
            nxt += s != 2020
        day = pd.Timestamp(f"{s}-08-15 14:00:00")
        for i in teams:
            for j in teams:
                if i != j:
                    mid += 1
                    (ai, di), (aj, dj) = strength[i], strength[j]
                    ko = day + pd.Timedelta(days=mid % 56 * 5)
                    rows.append((mid, "E0", s, i, j, ko.strftime("%Y-%m-%d %H:%M:%S"),
                                 int(rng.poisson(np.exp(0.15 + 0.25 + ai - dj))), int(rng.poisson(np.exp(0.15 + aj - di)))))
    return history(rows)


def run_weekly(m, h, cutoffs, t):
    """模拟回测：每个截止时间先 fit，再出预测。返回每次的预测。"""
    out = []
    for c in cutoffs:
        m.fit(h[h["kickoff_earliest_utc"] < c], c)
        out.append(m.predict(t, c))
    return out


CUTS = ["2019-10-01 00:00:00", "2020-08-20 00:00:00", "2020-12-01 00:00:00", "2021-11-01 00:00:00"]
KW = dict(min_matches=20, burn_in_seasons=2, season_bump=0.01)


@pytest.mark.parametrize("qs", [[1e-5], [3e-5, 3e-5, 3e-5]])
def test_same_as_dyn_when_candidates_equal(qs):
    h = promo_history()
    teams = sorted(set(h["home_team"]))
    t = targets([(9000 + k, "E0", s, a, b) for k, (s, a, b) in enumerate(
        [(2019, "T3", "T4"), (2020, "T0", "T5"), (2020, "T9", "T6"), (2021, "T10", "T7"), (2021, "T11", "T9")])])
    dyn = DynamicModel(q=qs[0], **KW)
    ada = AdaptiveDynamicModel(candidates_q=qs, alpha=0.95, **KW)
    for pd_, pa in zip(run_weekly(dyn, h, CUTS, t), run_weekly(ada, h, CUTS, t)):
        for x, y in zip(pd_, pa):
            assert (x is None) == (y is None)
            if x is not None:
                assert np.allclose(x, y, atol=1e-10)
    assert ada.rho["E0"] == pytest.approx(dyn.leagues["E0"].rho, abs=1e-10)
    assert len(teams) > 8                                   # 确实有升班马


def test_weight_update_by_hand():
    h = promo_history(seasons=(2017, 2018))
    m = AdaptiveDynamicModel(candidates_q=[1e-5, 1e-4], alpha=0.9, **KW)
    first, rest = h.iloc[:-1], h.iloc[-1:]
    m._process(first)
    comp, season, home, away = "E0", 2018, rest["home_team"].iloc[0], rest["away_team"].iloc[0]
    hg, ag = int(rest["home_goals"].iloc[0]), int(rest["away_goals"].iloc[0])
    before = {k: m.logw["E0"][k].copy() for k in (home, away, LEAGUE_KEY)}
    for f in m.filters:                                     # 先把时间走到开球，才能拿到每个候选赛前的预期进球
        f._prepare(comp, season, home, away, rest["kickoff_earliest_utc"].iloc[0])
    own = m._mixed(comp, season, home, away)[2]
    m._process(rest)
    for k, lw in before.items():
        w = np.exp(0.9 * lw) * np.exp(hg * np.log(own[:, 0]) - own[:, 0] + ag * np.log(own[:, 1]) - own[:, 1])
        assert np.exp(m.logw["E0"][k]) == pytest.approx(w / w.sum(), rel=1e-9)
    assert m.rec["E0"][-1][3:] == (hg, ag)


def test_jump_team_prefers_fast_candidate():
    h = jump_history(jump=0.6, n_rounds=60)
    m = AdaptiveDynamicModel(candidates_q=[1e-6, 1e-3], alpha=0.98, min_matches=20)
    m._process(h)
    w0 = m._weights("E0", "T0")
    others = np.mean([m._weights("E0", f"T{k}")[1] for k in range(1, 8)])
    assert w0[1] > 0.6                                       # 突然变好的 T0 偏向 q 大的候选
    assert w0[1] > others + 0.2                              # 比实力不变的队更偏向快的
    tab = m.ratings_table()
    assert tab.set_index("team").loc["T0", "q_weighted"] > tab.set_index("team").drop("T0")["q_weighted"].max()


def test_promoted_reset_and_league_control():
    h = promo_history()
    m = AdaptiveDynamicModel(candidates_q=[1e-5, 1e-4], alpha=0.95, **KW)
    m._process(h[h["season"] <= 2019])
    assert not np.allclose(m.logw["E0"]["T0"], 0)           # T0 2017 踢过，有旧权重
    assert np.allclose(m._team_weights("E0", 2020, "T0"), 0.5)   # 2020 重新升上来：出预测时用 1/K
    assert np.array_equal(m._team_weights("E0", 2020, "T5"), m._weights("E0", "T5"))   # 一直在的队接着用自己的
    lg = m.filters[0].leagues["E0"]
    before = ({k: v.copy() for k, v in m.logw["E0"].items()}, lg.m.copy(), lg.season)
    m.predict(targets([(1, "E0", 2020, "T0", "T5"), (2, "E0", 2020, "NEW", "T5")]), "2020-08-01 00:00:00")
    assert all(np.array_equal(before[0][k], v) for k, v in m.logw["E0"].items())   # 出预测不改状态
    assert np.array_equal(lg.m, before[1]) and lg.season == before[2]
    s20 = h[h["season"] == 2020].sort_values("kickoff_earliest_utc")
    first_t0 = s20[(s20["home_team"] == "T0") | (s20["away_team"] == "T0")].index[0]
    m._process(s20[s20["kickoff_earliest_utc"] < s20.loc[first_t0, "kickoff_earliest_utc"]])
    assert not np.allclose(m.logw["E0"]["T0"], 0)           # 第一场还没踢：旧权重还在（只是出预测时不用）
    m._process(s20.loc[[first_t0]])
    assert (2020, "T0") in m.reset_done["E0"]
    league = AdaptiveDynamicModel(candidates_q=[1e-5, 1e-4], alpha=0.95, per_team=False, **KW)
    league._process(h)
    assert set(league.logw["E0"]) == {LEAGUE_KEY}
    assert np.allclose(league._team_weights("E0", 2021, "T3"), league._weights("E0", LEAGUE_KEY))


def test_registry_and_matrices():
    s = model_config("ada_a98")
    assert "q" not in s and s["candidates_q"] == [3e-6, 1e-5, 3e-5, 1e-4] and s["season_bump"] == 0.01
    assert {"ada_a95", "ada_a98", "ada_a99"} <= set(available())
    m = build_model("ada_a95")
    assert isinstance(m, AdaptiveDynamicModel) and m.alpha == 0.95 and [f.q for f in m.filters] == s["candidates_q"]
    h = promo_history()
    m2 = AdaptiveDynamicModel(**KW)
    out = run_weekly(m2, h, CUTS[-1:], targets([(1, "E0", 2021, "T3", "T4"), (2, "E0", 2018, "T3", "T4")]))[0]
    assert out[1] is None                                    # 头两个赛季不出
    assert out[0].shape == (11, 11) and out[0].sum() == pytest.approx(1)
    assert set(m2.params()["E0"]["联赛权重（每个 q）"]) == {"3e-06", "1e-05", "3e-05", "0.0001"}
