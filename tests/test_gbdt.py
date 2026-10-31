"""第 4 节步骤 6：特征表 + 梯度提升树（src/models/gbdt.py、src/features/team_form.py、src/features/rest_days.py）的自动检查。

不需要数据库，用模拟比赛。检查的东西：
  - 休息天数：按日期算、同一天的不算、最多记 cap 天、欧战后标签；
  - pi-rating 一场赛后的变化和手算一样；升班马初始分；view 不改状态；
  - 积分榜、近期状态和手算一样（射正有空值也行）；
  - dyn 的“赛前看一眼”（pre_match_view）和真的走到开球后看状态完全一样（换赛季、升班马、补赛、挪平均值后都一样）；
  - 里面带的 dyn、Elo 和单独跑的完全一样；
  - 一场比赛的特征里没有这场自己的比分；出预测时算的特征和这场后来处理时存进训练表的一样；
  - 同一时刻开球的几场，互相看不到对方的比分；
  - 只在 dyn 出预测的比赛上出预测，比分矩阵加起来是 1，同样的数据跑两次结果一样；胜平负对照组出三个概率；
  - 登记表能按名字造出来，里面 dyn 的设置和 dyn_q1e5_b01 一样。
"""
import copy
import math

import numpy as np
import pandas as pd
import pytest

from src.features.rest_days import RestDays
from src.features.team_form import Form, PiRating, Standings
from src.models.dynamic import DynamicModel
from src.models.elo import EloModel
from src.models.gbdt import GBDTModel
from src.models.registry import available, build_model, model_config
from tests.test_adaptive import promo_history
from tests.test_goals_models import targets

DYN = dict(q=3e-5, season_bump=0.01, min_matches=20, burn_in_seasons=2)
ELO = dict(goal_diff=True, min_matches=20, burn_in_seasons=2)
PI = dict(lam=0.035, gamma=0.7, b=10, c=3, history_seasons=3)
FORM = dict(windows=(3, 5), half_lives=(3,), within_days=365)
SMALL = dict(num_leaves=4, min_data_in_leaf=5, max_rounds=40, early_stopping_rounds=5, valid_days=120,
             retrain_days=60, min_train_rows=50, seed=1)


def gb(name="gb", **kw):
    return GBDTModel(name=name, dyn=DYN, elo=ELO, pi=PI, form=FORM, leagues=["E0"], **{**SMALL, **kw})


# ======================= 休息天数 =======================
def sched(rows):
    return pd.DataFrame(rows, columns=["match_id", "competition", "match_date", "home_team", "away_team"])


def test_rest_days():
    r = RestDays(["UCL"], cap_days=30, euro_within_days=4)
    r.add(sched([(1, "E0", "2024-01-06", "A", "B"), (2, "UCL", "2024-01-09", "A", "C"),
                 (3, "E0", "2024-01-13", "B", "A"), (4, "E0", "2024-03-30", "B", "D")]))
    r.add(sched([(3, "E0", "2024-01-13", "B", "A")]))          # 重复的跳过
    assert r.team("A", "2024-01-13") == (4.0, 1)                # 周二欧战 → 周六：4 天，欧战后
    assert r.team("B", "2024-01-13") == (7.0, 0)                # 同一天那场（3 号自己）不算
    assert r.team("A", "2024-01-14") == (1.0, 0)
    assert r.team("C", "2024-01-15") == (6.0, 0)                # 欧战后但隔了 6 天 → 不算
    assert r.team("D", "2024-03-30") == (30.0, 0)               # 没有上一场 → 记 30
    assert r.team("B", "2024-03-30") == (30.0, 0)               # 隔了 77 天 → 最多 30
    out = r.annotate(pd.DataFrame({"match_date": ["2024-01-13"], "home_team": ["B"], "away_team": ["A"]}))
    assert out[["home_rest_days", "away_rest_days", "home_after_euro", "away_after_euro"]].iloc[0].tolist() == [7, 4, 0, 1]


# ======================= pi-rating、积分榜、近期状态 =======================
def test_pi_rating_by_hand():
    p = PiRating(**PI)
    p.observe("E0", 2020, "A", "B", 3, 0)
    step = 3 * math.log10(1 + 3)                                # 预期 0、误差 3
    assert p.view("E0", 2020, "A") == pytest.approx((step * 0.035, step * 0.035 * 0.7))
    assert p.view("E0", 2020, "B") == pytest.approx((-step * 0.035 * 0.7, -step * 0.035))
    # 第二场：A 主场分 > 0 → 预期净胜球 ψ(H_A) − ψ(A_C)
    hA = p.view("E0", 2020, "A")[0]
    egd = math.copysign(10 ** (abs(hA) / 3) - 1, hA)
    before = copy.deepcopy(p.r)
    p.view("E0", 2020, "C")
    assert p.r == before                                        # view 不改状态
    p.observe("E0", 2020, "A", "C", 0, 0)
    assert p.view("E0", 2020, "A")[0] == pytest.approx(hA - 3 * math.log10(1 + egd) * 0.035)


def test_pi_rating_promoted_start():
    """升班马初始分 = 最近几个赛季升班马赛季末的平均；第一个赛季的队、留队的队不算升班马。"""
    p = PiRating(**PI)
    for s, teams in [(2018, "ABC"), (2019, "ABD"), (2020, "ABE")]:
        for i, h in enumerate(teams):
            for a in teams:
                if h != a:
                    p.observe("E0", s, h, a, 2 if h == "D" else 1, 1)
    assert not p.is_promoted("E0", 2018, "A") and p.is_promoted("E0", 2019, "D") and not p.is_promoted("E0", 2019, "A")
    d_end = p.end["E0"][2019][0]                                # D 在 2019 赛季末的分
    assert p._prior("E0", 2021) == pytest.approx(np.mean([d_end, p.r[("E0", "E")]], axis=0))
    assert p.view("E0", 2021, "NEW") == pytest.approx(tuple(p._prior("E0", 2021)))
    assert p.view("E0", 2021, "A") == tuple(p.r[("E0", "A")])  # 留队的接着用


def test_standings_by_hand():
    t = Standings()
    t.observe("E0", 2020, "A", "B", 2, 0)
    t.observe("E0", 2020, "C", "A", 1, 1)
    t.observe("E0", 2020, "B", "C", 3, 0)
    assert t.view("E0", 2020, "A") == (2.0, 2.0, 1.0, 0.0)       # 4 分、净胜 +2 → 第一
    assert t.view("E0", 2020, "B") == (2.0, 1.5, 0.5, 0.5)       # 3 分
    assert t.view("E0", 2020, "C")[3] == 1.0                     # 1 分 → 最后
    assert t.view("E0", 2020, "D")[0] == 0 and np.isnan(t.view("E0", 2020, "D")[1])
    assert t.view("E0", 2021, "A")[0] == 0                       # 新赛季从 0 开始


def test_form_by_hand():
    f = Form(windows=(2, 5), half_lives=(1,), within_days=30)
    f.observe("E0", "A", 0, 1, 0, 5, 2)
    f.observe("E0", "A", 7, 0, 2, float("nan"), 4)
    f.observe("E0", "A", 14, 3, 3, 7, 1)
    v = dict(zip(f.names, f.view("E0", "A", 20)))
    assert v["form_n"] == 3
    assert v["form2_gf"] == 1.5 and v["form2_pts"] == 0.5 and v["form2_sotf"] == 7   # 射正空值不算
    assert v["form5_gf"] == pytest.approx(4 / 3) and v["form5_sotf"] == 6
    w = np.array([0.25, 0.5, 1.0])                               # 半衰期 1 场：越早的权重减半
    assert v["ewm1_gf"] == pytest.approx((w * [1, 0, 3]).sum() / w.sum())
    assert dict(zip(f.names, f.view("E0", "A", 40)))["form_n"] == 1   # 30 天以外的不算
    assert np.isnan(f.view("E0", "B", 20)[1])


# ======================= dyn 的赛前看一眼 =======================
def _state_view(lg, home, away):
    """真的走到开球（_prepare 之后）再从状态算：log λ、标准差、主场优势。"""
    ah, dh = lg.pos(home)
    aa, da = lg.pos(away)
    H = np.zeros((2, len(lg.m)))
    H[0, [0, 1, ah, da]] = (1, 1, 1, -1)
    H[1, [0, aa, dh]] = (1, 1, -1)
    return (*(H @ lg.m), *np.sqrt(np.diag(H @ lg.P @ H.T)), lg.m[1])


def test_pre_match_view_same_as_prepare():
    h = promo_history(seasons=(2017, 2018, 2019, 2020))
    late = pd.DataFrame([{"match_id": 9999, "competition": "E0", "season": 2018, "home_team": "T1", "away_team": "T2",
                          "kickoff_earliest_utc": "2019-09-20 14:00:00", "home_goals": 1, "away_goals": 0, "result": "H"}])
    h = pd.concat([h, late], ignore_index=True).sort_values(["kickoff_earliest_utc", "match_id"])
    m = DynamicModel(**DYN)
    checked = 0
    for k, (mid, comp, season, home, away, ko, hg, ag) in enumerate(h[
            ["match_id", "competition", "season", "home_team", "away_team", "kickoff_earliest_utc", "home_goals",
             "away_goals"]].itertuples(index=False, name=None)):
        v = m.pre_match_view(comp, int(season), home, away, ko)
        lg = m._prepare(comp, int(season), home, away, ko)
        if k > 0:
            assert v == pytest.approx(_state_view(lg, home, away), rel=1e-10, abs=1e-12)
            checked += 1
        m._observe(lg, ko, home, away, hg, ag)
        if k % 25 == 24:
            m._recentre(lg)                                     # 挪平均值不影响 log λ 和它的不确定
    assert checked == len(h) - 1


# ======================= 梯度提升树 =======================
def weekly_run(models, h, start, end):
    """模拟回测：每周一个截止时间，先 fit，再预测下一周的比赛。返回每个模型每周的预测。"""
    out = [[] for _ in models]
    cuts = []
    for c in pd.date_range(start, end, freq="7D"):
        c, nxt = c.strftime("%Y-%m-%d %H:%M:%S"), (c + pd.Timedelta(days=7)).strftime("%Y-%m-%d %H:%M:%S")
        t = h[(h["kickoff_earliest_utc"] >= c) & (h["kickoff_earliest_utc"] < nxt)]
        cuts.append(c)
        for m, o in zip(models, out):
            m.fit(h[h["kickoff_earliest_utc"] < c], c)
            o.append(None if m.predict is None else m.predict(t, c))
    return cuts, out


def test_inner_dyn_elo_same_as_standalone():
    h = promo_history()
    g, d, e = gb(), DynamicModel(**DYN), EloModel(**ELO)
    e.predict = None                                            # Elo 只更新、不出预测（gbdt 里的 Elo 也不拟合）
    cuts, (pg, pd_, _) = weekly_run([g, d, e], h, "2018-08-01", "2022-06-01")
    for c in d.leagues:
        assert np.array_equal(g.dyn.leagues[c].m, d.leagues[c].m) and np.array_equal(g.dyn.leagues[c].P, d.leagues[c].P)
        assert g.dyn.leagues[c].rho == d.leagues[c].rho and g.dyn.leagues[c].rec == d.leagues[c].rec
    assert g.elo.ratings == e.ratings
    n = 0
    first = g.train_log[0]["cutoff"]
    for c, a, b in zip(cuts, pg, pd_):                          # 树训练好以后：dyn 出预测的比赛它才出
        if c >= first:
            assert [x is None for x in a] == [x is None for x in b]
            for x in a:
                if x is not None:
                    assert np.isclose(x.sum(), 1)
                    n += 1
    assert n > 50 and len(g.train_log) >= 3


def test_features_never_see_own_result_and_predict_matches_training():
    h = promo_history()
    first2020 = h[h["season"] == 2020].sort_values(["kickoff_earliest_utc", "match_id"]).head(4)
    before = h[h["kickoff_earliest_utc"] < first2020["kickoff_earliest_utc"].min()]
    g = gb()
    g.update(before, "2020-08-01 00:00:00")
    rest = (3.0, 8.0, 1.0, 0.0)
    for _, row in first2020.iterrows():                         # 新赛季第一场起，有升班马 T0
        f = g._features("E0", 2020, row["home_team"], row["away_team"], row["kickoff_earliest_utc"], rest)
        one = row.to_frame().T.assign(home_rest_days=3.0, away_rest_days=8.0, home_after_euro=1.0, away_after_euro=0.0)
        g.update(one, "2020-08-01 00:00:00")
        assert np.array_equal(g.rows[-2], f[0], equal_nan=True) and np.array_equal(g.rows[-1], f[1], equal_nan=True)
    v = dict(zip(g.names, g.rows[-2]))
    assert v["is_home"] == 1 and v["rest_days_own"] == 3 and v["rest_days_opp"] == 8 and v["after_euro_own"] == 1
    assert len(g.names) == len(g.rows[-1])
    # 改最后一场的比分：它自己和之前所有比赛的特征都不变
    k = 200
    order = h.sort_values(["kickoff_earliest_utc", "match_id"]).reset_index(drop=True)
    changed = order.copy()
    changed.loc[k, ["home_goals", "away_goals", "result"]] = [9, 0, "H"]
    a, b = gb(), gb()
    a.update(order, "2030-01-01 00:00:00")
    b.update(changed, "2030-01-01 00:00:00")
    n = 2 * int((order.loc[:k, "season"] >= 2019).sum())        # 2019 起才进训练表（前两季只让评分稳定）
    assert np.array_equal(np.vstack(a.rows[:n]), np.vstack(b.rows[:n]), equal_nan=True)
    assert not np.array_equal(np.vstack(a.rows[n:]), np.vstack(b.rows[n:]), equal_nan=True)


def test_same_kickoff_matches_dont_see_each_other():
    """同一时刻开球的两场：改其中一场的比分，另一场存进训练表的特征不变（出预测时两场的比分都还不知道）。
    把模拟比赛里 k、k+1 两场挪到同一个开球时间，k 是 match_id 小的那场（先更新的）。
    里面的 dyn、Elo 和单独跑的照样一样（更新顺序没变）。"""
    h = promo_history()
    order = h.sort_values(["kickoff_earliest_utc", "match_id"]).reset_index(drop=True)
    k = 200
    order.loc[k + 1, "kickoff_earliest_utc"] = order.loc[k, "kickoff_earliest_utc"]
    assert order.loc[k, "match_id"] < order.loc[k + 1, "match_id"] and order.loc[k, "season"] >= 2019
    changed = order.copy()
    changed.loc[k, ["home_goals", "away_goals", "result"]] = [9, 0, "H"]
    a, b = gb(), gb()
    a.update(order, "2030-01-01 00:00:00")
    b.update(changed, "2030-01-01 00:00:00")
    n = 2 * int((order.loc[:k + 1, "season"] >= 2019).sum())      # k、k+1 两场的特征都在前 n 行里
    assert np.array_equal(np.vstack(a.rows[:n]), np.vstack(b.rows[:n]), equal_nan=True)
    assert not np.array_equal(np.vstack(a.rows[n:]), np.vstack(b.rows[n:]), equal_nan=True)
    # 和单独跑的 dyn、Elo 一样
    d, e = DynamicModel(**DYN), EloModel(**ELO)
    d.update(order, "2030-01-01 00:00:00")
    e.update(order, "2030-01-01 00:00:00")
    assert a.elo.ratings == e.ratings
    for c in d.leagues:
        assert np.array_equal(a.dyn.leagues[c].m, d.leagues[c].m) and np.array_equal(a.dyn.leagues[c].P, d.leagues[c].P)


def test_deterministic_and_1x2():
    h = promo_history()
    t = targets([(1, "E0", 2021, "T3", "T4"), (2, "E0", 2021, "T5", "NEW"), (3, "E0", 2016, "T1", "T2")]).assign(
        kickoff_earliest_utc="2021-08-20 14:00:00")
    outs = []
    for kw in ({}, {}, {"target": "result"}):
        m = gb(**kw)
        for c in pd.date_range("2019-08-01", "2021-08-01", freq="7D"):
            c = c.strftime("%Y-%m-%d %H:%M:%S")
            m.fit(h[h["kickoff_earliest_utc"] < c], c)
        outs.append(m.predict(t, "2021-08-01 00:00:00"))
    a, b, r = outs
    assert a[0] is not None and a[1] is not None and a[2] is None    # 2016 赛季（联赛还没数据）→ 不出
    assert np.isclose(a[0].sum(), 1) and np.array_equal(a[0], b[0]) and np.array_equal(a[1], b[1])
    assert r[0].shape == (3,) and np.isclose(r[0].sum(), 1) and r[2] is None
    nd = gb(features="nodyn")
    assert not any(nd.names[i].startswith("dyn_") for i in nd.use) and len(nd.use) == len(nd.names) - 5
    do = gb(features="dynonly")
    assert {do.names[i] for i in do.use} == {"league", "is_home", "dyn_loglam_for", "dyn_loglam_against", "dyn_sd_for",
                                             "dyn_sd_against", "dyn_home"}
    with pytest.raises(ValueError):
        gb(features="xg")


def test_registry_builds_gbdt():
    names = available()
    assert {"gb_l7", "gb_l15", "gb_l31"} <= set(names)
    cfg = model_config("gb_l7")
    assert cfg["dyn"] == model_config("dyn_q1e5_b01") and cfg["elo"] == model_config("elo_gd")
    assert cfg["num_leaves"] == 7 and cfg["min_data_in_leaf"] == 200
    m = build_model("gb_l31")
    assert isinstance(m, GBDTModel) and m.uses_rest_days and m.lgb_params["num_leaves"] == 31
    assert m.data == ("比分", "射门、xG") and m.usage == "A版输入"
