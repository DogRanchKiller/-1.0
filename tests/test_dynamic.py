"""第 4 节步骤 5：动态实力模型（src/models/dynamic.py）的自动检查（不需要数据库，用模拟比赛）。

检查的东西：
  - 一场比赛的更新和课本上“先验 + 泊松似然、走一步牛顿法”的式子算出来一样（用矩阵求逆直接算，对照程序里的快算法）；
  - 挪平均值（_recentre）前后，任意两队的预期进球一点不变，挪完本赛季各队平均是 0；
  - 实力不变的模拟联赛，估出的进攻力、防守力和真实值很接近；
  - 一队实力中途突然变好：q 越大，跟得越快；
  - 升班马：没有往年记录时用默认值，有了就用往年升班马赛季末的平均；
  - 每场赛后更新和只在每周拟合时更新：到同一个拟合时间，状态完全一样；
  - 头两个赛季不出预测；登记表能按名字造出来，比分矩阵加起来是 1。
"""
import numpy as np
import pandas as pd
import pytest

from src.models.dynamic import DynamicModel, _days as m_days
from src.models.registry import available, build_model, model_config
from tests.test_goals_models import history, simulate, targets


def league_with(n_teams=4, seed=0):
    """一个手工造的小联赛：n 队、随机的估计值和一个随机的正定协方差矩阵。"""
    rng = np.random.default_rng(seed)
    m = DynamicModel(min_matches=1)
    lg = m._league("E0", 2020)
    for k in range(n_teams):
        m._ensure(lg, 2020, f"T{k}")
    n = len(lg.m)
    lg.m = rng.normal(0, 0.2, n)
    A = rng.normal(0, 0.2, (n, n))
    lg.P = A @ A.T + 0.05 * np.eye(n)
    return m, lg


def test_update_is_one_newton_step():
    m, lg = league_with()
    m0, P0 = lg.m.copy(), lg.P.copy()
    m._observe(lg, "2020-09-01 15:00:00", "T0", "T2", 3, 1)
    ah, dh = 2, 3          # T0 的 a、d 位置
    aa, da = 6, 7          # T2
    H = np.zeros((2, len(m0)))
    H[0, [0, 1, ah, da]] = (1, 1, 1, -1)
    H[1, [0, aa, dh]] = (1, 1, -1)
    lam = np.exp(H @ m0)
    g = np.array([3, 1]) - lam
    info = np.linalg.inv(P0) + H.T @ np.diag(lam) @ H        # 后验的“信息矩阵”= 先验 + 似然的弯曲程度
    P1 = np.linalg.inv(info)
    m1 = m0 + P1 @ H.T @ g
    assert np.allclose(lg.P, P1, atol=1e-10)
    assert np.allclose(lg.m, m1, atol=1e-10)
    assert lg.rec[-1][1:3] == pytest.approx(tuple(lam))     # 记下的是赛前的预期进球


def test_recentre_keeps_expected_goals():
    m, lg = league_with(n_teams=5, seed=3)
    teams = lg.members[2020]
    before = {(h, a): m._lambdas(lg, h, a) for h in teams for a in teams if h != a}
    m._recentre(lg)
    after = {k: m._lambdas(lg, *k) for k in before}
    for k in before:
        assert after[k] == pytest.approx(before[k], rel=1e-12)
    a = [lg.m[lg.pos(t)[0]] for t in teams]
    d = [lg.m[lg.pos(t)[1]] for t in teams]
    assert np.mean(a) == pytest.approx(0, abs=1e-12) and np.mean(d) == pytest.approx(0, abs=1e-12)
    assert np.allclose(lg.P, lg.P.T)


def test_recovers_static_strengths():
    h, ta, td = simulate(n_teams=12, seasons=(2018, 2019, 2020, 2021), seed=7)
    m = DynamicModel(q=1e-6, min_matches=50)
    m.fit(h, "2025-01-01 00:00:00")                          # 模拟赛程最后一场在 2024 年
    lg = m.leagues["E0"]
    teams = sorted(ta)
    est_a = [lg.m[lg.pos(t)[0]] for t in teams]
    est_d = [lg.m[lg.pos(t)[1]] for t in teams]
    assert np.corrcoef(est_a, [ta[t] for t in teams])[0, 1] > 0.9
    assert np.corrcoef(est_d, [td[t] for t in teams])[0, 1] > 0.9
    assert lg.m[1] == pytest.approx(0.25, abs=0.08)         # 主场优势
    assert lg.ready and -0.3 <= lg.rho <= 0.3

def jump_history(jump=0.6, n_teams=8, n_rounds=40):
    """8 队每天一场轮着踢（不分赛季）；第 20 轮起 T0 的进攻力突然 + jump。
    比分直接用预期进球（小数）：没有随机波动，只看模型跟得快不快。"""
    rows, mid = [], 0
    day = pd.Timestamp("2020-08-15 14:00:00")
    pairs = [(i, j) for i in range(n_teams) for j in range(n_teams) if i != j]
    for r in range(n_rounds):
        for i, j in pairs[r % 7 * 8:(r % 7 + 1) * 8]:
            mid += 1
            a0 = jump if (i == 0 and r >= 20) else 0.0
            a1 = jump if (j == 0 and r >= 20) else 0.0
            rows.append((mid, "E0", 2020, f"T{i}", f"T{j}",
                         (day + pd.Timedelta(days=mid)).strftime("%Y-%m-%d %H:%M:%S"),
                         float(np.exp(0.15 + 0.25 + a0)), float(np.exp(0.15 + a1))))
    return history(rows)


def test_larger_q_tracks_jump_faster():
    h = jump_history()
    est = {}
    for q in (1e-5, 1e-3):
        m = DynamicModel(q=q)
        m._process(h[h["match_id"] <= 240])            # 跳了以后又踢了大约 10 场 T0 的比赛
        lg = m.leagues["E0"]
        a = [lg.m[lg.pos(f"T{k}")[0]] for k in range(8)]
        est[q] = a[0] - np.mean(a[1:])
    assert 0 < est[1e-5] < est[1e-3] < 0.6 + 0.05


def test_promoted_prior():
    m = DynamicModel(promoted_attack=-0.3, promoted_defence=-0.1, promoted_var=0.2)
    rows, mid = [], 0
    teams = {2020: ["A", "B", "C", "D"], 2021: ["A", "B", "C", "E"], 2022: ["A", "B", "E", "F"]}
    for s, ts in teams.items():
        for i, h in enumerate(ts):
            for a in ts:
                if h != a:
                    mid += 1
                    rows.append((mid, "E0", s, h, a, f"{s}-09-{mid % 28 + 1:02d} 15:00:00", 2 if h == "A" else 1, 1))
    hist = history(rows)
    m._process(hist[hist["season"] == 2020])
    lg = m.leagues["E0"]
    i, j = lg.pos("A")
    assert m._strength(lg, 2021, "E") == (-0.3, -0.1)             # 还没有往年升班马：用默认值
    assert m._strength(lg, 2021, "A") == (lg.m[i], lg.m[j])       # 留在联赛的队接着用
    first = hist[hist["season"] == 2021].sort_values("kickoff_earliest_utc")
    first_e = first[(first["home_team"] == "E") | (first["away_team"] == "E")].iloc[:1]
    m._process(first[first["kickoff_earliest_utc"] < first_e["kickoff_earliest_utc"].iloc[0]])
    m._advance(lg, m_days(first_e["kickoff_earliest_utc"].iloc[0]))
    m._ensure(lg, 2021, "E")
    i, j = lg.pos("E")
    assert lg.P[i, i] == lg.P[j, j] == 0.2                        # 加进状态时（开球时间）方差正好是 promoted_var
    assert not np.any(lg.P[i, :i]) and not np.any(lg.P[:i, i])    # 和其他队的牵连清零
    assert lg.promoted == {2021: ["E"]}
    m._process(hist[hist["season"] == 2021])
    e_end = (lg.m[i], lg.m[j])
    m._process(hist[hist["season"] == 2022])
    assert lg.end_strength[2021] == [e_end]                       # 进入 2022 时记下 E 在 2021 赛季末的实力
    k, l = lg.pos("F")
    assert lg.promoted[2022] == ["F"]
    assert lg.rec and lg.rec[-1][0].startswith("2022")
    m2 = DynamicModel(promoted_attack=-0.3, promoted_defence=-0.1, promoted_var=0.2)
    m2._process(hist[hist["season"] <= 2021])
    lg2 = m2.leagues["E0"]
    assert m2._strength(lg2, 2022, "F") == pytest.approx(e_end)  # F 的初始估计 = 往年升班马（E）赛季末的平均
    m2._process(hist[hist["season"] == 2022])
    f_end = m2._strength(lg2, 2022, "F")
    d_new = m2._strength(lg2, 2023, "D")                          # 2020 踢过、2021–22 不在：按升班马算
    assert d_new[0] == pytest.approx(np.mean([e_end[0], f_end[0]]))
    assert d_new[1] == pytest.approx(np.mean([e_end[1], f_end[1]]))


def test_weekly_same_state_as_per_match():
    h, _, _ = simulate(n_teams=8, seasons=(2018, 2019, 2020), seed=4)
    cut1, cut2 = "2020-10-01 00:00:00", "2020-11-01 00:00:00"
    kw = dict(q=3e-5, min_matches=20)
    fast, slow = DynamicModel(**kw), DynamicModel(per_match=False, **kw)
    for m in (fast, slow):
        m.fit(h[h["kickoff_earliest_utc"] < cut1], cut1)
        m.update(h[(h["kickoff_earliest_utc"] >= cut1) & (h["kickoff_earliest_utc"] < cut2)], cut2)
    t = targets([(9999, "E0", 2020, "T00", "T01")])
    assert not np.allclose(fast.predict(t, cut2)[0], slow.predict(t, cut2)[0])    # 每场更新的已经算进去了
    for m in (fast, slow):
        m.fit(h[h["kickoff_earliest_utc"] < cut2], cut2)
    lf, ls = fast.leagues["E0"], slow.leagues["E0"]
    assert np.allclose(lf.m, ls.m) and np.allclose(lf.P, ls.P)
    assert fast.predict(t, cut2)[0] == pytest.approx(slow.predict(t, cut2)[0])


def test_burn_in_and_matrices():
    h, _, _ = simulate(n_teams=6, seasons=(2018, 2019, 2020), seed=5)
    m = DynamicModel(min_matches=20)
    t = targets([(1, "E0", 2019, "T00", "T01"), (2, "E0", 2020, "T02", "T03"), (3, "XX", 2020, "T00", "T01")])
    cut = "2021-03-01 00:00:00"                              # 模拟的赛程：2019 赛季踢到 2020 年 12 月
    m.fit(h[h["season"] <= 2019], cut)
    out = m.predict(t, cut)
    assert out[0] is None and out[2] is None                 # 2019 是第 2 个赛季：不出；没见过的联赛：不出
    assert out[1] is not None                                # 2020 前面已经有 2 个赛季：新赛季第一场就出
    assert out[1].shape == (11, 11) and out[1].sum() == pytest.approx(1)
    lg = m.leagues["E0"]
    before = (lg.m.copy(), lg.P.copy(), lg.season, {k: list(v) for k, v in lg.members.items()})
    m.predict(targets([(4, "E0", 2020, "T00", "NEW")]), cut)  # 新赛季、没见过的队：当场按升班马算
    assert np.array_equal(lg.m, before[0]) and np.array_equal(lg.P, before[1])   # 出预测不改状态
    assert lg.season == before[2] == 2019 and lg.members == before[3]
    tab = m.ratings_table()
    assert len(tab) == 6 and m.params()["E0"]["赛季"] == 2019


def test_predict_new_season_matches_process():
    """出预测时当场算的升班马估计，和真正处理到那场时加进状态的一样（方差不影响预期进球）。"""
    h, _, _ = simulate(n_teams=6, seasons=(2018, 2019), seed=6)
    m = DynamicModel(min_matches=20, burn_in_seasons=1)
    cut = "2021-03-01 00:00:00"
    m.fit(h, cut)
    t = targets([(1, "E0", 2020, "NEW", "T01"), (2, "E0", 2020, "T02", "T03")])
    pred = m.predict(t, cut)
    lg = m.leagues["E0"]
    m._new_season(lg, 2020)
    for team in ("NEW", "T01", "T02", "T03"):
        m._ensure(lg, 2020, team)
    expect = [m._lambdas(lg, "NEW", "T01"), m._lambdas(lg, "T02", "T03")]
    from src.models.goals import score_matrices
    for p, (lh, la) in zip(pred, expect):
        assert p == pytest.approx(score_matrices(np.array([lh]), np.array([la]), lg.rho, 0.0, 10)[0])


def test_registry_builds_variants():
    names = [n for n in available() if n.startswith("dyn_")]
    assert len(names) >= 6
    for n in names:
        m = build_model(n)
        cfg = model_config(n)
        assert isinstance(m, DynamicModel) and m.name == n
        assert m.q == cfg["q"] and m.per_match == cfg["per_match"]
    assert build_model("dyn_q1e4_b01").season_bump == 0.01
