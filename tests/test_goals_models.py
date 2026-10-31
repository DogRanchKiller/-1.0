"""第 2 阶段步骤 1–3 的模型（src/models/goals.py：泊松 / Dixon-Coles / DC + 负二项）的自动检查。

不需要数据库：用手造的小数据或按已知参数模拟出来的比赛。
检查的东西：
  - 比分矩阵加起来是 1；rho = 0、alpha = 0 时和 markets.poisson_score_matrix 一模一样；DC 改的 4 个格子和手算一样；
  - 负二项 alpha 很小时和泊松几乎一样；
  - 拟合用的“斜率”（解析公式）和数值差分一样（四种组合都查）；拟合用的对数似然和“直接查矩阵格子取 log”一样；
  - 时间衰减：过了一个半衰期权重正好是 0.5；
  - 模拟数据能把参数大致找回来；
  - 升班马用先验；比赛太少的联赛不出预测；登记表能造出这几个模型。
"""
import numpy as np
import pandas as pd
import pytest
from scipy.optimize import approx_fprime

from src.common.config import load_settings
from src.common.markets import poisson_score_matrix
from src.models.goals import (FitData, GoalsModel, _nll, fit_league, goal_probs, rho_range, score_matrices,
                              time_weights)
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
    return pd.DataFrame(rows, columns=["match_id", "competition", "season", "home_team", "away_team"])


def simulate(n_teams=20, seasons=(2020, 2021), seed=0, mu=0.15, home=0.25, sd=0.3, comp="E0"):
    """按已知参数模拟双循环联赛：每个赛季每对球队主客各一场，比分是泊松。返回 (历史表, 真实 a, 真实 d)。"""
    rng = np.random.default_rng(seed)
    teams = [f"T{i:02d}" for i in range(n_teams)]
    a = rng.normal(0, sd, n_teams)
    d = rng.normal(0, sd, n_teams)
    a, d = a - a.mean(), d - d.mean()
    rows, mid = [], 0
    for s in seasons:
        day = pd.Timestamp(f"{s}-08-15 14:00:00")
        for i in range(n_teams):
            for j in range(n_teams):
                if i == j:
                    continue
                mid += 1
                lh, la = np.exp(mu + home + a[i] - d[j]), np.exp(mu + a[j] - d[i])
                ko = day + pd.Timedelta(days=int(mid * 280 / (n_teams * (n_teams - 1))))
                rows.append((mid, comp, s, teams[i], teams[j], ko.strftime("%Y-%m-%d %H:%M:%S"),
                             int(rng.poisson(lh)), int(rng.poisson(la))))
    return history(rows), dict(zip(teams, a)), dict(zip(teams, d))


def small_fit_data(dc, nb, seed=1):
    rng = np.random.default_rng(seed)
    n, m = 6, 120
    hi = rng.integers(0, n, m)
    ai = (hi + rng.integers(1, n, m)) % n
    return FitData(hi=hi, ai=ai, x=rng.poisson(1.5, m), y=rng.poisson(1.1, m), w=rng.uniform(0.3, 1, m),
                   n_teams=n, center_a=np.zeros(n), center_d=np.full(n, 0.1), ridge=2.0, dc=dc, nb=nb)


def theta_for(f: FitData, seed=2, alpha=0.05):
    rng = np.random.default_rng(seed)
    parts = [[0.1, 0.25], rng.normal(0, 0.2, f.n_teams), rng.normal(0, 0.2, f.n_teams)]
    if f.dc:
        parts.append([-0.07])
    if f.nb:
        parts.append([alpha])
    return np.concatenate([np.asarray(p, float) for p in parts])


# ======================= 比分矩阵 =======================
def test_matrix_sums_to_one_and_matches_markets_poisson():
    m = score_matrices([1.6, 0.4], [1.1, 2.9], n=10)
    assert m.shape == (2, 11, 11)
    assert np.allclose(m.sum(axis=(1, 2)), 1)
    assert np.allclose(m[0], poisson_score_matrix(1.6, 1.1, 10))
    assert np.allclose(m[1], poisson_score_matrix(0.4, 2.9, 10))


def test_dc_changes_four_cells_like_hand_calculation():
    """λ主 = 1.5、λ客 = 1.2、ρ = −0.1：0:0 × (1 − 1.5·1.2·(−0.1)) = × 1.18；0:1 × (1 + 1.5·(−0.1)) = × 0.85；
    1:0 × (1 + 1.2·(−0.1)) = × 0.88；1:1 × 1.1。泊松下 4 个格子改完总和还是 1（不用再除）。"""
    p = poisson_score_matrix(1.5, 1.2, 10)
    m = score_matrices(1.5, 1.2, rho=-0.1, n=10)[0]
    assert m.sum() == pytest.approx(1)
    for (i, j), k in {(0, 0): 1.18, (0, 1): 0.85, (1, 0): 0.88, (1, 1): 1.1, (2, 2): 1.0, (3, 0): 1.0}.items():
        assert m[i, j] == pytest.approx(p[i, j] * k, rel=1e-9)


def test_negative_binomial_tiny_alpha_is_poisson_and_large_alpha_fattens_tail():
    pois = goal_probs([1.4], 0.0, 10)
    assert np.allclose(goal_probs([1.4], 1e-7, 10), pois, atol=1e-6)
    nb = goal_probs([1.4], 0.3, 10)
    assert nb.sum() == pytest.approx(1)
    # 平均数一样（1.4），但负二项更“分散”：0 球和 4 球以上都更多
    assert nb[0, 0] > pois[0, 0] and nb[0, 4:].sum() > pois[0, 4:].sum()
    mean = (np.arange(10) * nb[0, :10]).sum()
    assert mean == pytest.approx(1.4, abs=0.01)


def test_nb_dc_matrix_sums_to_one():
    m = score_matrices([1.3, 2.2], [0.9, 1.5], rho=-0.08, alpha=0.1, n=10)
    assert np.allclose(m.sum(axis=(1, 2)), 1)
    assert (m >= 0).all()


# ======================= 拟合用的对数似然 =======================
@pytest.mark.parametrize("dc,nb", [(False, False), (True, False), (False, True), (True, True)])
def test_gradient_matches_numerical(dc, nb):
    f = small_fit_data(dc, nb)
    for alpha in ([0.05, 1e-5] if nb else [0.05]):
        th = theta_for(f, alpha=alpha)
        _, g = _nll(th, f)
        num = approx_fprime(th, lambda t: _nll(t, f)[0], 1e-7)
        assert np.allclose(g, num, rtol=1e-4, atol=1e-4), (dc, nb, alpha, np.abs(g - num).max())


@pytest.mark.parametrize("dc,nb", [(False, False), (True, False), (False, True), (True, True)])
def test_likelihood_equals_matrix_cell(dc, nb):
    """没有收缩、权重都是 1 时：拟合用的 −对数似然 = Σ −log(比分矩阵里真实比分那一格)（矩阵开到 40 球，尾巴可以忽略）。"""
    f = small_fit_data(dc, nb)
    f.ridge, f.w = 0.0, np.ones(len(f.x))
    th = theta_for(f, alpha=0.08)
    n = f.n_teams
    mu, home, a, d = th[0], th[1], th[2:2 + n], th[2 + n:2 + 2 * n]
    rho = th[2 + 2 * n] if dc else 0.0
    alpha = th[-1] if nb else 0.0
    lh = np.exp(mu + home + a[f.hi] - d[f.ai])
    la = np.exp(mu + a[f.ai] - d[f.hi])
    m = score_matrices(lh, la, rho, alpha, n=40)
    brute = -np.log(m[np.arange(len(f.x)), f.x, f.y]).sum()
    assert _nll(th, f)[0] == pytest.approx(brute, rel=1e-9)


def test_rho_range_hand_calculation():
    """一场 λ主 = e^(0.5+0.3+0.4) ≈ 3.32、λ客 = e^0.5 ≈ 1.65：
    下限 = (0.05 − 1) ÷ 3.32 ≈ −0.286（强队主场 0:1 先撑不住）；上限 = min(0.95 ÷ (3.32 × 1.65), 0.95) ≈ 0.173。"""
    f = FitData(hi=np.array([0]), ai=np.array([1]), x=np.array([0]), y=np.array([0]), w=np.ones(1), n_teams=2,
                center_a=np.zeros(2), center_d=np.zeros(2), ridge=0.0, dc=True, nb=False)
    th = np.array([0.5, 0.3, 0.4, 0.0, 0.0, 0.0, -0.1])
    lo, hi = rho_range(th, f)
    lh, la = np.exp(1.2), np.exp(0.5)
    assert lo == pytest.approx(-0.95 / lh) and hi == pytest.approx(0.95 / (lh * la))


def test_clipped_tau_has_zero_slope():
    """0:1 的系数 1 + λ主ρ 被压到下限的那场：数值和斜率要一致（斜率用数值差分核对）。"""
    f = FitData(hi=np.array([0, 1]), ai=np.array([1, 0]), x=np.array([0, 1]), y=np.array([1, 1]), w=np.ones(2),
                n_teams=2, center_a=np.zeros(2), center_d=np.zeros(2), ridge=0.0, dc=True, nb=False)
    th = np.array([0.5, 0.3, 0.9, 0.0, 0.0, 0.0, -0.3])           # λ主 ≈ 5.5，1 + 5.5 × (−0.3) < 0
    _, g = _nll(th, f)
    num = approx_fprime(th, lambda t: _nll(t, f)[0], 1e-7)
    assert np.allclose(g, num, rtol=1e-4, atol=1e-4)


def test_fit_keeps_rho_in_valid_range():
    """造一个联赛：T0 主场总是 4–6 : 0 大胜（从来没有 0:1），其他比赛全是 0:0、1:1、1:0、0:1 的小比分。
    只看似然 rho 会一直往 −0.3 跑（强队主场 0:1 的格子变成负数也没人管）；拟合要停在合法范围里。"""
    rng = np.random.default_rng(0)
    n, hi, ai, x, y = 8, [], [], [], []
    for _ in range(6):
        for i in range(n):
            for j in range(n):
                if i == j:
                    continue
                hi.append(i)
                ai.append(j)
                if i == 0:
                    x.append(int(rng.choice([4, 5, 6]))), y.append(0)
                elif j == 0:
                    x.append(0), y.append(int(rng.choice([2, 3])))
                else:
                    k = rng.choice(4, p=[0.4, 0.35, 0.15, 0.1])
                    x.append([0, 1, 1, 0][k]), y.append([0, 1, 0, 1][k])
    f = FitData(hi=np.array(hi), ai=np.array(ai), x=np.array(x), y=np.array(y), w=np.ones(len(x)), n_teams=n,
                center_a=np.zeros(n), center_d=np.zeros(n), ridge=0.5, dc=True, nb=False)
    th = fit_league(f, np.concatenate([[0, 0.2], np.zeros(2 * n), [-0.05]]))
    lo, hi_ = rho_range(th, f)
    assert lo - 1e-9 <= th[-1] <= hi_ + 1e-9
    assert th[-1] > -0.25                              # 没有跑到 −0.3


def test_time_weight_half_life():
    w = time_weights(pd.Series(["2024-01-01 00:00:00", "2023-01-01 00:00:00"]), "2024-01-01 00:00:00", 365)
    assert w == pytest.approx([1.0, 0.5])
    assert (time_weights(pd.Series(["2020-01-01 00:00:00"]), "2024-01-01 00:00:00", None) == 1).all()


# ======================= 模拟数据：能把参数找回来 =======================
def test_fit_recovers_simulated_parameters():
    # 1520 场：主场优势、mu 的抽样误差大约 ±0.035（10 个随机种子试过，平均 0.263，没有偏差），允许差 0.08
    h, a_true, d_true = simulate(seasons=(2018, 2019, 2020, 2021), seed=0)
    m = GoalsModel(dc=True, ridge=1.0, half_life_days=None, window_days=2000, min_matches=50)
    m.fit(h, "2022-07-01 00:00:00")
    f = m.fits["E0"]
    assert f["home"] == pytest.approx(0.25, abs=0.08)
    assert f["mu"] == pytest.approx(0.15, abs=0.08)
    assert abs(f["rho"]) < 0.1                      # 模拟数据是纯泊松，真实 rho = 0
    teams = sorted(a_true)
    assert np.corrcoef([a_true[t] for t in teams], [f["a"][t] for t in teams])[0, 1] > 0.85
    assert np.corrcoef([d_true[t] for t in teams], [f["d"][t] for t in teams])[0, 1] > 0.85


def test_nb_on_poisson_data_gives_small_alpha():
    h, _, _ = simulate(seasons=(2019, 2020, 2021), seed=4)
    m = GoalsModel(nb=True, half_life_days=None, window_days=2000, min_matches=50)
    m.fit(h, "2022-07-01 00:00:00")
    assert m.fits["E0"]["alpha"] < 0.03


# ======================= 预测：矩阵、升班马、没拟合的联赛 =======================
def test_predict_returns_matrices_promoted_prior_and_none():
    h, _, _ = simulate(n_teams=10, seasons=(2020, 2021), seed=5)
    m = GoalsModel(min_matches=50, promoted_attack=-0.3, promoted_defence=-0.25)
    m.fit(h, "2022-07-01 00:00:00")
    t = targets([(1, "E0", 2022, "T00", "T01"), (2, "E0", 2022, "NEW", "T01"), (3, "SP1", 2022, "X", "Y")])
    p = m.predict(t, "2022-07-01 00:00:00")
    assert p[0].shape == (11, 11) and p[0].sum() == pytest.approx(1)
    assert p[2] is None                              # SP1 没有拟合结果
    # NEW 没踢过 → 用升班马先验（第一次拟合没有往年升班马可以平均，就用设置里的默认值）
    f = m.fits["E0"]
    lh, la, ok = m.expected_goals(t.iloc[[1]])
    assert lh[0] == pytest.approx(np.exp(f["mu"] + f["home"] - 0.3 - f["d"]["T01"]))
    assert la[0] == pytest.approx(np.exp(f["mu"] + f["a"]["T01"] + 0.25))


def test_promoted_prior_from_previous_promoted_teams():
    """2021 赛季 T09 不在、P1 在 → P1 是 2021 的升班马；下一次拟合时 2022 的升班马先验 = P1 上一次拟合的 (a, d)。"""
    h, _, _ = simulate(n_teams=10, seasons=(2020,), seed=6)
    h21, _, _ = simulate(n_teams=10, seasons=(2021,), seed=7)
    h21 = h21.replace({"home_team": {"T09": "P1"}, "away_team": {"T09": "P1"}})
    h21["match_id"] += 10_000
    hist = pd.concat([h, h21], ignore_index=True)
    m = GoalsModel(min_matches=50)
    m.fit(hist, "2022-06-01 00:00:00")
    assert m._promoted("E0", 2021) == {"P1"}
    a_p1, d_p1 = m.fits["E0"]["a"]["P1"], m.fits["E0"]["d"]["P1"]
    assert m._new_prior("E0", 2022) == pytest.approx((a_p1, d_p1))


def test_too_few_matches_no_fit():
    h, _, _ = simulate(n_teams=6, seasons=(2021,), seed=8)       # 30 场
    m = GoalsModel(min_matches=200)
    m.fit(h, "2022-07-01 00:00:00")
    assert m.fits == {}
    assert m.predict(targets([(1, "E0", 2022, "T00", "T01")]), "2022-07-01 00:00:00") == [None]


def test_warm_start_gives_same_answer():
    """接着上一次拟合的参数往下算（warm start），结果和从头算一样（是同一个最优点）。"""
    h, _, _ = simulate(n_teams=10, seasons=(2020, 2021), seed=9)
    m1 = GoalsModel(dc=True, min_matches=50)
    m1.fit(h, "2022-07-01 00:00:00")
    m2 = GoalsModel(dc=True, min_matches=50)
    m2.fit(h[h["season"] == 2020], "2021-07-01 00:00:00")
    m2.fit(h, "2022-07-01 00:00:00")
    for k in ("mu", "home", "rho"):
        assert m1.fits["E0"][k] == pytest.approx(m2.fits["E0"][k], abs=1e-4)


# ======================= 登记表 =======================
def test_registry_builds_goals_variants():
    s = load_settings()
    names = available(s)
    for n in ("poisson", "dc", "dcnb"):
        assert n in names
    m = build_model("dcnb", s)
    assert isinstance(m, GoalsModel) and m.dc and m.nb and m.name == "dcnb"
    assert m.max_goals == s["calc"]["max_goals"]
    assert build_model("dc_nodecay", s).half_life_days is None
    assert not build_model("poisson", s).dc
