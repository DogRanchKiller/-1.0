"""第 4 节步骤 4：goals.py 的 target / two_stage / require 开关的自动检查（不需要数据库，用模拟比赛）。

检查的东西：
  - 默认（target = goals、一次拟合）和明写这些开关的结果一模一样（步骤 1–3 的 dc 没被改动）；
  - 两步拟合在“进球就是进球”时和一次拟合几乎一样（rho、mu、home 差很小）；
  - xG 列填成和进球一样时，target = xg / mix 和 goals 两步版一模一样（证明只是换了数据，别的没变）；
  - 射门系数 a、b：按已知系数造的数据能找回来；算出负数的一边按 0 算；
  - 小数“进球”的斜率公式和数值差分一样；
  - require：缺数据的比赛不进拟合；取来的数据里没有这列就报错；
  - 不合法的组合报错；登记表里实验一、实验二的组用途和数据对。
"""
import numpy as np
import pandas as pd
import pytest
from scipy.optimize import approx_fprime

from src.common.config import load_settings
from src.models.goals import FitData, GoalsModel, _nll, fit_rho, fit_targets, shot_coefficients, time_weights
from src.models.registry import build_model, model_config
from tests.test_goals_models import simulate, targets

CUT = "2022-07-01 00:00:00"


def with_stats(h: pd.DataFrame, seed=0) -> pd.DataFrame:
    """给模拟比赛加上射门、射正和 Understat xG 列（xG 先填成和进球一样，射门随便造）。"""
    rng = np.random.default_rng(seed)
    h = h.copy()
    for side in ("home", "away"):
        sot = h[f"{side}_goals"] + rng.integers(0, 5, len(h))
        h[f"{side}_shots_on_target"] = sot
        h[f"{side}_shots"] = sot + rng.integers(0, 9, len(h))
        h[f"us_{side}_xg"] = h[f"{side}_goals"].astype(float)
    return h


def fit(h, **kw):
    m = GoalsModel(dc=True, min_matches=50, **kw)
    m.fit(h, CUT)
    return m.fits["E0"]


def test_default_is_unchanged():
    h = with_stats(simulate(n_teams=10, seasons=(2020, 2021), seed=3)[0])
    a = fit(h)
    b = fit(h, target="goals", two_stage=False, require=())
    for k in ("mu", "home", "rho"):
        assert a[k] == b[k]
    assert a["a"] == b["a"] and a["d"] == b["d"]
    assert GoalsModel(dc=True).data == ("比分",) and GoalsModel(dc=True).usage == "A版输入"


def test_two_stage_close_to_joint_fit():
    h = simulate(n_teams=12, seasons=(2019, 2020, 2021), seed=5)[0]
    joint, two = fit(h), fit(h, two_stage=True)
    assert two["rho"] == pytest.approx(joint["rho"], abs=0.02)
    assert two["mu"] == pytest.approx(joint["mu"], abs=0.02)
    assert two["home"] == pytest.approx(joint["home"], abs=0.02)


def test_fit_rho_finds_joint_optimum_when_strengths_are_optimal():
    """一次拟合的最优点上，固定实力再单独找 rho，应该回到同一个 rho。"""
    h = simulate(n_teams=10, seasons=(2020, 2021), seed=8)[0]
    m = GoalsModel(dc=True, min_matches=50)
    m.fit(h, CUT)
    f = m.fits["E0"]
    teams = sorted(f["a"])
    idx = {t: i for i, t in enumerate(teams)}
    fd = FitData(hi=h["home_team"].map(idx).to_numpy(int), ai=h["away_team"].map(idx).to_numpy(int),
                 x=h["home_goals"].to_numpy(int), y=h["away_goals"].to_numpy(int),
                 w=time_weights(h["kickoff_earliest_utc"], CUT, 365),   # 和一次拟合同样的时间权重，才是同一个问题
                 n_teams=len(teams), center_a=np.zeros(len(teams)), center_d=np.zeros(len(teams)), ridge=3.0,
                 dc=True, nb=False)
    theta = np.concatenate([[f["mu"], f["home"]], [f["a"][t] for t in teams], [f["d"][t] for t in teams]])
    assert fit_rho(fd, theta) == pytest.approx(f["rho"], abs=1e-3)


@pytest.mark.parametrize("target", ["xg", "mix"])
def test_xg_equal_to_goals_gives_same_as_two_stage_goals(target):
    h = with_stats(simulate(n_teams=10, seasons=(2020, 2021), seed=6)[0])
    base = fit(h, two_stage=True, require=["understat"])
    new = fit(h, target=target)
    for k in ("mu", "home", "rho"):
        assert new[k] == pytest.approx(base[k], abs=1e-6)
    assert all(new["a"][t] == pytest.approx(base["a"][t], abs=1e-6) for t in base["a"])


def test_shot_coefficients_recovered():
    rng = np.random.default_rng(1)
    n = 4000
    sot_h, sot_a = rng.integers(0, 10, n), rng.integers(0, 10, n)
    off_h, off_a = rng.integers(0, 12, n), rng.integers(0, 12, n)
    part = pd.DataFrame({"home_shots_on_target": sot_h, "away_shots_on_target": sot_a,
                         "home_shots": sot_h + off_h, "away_shots": sot_a + off_a,
                         "home_goals": rng.poisson(0.3 * sot_h + 0.03 * off_h),
                         "away_goals": rng.poisson(0.3 * sot_a + 0.03 * off_a)})
    a, b = shot_coefficients(part, np.ones(n))
    assert a == pytest.approx(0.3, abs=0.02)
    assert b == pytest.approx(0.03, abs=0.01)
    x, y = fit_targets(part, "shots", (a, b))
    assert x[0] == pytest.approx(a * sot_h[0] + b * off_h[0])


def test_shot_target_never_negative():
    part = pd.DataFrame({"home_shots_on_target": [0, 3], "away_shots_on_target": [1, 0],
                         "home_shots": [10, 3], "away_shots": [1, 2], "home_goals": [0, 1], "away_goals": [1, 0]})
    x, y = fit_targets(part, "shots", (0.3, -0.05))
    assert (x >= 0).all() and (y >= 0).all()
    assert x[0] == 0.0 and x[1] == pytest.approx(0.9)


def test_gradient_with_fractional_goals():
    rng = np.random.default_rng(2)
    n, m = 6, 120
    hi = rng.integers(0, n, m)
    ai = (hi + rng.integers(1, n, m)) % n
    f = FitData(hi=hi, ai=ai, x=rng.uniform(0, 3, m), y=rng.uniform(0, 2.5, m), w=rng.uniform(0.3, 1, m),
                n_teams=n, center_a=np.zeros(n), center_d=np.zeros(n), ridge=2.0, dc=False, nb=False)
    th = np.concatenate([[0.1, 0.25], rng.normal(0, 0.2, n), rng.normal(0, 0.2, n)])
    _, g = _nll(th, f)
    num = approx_fprime(th, lambda t: _nll(t, f)[0], 1e-7)
    assert np.allclose(g, num, rtol=1e-4, atol=1e-4)


def test_require_drops_matches_without_data():
    h = with_stats(simulate(n_teams=10, seasons=(2020, 2021), seed=7)[0])
    h.loc[h.index[:40], ["us_home_xg", "us_away_xg"]] = np.nan
    m = GoalsModel(dc=True, min_matches=50, target="xg")
    m.fit(h, CUT)
    assert m.fits["E0"]["n"] == len(h) - 40
    with pytest.raises(ValueError, match="Understat"):
        GoalsModel(dc=True, min_matches=50, target="xg").fit(h.drop(columns=["us_home_xg"]), CUT)
    # 预测时照样每场都有（不需要本场的 xG）
    p = m.predict(targets([(9001, "E0", 2021, "T00", "T01")]), CUT)
    assert p[0] is not None and p[0].sum() == pytest.approx(1)


def test_bad_combinations():
    with pytest.raises(ValueError):
        GoalsModel(target="corners")
    with pytest.raises(ValueError):
        GoalsModel(nb=True, target="shots")
    with pytest.raises(ValueError):
        GoalsModel(nb=True, two_stage=True)
    with pytest.raises(ValueError):
        GoalsModel(require=["lineups"])


def test_registry_step4_variants():
    s = load_settings()
    assert "target" not in model_config("dc", s)          # 原来的 dc 设置记录不变
    for name in ("e1_dc", "e1_goals", "e1_goals_2s", "e1_shots", "e1_xg", "e1_mix"):
        m = build_model(name, s)
        assert m.usage == "研究实验" and m.dc, name
    assert build_model("e1_dc", s).data == ("比分",)
    assert set(build_model("e1_goals", s).data) == {"比分", "射门、xG", "Understat xG（研究）"}
    assert not build_model("e1_goals", s).two_stage and build_model("e1_goals_2s", s).two_stage
    assert build_model("e1_xg", s).target == "xg" and build_model("e1_mix", s).target == "mix"
    sh = build_model("dc_shots", s)
    assert sh.usage == "A版输入" and sh.target == "shots" and sh.two_stage and sh.data == ("比分", "射门、xG")
    assert build_model("dc_2s", s).two_stage and build_model("dc_2s", s).target == "goals"


SCRIPT = """
import pandas as pd
from src.models.goals import GoalsModel
from tests.test_goals_models import simulate
# 每季换掉 4 个队（4 个升班马），连续拟合几次：升班马先验要把几个队的 a、d 加起来求平均
parts = []
for i, s in enumerate((2018, 2019, 2020, 2021)):
    h = simulate(n_teams=12, seasons=(s,), seed=20 + i)[0]
    ren = {f"T{k:02d}": f"N{s}_{k}" for k in range(4)} if i else {}
    h = h.replace({"home_team": ren, "away_team": ren})
    h["match_id"] += 10_000 * i
    parts.append(h)
hist = pd.concat(parts, ignore_index=True)
m = GoalsModel(dc=True, min_matches=50, window_days=800)
for cut in ("2020-06-01 00:00:00", "2021-06-01 00:00:00", "2022-06-01 00:00:00"):
    m.fit(hist[hist["kickoff_earliest_utc"] < cut], cut)
f = m.fits["E0"]
print(repr((f["mu"], f["home"], f["rho"], f["prior"], sorted(f["a"].items()))))
"""


def test_same_result_whatever_hash_seed():
    """集合的顺序跟着 PYTHONHASHSEED 变；升班马先验按队名排序后求平均，换种子结果也一模一样（一位小数都不差）。"""
    import os
    import subprocess
    import sys
    outs = []
    for seed in ("0", "1", "12345"):
        env = {**os.environ, "PYTHONHASHSEED": seed, "PYTHONPATH": os.getcwd()}
        r = subprocess.run([sys.executable, "-c", SCRIPT], env=env, capture_output=True, text=True, check=True)
        outs.append(r.stdout)
    assert outs[0] == outs[1] == outs[2] and "N2021_0" in outs[0]
