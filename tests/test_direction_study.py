"""选方向研究（src/evaluate/direction_study.py）的检查。只用手造的假比赛，不读真回测、不连数据库。"""
import json

import numpy as np
import pandas as pd
import pytest

from src.common.config import load_settings
from src.common.markets import asian_handicap, dc_matrices, over_under
from src.common.settlement import settle_asian_handicap, settle_over_under
from src.evaluate import direction_study as ds
from src.features.errors import SeasonLockedError


@pytest.fixture(scope="module")
def settings():
    s = load_settings()
    c = {**s["direction_study"], "grids": {"a_k": [0.8, 1.2, 0.2], "a_b": [-0.04, 0.04, 0.04], "a_d": [-0.1, 0.1, 0.1],
                                           "b_a": [0.0, 1.0, 0.5], "c_tau": [0.0, 0.04, 0.02], "e_c": [0.0, 2.0, 1.0]}}
    return {**s, "direction_study": c, "compare": {**s["compare"], "n_boot": 200}}


def _probs(lh, la, rho):
    m = ds.flat(dc_matrices(lh, la, rho, n=ds.N))
    return np.column_stack([m @ ds.P3_IND, m @ ds.OVER25])


def _table(per_season=40, seed=1, seasons=range(2014, 2024)):
    """假比赛：每个赛季 per_season 场，每周一场。赔率、gb_l7、gb_inj 各有自己的预期进球；比分按赔率的矩阵抽。"""
    rng = np.random.default_rng(seed)
    ss = np.repeat(list(seasons), per_season)
    n = len(ss)
    lh, la = rng.uniform(0.8, 2.2, n), rng.uniform(0.5, 1.6, n)
    rho = np.full(n, -0.05)
    srcs = {"market": (lh, la), "gb_l7": (lh * rng.uniform(0.8, 1.25, n), la * rng.uniform(0.8, 1.25, n))}
    srcs["gb_l7_r"] = srcs["gb_l7"]
    srcs["gb_inj"] = (srcs["gb_l7"][0] * rng.uniform(0.95, 1.05, n), srcs["gb_l7"][1] * rng.uniform(0.95, 1.05, n))
    mk = dc_matrices(lh, la, rho, n=ds.N).reshape(n, -1)
    cell = np.array([rng.choice(mk.shape[1], p=mk[k] / mk[k].sum()) for k in range(n)])
    hg, ag = cell // (ds.N + 1), cell % (ds.N + 1)
    dates = [pd.Timestamp(f"{s}-08-12") + pd.Timedelta(weeks=int(k % per_season)) for k, s in enumerate(ss)]
    t = pd.DataFrame({"match_id": np.arange(n) + 1, "competition": rng.choice(["E0", "SP1", "D1", "I1", "F1"], n),
                      "season": ss, "match_date": [d.strftime("%Y-%m-%d") for d in dates],
                      "home_goals": hg, "away_goals": ag,
                      "ah_line": rng.choice([-1.0, -0.75, -0.5, -0.25, 0.0, 0.25], n),
                      "ah_home_odds": rng.uniform(1.8, 2.1, n), "ah_away_odds": rng.uniform(1.8, 2.1, n)})
    for name, (h, a) in srcs.items():
        p = _probs(h, a, rho)
        for i, col in enumerate(ds.PCOLS):
            t[f"{name}:{col}"] = p[:, i]
    return t


@pytest.fixture(scope="module")
def data(settings):
    return ds.prepare(_table(), settings)


# ---------- 让球赢半、走盘的算法 ----------
def test_ah_ou_values_half_and_push():
    v = dict(zip(ds.AH_X, ds.ah_values(-0.25)))       # 主队让平半
    assert v[1] == 1 and v[0] == -0.5 and v[-1] == -1  # 赢 1 球全赢；平局输半；输球全输
    v = dict(zip(ds.AH_X, ds.ah_values(0.0)))           # 平手盘：平局走盘
    assert v[0] == 0 and v[1] == 1
    v = dict(zip(ds.AH_X, ds.ah_values(-0.75)))         # 让半一：赢 1 球赢半
    assert v[1] == 0.5 and v[2] == 1
    v = dict(zip(ds.OU_X, ds.ou_values(2.25)))          # 大 2.25：2 球输半、3 球全赢
    assert v[2] == -0.5 and v[3] == 1
    assert dict(zip(ds.OU_X, ds.ou_values(3.0)))[3] == 0  # 整数线进 3 球走盘


def test_hit_rate_half_and_push():
    assert ds.hit_rate([1, 0.5, 0, -0.5, -1]) == pytest.approx(1.5 / 3)   # 赢半算半个中、输半算半个没中、走盘不算
    assert ds.hit_rate([0.5, 0, 0]) == pytest.approx(1.0)
    assert np.isnan(ds.hit_rate([0, 0]))


def test_value_of_both_sides_match_settlement(data):
    """买主队 / 买客队、买大 / 买小的结算分值和 settlement 一样（同一条线两边正好相反）。"""
    t, st = data.t, data.setup
    for k in range(60):
        h, a = int(t["home_goals"][k]), int(t["away_goals"][k])
        for side, name in ((1, "home"), (-1, "away")):
            got = ds.value_of("ah", np.array([side]), {"v_ah_home": st["v_ah_home"][k:k + 1]})[0]
            assert got == ds.VALUE[settle_asian_handicap(h, a, float(st["ah_line"][k]), name)["outcome"]]
        for side, name in ((1, "over"), (-1, "under")):
            got = ds.value_of("ou", np.array([side]), {"v_ou_over": st["v_ou_over"][k:k + 1]})[0]
            assert got == ds.VALUE[settle_over_under(h, a, float(st["ou_line"][k]), name)["outcome"]]


def test_shares_match_markets(data):
    """向量算的让球、大小球胜算和 markets.asian_handicap / over_under 一样（含四分之一线、平手盘）。"""
    lam = data.lam["gb_l7"]
    for k in range(0, 80, 7):
        m = dc_matrices(lam[0][k:k + 1], lam[1][k:k + 1], lam[2][k:k + 1], n=ds.N)[0]
        assert data.probs["gb_l7"]["ah"][k] == pytest.approx(
            asian_handicap(m, float(data.setup["ah_line"][k]), "home")["win_share"], abs=1e-12)
        assert data.probs["gb_l7"]["ou"][k] == pytest.approx(
            over_under(m, float(data.setup["ou_line"][k]), "over")["win_share"], abs=1e-12)


def test_exact_half_goes_home():
    """正好五五开（计算误差让它是 0.4999999999）时按规则选主队 / 大。"""
    assert list(ds.side_of("ah", [0.5 - 1e-12, 0.5 + 1e-12, 0.49])) == [1, 1, -1]
    assert list(ds.side_of("1x2", np.array([[0.4 - 1e-12, 0.2, 0.4]]))) == [0]


# ---------- 期末考试赛季拦着 ----------
def test_exam_season_blocked(settings, tmp_path):
    """设置里检验季写到 2024 就停下；文件里有 2024 的行，读的时候筛掉、不往下传；直接给 2024 的比赛也停下。"""
    bad = {**settings, "direction_study": {**settings["direction_study"], "test_last": 2024}}
    with pytest.raises(ValueError, match="期末考试"):
        ds.seasons_of(bad)
    t = _table(per_season=5, seasons=[2022, 2023, 2024])
    rows = []
    for _, r in t.iterrows():
        rows.append({"match_id": r["match_id"], "competition": "E0", "season": r["season"], "model": "gb_l7",
                     "horizon": "T-24", "p_home": 0.4})
    pd.DataFrame(rows).to_csv(tmp_path / "predictions.csv", index=False, encoding="utf-8-sig")
    got = ds.read_run(tmp_path, ["gb_l7"], settings)
    assert sorted(got["season"].unique()) == [2022, 2023]
    with pytest.raises(ValueError, match="不是开发赛季"):
        ds.prepare(t, settings)


def test_usage_check_refuses_exam(settings):
    assert ds.check_usage([2019, 2023], settings) == "研究实验"
    with pytest.raises(SeasonLockedError):
        ds.check_usage([2024], settings)


# ---------- 选参数不碰检验季 ----------
def test_select_never_sees_test_seasons(settings, data):
    """把检验季的比分全换掉，选出来的参数一模一样；把检验季的比赛交给 select 就停下。"""
    t2 = _table()
    test = t2["season"] >= 2020
    rng = np.random.default_rng(99)
    t2.loc[test, "home_goals"] = rng.integers(0, 5, test.sum())
    t2.loc[test, "away_goals"] = rng.integers(0, 5, test.sum())
    d2 = ds.prepare(t2, settings)
    plast = settings["direction_study"]["param_last"]
    c1, _ = ds.select(ds.take(data, data.setup["season"] <= plast), settings)
    c2, _ = ds.select(ds.take(d2, d2.setup["season"] <= plast), settings)
    assert c1 == c2
    with pytest.raises(ValueError, match="检验季"):
        ds.select(data, settings)


def test_identity_parameters(settings, data):
    """a = 0、c = 0、τ = 0 就是照赔率选；A 的 k = 1、b = 0、d = 0 就是 gb_l7 自己。"""
    mk = ds.decide(data, "market", {}, settings)
    base = ds.decide(data, "gb_l7", {}, settings)
    for x, prm in (("B", {"a": 0.0}), ("E", {"c": 0.0}), ("C", {"tau": 0.0})):
        got = ds.decide(data, x, prm, settings)
        for m in ds.MARKETS:
            assert (got[m][0] == mk[m][0]).all()
    got = ds.decide(data, "A", {"k": 1.0, "b": 0.0, "d": 0.0}, settings)
    for m in ds.MARKETS:
        assert (got[m][0] == base[m][0]).all()


# ---------- bootstrap、判断 ----------
def test_bootstrap_reproducible(settings):
    rng = np.random.default_rng(3)
    num, den = rng.normal(0.01, 0.5, 500), np.ones(500)
    weeks = np.repeat(np.arange(100), 5)
    a, b = ds.boot_ratio(num, den, weeks, settings), ds.boot_ratio(num, den, weeks, settings)
    assert a == b                                            # 同一个种子：每次一样
    assert a[0] == pytest.approx(num.mean()) and a[1] < a[0] < a[2]
    other = {**settings, "compare": {**settings["compare"], "seed": 1}}
    assert ds.boot_ratio(num, den, weeks, other)[1] != a[1]


def test_judge_rules(settings):
    def res(diff, lo, seasons, sd=0.001):
        return {"差": diff, "区间": [lo, diff + 0.01], "标准差": sd, "每个赛季的差": dict(zip(range(2020, 2024), seasons))}
    assert ds.judge(res(0.01, 0.002, [1, 1, 1, -1]), settings)["结论"] == "过了，比较可信"
    assert ds.judge(res(0.01, 0.002, [1, 1, 1, -1], sd=0.005), settings)["结论"] == "过了，可能是碰巧"
    assert ds.judge(res(0.01, -0.001, [1, 1, 1, 1]), settings)["结论"] == "没看到"      # 下限没到 0
    assert ds.judge(res(0.01, 0.002, [1, 1, -1, -1]), settings)["结论"] == "没看到"     # 只有两个赛季 > 0
    assert ds.judge(res(0.004, 0.001, [1, 1, 1, 1]), settings)["结论"] == "没看到"      # 差不到 0.5 个百分点


def test_diff_needs_same_line():
    with pytest.raises(ValueError, match="同一条线"):
        ds.diff_parts([1, 0.5], [1, 1])


# ---------- 核对、整个跑一遍、报告 ----------
def test_run_info_checks(settings):
    good = {"运行模式": "dev", "设置": {"models": settings["models"]}}
    assert "gbdt_variants.gb_inj" in ds.check_run_info(good, ["gb_l7_r", "gb_inj"], settings)["比过的设置"]
    with pytest.raises(ValueError, match="开发模式"):
        ds.check_run_info({**good, "运行模式": "exam"}, ["gb_inj"], settings)
    changed = json.loads(json.dumps(settings["models"]))
    changed["gbdt_variants"]["gb_inj"]["num_leaves"] = 15
    with pytest.raises(ValueError, match="gb_inj"):
        ds.check_run_info({"运行模式": "dev", "设置": {"models": changed}}, ["gb_inj"], settings)


def test_refit_tolerance():
    p = _probs(np.array([1.4]), np.array([1.1]), np.array([-0.05]))
    ds.fit_source(p, 1e-3, "gb_l7")
    p[0, 3] = 0.95                                           # 平局 28% 又要大 2.5 球 95%：DC 矩阵做不出来（rho 到边界）
    with pytest.raises(ValueError, match="超过"):
        ds.fit_source(p, 1e-3, "gb_l7")


def test_study_and_report(settings, data, tmp_path):
    res = ds.study(data, settings, log=lambda *a: None)
    s = res["summary"]
    assert set(s["检验"]) == set(ds.CANDIDATES)
    assert s["场数"]["检验"] == int((data.setup["season"] >= 2020).sum())
    assert all(s["检验"][x][m]["判断"]["结论"] in ("没看到", "过了，比较可信", "过了，可能是碰巧")
               for x in ds.CANDIDATES for m in ("ah", "1x2"))
    assert (res["picks"]["season"] >= 2020).all()
    ds.write(res, tmp_path, settings, {"检查": {}})
    text = (tmp_path / "report.md").read_text(encoding="utf-8")
    assert "推荐" not in text and "建议下注" not in text
    assert "实时伤停来源" in text
    for f in ("summary.json", "picks.csv", "grid.csv", "calibration.csv", "groups.csv", "run_info.json"):
        assert (tmp_path / f).exists()
