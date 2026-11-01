"""融合研究用到的市场比分矩阵（需求文档“用 A 版重选融合权重和倾向门槛怎么做”）。"""
import numpy as np

from src.common.markets import (dc_matrices, dc_rho_bounds, fit_market_dc, fit_market_lambdas, matrix_probs,
                                poisson_matrices)
from src.models.goals import score_matrices


def _probs(m):
    """矩阵 → [主胜, 平, 客胜, 大 2.5 球]。"""
    i, j = np.indices(m.shape[1:])
    return np.stack([(m * (i > j)).sum(axis=(1, 2)), (m * (i == j)).sum(axis=(1, 2)),
                     (m * (i < j)).sum(axis=(1, 2)), (m * (i + j > 2)).sum(axis=(1, 2))], axis=1)


def test_dc_matrices_same_as_model_score_matrices():
    lh, la = np.array([1.2, 1.8, 0.7]), np.array([1.1, 0.6, 1.5])
    for rho in (-0.1, 0.0, 0.08):
        assert np.allclose(dc_matrices(lh, la, rho), score_matrices(lh, la, rho=rho))
    assert np.allclose(dc_matrices(lh, la, 0.0), poisson_matrices(lh, la))


def test_dc_changes_only_draw_split_not_goals():
    """rho 只把概率在平局和胜负之间挪：主胜−客胜、大 2.5 球不变，平局多 2·rho·c。"""
    lh, la = np.array([1.4]), np.array([1.1])
    a, b = _probs(dc_matrices(lh, la, 0.0)), _probs(dc_matrices(lh, la, 0.1))
    c = lh * la * np.exp(-lh - la)
    assert np.isclose(b[0, 0] - b[0, 2], a[0, 0] - a[0, 2])
    assert np.isclose(b[0, 3], a[0, 3])
    assert np.isclose(a[0, 1] - b[0, 1], 2 * 0.1 * c[0])


def test_fit_market_dc_reproduces_targets():
    """真实常见的赔率（去水后）：胜平负和大 2.5 球都要正好对上。"""
    target = np.array([[0.45, 0.27, 0.28, 0.52],     # 普通比赛
                       [0.75, 0.16, 0.09, 0.65],     # 强队主场
                       [0.20, 0.25, 0.55, 0.55],     # 客队强
                       [0.36, 0.32, 0.32, 0.40]])    # 平局多、球少
    lh, la, rho = fit_market_dc(target)
    got = _probs(dc_matrices(lh, la, rho))
    assert np.abs(got - target).max() < 1e-6
    lo, hi = dc_rho_bounds(lh, la)
    assert ((rho >= lo) & (rho <= hi)).all()


def test_fit_market_dc_clips_rho_when_draw_impossible():
    """平局要得太多、rho 到边界也补不上时，取边界，矩阵仍然合法（格子不为负、加起来是 1）。"""
    lh, la, rho = fit_market_dc(np.array([[0.20, 0.60, 0.20, 0.30]]))
    lo, hi = dc_rho_bounds(lh, la)
    assert np.isclose(rho[0], lo[0])
    m = dc_matrices(lh, la, rho)
    assert (m >= 0).all() and np.isclose(m.sum(), 1)


def test_fit_market_lambdas_unchanged_after_refactor():
    """抽出公共的高斯-牛顿以后，原来的四个数拟合结果照旧：独立泊松造出来的目标能拟合回去。"""
    lh, la = np.array([1.3, 2.1]), np.array([0.9, 0.8])
    fh, fa = fit_market_lambdas(matrix_probs(lh, la))
    assert np.allclose(fh, lh, atol=1e-5) and np.allclose(fa, la, atol=1e-5)


# ======================= 融合研究 src/evaluate/fusion_study.py =======================
import pandas as pd  # noqa: E402

from src.common.config import load_settings  # noqa: E402
from src.evaluate import fusion_study as fsy  # noqa: E402


def _fake_run(n_per_season=300, seasons=(2005, 2006, 2007), seed=1):
    """造一次假回测：真实比分按“真”的 λ 抽；模型看到的 λ 带噪声，赔率看到的 λ 噪声小（赔率更准）。"""
    rng = np.random.default_rng(seed)
    rows, mats, odds = [], [], []
    mid = 0
    for s in seasons:
        for k in range(n_per_season):
            mid += 1
            lh, la = rng.uniform(0.8, 2.2), rng.uniform(0.6, 1.8)
            hg, ag = rng.poisson(lh), rng.poisson(la)
            m = score_matrices(np.array([lh * np.exp(rng.normal(0, 0.25))]), np.array([la * np.exp(rng.normal(0, 0.25))]))[0]
            mk = score_matrices(np.array([lh * np.exp(rng.normal(0, 0.05))]), np.array([la * np.exp(rng.normal(0, 0.05))]),
                                rho=-0.05)[0]
            p = _probs(m[None])[0]
            q = _probs(mk[None])[0]
            date = pd.Timestamp(f"{s}-08-15") + pd.Timedelta(days=k // 10 * 7)
            res = "H" if hg > ag else ("D" if hg == ag else "A")
            base = {"match_id": mid, "competition": "E0" if k % 2 else "D1", "season": s,
                    "match_date": date.strftime("%Y-%m-%d"), "result": res, "home_goals": hg, "away_goals": ag}
            rows.append({**base, "model": "gb_l7", "horizon": "T-24", "p_home": p[0], "p_draw": p[1], "p_away": p[2],
                         "cutoff_utc": (date - pd.Timedelta(days=1)).strftime("%Y-%m-%dT%H:%M:%SZ")})
            y = {"H": 0, "D": 1, "A": 2}[res]
            rows.append({**base, "model": "pinnacle_close", "horizon": "收盘", "p_home": q[0], "p_draw": q[1],
                         "p_away": q[2], "log_loss": -np.log(q[y]), "brier": 0.5, "rps": 0.2})
            mats.append(m.astype(np.float32))
            odds.append({"match_id": mid, "mo_p_home": q[0], "mo_p_draw": q[1], "mo_p_away": q[2], "mo_over25": q[3],
                         "mo_pinnacle": 1.0, "ah_line": -0.25 if lh > la else 0.25, "ah_home_odds": 1.9,
                         "ah_away_odds": 1.95 if k % 50 else 9.0})        # 每 50 场一场不是主盘口
    odds[0].update(mo_over25=np.nan)          # 一场缺大小球：不融合
    return pd.DataFrame(rows), np.arange(1, mid + 1), np.stack(mats), pd.DataFrame(odds)


def test_fusion_study_runs_and_market_matrix_reproduces_odds():
    settings = load_settings()
    pred, ids, mats, odds = _fake_run()
    res = fsy.run(pred, ids, mats, odds, settings)
    c = res["check"]
    assert c["融合"] == len(ids) - 1 and c["未融合（胜平负或大 2.5 球赔率缺）"] == 1
    assert c["市场矩阵还原误差（最大）"] < 1e-6
    assert c["亚盘线不是主盘口、不用的场数"] == 17 and c["没有亚盘线的场数"] == 0   # 第 1 场没融合
    assert c["两种融合算法胜平负概率最大差"] < 1e-6        # 市场矩阵还原得一样 → 矩阵上融合 = 各玩法各自融合
    g = res["grid"]
    assert len(g[(g["分法"] == "全部")]) == 21 and g[g["分法"] == "全部"]["最好"].sum() == 1
    # 赔率噪声小得多 → 选出的 w 偏向赔率
    assert c["全部融合场次选出的 w"] >= 0.5
    assert list(res["rolling"]["season"]) == [2006, 2007]
    comp = res["compare"]
    assert {"market_w1", "fused_rolling"} <= set(comp["base"])
    assert set(res["thresholds"]["玩法"]) == {"result", "draw", "over25", "btts", "ah"}


def test_rolling_w_uses_only_earlier_seasons():
    ws = fsy.w_grid(0.5)                                   # 0, 0.5, 1
    seasons = np.array([2005, 2005, 2006, 2007])
    ll = np.array([[1, 2, 3], [1, 2, 3], [3, 2, 1], [9, 9, 9]], float)
    rw = fsy.rolling_w(seasons, ll, ws, 2006)
    # 2006 只用 2005（w=0 最好）；2007 用 2005+2006（三档平均 1.67 / 2 / 2.33 → 还是 0），2007 自己那行（全 9）不能参与
    assert rw == {2006: 0.0, 2007: 0.0}
    rw2 = fsy.rolling_w(np.array([2005, 2006, 2007]), np.array([[3, 2, 1], [3, 2, 1], [0, 9, 9]], float), ws, 2006)
    assert rw2 == {2006: 1.0, 2007: 1.0}


def test_pick_threshold_needs_every_band_above():
    b = pd.DataFrame({"下沿": [0, 0.02, 0.04, 0.06], "n": [10, 10, 0, 10], "满足": [False, True, None, True]})
    assert fsy.pick_threshold(b) == 0.02                   # 空档跳过；0 那档不满足，停在 0.02
    b2 = b.assign(满足=[True, True, None, False])
    assert fsy.pick_threshold(b2) is None                  # 最高一档不满足 → 一律无明显倾向


def test_ah_lean_settles_quarter_lines():
    """主队 -0.25（平手/半球）：净胜 1 球以上主队赢全，平局主队输半，输球输全。"""
    n = 10
    diff = np.zeros((3, 2 * n + 1))
    diff[:, n + 1] = 0.6                                   # 主胜 1 球 60%
    diff[:, n] = 0.4                                       # 平局 40%
    line = np.array([-0.25, -0.25, np.nan])
    s, v = fsy.ah_lean(diff, line, np.array([0, 2, 1]))
    # 主队胜算 = 0.6 ÷ (0.6 + 0.4 × 0.5) = 0.75 → 偏向 0.25，偏向主队
    assert np.isclose(s[0], 0.25) and np.isclose(s[1], 0.25) and np.isnan(s[2])
    assert v[0] == -0.5 and v[1] == 1.0                    # 踢平：主队输半；赢 2 球：赢全


def test_lean_values_result_and_sides():
    t = pd.DataFrame({"y": [0, 1, 2], "over": [1.0, 0.0, 0.0], "btts_yes": [0.0, 1.0, 1.0],
                      "home_goals": [2, 1, 0], "away_goals": [0, 1, 1], "ah_line": [np.nan] * 3})
    d = {"p3": np.array([[0.5, 0.3, 0.2], [0.5, 0.3, 0.2], [0.5, 0.3, 0.2]]), "over25": np.array([0.6, 0.6, 0.4]),
         "btts": np.array([0.4, 0.4, 0.6]), "diff": np.zeros((3, 21))}
    v = fsy.lean_values(t, d)
    assert np.allclose(v["result"][0], 0.2) and list(v["result"][1]) == [1.0, -1.0, 0.0]
    assert np.allclose(v["draw"][1], np.array([0, 1, 0]) - 1 / 3)
    assert list(v["over25"][1]) == [1.0, -1.0, 1.0] and np.allclose(v["over25"][0], 0.1)
    assert list(v["btts"][1]) == [1.0, -1.0, 1.0]
