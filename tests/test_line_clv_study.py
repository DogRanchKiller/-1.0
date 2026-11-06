'''亚盘、大小球收盘线研究 src/evaluate/line_clv_study.py 的检查（需求文档第 14 节“亚盘、大小球收盘线研究怎么做”）。

都用小的假数据（数据库在内存里，回测结果写在 tmp_path），不碰 data/。
检查：两个赔率去水和录错的价格、回归找回已知的系数、亚盘线没变 / 变了时收盘胜算怎么算、多家平均线不一样不比、
CLV 和真实结算、各条规则买哪边、结论按三条写、考试赛季读不进来、次级联赛核对对不上就停、整个流程能跑通。
'''
import copy
import sqlite3

import numpy as np
import pandas as pd
import pytest
from scipy.stats import poisson

from src.clean.schema import create_tables
from src.common.config import load_settings
from src.common.markets import asian_handicap, over_under
from src.common.odds import devig
from src.evaluate import line_clv_study as lc
from src.features import FeatureStore

PIN, AVG = "平博 Pinnacle", "多家平均"


@pytest.fixture
def settings():
    s = copy.deepcopy(load_settings())
    s["compare"]["n_boot"] = 200
    return s


def _matrix(lh, la, n=10):
    m = np.outer(poisson.pmf(np.arange(n + 1), lh), poisson.pmf(np.arange(n + 1), la))
    return m / m.sum()


# ---------------- 去水 ----------------
def test_two_way(settings):
    rng = settings["quality"]["overround"]["2way"]
    p = lc.two_way([2.0, 1.9, 3.0, np.nan, 1.0], [2.0, 2.1, 3.0, 2.0, 5.0], rng, "proportional")
    assert p[0] == pytest.approx(0.5)
    assert p[1] == pytest.approx((1 / 1.9) / (1 / 1.9 + 1 / 2.1))
    assert np.isnan(p[2:]).all()          # 倒数之和 0.67（录错）、缺一个、赔率 1.0 都不算


# ---------------- 回归 ----------------
def test_ols_boot_recovers_known_slope(settings):
    '''造 y = 0.001 + 0.2·x₁ + 0.6·x₂ + 噪声，b₁ 应该在 0.2 附近、区间包住 0.2 且整个 > 0。'''
    r = np.random.default_rng(1)
    n = 3000
    x1, x2 = r.normal(0, 0.05, n), r.normal(0, 0.02, n)
    y = 0.001 + 0.2 * x1 + 0.6 * x2 + r.normal(0, 0.01, n)
    dates = pd.Series(pd.date_range("2019-08-01", periods=n, freq="4h").strftime("%Y-%m-%d"))
    out = lc.ols_boot(y, np.column_stack([x1, x2]), dates, settings)
    assert out["b1"] == pytest.approx(0.2, abs=0.01) and out["b2"] == pytest.approx(0.6, abs=0.03)
    assert 0 < out["b1_下限"] < 0.2 < out["b1_上限"]
    few = lc.ols_boot(y[:3], np.column_stack([x1, x2])[:3], dates[:3], settings)
    assert np.isnan(few["b1"])            # 场太少不算


# ---------------- 每场的概率 ----------------
def _raw():
    '''两场：1 号亚盘线没变（-0.5），2 号赛前 -0.25、收盘变成 -0.5，多家平均赛前用了别的线。'''
    base = {"pre_pin_1x2_h": 2.0, "pre_pin_1x2_d": 3.5, "pre_pin_1x2_a": 4.0,
            "close_pin_1x2_h": 1.9, "close_pin_1x2_d": 3.6, "close_pin_1x2_a": 4.3,
            "pre_pin_ou_line": 2.5, "pre_pin_ou_x": 2.0, "pre_pin_ou_y": 1.85,
            "pre_avg_ou_line": 2.5, "pre_avg_ou_x": 1.95, "pre_avg_ou_y": 1.9,
            "close_pin_ou_line": 2.5, "close_pin_ou_x": 1.9, "close_pin_ou_y": 1.95}
    one = {**base, "match_id": 1, "pre_pin_ah_line": -0.5, "pre_pin_ah_x": 1.95, "pre_pin_ah_y": 1.95,
           "pre_avg_ah_line": -0.5, "pre_avg_ah_x": 2.0, "pre_avg_ah_y": 1.85,
           "close_pin_ah_line": -0.5, "close_pin_ah_x": 1.85, "close_pin_ah_y": 2.05, "home_goals": 2, "away_goals": 0}
    two = {**base, "match_id": 2, "pre_pin_ah_line": -0.25, "pre_pin_ah_x": 1.9, "pre_pin_ah_y": 2.0,
           "pre_avg_ah_line": -0.5, "pre_avg_ah_x": 2.0, "pre_avg_ah_y": 1.85,
           "close_pin_ah_line": -0.5, "close_pin_ah_x": 1.95, "close_pin_ah_y": 1.95, "home_goals": 1, "away_goals": 1}
    info = pd.DataFrame({"match_id": [1, 2], "group": "五大联赛", "competition": "E0", "season": 2019,
                         "match_date": ["2019-09-14", "2019-09-21"], "cutoff_utc": ["x", "x"]})
    return pd.DataFrame([one, two]), info


def test_per_match_probabilities(settings):
    raw, info = _raw()
    mats = {1: _matrix(1.6, 1.0), 2: _matrix(1.3, 1.1)}
    pm, ap = lc.per_match(raw, info, mats, settings)
    pm = pm.set_index(["match_id", "market"])
    d = lambda x, y: devig(np.array([[x, y]]), "proportional")[0, 0]   # noqa: E731
    # 1 号亚盘：线没变，c 就是收盘直接去水
    r = pm.loc[(1, "ah")]
    assert not r["moved"]
    assert r["m"] == pytest.approx(0.5) and r["a"] == pytest.approx(d(2.0, 1.85))
    assert r["c"] == pytest.approx(d(1.85, 2.05))
    assert r["p"] == pytest.approx(asian_handicap(mats[1], -0.5, "home")["win_share"])
    assert r["ev_a"] == pytest.approx(asian_handicap(mats[1], -0.5, "home", 1.95)["expected_profit"])
    assert r["clv_a"] == pytest.approx(1.95 * r["c"] - 1) and r["clv_b"] == pytest.approx(1.95 * (1 - r["c"]) - 1)
    assert (r["out_a"], r["profit_a"], r["out_b"], r["profit_b"]) == ("win", pytest.approx(0.95), "loss", -1.0)
    # 1 号大小球：2 球，小球赢
    r = pm.loc[(1, "ou")]
    assert r["c"] == pytest.approx(d(1.9, 1.95)) and r["p"] == pytest.approx(over_under(mats[1], 2.5)["win_share"])
    assert (r["profit_a"], r["profit_b"]) == (-1.0, pytest.approx(0.85))
    # 2 号亚盘：线变了 → 收盘去水（新线）+ 市场矩阵在旧线、新线的差；多家平均线不一样 → a 不算
    r = pm.loc[(2, "ah")]
    assert r["moved"] and np.isnan(r["a"])
    c3 = np.array([[1.9, 3.6, 4.3]])
    cm = lc.market_mats(c3, np.array([d(1.9, 1.95)]), np.array([True]), settings)[0]
    gap = asian_handicap(cm, -0.25, "home")["win_share"] - asian_handicap(cm, -0.5, "home")["win_share"]
    assert gap > 0                                               # 让得少，主队这边胜算更高
    assert r["c"] == pytest.approx(0.5 + gap)
    assert r["out_a"] == "half_loss" and r["profit_a"] == pytest.approx(-0.5)   # 1:1，主让 0.25 输一半
    # 换线近似的核对只用线没变的场（1 号）
    assert len(ap) == 1 and ap.iloc[0]["direct"] == pytest.approx(d(1.85, 2.05))
    # 网站倾向用赛前市场矩阵
    assert 0 < pm.loc[(1, "ah"), "site"] < 1


def test_per_match_missing_close(settings):
    '''收盘胜平负缺了：线变了的亚盘 c 算不了（nan），线没变的照常；没有比分不结算。'''
    raw, info = _raw()
    raw[["close_pin_1x2_h", "home_goals", "away_goals"]] = np.nan
    pm, _ = lc.per_match(raw, info, {1: _matrix(1.6, 1.0), 2: _matrix(1.3, 1.1)}, settings)
    pm = pm.set_index(["match_id", "market"])
    assert np.isnan(pm.loc[(2, "ah"), "c"]) and np.isfinite(pm.loc[(1, "ah"), "c"])
    assert "profit_a" not in pm or pm["profit_a"].isna().all()
    raw2, _ = _raw()
    raw2.loc[0, "pre_pin_ah_x"] = 1.2                            # 倒数之和 1.34：录错 → 这场亚盘不算
    pm2, _ = lc.per_match(raw2, info, {1: _matrix(1.6, 1.0), 2: _matrix(1.3, 1.1)}, settings)
    assert not ((pm2["match_id"] == 1) & (pm2["market"] == "ah")).any()


# ---------------- 虚拟注 ----------------
def test_picks(settings):
    s = pd.DataFrame({"match_id": [1, 2, 3, 4], "ev_a": [0.06, -0.10, 0.04, 0.09], "ev_b": [-0.08, 0.02, -0.05, 0.10],
                      "p": [0.6, 0.45, 0.5, 0.52], "site": [0.53, 0.47, 0.51, np.nan]})
    pk = lc.picks(s, settings)
    assert list(pk["模型价值 5%"]) == [1, 0, 0, -1]               # 两边里大的 ≥ 5% 才下，下大的那边
    assert list(pk["模型价值 3%"]) == [1, 0, 1, -1]
    assert list(pk["模型价值 8%"]) == [0, 0, 0, -1]
    assert list(pk["跟模型"]) == [1, -1, 0, 1]
    assert list(pk["网站倾向"]) == [1, -1, 0, 0]                   # 0.52 以上买这一边，0.48 以下买另一边
    assert list(pk["总买主队/大球"]) == [1, 1, 1, 1]
    many = pd.DataFrame({"match_id": range(2000), "ev_a": 0.0, "ev_b": 0.0, "p": 0.5, "site": 0.5})
    r1, r2 = lc.picks(many, settings)["随机"], lc.picks(many, settings)["随机"]
    assert (r1 == r2).all() and abs((r1 == 1).mean() - 0.5) < 0.04   # 同一场每次一样，两边各一半左右


def test_bot_stats(settings):
    s = pd.DataFrame({"match_date": ["2019-09-14", "2019-09-21", "2019-09-28"],
                      "clv_a": [0.02, -0.01, 0.03], "clv_b": [-0.05, -0.03, -0.06],
                      "profit_a": [0.9, -1.0, 0.0], "profit_b": [-1.0, 0.9, 0.0],
                      "out_a": ["win", "loss", "push"], "out_b": ["loss", "win", "push"]})
    st = lc.bot_stats(s, np.array([1, -1, 1]), settings)
    assert st["下注场数"] == 3
    assert st["CLV"] == pytest.approx((0.02 - 0.03 + 0.03) / 3)
    assert st["比另一边"] == pytest.approx(((0.02 + 0.05) + (-0.03 + 0.01) + (0.03 + 0.06)) / 3)
    assert st["ROI"] == pytest.approx((0.9 + 0.9 + 0.0) / 3)
    assert st["赢的比例"] == pytest.approx(1.0)                    # 走盘不算进分母


# ---------------- 结论 ----------------
def _beta(b1, lo, seasons):
    rows = [{"组": "五大联赛", "玩法": "亚盘", "分法": "主判断", "取值": "全部", "场数": 100,
             "b1": b1, "b1_下限": lo, "b1_上限": b1 + 0.05, "b2": 0.5}]
    rows += [{"组": "五大联赛", "玩法": "亚盘", "分法": "赛季", "取值": str(2019 + i), "场数": 20,
              "b1": v, "b1_下限": np.nan, "b1_上限": np.nan, "b2": 0.5} for i, v in enumerate(seasons)]
    return pd.DataFrame(rows)


def _bots(clv_lo):
    return pd.DataFrame([{"组": "五大联赛", "玩法": "亚盘", "规则": "模型价值 5%", "下注场数": 50,
                          "CLV": clv_lo + 0.01, "CLV_下限": clv_lo, "CLV_上限": clv_lo + 0.02,
                          "ROI": -0.02, "ROI_下限": -0.08, "ROI_上限": 0.04}])


def test_verdict_rules(settings):
    v = lc.verdict(_beta(0.10, 0.03, [0.1, 0.2, 0.05, 0.1, -0.1]), _bots(-0.03), settings).iloc[0]
    assert v["判断1 收盘往模型走"] and not v["判断2 价格比收盘好"]
    assert v["结论"] == "方向对，但不够抵掉抽水" and v["ROI 说明"] == "可能只是运气" and not v["要先查偷看未来"]
    # 只有 3 个赛季 b₁ > 0 → 判断 1 不过
    v = lc.verdict(_beta(0.10, 0.03, [0.1, 0.2, -0.05, 0.1, -0.1]), _bots(-0.03), settings).iloc[0]
    assert v["结论"] == "没看到"
    # b₁ 区间 > 0 但太小（< 0.05）→ 不过
    v = lc.verdict(_beta(0.04, 0.01, [0.1] * 5), _bots(-0.03), settings).iloc[0]
    assert not v["判断1 收盘往模型走"]
    # 两条都过 → 能预判收盘，但 CLV 明显为正要先查偷看未来
    v = lc.verdict(_beta(0.10, 0.03, [0.1] * 5), _bots(0.01), settings).iloc[0]
    assert v["结论"] == "模型在这一组能预判收盘" and v["要先查偷看未来"]
    v = lc.verdict(_beta(0.6, 0.5, [0.6] * 5), _bots(-0.03), settings).iloc[0]
    assert v["要先查偷看未来"]                                     # b₁ > 0.5 也要先查


# ---------------- 读回测、核对、整个流程 ----------------
def _fake_db(settings, n=14):
    '''两组联赛（E0、E1）2019/20 各 n 场、每周一场，另加一场考试赛季（2024/25）。价格是随便造的、但都像样。'''
    r = np.random.default_rng(7)
    c = sqlite3.connect(":memory:")
    create_tables(c)
    # 都是周日：football-data 周五傍晚采的赛前价格，T-24 已经能知道（周六的比赛 T-24 还没采，需求文档第 16 节时间核对）
    games = [(100 * (k + 1) + i, lg, 2019, (pd.Timestamp("2019-08-11") + pd.Timedelta(weeks=i)).strftime("%Y-%m-%d"))
             for k, lg in enumerate(["E0", "E1"]) for i in range(n)]
    games.append((999, "E0", 2024, "2024-08-17"))
    c.executemany("insert into teams (team_id, country) values (?, 'England')",
                  [(f"H{g[0]}",) for g in games] + [("B",)])
    c.executemany("insert into matches (match_id, competition, season, home_team, away_team, match_date, status)"
                  " values (?, ?, ?, ?, 'B', ?, 'finished')", [(g[0], g[1], g[2], f"H{g[0]}", g[3]) for g in games])
    c.executemany("insert into match_stats (match_id, source, home_goals, away_goals, result)"
                  " values (?, 'football-data', ?, ?, ?)",
                  [(g[0], hg, ag, "H" if hg > ag else "D" if hg == ag else "A")
                   for g in games for hg, ag in [tuple(int(v) for v in r.integers(0, 4, 2))]])
    ins = ("insert into odds (match_id, source, bookmaker, market, timing, line, home, draw, away, over, under)"
           " values (?, 'football-data', ?, ?, ?, ?, ?, ?, ?, ?, ?)")
    rows = []
    for g in games:
        mid = g[0]
        line = -0.5 if mid % 3 else -0.25
        for book, timing, l in ((PIN, "pre", line), (AVG, "pre", line), (PIN, "close", -0.5)):
            h = float(r.uniform(1.8, 2.1))
            rows.append((mid, book, "ah", timing, l, h, None, 1 / (1.04 - 1 / h), None, None))
            o = float(r.uniform(1.8, 2.1))
            rows.append((mid, book, "ou", timing, 2.5, None, None, None, o, 1 / (1.04 - 1 / o)))
            rows.append((mid, book, "1x2", timing, None, 2.0 + r.uniform(-0.2, 0.2), 3.5, 4.0, None, None))
    rows.append((100, "Bet365", "ah", "pre", -0.5, 1.9, None, 1.9, None, None))   # 别的公司：不用
    c.executemany(ins, rows)
    c.commit()
    return FeatureStore(c, "开发", settings), games


def _fake_run(tmp_path, fs, games, league, name, extra_rows=()):
    out, ids, mats = [], [], []
    for k, (mid, lg, season, date) in enumerate([g for g in games if g[1] == league]):
        m = _matrix(1.1 + 0.05 * k, 1.2)
        p = [np.tril(m, -1).sum(), np.trace(m), np.triu(m, 1).sum()]
        cut = fs.cutoff_before(mid, 24) if season <= 2023 else "2024-08-16 00:00:00"
        for model, hz, ll in (("gb_l7", "T-24", 1.0 + 0.01 * k), ("pinnacle_close", "收盘", 0.99 + 0.01 * k)):
            out.append({"match_id": mid, "competition": lg, "season": season, "match_date": date, "horizon": hz,
                        "cutoff_utc": cut, "model": model, "p_home": p[0], "p_draw": p[1], "p_away": p[2],
                        "log_loss": ll})
        ids.append(mid)
        mats.append(m.astype(np.float32))
    d = tmp_path / name
    d.mkdir()
    pd.DataFrame(out).to_csv(d / "predictions.csv", index=False, encoding="utf-8-sig")
    np.savez_compressed(d / "matrices_gb_l7.npz", match_id=np.array(ids, dtype=np.int64), matrix=np.stack(mats))
    return d


def _groups(settings):
    settings["line_clv_study"]["groups"] = {"五大联赛": {"run": "a", "leagues": ["E0"]},
                                            "次级联赛": {"run": "b", "leagues": ["E1"]}}
    return settings


def test_load_group_drops_exam_seasons(settings, tmp_path):
    fs, games = _fake_db(settings)
    d = _fake_run(tmp_path, fs, games, "E0", "a")
    m, mats = lc.load_group("五大联赛", {"run": "a", "leagues": ["E0"]}, settings, d)
    assert 999 not in set(m["match_id"]) and len(m) == len(mats) == 14
    assert np.allclose(mats.sum(axis=(1, 2)), 1)
    bad = copy.deepcopy(settings)
    bad["line_clv_study"]["last_season"] = 2024
    with pytest.raises(ValueError):
        lc.load_group("五大联赛", {"run": "a", "leagues": ["E0"]}, bad, d)
    z = np.load(d / "matrices_gb_l7.npz")
    np.savez_compressed(d / "matrices_gb_l7.npz", match_id=z["match_id"], matrix=z["matrix"][::-1])
    with pytest.raises(ValueError):                               # 矩阵和 predictions.csv 对不上
        lc.load_group("五大联赛", {"run": "a", "leagues": ["E0"]}, settings, d)


def test_second_check(settings, tmp_path):
    fs, games = _fake_db(settings)
    d = _fake_run(tmp_path, fs, games, "E1", "b")
    settings["line_clv_study"]["second_check"]["log_loss"] = 1.0 + 0.01 * 13 / 2
    assert lc.second_check(settings, d)["对得上"]
    settings["line_clv_study"]["second_check"]["log_loss"] = 1.1
    with pytest.raises(ValueError):
        lc.second_check(settings, d)


def test_collect_and_study(settings, tmp_path):
    fs, games = _fake_db(settings)
    _groups(settings)
    dirs = {"五大联赛": _fake_run(tmp_path, fs, games, "E0", "a"), "次级联赛": _fake_run(tmp_path, fs, games, "E1", "b")}
    m, _ = lc.load_group("五大联赛", settings["line_clv_study"]["groups"]["五大联赛"], settings, dirs["五大联赛"])
    raw, info = lc.collect(fs, m, settings)
    assert info == {"取数层不认的比赛": 0, "截止时间和回测不一样的比赛": 0}
    r = raw.set_index("match_id").loc[101]
    assert r["pre_pin_ah_line"] == -0.5 and r["close_pin_ah_line"] == -0.5 and r["pre_pin_ou_line"] == 2.5
    assert np.isfinite(r[["pre_pin_1x2_h", "close_pin_1x2_h", "pre_avg_ah_x", "home_goals"]].astype(float)).all()
    assert raw.set_index("match_id").loc[102, "pre_pin_ah_line"] == -0.25   # 102 能被 3 整除 → 赛前 -0.25
    tables, _ = lc.study(fs, settings, dirs)
    pm = tables["per_match"]
    assert 999 not in set(pm["match_id"]) and set(pm["group"]) == {"五大联赛", "次级联赛"}
    assert len(pm) == 2 * 2 * 14                                  # 两组 × 两个玩法 × 14 场
    assert pm.loc[pm["market"] == "ah", "moved"].sum() == sum(1 for g in games[:-1] if g[0] % 3 == 0)
    v = tables["verdict"]
    assert len(v) == 4 and set(v["结论"]) <= {"模型在这一组能预判收盘", "方向对，但不够抵掉抽水", "没看到",
                                             "没看到（CLV 过了但回归没过，按规则不算）"}
    assert set(tables["bots"]["规则"]) >= {"模型价值 5%", "模型价值 3%", "模型价值 8%", "随机", "网站倾向"}
    text = lc.report(v, tables["beta"], tables["bots"], tables["approx_check"], tables["counts"], {})
    assert "## 结论" in text and "推荐" not in text
