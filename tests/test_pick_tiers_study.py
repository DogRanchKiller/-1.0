"""四个玩法“把握”等级研究（src/evaluate/pick_tiers_study.py）的检查。用手造的小比分矩阵，不读真回测、不连数据库。"""
import numpy as np
import pandas as pd
import pytest

from src.common.config import load_settings
from src.common.markets import over_under, poisson_score_matrix
from src.evaluate import pick_tiers_study as pt


@pytest.fixture(scope="module")
def settings():
    s = load_settings()
    s = {**s, "pick_tiers_study": {**s["pick_tiers_study"], "boot": 200}}
    return s


def _mats(pairs, n=10):
    m = np.stack([poisson_score_matrix(h, a, n) for h, a in pairs])
    return m / m.sum(axis=(1, 2), keepdims=True)


def test_over_shares_match_markets(settings):
    """向量算法和 markets.over_under 一样（含四分之一线、整数线）。"""
    mats = _mats([(1.4, 1.1), (2.2, 0.6)])
    tot = pt.totals(mats)
    lines = [1.75, 2.0, 2.5, 3.25]
    got = pt.over_shares(tot, lines)
    for k in range(2):
        for c, ln in enumerate(lines):
            assert got[k, c] == pytest.approx(over_under(mats[k], ln, "over")["win_share"], abs=1e-12)


def test_implied_line_closest_to_half(settings):
    mats = _mats([(0.8, 0.6), (2.5, 1.8)])
    line, over = pt.implied_lines(pt.totals(mats), pt.ou_grid(settings))
    assert line[0] < line[1]                      # 进球少的比赛线低
    grid = pt.ou_grid(settings)
    sh = pt.over_shares(pt.totals(mats), grid)
    for k in range(2):
        assert abs(over[k] - 0.5) == pytest.approx(np.nanmin(np.abs(sh[k] - 0.5)))


def test_cuts_and_tier():
    p = pd.Series(np.linspace(0.3, 0.9, 301))
    c = pt.cuts(p)
    assert c == [0.5, 0.7]
    t = pt.tier(pd.Series([0.4, 0.5, 0.69, 0.7, np.nan]), c)
    assert list(t[:4]) == ["低", "中", "中", "高"] and pd.isna(t[4])


def test_hit_rate_ignores_push():
    assert pt.hit_rate([1, 0.5, 0, -1, np.nan]) == pytest.approx(1.5 / 2.5)
    assert np.isnan(pt.hit_rate([0, 0]))


def _table(n, seed=1):
    """n 场假比赛：比分矩阵、比分按矩阵抽，日期每周一场，2012–2023 轮着排。"""
    rng = np.random.default_rng(seed)
    pairs = [(float(rng.uniform(0.6, 2.6)), float(rng.uniform(0.4, 1.8))) for _ in range(n)]
    mats = _mats(pairs)
    flat = mats.reshape(n, -1)
    cell = np.array([rng.choice(flat.shape[1], p=f / f.sum()) for f in flat])
    hg, ag = np.unravel_index(cell, mats.shape[1:])
    dates = pd.date_range("2012-08-06", periods=n, freq="7D")
    t = pd.DataFrame({"match_id": np.arange(n) + 1, "competition": "E0",
                      "season": np.clip(2012 + np.arange(n) * 12 // n, 2012, 2023),
                      "match_date": dates.strftime("%Y-%m-%d"), "home_goals": hg, "away_goals": ag,
                      "ah_line": np.where(np.arange(n) % 7 == 0, np.nan, -0.5)})
    return t, mats


def test_picks_columns_and_settle(settings):
    t, mats = _table(60)
    per = pt.picks(t, mats, settings)
    assert len(per) == 60
    for ch in pt.CHANNELS:
        assert per[f"p_{ch}"].dropna().between(0, 1).all()
    assert per["p_1x2"].ge(1 / 3 - 1e-9).all() and per["p_ah"].dropna().ge(0.5).all()
    assert per["pick_ah"].isna().sum() == t["ah_line"].isna().sum()        # 没有主盘口的场不预测让球
    k = 3                                                                  # 逐场核对一场：大小球的结算
    h, a, ln = per.loc[k, ["home_goals", "away_goals", "ou_line"]]
    over = (h + a) > ln
    assert per.loc[k, "v_ou"] in (1.0, 0.5, 0.0, -0.5, -1.0)
    if ln % 1 == 0.5:                                                      # 半球线没有走盘、赢半
        assert per.loc[k, "v_ou"] == (1.0 if over == (per.loc[k, "pick_ou"] == "over") else -1.0)
    # 给了线就用那条线
    fixed = pt.picks(t, mats, settings, ou_line=np.full(60, 2.5))
    assert (fixed["ou_line"] == 2.5).all()
    assert (fixed["v_ou"] == per["v_ou25"]).all()


def test_tier_stats_verdict_and_study_shape(settings):
    t, mats = _table(400)
    per = pt.picks(t, mats, settings)
    mask = (per["season"] <= 2019).to_numpy()
    df, cut = pt.tiered(per, mask, settings)
    st = pt.tier_stats(df[~mask], "1x2", settings)
    assert sum(st[tr]["场数"] for tr in pt.TIERS) == st["全部"]["场数"] == int((~mask).sum())
    assert st["高减低"]["下限"] <= st["高减低"]["差"] <= st["高减低"]["上限"]
    v = pt.verdict(st, settings)
    assert isinstance(v["分得开"], bool) and set(v["准不准"]) == set(pt.TIERS)
    c = pt.controls(df[mask], df[~mask])
    assert len(c["最常见4个比分"]) == 4 and 0 <= c["一直猜主胜"] <= 1


def test_verdict_rules(settings):
    good = {tr: {"命中率": r, "差": 0.0, "差_下限": -0.01, "差_上限": 0.01} for tr, r in zip(pt.TIERS, (0.7, 0.5, 0.4))}
    good["高减低"] = {"差": 0.3, "下限": 0.2, "上限": 0.4}
    assert pt.verdict(good, settings)["分得开"] is True
    bad = {**good, "中": {**good["中"], "命中率": 0.8}}                     # 不是 高 ≥ 中 ≥ 低
    assert pt.verdict(bad, settings)["分得开"] is False
    off = {**good, "高": {**good["高"], "差": 0.05, "差_下限": 0.03, "差_上限": 0.07}}
    assert pt.verdict(off, settings)["准不准"]["高"] is False


def test_load_refuses_exam_seasons(settings, tmp_path):
    s = {**settings, "pick_tiers_study": {**settings["pick_tiers_study"], "test_last": 2024}}
    with pytest.raises(ValueError, match="考试赛季"):
        pt.load(s, run_dir=tmp_path)
