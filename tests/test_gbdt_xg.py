"""xG 研究第 1 段（需求文档第 4 节“xG 研究第 1 段怎么做”）：gbdt 里的 dyn 用“一半进球一半 xG”更新的自动检查。

不需要数据库，用模拟比赛。检查的东西：
  - xg_mix = 0（不写）时，多给 xG 列也和原来完全一样（gb_l7 不受影响）；
  - xG 和真实进球一样时，xg_mix = 0.5 的结果和不用 xG 完全一样（混合值 = 进球）；
  - xG 不一样时：只有 dyn 的 m、P 变了；dyn 记下的比分（拟合 rho 用）、Elo、pi-rating、积分榜、树的目标都还是真实比分；
  - 混合值和手算的 0.5 × 进球 + 0.5 × xG 一样（直接拿单独的 dyn 喂混合值对照）；
  - 没有 xG 的比赛照常用真实进球；
  - 用途不是“研究实验”、xg_mix 不在 0～1 之间都报错；登记表里 gb_xg_r 的设置对、gb_l7 没变。
"""
import numpy as np
import pytest

from src.models.dynamic import DynamicModel
from src.models.registry import build_model, model_config
from tests.test_adaptive import promo_history
from tests.test_gbdt import DYN, gb

CUT = "2030-01-01 00:00:00"


def with_xg(h, shift=0.0, seed=0):
    """给模拟比赛加上 Understat xG 两列：shift = 0 时就等于进球，不然加一点随机数（不小于 0）。"""
    rng = np.random.default_rng(seed)
    n = len(h)
    return h.assign(us_home_xg=np.clip(h["home_goals"] + shift * rng.normal(size=n), 0, None),
                    us_away_xg=np.clip(h["away_goals"] + shift * rng.normal(size=n), 0, None))


def same_dyn(a, b):
    return all(np.array_equal(a.dyn.leagues[c].m, b.dyn.leagues[c].m)
               and np.array_equal(a.dyn.leagues[c].P, b.dyn.leagues[c].P) for c in a.dyn.leagues)


def test_no_mix_ignores_xg_columns():
    h = promo_history()
    a, b = gb(), gb(usage="研究实验")
    a.update(h, CUT)
    b.update(with_xg(h, shift=1.0), CUT)
    assert same_dyn(a, b) and np.array_equal(np.vstack(a.rows), np.vstack(b.rows), equal_nan=True)
    assert "Understat xG（研究）" not in b.data


def test_mix_with_xg_equal_goals_is_unchanged():
    h = promo_history()
    a, b = gb(), gb(usage="研究实验", xg_mix=0.5)
    a.update(h, CUT)
    b.update(with_xg(h, shift=0.0), CUT)
    assert same_dyn(a, b) and np.array_equal(np.vstack(a.rows), np.vstack(b.rows), equal_nan=True)


def test_mix_changes_only_dyn_update():
    h = with_xg(promo_history(), shift=0.8)
    a, b = gb(usage="研究实验"), gb(usage="研究实验", xg_mix=0.5)
    a.update(h, CUT)
    b.update(h, CUT)
    assert not same_dyn(a, b)                                    # dyn 的状态变了
    for c in a.dyn.leagues:                                      # dyn 记下的还是真实比分（只有赛前 λ 不同）
        ra, rb = a.dyn.leagues[c].rec, b.dyn.leagues[c].rec
        assert [(r[0], r[3], r[4]) for r in ra] == [(r[0], r[3], r[4]) for r in rb]
    assert a.elo.ratings == b.elo.ratings and a.pi.r == b.pi.r
    assert a.goals == b.goals and a.results == b.results         # 树的目标是真实进球
    names = b.names
    ra_, rb_ = np.vstack(a.rows), np.vstack(b.rows)
    diff = ~((ra_ == rb_) | (np.isnan(ra_) & np.isnan(rb_)))     # 两边都是空值（没给休息天数）算一样
    changed = {names[i] for i in np.where(diff.any(axis=0))[0]}
    assert changed and all(n.startswith("dyn_") for n in changed)   # 特征里只有 dyn 的几列不同


def test_mix_equals_hand_fed_dyn():
    """单独的 dyn 每场喂 0.5 × 进球 + 0.5 × xG（rec 记真实比分），和 gbdt 里的 dyn 一样。"""
    h = with_xg(promo_history(), shift=0.8).sort_values(["kickoff_earliest_utc", "match_id"])
    g = gb(usage="研究实验", xg_mix=0.5)
    g.update(h, CUT)
    d = DynamicModel(**DYN)
    for r in h.itertuples(index=False):
        d.processed.add(int(r.match_id))
        lg = d._prepare(r.competition, int(r.season), r.home_team, r.away_team, r.kickoff_earliest_utc)
        if lg is not None:
            t = (0.5 * r.home_goals + 0.5 * r.us_home_xg, 0.5 * r.away_goals + 0.5 * r.us_away_xg)
            d._observe(lg, r.kickoff_earliest_utc, r.home_team, r.away_team, int(r.home_goals), int(r.away_goals), t)
    for c in d.leagues:
        assert np.allclose(g.dyn.leagues[c].m, d.leagues[c].m) and np.allclose(g.dyn.leagues[c].P, d.leagues[c].P)
        assert g.dyn.leagues[c].rec == d.leagues[c].rec


def test_missing_xg_falls_back_to_goals():
    h = with_xg(promo_history(), shift=0.8)
    h["us_away_xg"] = np.nan                                     # 只缺一队也算没有
    a, b = gb(), gb(usage="研究实验", xg_mix=0.5)
    a.update(h, CUT)
    b.update(h, CUT)
    assert same_dyn(a, b)
    c = gb(usage="研究实验", xg_mix=0.5)
    c.update(h.drop(columns=["us_home_xg", "us_away_xg"]), CUT)  # 根本没有这两列也行
    assert same_dyn(a, c)


def test_dyn_observe_target_by_hand():
    """dyn 的 _observe：target 等于真实进球时和不给一样；rec 一直记真实比分。"""
    h = promo_history().sort_values(["kickoff_earliest_utc", "match_id"])
    d1, d2 = DynamicModel(**DYN), DynamicModel(**DYN)
    d1.update(h.head(200), CUT)
    d2.update(h.head(200), CUT)
    r = h.iloc[200]
    args = (r.competition, int(r.season), r.home_team, r.away_team, r.kickoff_earliest_utc)
    l1, l2 = d1._prepare(*args), d2._prepare(*args)
    d1._observe(l1, r.kickoff_earliest_utc, r.home_team, r.away_team, 2, 1)
    d2._observe(l2, r.kickoff_earliest_utc, r.home_team, r.away_team, 2, 1, (2.0, 1.0))
    assert np.array_equal(l1.m, l2.m) and l1.rec == l2.rec
    d2._observe(l2, r.kickoff_earliest_utc, r.home_team, r.away_team, 0, 0, (1.4, 0.3))
    assert l2.rec[-1][3:] == (0, 0)


def test_mix_guards_and_registry():
    with pytest.raises(ValueError):
        gb(xg_mix=0.5)                                           # A版输入不能用 Understat xG
    with pytest.raises(ValueError):
        gb(usage="B版输入", xg_mix=0.5)
    with pytest.raises(ValueError):
        gb(usage="研究实验", xg_mix=1.5)
    base, r, x = model_config("gb_l7"), model_config("gb_l7_r"), model_config("gb_xg_r")
    assert "xg_mix" not in base and "xg_mix" not in r
    assert {k: v for k, v in x.items() if k not in ("usage", "xg_mix")} == base and x["xg_mix"] == 0.5
    m = build_model("gb_xg_r")
    assert m.xg_mix == 0.5 and m.usage == "研究实验"
    assert m.data == ("比分", "射门、xG", "Understat xG（研究）") and not hasattr(m, "pre_match_data")
    assert build_model("gb_l7").xg_mix == 0.0
