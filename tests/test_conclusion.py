"""预测页的结论（src/predict/conclusion.py）和融合用的赔率（src/predict/preview.py 的 fusion_odds）的自动检查。

需求文档第 5 节“预测页怎么接融合、倾向、信心等级”。不用数据库：赔率、模型预测都是手写的。
"""
import copy

import numpy as np
import pandas as pd
import pytest

from src.common.config import load_settings
from src.common.markets import (dc_matrices, fit_market_dc, matrix_probs, over_shares, over_under, result_probs)
from src.common.odds import devig
from src.predict.conclusion import FUSED, NO_ODDS, PARTIAL, conclude, fuse, leans, odds_move, season_flags
from src.predict.preview import fusion_odds


@pytest.fixture
def s():
    return copy.deepcopy(load_settings())


def fo_of(p_1x2=None, ou=None, ah=None, pinnacle=False):
    """手写一份“融合用的赔率”（和 fusion_odds 返回的样子一样）。"""
    return {"p_1x2": p_1x2, "pinnacle": pinnacle, "n_books_1x2": 1 if p_1x2 else 0, "ou": ou, "ah": ah}


def odds_rows(rows):
    return pd.DataFrame(rows, columns=["bookmaker", "market", "line", "home", "draw", "away", "over", "under"])


def model_matrix():
    return dc_matrices(np.array([1.6]), np.array([1.0]), np.array([-0.05]))[0]


# ---------- 市场比分矩阵：大小球不是 2.5 的线 ----------
def test_fit_market_dc_other_ou_line():
    """没有 2.5 球赔率时用 2.75 这种主盘口：胜平负和 2.75 上“大”的胜算都要对上。"""
    lh, la, rho = np.array([1.5, 2.2, 0.9]), np.array([1.1, 0.7, 1.4]), np.array([-0.06, 0.02, -0.1])
    ms = dc_matrices(lh, la, rho)
    target = np.array([[*result_probs(m), over_under(m, 2.75, "over")["win_share"]] for m in ms])
    fh, fa, fr = fit_market_dc(target, ou_line=2.75)
    got = dc_matrices(fh, fa, fr)
    for m, t in zip(got, target):
        assert np.abs([*result_probs(m), over_under(m, 2.75, "over")["win_share"]] - t).max() < 1e-6
    assert np.allclose(fh, lh, atol=1e-5) and np.allclose(fa, la, atol=1e-5)


def test_over_shares_matches_matrix_and_rejects_low_lines():
    lh, la = np.array([1.3, 2.0]), np.array([1.0, 0.8])
    assert np.allclose(over_shares(lh, la, 2.5), matrix_probs(lh, la)[:, 3])
    # rho 不改 2.5 以上的线：DC 矩阵的胜算和泊松算的一样
    m = dc_matrices(lh, la, np.array([-0.1, -0.1]))
    assert np.allclose(over_shares(lh, la, 3.25), [over_under(x, 3.25, "over")["win_share"] for x in m])
    with pytest.raises(ValueError):
        fit_market_dc(np.array([[0.45, 0.27, 0.28, 0.6]]), ou_line=2.25)    # 2.5 以下 rho 会影响，不支持


# ---------- 融合用的赔率 ----------
def test_fusion_odds_prefers_pinnacle(s):
    o = odds_rows([["平博 Pinnacle", "1x2", None, 2.0, 3.5, 4.0, None, None],
                   ["Bet365", "1x2", None, 1.8, 3.6, 4.6, None, None]])
    fo = fusion_odds(o, s)
    assert fo["pinnacle"] and fo["n_books_1x2"] == 1 and fo["p_1x2"] == pytest.approx(devig([2.0, 3.5, 4.0]).tolist())
    fo = fusion_odds(o.iloc[1:], s)                       # 没有平博：各家去水再平均（这里只剩一家）
    assert not fo["pinnacle"] and fo["p_1x2"] == pytest.approx(devig([1.8, 3.6, 4.6]).tolist())
    assert fusion_odds(o.iloc[0:0], s)["p_1x2"] is None


def test_fusion_odds_ou_line_choice(s):
    # 有一家开 2.5：只用 2.5（哪怕更多家开 2.75）
    o = odds_rows([["A", "ou", 2.5, None, None, None, 1.9, 1.9],
                   ["B", "ou", 2.75, None, None, None, 2.0, 1.8],
                   ["C", "ou", 2.75, None, None, None, 2.1, 1.75]])
    fo = fusion_odds(o, s)
    assert fo["ou"]["line"] == 2.5 and fo["ou"]["n_books"] == 1 and fo["ou"]["p_over"] == pytest.approx(0.5)
    # 没有 2.5：主盘口 2.75 照用
    fo = fusion_odds(o.iloc[1:], s)
    assert fo["ou"]["line"] == 2.75 and fo["ou"]["n_books"] == 2
    assert fo["ou"]["p_over"] == pytest.approx(((devig([2.0, 1.8]) + devig([2.1, 1.75])) / 2)[0])
    # 主盘口 2.25：当作没有
    low = odds_rows([["A", "ou", 2.25, None, None, None, 1.9, 1.9]])
    assert fusion_odds(low, s)["ou"] is None


def test_fusion_odds_ah_needs_main_odds_range(s):
    ok = odds_rows([["A", "ah", -0.5, 1.95, None, 1.90, None, None]])
    assert fusion_odds(ok, s)["ah"]["line"] == -0.5
    bad = odds_rows([["A", "ah", 0.0, 1.04, None, 9.05, None, None]])    # 不是真正的主盘口（赔率差太多）
    assert fusion_odds(bad, s)["ah"] is None


# ---------- 融合 ----------
TARGET = [0.45, 0.27, 0.28]


def test_fuse_both_odds_w1_is_market(s):
    f = fuse(model_matrix(), fo_of(TARGET, {"line": 2.5, "p_over": 0.52, "n_books": 2}), s)
    assert f["status"] == FUSED and f["w"] == 1
    assert f["p_1x2"] == pytest.approx(TARGET, abs=1e-6) and f["over25"] == pytest.approx(0.52, abs=1e-6)
    assert f["fit_error"] < 1e-6 and 0 < f["btts"] < 1


def test_fuse_mixes_with_model_when_w_below_1(s):
    s["fusion"]["w"] = 0.5
    mm = model_matrix()
    fo = fo_of(TARGET, {"line": 2.5, "p_over": 0.52, "n_books": 2})
    f = fuse(mm, fo, s)
    want = 0.5 * np.array(TARGET) + 0.5 * result_probs(mm)       # 胜平负是矩阵的线性函数，矩阵混合 = 概率混合
    assert f["p_1x2"] == pytest.approx(want.tolist(), abs=1e-6)
    # 模型只给胜平负（没有比分矩阵）：只能用市场矩阵
    assert fuse([0.6, 0.2, 0.2], fo, s)["p_1x2"] == pytest.approx(TARGET, abs=1e-6)


def test_fuse_partial_and_no_odds(s):
    f = fuse(model_matrix(), fo_of(TARGET), s)                  # 只有胜平负
    assert f["status"] == PARTIAL and f["p_1x2"] == pytest.approx(TARGET) and f["over25"] is None and f["btts"] is None
    mm = model_matrix()
    f = fuse(mm, fo_of(), s)                                     # 没有赔率：模型自己的
    assert f["status"] == NO_ODDS and f["p_1x2"] == pytest.approx(result_probs(mm).tolist())
    assert f["over25"] == pytest.approx(over_under(mm, 2.5, "over")["win_share"])
    assert fuse(None, fo_of(), s)["status"] is None              # 两边都没有


# ---------- 倾向 ----------
def lean_input(p, over25=None, btts=None, matrix=None):
    return {"p_1x2": p, "over25": over25, "btts": btts, "matrix": matrix}


def test_leans_thresholds(s):
    ln = leans(lean_input([0.45, 0.27, 0.28], over25=0.53, btts=0.47), None, s)
    assert ln["result"] == {"side": 0, "p": 0.45, "lean": True}            # 0.45 − 0.28 = 0.17 ≥ 0.04
    assert ln["draw"]["risk"] is False                                       # 0.27 < 0.28
    assert ln["over25"] == {"side": "over", "p": 0.53, "lean": True}
    assert ln["btts"]["side"] == "no" and ln["btts"]["p"] == pytest.approx(0.53) and ln["btts"]["lean"]
    assert ln["ah"] is None                                                  # 没有盘口
    ln = leans(lean_input([0.36, 0.33, 0.31], over25=0.515), None, s)
    assert ln["result"]["lean"] is False and ln["draw"]["risk"] is True     # 差 0.03 不够；平局 0.33 ≥ 0.28
    assert ln["over25"]["lean"] is False and ln["btts"] is None             # 差 0.015 不够；没有双方进球
    s["lean"]["result"] = None                                               # 门槛是 null：一律不够
    assert leans(lean_input([0.7, 0.2, 0.1]), None, s)["result"]["lean"] is False
    assert leans(lean_input(None), None, s)["result"] is None


def test_leans_ah_from_matrix(s):
    m = np.zeros((4, 4))
    m[2, 0], m[1, 0], m[1, 1], m[0, 1] = 0.3, 0.25, 0.25, 0.2
    ln = leans(lean_input(result_probs(m).tolist(), matrix=m), -0.5, s)
    assert ln["ah"] == {"side": "home", "p": pytest.approx(0.55), "line": -0.5, "lean": True}   # 主让半球：赢 1 球以上
    ln = leans(lean_input(result_probs(m).tolist(), matrix=m), -1.5, s)
    assert ln["ah"]["side"] == "away" and ln["ah"]["p"] == pytest.approx(0.7)                    # 主让球半：只有 2:0 赢


# ---------- 信心等级、不确定因素 ----------
def test_conclude_levels_and_flags(s):
    mm = model_matrix()
    mp = result_probs(mm).tolist()
    fo = fo_of(TARGET, {"line": 2.5, "p_over": 0.52, "n_books": 2}, {"line": -0.5, "home": 1.9, "away": 1.95})
    c = conclude(mm, mp, fo, fo_of([0.45, 0.27, 0.28]), {"early": True, "promoted": False}, s)
    assert c["status"] == FUSED and c["level"] == 3 and c["level_name"].startswith("●●●")    # 现在没有算数的因素
    assert c["flags"]["early"] and c["flags"]["no_pinnacle"] and not c["flags"]["odds_move"]
    assert "赛季前几轮" in c["uncertain"] and c["leans"]["ah"] is not None
    gap = max(abs(a - b) for a, b in zip(mp, TARGET))
    assert c["flags"]["gap"] == (gap >= s["confidence_study"]["gap"])
    # 算数的因素：碰上一个降一级
    s["confidence"]["downgrade_factors"] = ["early", "no_pinnacle"]
    assert conclude(mm, mp, fo, fo, {"early": True}, s)["level"] == 1
    # 没有赔率：降一级，分歧、没有平博不算
    s["confidence"]["downgrade_factors"] = []
    c = conclude(mm, mp, fo_of(), fo_of(), {}, s)
    assert c["status"] == NO_ODDS and c["level"] == 2 and not c["flags"]["gap"] and not c["flags"]["no_pinnacle"]
    # 什么都没有：没有结论
    assert conclude(None, None, fo_of(), fo_of(), {}, s)["level"] is None


def test_conclude_gap_and_odds_move(s):
    fo = fo_of(TARGET, pinnacle=True)
    c = conclude([0.6, 0.2, 0.2], [0.6, 0.2, 0.2], fo, fo_of([0.52, 0.24, 0.24]), {}, s)
    assert c["flags"]["gap"] and c["flags"]["odds_move"] and not c["flags"]["no_pinnacle"]
    assert c["odds_move"] == pytest.approx(0.07)
    assert c["uncertain"] == ["模型和赔率分歧大", "赔率最近大幅变动"]
    assert odds_move(TARGET, None) is None


def test_conclude_rejects_unknown_factor(s):
    s["confidence"]["downgrade_factors"] = ["weather"]
    with pytest.raises(ValueError, match="weather"):
        conclude(None, None, fo_of(), fo_of(), {}, s)


def test_season_flags(s):
    cols = ["match_id", "competition", "season", "match_date", "kickoff_utc", "home_team", "away_team"]
    hist = pd.DataFrame([[1, "E0", 2025, "2025-08-10", "2025-08-10 14:00", "A", "B"],
                         [2, "E0", 2025, "2025-08-17", "2025-08-17 14:00", "C", "D"]], columns=cols)
    targets = pd.DataFrame([[3, "E0", 2026, "2026-08-10", "2026-08-10 14:00", "A", "B"],
                            [4, "E0", 2026, "2026-08-10", "2026-08-10 14:00", "C", "E"]], columns=cols)
    out = season_flags(hist, targets, s)
    assert out[3] == {"early": True, "promoted": False} and out[4]["promoted"] is True     # E 上赛季不在
    assert season_flags(hist, targets.iloc[0:0], s) == {}
