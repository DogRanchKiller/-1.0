"""计算校验（需求文档第 16 节“自动检查”“代码算得对不对”）：去水、亚盘/大小球结算、比分矩阵。

这些地方算错了不会报错，只会让回测和机器人的结果悄悄出错，所以每个函数都用手算的小例子核对。
每个例子的注释里写了手算过程，不会编程也可以拿计算器跟着算一遍。
评分指标（log loss、Brier、RPS、校准）的检查在 tests/test_metrics.py。

运行方法：.venv\\Scripts\\python -m pytest
"""
import math

import numpy as np
import pytest
from scipy.stats import poisson

from src.common import markets, odds, settlement
from src.common.config import load_settings

approx = pytest.approx


# ───────────────────────── 一、赔率和去水 ─────────────────────────

def test_implied_probs_and_overround():
    # 隐含概率 = 1 ÷ 赔率：1/2.0 = 0.5，1/3.5 = 0.285714，1/4.0 = 0.25
    assert odds.implied_probs([2.0, 3.5, 4.0]) == approx([0.5, 0.2857142857, 0.25])
    # 赔率倒数之和 = 0.5 + 0.285714 + 0.25 = 1.035714（= 29/28），抽水约 3.6%
    assert odds.overround([2.0, 3.5, 4.0]) == approx(29 / 28)


def test_devig_proportional_hand_example():
    # 三项赔率 2.0 / 3.5 / 4.0 按比例去水：
    # 隐含概率写成分数：1/2 = 14/28，1/3.5 = 8/28，1/4 = 7/28，加起来 29/28。
    # 每个除以 29/28 → 14/29 = 0.4828，8/29 = 0.2759，7/29 = 0.2414，加起来正好 1。
    p = odds.devig_proportional([2.0, 3.5, 4.0])
    assert p == approx([14 / 29, 8 / 29, 7 / 29])
    assert p.sum() == approx(1)


def test_devig_two_way():
    # 两项（大小球、亚盘）：1.90 / 1.90 → 隐含概率各 0.5263，和 1.0526；各除以 1.0526 → 0.5、0.5
    assert odds.devig_proportional([1.90, 1.90]) == approx([0.5, 0.5])


def test_shin_two_way_equals_additive():
    # 已知的性质：只有两个选项时，Shin 法 = “两边各减去同样多的概率”。
    # 赔率 1.5 / 2.5：隐含概率 0.6667、0.4，和 1.0667，多出 0.0667，两边各减一半 0.0333
    # → 0.6333、0.3667
    assert odds.devig_shin([1.5, 2.5]) == approx([0.6333333333, 0.3666666667])


def test_power_hand_example():
    # 赔率 5/3（= 1.667）和 1.25：隐含概率 0.6、0.8，和 1.4。
    # 取 2 次方：0.36 + 0.64 = 1，正好等于 1，所以 k = 2，去水后 0.36、0.64。
    p, k = odds.devig_power([5 / 3, 1.25], return_k=True)
    assert k == approx(2)
    assert p == approx([0.36, 0.64])


def test_shin_recovers_known_truth():
    # 反过来验证：先假设真实概率 [0.5, 0.3, 0.2]、内幕比例 z = 0.03，按 Shin 模型造出庄家的价格，
    # 再用 devig_shin 从赔率解回来，应该得到原来的概率和 z。
    # Shin 模型里庄家的价格 π_i 和 s_i = √(z·p_i + (1−z)·p_i²) 成正比，且 π_i = s_i × (所有 s 之和)。
    p_true, z_true = np.array([0.5, 0.3, 0.2]), 0.03
    s = np.sqrt(z_true * p_true + (1 - z_true) * p_true ** 2)
    price = s * s.sum()
    p, z = odds.devig_shin(1 / price, return_z=True)
    assert z == approx(z_true)
    assert p == approx(p_true)


def test_zero_margin_all_methods_unchanged():
    # 没有抽水（赔率倒数之和正好 1）时，三种方法都应该原样返回隐含概率；Shin 的 z = 0，幂函数的 k = 1
    o = [2.0, 4.0, 4.0]            # 0.5 + 0.25 + 0.25 = 1
    for method in odds.METHODS:
        assert odds.devig(o, method) == approx([0.5, 0.25, 0.25])
    assert odds.devig_shin(o, return_z=True)[1] == approx(0, abs=1e-12)
    assert odds.devig_power(o, return_k=True)[1] == approx(1)


@pytest.mark.parametrize("method", odds.METHODS)
@pytest.mark.parametrize("o", [
    [2.0, 3.5, 4.0], [1.25, 6.0, 12.0], [1.01, 21.0, 41.0],   # 胜平负，含大热门
    [1.90, 1.95], [1.01, 30.0],                                  # 两项
    [2.1, 2.1], [2.6, 3.6, 3.3],                                 # 倒数之和小于 1（交易所、多家最高）
    [7.0, 8.0, 9.0, 10.0, 11.0, 12.0, 13.0],                     # 很多选项
])
def test_devig_sums_to_one(method, o):
    # 性质：不管哪种方法，去水后加起来都是 1，每个概率在 0 和 1 之间
    p = odds.devig(o, method)
    assert p.sum() == approx(1, abs=1e-12)
    assert ((p > 0) & (p < 1)).all()


def test_shin_and_power_shrink_longshots_more():
    # Shin 法、幂函数法的特点：冷门的概率比按比例去水压得更多，热门的更高。
    # 2.0 / 3.5 / 4.0：按比例 客胜 0.2414；Shin ≈ 0.2389；幂函数 ≈ 0.2383
    prop = odds.devig_proportional([2.0, 3.5, 4.0])
    for method in ("shin", "power"):
        p = odds.devig([2.0, 3.5, 4.0], method)
        assert p[2] < prop[2] and p[0] > prop[0]


def test_negative_margin_signs():
    # 倒数之和小于 1：Shin 的 z 是负数（没有实际意义，但概率照样加起来等于 1），幂函数的 k < 1
    assert odds.devig_shin([2.1, 2.1], return_z=True)[1] < 0
    assert odds.devig_power([2.6, 3.6, 3.3], return_k=True)[1] < 1


def test_devig_many_matches_at_once():
    # 二维输入：每行一场，结果和一场一场算一样
    rows = [[2.0, 3.5, 4.0], [1.25, 6.0, 12.0], [2.6, 3.6, 3.3]]
    for method in odds.METHODS:
        together = odds.devig(rows, method)
        assert together.shape == (3, 3)
        for i, r in enumerate(rows):
            assert together[i] == approx(odds.devig(r, method))


def test_devig_default_comes_from_settings():
    assert load_settings()["calc"]["devig_method"] == "proportional"
    assert odds.devig([2.0, 3.5, 4.0]) == approx(odds.devig_proportional([2.0, 3.5, 4.0]))


@pytest.mark.parametrize("bad", [[0.85, 1.05], [2.0, float("nan")], [2.0], [[[2.0, 2.0]]]])
def test_devig_rejects_bad_odds(bad):
    # 香港盘写法 0.85 没换成 1.85、空值、只有一个选项、形状不对，都要报错，不能悄悄算出错的数
    with pytest.raises(ValueError):
        odds.devig(bad)


def test_fair_odds():
    # 公平赔率 = 1 ÷ 概率：0.4 → 2.5；0.25 → 4
    assert odds.fair_odds([0.4, 0.25]) == approx([2.5, 4.0])
    with pytest.raises(ValueError):
        odds.fair_odds([0.0])


# ───────────────────────── 二、亚盘、大小球结算 ─────────────────────────
# 盘口写法和数据库一样：亚盘的 line 是主队的盘口，负数 = 主队让球。

@pytest.mark.parametrize("line, halves", [
    (-0.25, (-0.5, 0.0)),    # 平半：一半 0（平手）、一半 -0.5（半球）
    (-0.75, (-1.0, -0.5)),   # 半一：一半 -0.5、一半 -1
    (0.25, (0.0, 0.5)),      # 受让平半
    (2.75, (2.5, 3.0)),      # 大小 2.75 = 一半 2.5、一半 3
    (-0.5, (-0.5, -0.5)),    # 半球盘不用拆
    (-1.0, (-1.0, -1.0)),    # 整数盘不用拆
    (0.0, (0.0, 0.0)),
])
def test_split_line(line, halves):
    assert settlement.split_line(line) == halves


def test_split_line_rejects_bad_line():
    with pytest.raises(ValueError):
        settlement.split_line(-0.3)


@pytest.mark.parametrize("score, outcome, pnl", [
    # 主队 -0.25（平半），买主队，赔率 1.90，下 1 元，一半 0.5 元下 0 盘、一半 0.5 元下 -0.5 盘：
    ((1, 0), "win", 0.90),        # 主赢 1 球：0 盘赢（0.5×0.9=0.45）、-0.5 盘赢（0.45）→ 共 +0.90
    ((1, 1), "half_loss", -0.5),  # 平局：0 盘走（退回 0.5，盈亏 0）、-0.5 盘输（−0.5）→ 共 −0.5
    ((0, 1), "loss", -1.0),       # 主负：两半都输 → −1
])
def test_ah_minus_quarter_home(score, outcome, pnl):
    r = settlement.settle_asian_handicap(*score, line=-0.25, side="home", odds=1.90)
    assert r["outcome"] == outcome
    assert r["profit"] == approx(pnl)


@pytest.mark.parametrize("score, outcome, pnl", [
    # 同一场买客队：主队 -0.25 = 客队 +0.25，客队赔率 2.00
    ((1, 0), "loss", -1.0),       # 客队输 1 球：+0 那半输、+0.5 那半也输（-1 + 0.5 < 0）
    ((1, 1), "half_win", 0.5),    # 平局：+0 那半走、+0.5 那半赢（0.5 × 1.0 = 0.5）→ +0.5
    ((0, 1), "win", 1.0),         # 客队赢：两半都赢 → +1.0
])
def test_ah_minus_quarter_away(score, outcome, pnl):
    r = settlement.settle_asian_handicap(*score, line=-0.25, side="away", odds=2.00)
    assert r["outcome"] == outcome
    assert r["profit"] == approx(pnl)


@pytest.mark.parametrize("score, outcome, pnl", [
    # 主队 -0.75（半一），赔率 1.90：一半 -0.5、一半 -1
    ((2, 0), "win", 0.90),        # 赢 2 球：-0.5 赢、-1 赢 → +0.90
    ((1, 0), "half_win", 0.45),   # 赢 1 球：-0.5 赢（+0.45）、-1 走（0）→ 赢半 +0.45
    ((1, 1), "loss", -1.0),       # 平：两半都输
])
def test_ah_minus_three_quarters(score, outcome, pnl):
    r = settlement.settle_asian_handicap(*score, line=-0.75, odds=1.90)
    assert (r["outcome"], r["profit"]) == (outcome, approx(pnl))


@pytest.mark.parametrize("score, line, outcome", [
    ((1, 0), -1.0, "push"),       # 让 1 球赢 1 球：走盘
    ((2, 1), -1.25, "half_loss"), # 让 1.25（一半 -1、一半 -1.5）赢 1 球：-1 走、-1.5 输 → 输半
    ((3, 1), -1.25, "win"),
    ((0, 0), 0.0, "push"),        # 平手盘打平：走盘
    ((0, 0), 0.5, "win"),         # 主队受让半球，打平就赢
    ((0, 1), 1.0, "push"),        # 主队受让 1 球，输 1 球：走盘
    ((0, 1), 0.75, "half_loss"),  # 受让 0.75（一半 0.5、一半 1）输 1 球：0.5 那半输、1 那半走 → 输半
])
def test_ah_more_lines(score, line, outcome):
    assert settlement.settle_asian_handicap(*score, line=line)["outcome"] == outcome


@pytest.mark.parametrize("goals, line, side, outcome, pnl", [
    # 大小球，赔率 2.00
    (3, 2.75, "over", "half_win", 0.5),   # 大 2.75 进 3 球：大 2.5 那半赢（+0.5）、大 3 那半走 → 赢半 +0.5
    (3, 2.75, "under", "half_loss", -0.5),# 同一场买小：小 2.5 那半输、小 3 那半走 → 输半 −0.5
    (2, 2.25, "over", "half_loss", -0.5), # 大 2.25 进 2 球：大 2 那半走、大 2.5 那半输 → 输半
    (2, 2.25, "under", "half_win", 0.5),
    (3, 2.5, "over", "win", 1.0),
    (2, 2.5, "over", "loss", -1.0),
    (3, 3.0, "over", "push", 0.0),        # 整数线打平：走盘
    (4, 2.75, "over", "win", 1.0),
])
def test_over_under_settlement(goals, line, side, outcome, pnl):
    r = settlement.settle_over_under(goals, 0, line, side, odds=2.00)
    assert (r["outcome"], r["profit"]) == (outcome, approx(pnl))


@pytest.mark.parametrize("line", [x / 4 for x in range(-12, 13)])
@pytest.mark.parametrize("score", [(0, 0), (1, 0), (0, 1), (2, 0), (3, 1), (0, 3)])
def test_ah_home_and_away_mirror(line, score):
    # 性质：同一场同一个盘口，主队“赢”时客队一定“输”，赢半对输半，走对走
    mirror = {"win": "loss", "half_win": "half_loss", "push": "push", "half_loss": "half_win", "loss": "win"}
    home = settlement.settle_asian_handicap(*score, line=line, side="home")["outcome"]
    away = settlement.settle_asian_handicap(*score, line=line, side="away")["outcome"]
    assert away == mirror[home]


def test_profit_table():
    # 赔率 1.90 每 1 元：赢 +0.90，赢半 +0.45，走 0，输半 −0.5，输 −1
    assert [settlement.profit(o, 1.90) for o in settlement.OUTCOMES] == approx([0.90, 0.45, 0.0, -0.5, -1.0])


def test_settlement_bad_input():
    with pytest.raises(ValueError):
        settlement.settle_asian_handicap(1, 0, -0.5, side="draw")
    with pytest.raises(ValueError):
        settlement.settle_over_under(1, 0, -2.5)
    with pytest.raises(ValueError):
        settlement.profit("win", 0.9)


# ───────────────────────── 三、比分矩阵 → 各玩法 ─────────────────────────

# 手算用的 3×3 小矩阵：行 = 主队进 0/1/2 球，列 = 客队进 0/1/2 球。九个数加起来 = 1。
#            客 0   客 1   客 2
#   主 0     0.10   0.08   0.04
#   主 1     0.15   0.12   0.03
#   主 2     0.20   0.17   0.11
SMALL = np.array([[0.10, 0.08, 0.04],
                  [0.15, 0.12, 0.03],
                  [0.20, 0.17, 0.11]])


def test_small_matrix_result_probs():
    # 主胜 = 主进球多的格子：1–0 0.15 + 2–0 0.20 + 2–1 0.17 = 0.52
    # 平   = 对角线：0–0 0.10 + 1–1 0.12 + 2–2 0.11 = 0.33
    # 客胜 = 客进球多的格子：0–1 0.08 + 0–2 0.04 + 1–2 0.03 = 0.15
    # 0.52 + 0.33 + 0.15 = 1
    assert markets.result_probs(SMALL) == approx([0.52, 0.33, 0.15])


def test_small_matrix_over_under_1_5():
    # 总进球 0 或 1 球的格子：0–0 0.10、1–0 0.15、0–1 0.08，小 1.5 = 0.33；大 1.5 = 1 − 0.33 = 0.67
    over = markets.over_under(SMALL, 1.5, "over")
    under = markets.over_under(SMALL, 1.5, "under")
    assert over["win"] == approx(0.67) and over["loss"] == approx(0.33)
    assert under["win"] == approx(0.33)


def test_small_matrix_btts():
    # 两队都进球 = 去掉第 0 行和第 0 列：1–1 0.12 + 1–2 0.03 + 2–1 0.17 + 2–2 0.11 = 0.43
    assert markets.btts(SMALL) == approx({"yes": 0.43, "no": 0.57})


def test_small_matrix_asian_handicap_minus_0_75():
    # 主队 -0.75（一半 -0.5、一半 -1）。先按净胜球（主 − 客）加起来：
    #   净胜 2 球：2–0 = 0.20                    → 两半都赢 = 赢
    #   净胜 1 球：1–0 0.15 + 2–1 0.17 = 0.32    → -0.5 赢、-1 走 = 赢半
    #   平或输：剩下的 1 − 0.20 − 0.32 = 0.48    → 输
    # 胜算 = (0.20 + 0.32×0.5) ÷ (0.20 + 0.16 + 0.48) = 0.36 ÷ 0.84 = 0.4286，公平赔率 = 1 ÷ 0.4286 = 2.333
    # 赔率 2.00 的期望盈亏 = 0.20×1 + 0.32×0.5 − 0.48×1 = 0.20 + 0.16 − 0.48 = −0.12
    r = markets.asian_handicap(SMALL, -0.75, "home", odds=2.00)
    assert (r["win"], r["half_win"], r["push"], r["half_loss"], r["loss"]) == approx((0.20, 0.32, 0, 0, 0.48))
    assert r["win_share"] == approx(0.36 / 0.84)
    assert r["fair_odds"] == approx(0.84 / 0.36)
    assert r["expected_profit"] == approx(-0.12)


def test_small_matrix_asian_handicap_away_side():
    # 同一场买客队（客队 +0.75），结果正好反过来：主队赢 → 客队输；主队赢半 → 客队输半
    r = markets.asian_handicap(SMALL, -0.75, "away")
    assert (r["win"], r["half_loss"], r["loss"]) == approx((0.48, 0.32, 0.20))


def test_small_matrix_over_2_75():
    # 大 2.75：总进球 4 球（2–2）0.11 → 赢；3 球（1–2 0.03 + 2–1 0.17 = 0.20）→ 赢半；其他 0.69 → 输
    r = markets.over_under(SMALL, 2.75, "over")
    assert (r["win"], r["half_win"], r["loss"]) == approx((0.11, 0.20, 0.69))


def test_small_matrix_draw_no_bet():
    # 0 盘（平手盘）：平局走盘。主赢 0.52 → 赢，平 0.33 → 走，主负 0.15 → 输
    r = markets.asian_handicap(SMALL, 0.0)
    assert (r["win"], r["push"], r["loss"]) == approx((0.52, 0.33, 0.15))
    # 胜算不算走盘：0.52 ÷ (0.52 + 0.15) = 0.7761
    assert r["win_share"] == approx(0.52 / 0.67)


def test_correct_score_and_top_scores():
    assert markets.correct_score(SMALL, 2, 1) == approx(0.17)
    assert markets.top_scores(SMALL, 2) == [(2, 0, approx(0.20)), (2, 1, approx(0.17))]
    with pytest.raises(ValueError):
        markets.correct_score(SMALL, 3, 0)


def test_poisson_matrix_tiny_by_hand():
    # 最小的泊松矩阵：两队预期进球都是 1，只算到 1 球（最后一格 = “1 球及以上”）。
    # 进 0 球 = e^(−1) = 0.3679；“1 球及以上” = 1 − 0.3679 = 0.6321
    # 0–0 = 0.3679² = 0.1353；1–0 = 0–1 = 0.3679 × 0.6321 = 0.2325；“1–1 及以上” = 0.6321² = 0.3996
    m = markets.poisson_score_matrix(1.0, 1.0, max_goals=1)
    e = math.exp(-1)
    assert m == approx(np.array([[e * e, e * (1 - e)], [(1 - e) * e, (1 - e) ** 2]]))
    assert m.sum() == approx(1)


def test_requirements_doc_example_1_7_vs_1_0():
    # 需求文档第 3 节的例子：主队预期进球 1.7、客队 1.0，独立泊松，每队算到 10 球。
    # 1–0 = P(主进 1) × P(客进 0) = (e^−1.7 × 1.7) × e^−1.0 = 0.3106 × 0.3679 = 0.1142
    m = markets.poisson_score_matrix(1.7, 1.0)
    assert m.shape == (11, 11)                    # 0 到 10 球
    assert m[1, 0] == approx(poisson.pmf(1, 1.7) * poisson.pmf(0, 1.0))
    assert m[1, 0] == approx(0.114, abs=5e-4) and m[1, 1] == approx(0.114, abs=5e-4)
    assert m[2, 1] == approx(0.097, abs=5e-4)
    assert markets.result_probs(m) == approx([0.538, 0.240, 0.222], abs=5e-4)
    ah = markets.asian_handicap(m, -0.75)
    assert (ah["win"], ah["half_win"], ah["loss"]) == approx((0.295, 0.243, 0.462), abs=5e-4)
    assert ah["win_share"] == approx(0.474, abs=5e-4)   # 文档写 47.4%
    assert markets.over_under(m, 2.5)["win"] == approx(0.506, abs=5e-4)
    assert markets.btts(m)["yes"] == approx(0.517, abs=5e-4)


def test_poisson_matrix_tail_is_folded_in():
    # 截断处理：最后一行 = “主队 10 球及以上”，所以矩阵加起来正好 1，不丢概率
    m = markets.poisson_score_matrix(3.5, 2.5)
    assert m.sum() == approx(1, abs=1e-12)
    assert m[10, :].sum() == approx(poisson.sf(9, 3.5))   # P(主队 ≥ 10 球)
    assert load_settings()["calc"]["max_goals"] == 10


def test_truncated_matrix_without_tail_is_rejected():
    # 只取 0–10 球、把 10 球以上直接丢掉的矩阵，预期进球大时加起来明显小于 1，要报错，不能悄悄放大
    g = np.arange(11)
    lossy = np.outer(poisson.pmf(g, 4.5), poisson.pmf(g, 4.0))   # 加起来约 0.989
    with pytest.raises(ValueError, match="尾部"):
        markets.result_probs(lossy)


@pytest.mark.parametrize("hxg, axg", [(1.7, 1.0), (0.3, 2.9), (1.35, 1.35), (3.2, 0.4)])
def test_matrix_consistency_properties(hxg, axg):
    m = markets.poisson_score_matrix(hxg, axg)
    h, d, a = markets.result_probs(m)
    assert h + d + a == approx(1)                                  # 主胜 + 平 + 客胜 = 1
    for line in [x / 4 for x in range(-10, 11)]:
        ah = markets.asian_handicap(m, line)
        assert sum(ah[o] for o in settlement.OUTCOMES) == approx(1)  # 五种结果加起来 = 1
        # 主队这边和客队这边正好反过来
        away = markets.asian_handicap(m, line, "away")
        assert (away["win"], away["half_win"], away["push"]) == approx((ah["loss"], ah["half_loss"], ah["push"]))
        # 按公平赔率下注，期望盈亏 = 0（公平赔率 = 1 ÷ 胜算 的意义）
        if 1 < ah["fair_odds"] < math.inf:
            assert markets.expected_profit(ah, ah["fair_odds"]) == approx(0, abs=1e-12)
    # 0 盘：平局全部走盘；主队 -0.5 赢 = 主胜；主队 +0.5 赢 = 主胜 + 平
    assert markets.asian_handicap(m, 0.0)["push"] == approx(d)
    assert markets.asian_handicap(m, -0.5)["win"] == approx(h)
    assert markets.asian_handicap(m, 0.5)["win"] == approx(h + d)
    # 大小球：大 + 小 = 1；大 2.5 = 总进球 ≥ 3 的格子
    for line in [0.5, 1.75, 2.25, 2.5, 3.0, 4.75]:
        over, under = markets.over_under(m, line, "over"), markets.over_under(m, line, "under")
        assert over["win"] == approx(under["loss"]) and over["half_win"] == approx(under["half_loss"])
    total = np.add.outer(np.arange(11), np.arange(11))
    assert markets.over_under(m, 2.5)["win"] == approx(m[total >= 3].sum())
    assert markets.btts(m)["yes"] == approx((1 - poisson.pmf(0, hxg)) * (1 - poisson.pmf(0, axg)))


@pytest.mark.parametrize("score", [(0, 0), (1, 0), (2, 1), (0, 2), (4, 1)])
@pytest.mark.parametrize("line", [-1.75, -1.0, -0.75, -0.25, 0.0, 0.25, 0.5, 1.25])
def test_markets_agree_with_settlement(score, line):
    # 比分确定（矩阵只有一个格子是 1）时，玩法函数的结果必须和逐场结算完全一样：两处用的是同一套规则
    m = np.zeros((6, 6))
    m[score] = 1.0
    expected = settlement.settle_asian_handicap(*score, line=line)["outcome"]
    assert markets.asian_handicap(m, line)[expected] == 1.0
    ou_line = abs(line) + 2
    expected = settlement.settle_over_under(*score, line=ou_line)["outcome"]
    assert markets.over_under(m, ou_line)[expected] == 1.0


def test_non_square_matrix_works():
    # 矩阵不必是正方形：主队算到 2 球、客队算到 1 球
    m = np.array([[0.2, 0.1], [0.3, 0.1], [0.2, 0.1]])
    assert markets.result_probs(m) == approx([0.3 + 0.2 + 0.1, 0.2 + 0.1, 0.1])


@pytest.mark.parametrize("bad", [np.full((3, 3), 0.2), -SMALL, np.array([0.5, 0.5])])
def test_bad_matrix_rejected(bad):
    with pytest.raises(ValueError):
        markets.check_score_matrix(bad)
