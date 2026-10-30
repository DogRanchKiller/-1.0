"""评分指标的手算校验：log loss、Brier、RPS、校准表、ECE（src/evaluate/metrics.py）。

每个例子的注释写了手算过程。结果编号：0 = 主胜，1 = 平，2 = 客胜。
"""
import math

import numpy as np
import pytest

from src.common import markets
from src.common.config import load_settings
from src.evaluate import metrics

approx = pytest.approx

# 两场比赛的预测（主胜、平、客胜）和实际结果
PROBS = [[0.5, 0.3, 0.2],    # 第 1 场，实际主胜
         [0.2, 0.3, 0.5]]    # 第 2 场，实际平
RESULTS = [0, 1]


def test_encode_results():
    assert list(metrics.encode_results(["H", "D", "A", "H"])) == [0, 1, 2, 0]
    with pytest.raises(ValueError):
        metrics.encode_results(["X"])


def test_log_loss_by_hand():
    # 第 1 场实际主胜，主胜给了 0.5：−ln 0.5 = 0.6931
    # 第 2 场实际平，平给了 0.3：−ln 0.3 = 1.2040
    # 平均 = (0.6931 + 1.2040) ÷ 2 = 0.9486
    assert metrics.log_loss_per_match(PROBS, RESULTS) == approx([0.693147, 1.203973], abs=1e-6)
    assert metrics.log_loss(PROBS, RESULTS) == approx(0.948560, abs=1e-6)


def test_log_loss_zero_probability_is_clipped():
    # 实际客胜，客胜却给了 0：ln 0 是负无穷，截到 1e-15（settings.yaml calc.logloss_eps）
    # → 扣 −ln(1e-15) = 15 × ln 10 = 34.539 分，不是无穷大
    assert load_settings()["calc"]["logloss_eps"] == 1e-15
    assert metrics.log_loss([[0.6, 0.4, 0.0]], [2]) == approx(15 * math.log(10))
    assert math.isfinite(metrics.log_loss([[1.0, 0.0, 0.0]], [1]))


def test_brier_by_hand():
    # 第 1 场：预测 [0.5, 0.3, 0.2]，实际 [1, 0, 0]
    #   (0.5−1)² + (0.3−0)² + (0.2−0)² = 0.25 + 0.09 + 0.04 = 0.38
    # 第 2 场：预测 [0.2, 0.3, 0.5]，实际 [0, 1, 0]
    #   0.2² + (0.3−1)² + 0.5² = 0.04 + 0.49 + 0.25 = 0.78
    # 平均 = (0.38 + 0.78) ÷ 2 = 0.58
    assert metrics.brier_per_match(PROBS, RESULTS) == approx([0.38, 0.78])
    assert metrics.brier(PROBS, RESULTS) == approx(0.58)


def test_brier_binary_by_hand():
    # BTTS 只给“是”的概率：0.6 实际是 → (0.6−1)² = 0.16；0.3 实际否 → 0.3² = 0.09。平均 0.125
    assert metrics.brier([0.6, 0.3], [1, 0]) == approx(0.125)
    # 同一件事写成两列 [否, 是]，每场要加两项，分数正好翻倍
    assert metrics.brier([[0.4, 0.6], [0.7, 0.3]], [1, 0]) == approx(0.25)


def test_rps_by_hand():
    # 第 1 场：累计预测 [0.5, 0.5+0.3=0.8]，实际主胜，累计实际 [1, 1]
    #   ((0.5−1)² + (0.8−1)²) ÷ 2 = (0.25 + 0.04) ÷ 2 = 0.145
    # 第 2 场：累计预测 [0.2, 0.5]，实际平，累计实际 [0, 1]
    #   ((0.2−0)² + (0.5−1)²) ÷ 2 = (0.04 + 0.25) ÷ 2 = 0.145
    assert metrics.rps_per_match(PROBS, RESULTS) == approx([0.145, 0.145])
    assert metrics.rps(PROBS, RESULTS) == approx(0.145)


def test_rps_cares_about_order_brier_does_not():
    # 实际主胜。两个预测都只给主胜 0.4，剩下 0.6 一个放在“平”、一个放在“客胜”。
    # Brier 两个一样：0.36 + 0.36 + 0 = 0.72
    # RPS：放在平 → 累计 [0.4, 1.0]，((0.4−1)² + 0) ÷ 2 = 0.18
    #      放在客胜 → 累计 [0.4, 0.4]，((0.4−1)² + (0.4−1)²) ÷ 2 = 0.36
    # “差一点就赢”（平）比“完全反了”（客胜）扣分少，这就是 RPS 考虑顺序的意思。
    near, far = [[0.4, 0.6, 0.0]], [[0.4, 0.0, 0.6]]
    assert metrics.brier(near, [0]) == approx(metrics.brier(far, [0])) == approx(0.72)
    assert metrics.rps(near, [0]) == approx(0.18)
    assert metrics.rps(far, [0]) == approx(0.36)


def test_perfect_and_uniform_predictions():
    # 性质：完全猜中（真实结果给 1）三个指标都是 0；三个结果各 1/3 时 log loss = ln 3 = 1.0986
    assert metrics.log_loss([[1, 0, 0]], [0]) == approx(0, abs=1e-12)
    assert metrics.brier([[0, 0, 1]], [2]) == 0
    assert metrics.rps([[0, 1, 0]], [1]) == 0
    assert metrics.log_loss([[1 / 3] * 3] * 3, [0, 1, 2]) == approx(math.log(3))


def test_metrics_accept_matrix_output():
    # 和比分矩阵接起来用：独立泊松 1.7 对 1.0 → 胜平负 [0.538, 0.240, 0.222]；实际主胜 → log loss = −ln 0.538
    p = markets.result_probs(markets.poisson_score_matrix(1.7, 1.0))
    assert metrics.log_loss([p], [0]) == approx(-math.log(p[0]))


@pytest.mark.parametrize("probs, outcomes", [
    ([[0.5, 0.3, 0.1]], [0]),        # 加起来不是 1
    ([[0.5, 0.5, 0.0]], [3]),        # 结果编号超出范围
    ([[0.5, 0.3, 0.2]], [0, 1]),     # 场数对不上
    ([1.2], [1]),                    # 概率大于 1
])
def test_metrics_reject_bad_input(probs, outcomes):
    with pytest.raises(ValueError):
        metrics.log_loss(probs, outcomes)


def test_calibration_table_by_hand():
    # 4 个箱（0–25%、25–50%、50–75%、75–100%）：
    #   预测 0.30、0.40 落在第 2 箱，实际 [0, 1] → 平均预测 0.35，实际 0.5，gap +0.15（低估了）
    #   预测 0.70、0.80 分别落在第 3、4 箱；1.0 也算进第 4 箱
    #   第 1 箱没有比赛：场数 0，平均值为空
    t = metrics.calibration_table([0.30, 0.40, 0.70, 0.80, 1.0], [0, 1, 1, 0, 1], n_bins=4)
    assert list(t["n"]) == [0, 2, 1, 2]
    assert np.isnan(t.loc[0, "mean_pred"])
    assert t.loc[1, "mean_pred"] == approx(0.35) and t.loc[1, "observed"] == approx(0.5)
    assert t.loc[1, "gap"] == approx(0.15)
    assert t.loc[3, "mean_pred"] == approx(0.9) and t.loc[3, "observed"] == approx(0.5)


def test_calibration_default_bins_from_settings():
    t = metrics.calibration_table([0.05, 0.55, 0.95], [0, 1, 1])
    assert len(t) == load_settings()["calc"]["calibration_bins"] == 10


def test_ece_by_hand():
    # 箱 A（20–30%）10 场，每场预测 0.25，其中 2 场发生 → 实际 0.20，差 0.05
    # 箱 B（60–70%）30 场，每场预测 0.65，其中 21 场发生 → 实际 0.70，差 0.05
    # ECE = (10 × 0.05 + 30 × 0.05) ÷ 40 = 0.05
    probs = [0.25] * 10 + [0.65] * 30
    happened = [1] * 2 + [0] * 8 + [1] * 21 + [0] * 9
    assert metrics.ece(probs, happened) == approx(0.05)


def test_ece_perfectly_calibrated_is_zero():
    # 预测 0.2 的 10 场里正好 2 场发生，预测 0.8 的 10 场里正好 8 场发生 → 完全校准，ECE = 0
    probs = [0.2] * 10 + [0.8] * 10
    happened = [1] * 2 + [0] * 8 + [1] * 8 + [0] * 2
    assert metrics.ece(probs, happened) == approx(0, abs=1e-12)
