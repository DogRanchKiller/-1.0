"""评分指标（需求文档第 5 节“指标”）：log loss、Brier、RPS、校准表、ECE。

这些指标都是“分数越小越好”（校准表除外，它是一张表）。它们回答的问题不一样：
- log loss（对数损失）：真实发生的那个结果，你给了多少概率？给得越少扣分越重，给 0 会扣到无穷大。
- Brier 分数：每个结果“预测概率”和“实际 0 或 1”差多少，差的平方加起来。
- RPS（排序概率分数）：胜平负是有顺序的（主胜 → 平 → 客胜）。真实是主胜时，把概率错放在“平”比错放在“客胜”扣得少。
- 校准：模型说 60% 的事情，实际是不是大约六成发生了。

输入的约定：
- probs：每场比赛一行，每列是一个结果的概率，例如胜平负 [[0.5, 0.3, 0.2], ...]，顺序 [主胜, 平, 客胜]。
  两个结果的玩法（BTTS 是/否、大小球）也可以只给“是”的概率，一维 [0.6, 0.3, ...]。
- outcomes：每场实际发生的结果，用列序号表示（0 = 主胜，1 = 平，2 = 客胜）；一维概率时用 1 = 发生、0 = 没发生。
  数据库里的 'H' / 'D' / 'A' 用 encode_results 换成 0 / 1 / 2。
- 每个函数有两个版本：xxx_per_match 返回每场一个分数（第 5 节 bootstrap 比较两个模型时要用每场的分数差），
  不带 per_match 的返回所有比赛的平均分。
"""
from functools import lru_cache

import numpy as np
import pandas as pd

from src.common.config import load_settings

RESULT_ORDER = ("H", "D", "A")   # 胜平负的顺序：主胜、平、客胜（和 src/common/markets.result_probs 一致）


@lru_cache(maxsize=1)
def _calc_settings() -> dict:
    """读 settings.yaml 的 calc 段（只读一次）。"""
    return load_settings()["calc"]


def encode_results(results) -> np.ndarray:
    """'H' / 'D' / 'A'（数据库 match_stats.result 列）→ 0 / 1 / 2。"""
    index = {r: i for i, r in enumerate(RESULT_ORDER)}
    try:
        return np.array([index[r] for r in results])
    except KeyError as e:
        raise ValueError(f"结果只能是 H / D / A，收到 {e.args[0]!r}") from None


def _prepare(probs, outcomes) -> tuple[np.ndarray, np.ndarray]:
    """检查输入并统一成二维：一维的“是”的概率 p 换成两列 [1−p, p]，这样 outcomes 的 1 正好对应第 2 列。"""
    p = np.asarray(probs, dtype=float)
    y = np.asarray(outcomes)
    if p.ndim == 1:
        p = np.column_stack([1 - p, p])
    if p.ndim != 2 or len(p) != len(y) or len(y) == 0:
        raise ValueError(f"概率要每场一行，和结果的场数一样；收到概率 {np.shape(probs)}、结果 {np.shape(outcomes)}")
    if np.isnan(p).any() or (p < 0).any() or (p > 1).any():
        raise ValueError("概率要在 0 到 1 之间，不能有空值")
    worst = float(np.abs(p.sum(axis=1) - 1).max())
    if worst > _calc_settings()["prob_sum_tol"]:
        raise ValueError(f"每场的概率加起来要等于 1，最大差了 {worst:.2e}")
    if not np.isin(y, np.arange(p.shape[1])).all():
        raise ValueError(f"结果要是 0 到 {p.shape[1] - 1} 的整数")
    return p, y.astype(int)


def one_hot(outcomes, n_classes: int) -> np.ndarray:
    """结果 → 0/1 表：发生的那一列是 1，其他是 0。例：结果 0（主胜）→ [1, 0, 0]。"""
    y = np.asarray(outcomes, dtype=int)
    return np.eye(n_classes)[y]


def log_loss_per_match(probs, outcomes, eps: float | None = None) -> np.ndarray:
    """每场的 log loss = −ln(真实结果的预测概率)。

    例：真实主胜，主胜概率 0.5 → −ln 0.5 = 0.693；主胜概率 0.9 → 0.105；0.1 → 2.303。
    概率是 0 时 ln 0 = 负无穷，所以先把概率截到不低于 eps（settings.yaml 的 calc.logloss_eps，1e-15），
    这时一场扣 −ln(1e-15) ≈ 34.5 分，很重但不是无穷大，平均分还能算。只截真实结果那一格，其他格不动。
    """
    p, y = _prepare(probs, outcomes)
    eps = _calc_settings()["logloss_eps"] if eps is None else eps
    return -np.log(np.clip(p[np.arange(len(y)), y], eps, 1.0))


def log_loss(probs, outcomes, eps: float | None = None) -> float:
    """平均 log loss（越小越好）。"""
    return float(log_loss_per_match(probs, outcomes, eps).mean())


def brier_per_match(probs, outcomes) -> np.ndarray:
    """每场的 Brier 分数。

    多个结果（胜平负）：每个结果 (预测概率 − 实际 0 或 1)² 加起来，范围 0–2。
      例：预测 [0.5, 0.3, 0.2]，真实主胜 [1, 0, 0] → 0.25 + 0.09 + 0.04 = 0.38。
    一维“是”的概率（BTTS 等）：只算一项 (p − 实际)²，范围 0–1，这是二分类常用的写法。
      例：预测 BTTS 是 0.6，真的两队都进了 → (0.6 − 1)² = 0.16。
      注意两种写法差 2 倍：同一个二分类问题写成两列 [0.4, 0.6] 会得到 0.32。
    """
    one_dim = np.asarray(probs).ndim == 1
    p, y = _prepare(probs, outcomes)
    if one_dim:
        return (p[:, 1] - y) ** 2
    return ((p - one_hot(y, p.shape[1])) ** 2).sum(axis=1)


def brier(probs, outcomes) -> float:
    """平均 Brier 分数（越小越好）。"""
    return float(brier_per_match(probs, outcomes).mean())


def rps_per_match(probs, outcomes) -> np.ndarray:
    """每场的 RPS（排序概率分数），结果要按顺序排（主胜、平、客胜）。

    做法：把预测和实际都改成“累计”的：第 1 项 = 主胜，第 2 项 = 主胜 + 平（第 3 项一定是 1，不用算）。
    每项差的平方加起来，再除以 (结果个数 − 1)，范围 0–1。
    例：预测 [0.5, 0.3, 0.2]，真实主胜。累计预测 [0.5, 0.8]，累计实际 [1, 1]。
        RPS = ((0.5 − 1)² + (0.8 − 1)²) ÷ 2 = (0.25 + 0.04) ÷ 2 = 0.145。
    """
    p, y = _prepare(probs, outcomes)
    cum_p = np.cumsum(p, axis=1)[:, :-1]
    cum_y = np.cumsum(one_hot(y, p.shape[1]), axis=1)[:, :-1]
    return ((cum_p - cum_y) ** 2).sum(axis=1) / (p.shape[1] - 1)


def rps(probs, outcomes) -> float:
    """平均 RPS（越小越好）。"""
    return float(rps_per_match(probs, outcomes).mean())


def calibration_table(probs, happened, n_bins: int | None = None) -> pd.DataFrame:
    """校准表：把预测概率分成等宽的箱（10 个箱就是 0–10%、10–20%……），每个箱里比较“平均预测”和“实际发生的比例”。

    probs：一维，每场一个概率，例如每场的主胜概率；happened：一维，这件事发生了是 1，没发生是 0。
    胜平负三个结果一起看时，把三列概率和三列 0/1 都拉平成一维再传进来（每场贡献 3 个点）。
    每个箱：[下限, 上限)，最后一个箱包含 1.0。空箱也列出来，场数为 0，平均值为空。

    返回的表每行一个箱：bin_low 下限、bin_high 上限、n 场数、mean_pred 平均预测概率、observed 实际发生比例、
    gap = observed − mean_pred（正数 = 模型低估了，负数 = 高估了）。校准好的模型每个箱的 gap 都接近 0。
    """
    p = np.asarray(probs, dtype=float)
    y = np.asarray(happened, dtype=float)
    if p.ndim != 1 or p.shape != y.shape or len(p) == 0:
        raise ValueError("校准表要一维的概率和一维的 0/1 结果，长度一样")
    if np.isnan(p).any() or (p < 0).any() or (p > 1).any() or not np.isin(y, (0, 1)).all():
        raise ValueError("概率要在 0 到 1 之间，结果只能是 0 或 1")
    n_bins = _calc_settings()["calibration_bins"] if n_bins is None else n_bins
    which = np.minimum((p * n_bins).astype(int), n_bins - 1)   # 第几个箱；p = 1.0 放进最后一个箱
    rows = []
    for b in range(n_bins):
        inside = which == b
        n = int(inside.sum())
        rows.append({"bin_low": b / n_bins, "bin_high": (b + 1) / n_bins, "n": n,
                     "mean_pred": p[inside].mean() if n else np.nan,
                     "observed": y[inside].mean() if n else np.nan})
    table = pd.DataFrame(rows)
    table["gap"] = table["observed"] - table["mean_pred"]
    return table


def ece(probs, happened, n_bins: int | None = None) -> float:
    """ECE（期望校准误差）：校准表里每个箱 |gap| 按场数加权平均。0 = 完全校准；0.03 = 平均差 3 个百分点。

    例：箱 A 10 场、平均预测 0.25、实际 0.20；箱 B 30 场、平均预测 0.65、实际 0.70。
        ECE = (10 × 0.05 + 30 × 0.05) ÷ 40 = 0.05。
    注意箱分得越细、场数越少，ECE 越容易因为运气变大，只适合同样分箱、同样场数之间比较。
    """
    t = calibration_table(probs, happened, n_bins)
    t = t[t["n"] > 0]
    return float((t["gap"].abs() * t["n"]).sum() / t["n"].sum())
