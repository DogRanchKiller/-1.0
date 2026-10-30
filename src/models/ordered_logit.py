"""有序 logit：把“一个实力差的数”换成胜平负三个概率（Elo 用它；以后 pi-rating 等评分也可以用）。

为什么需要这一步：Elo 公式本身只给“预期得分”（赢 = 1、平 = 0.5、输 = 0 的平均），例如 0.6。
0.6 可以是“60% 赢、40% 输”，也可以是“45% 赢、30% 平、25% 输”，平局占多少它不知道。所以要另外拟合一个换算。

为什么选有序 logit（不按 Elo 差分箱数频率）：
- 胜平负有顺序：实力差越大，主胜越多、客胜越少，平局在两队差不多时最多。有序 logit 正好是这个形状，而且一定是平滑、单调的；
  分箱的话，箱边上差 1 分概率就跳一下，箱分细了每箱场数少、运气大，分粗了又不准，还要多选一个“分几箱”。
- 只有 3 个参数（斜率、两个切点），5 年 1900 场左右（一个联赛）拟合很稳，每周重新拟合一次也很快。
- 足球文献里用 Elo 差做有序 logit 预测胜平负是常见做法（例如 Hvattum & Arntzen 2010 用 Elo 差做有序 logit）。

公式（x = 主队评分 + 主场优势 − 客队评分，再除以 400，让数字在 −1 到 1 左右）：
    把结果按顺序排成 客胜 < 平 < 主胜。
    P(客胜)      = sigmoid(c1 − b·x)
    P(客胜或平)  = sigmoid(c2 − b·x)       （c2 > c1）
    所以 P(平) = sigmoid(c2 − b·x) − sigmoid(c1 − b·x)，P(主胜) = 1 − sigmoid(c2 − b·x)。
    sigmoid(z) = 1 ÷ (1 + e^(−z))，把任何数压到 0–1 之间。
    b > 0：实力差越大主胜越多。c1、c2 管“平局有多宽”：c2 − c1 越大，平局越多。
拟合：找 b、c1、c2，让历史比赛“真实结果的概率”连乘起来最大（最大似然，等价于让 log loss 最小）。
"""
from __future__ import annotations

import numpy as np
from scipy.optimize import minimize
from scipy.special import expit, log_expit

SCALE = 400.0   # 评分差除以这个数再进公式（和 Elo 公式里的 400 一样），只为了数字好看，不影响结果


def _unpack(theta: np.ndarray) -> tuple[float, float, float]:
    """内部参数 → (b, c1, c2)。c2 = c1 + e^g，保证 c2 一定大于 c1（平局概率不会是负数）。"""
    b, c1, g = theta
    return b, c1, c1 + np.exp(g)


def _neg_loglik(theta: np.ndarray, x: np.ndarray, y: np.ndarray) -> float:
    """负的对数似然 = 每场 −ln(真实结果的概率) 加起来。y：0 = 客胜，1 = 平，2 = 主胜。"""
    b, c1, c2 = _unpack(theta)
    z1, z2 = c1 - b * x, c2 - b * x
    # 用 log_expit 直接算 ln(sigmoid)，概率很小的时候也不会算成 ln 0
    ll_away = log_expit(z1)
    ll_home = log_expit(-z2)                       # 1 − sigmoid(z) = sigmoid(−z)
    p_draw = np.clip(expit(z2) - expit(z1), 1e-15, 1.0)
    ll = np.where(y == 0, ll_away, np.where(y == 2, ll_home, np.log(p_draw)))
    return -float(ll.sum())


def fit_ordered_logit(diff: np.ndarray, outcome: np.ndarray, start: np.ndarray | None = None) -> np.ndarray:
    """拟合有序 logit。diff：每场的评分差（含主场优势，没除 400）；outcome：0 = 主胜、1 = 平、2 = 客胜（和 metrics.py 一样）。

    返回内部参数 theta（给 result_probs 用，也可以下次拟合时当起点 start，加快速度）。
    """
    x = np.asarray(diff, dtype=float) / SCALE
    y = 2 - np.asarray(outcome, dtype=int)         # 换成 0 = 客胜、1 = 平、2 = 主胜（按“主队越强越往上”的顺序）
    if start is None:
        start = np.array([1.0, -0.6, np.log(1.1)])  # 大约 主胜 45%、平 27%、客胜 28% 的起点
    res = minimize(_neg_loglik, start, args=(x, y), method="L-BFGS-B")
    return res.x


def result_probs(theta: np.ndarray, diff) -> np.ndarray:
    """评分差 → 胜平负概率，每场一行 [主胜, 平, 客胜]。diff 可以是一个数或一组数。"""
    b, c1, c2 = _unpack(np.asarray(theta, dtype=float))
    x = np.atleast_1d(np.asarray(diff, dtype=float)) / SCALE
    p_away = expit(c1 - b * x)
    p_away_or_draw = expit(c2 - b * x)
    return np.column_stack([1 - p_away_or_draw, p_away_or_draw - p_away, p_away])


def describe(theta: np.ndarray) -> dict:
    """参数写成人能看的样子：斜率、两个切点，以及评分差为 0（两队加主场优势后一样强）时的胜平负。"""
    b, c1, c2 = _unpack(np.asarray(theta, dtype=float))
    p = result_probs(theta, 0.0)[0]
    return {"斜率b": round(float(b), 4), "切点c1": round(float(c1), 4), "切点c2": round(float(c2), 4),
            "评分差0时[主胜,平,客胜]": [round(float(v), 4) for v in p]}
