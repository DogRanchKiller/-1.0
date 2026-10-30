"""赔率换算和去水（需求文档第 16 节“去水方法”）。

几个词先说清楚：
- 赔率：一律是欧洲小数赔率，例如 1.90 = 下 1 元、赢了拿回 1.90 元（含本金）。数据库 odds 表里存的就是这种。
- 隐含概率：赔率的倒数，1 / 1.90 ≈ 0.526。它是“庄家开出的价格对应的概率”，还带着庄家的抽水。
- 赔率倒数之和（英文 overround）：一个玩法所有选项的隐含概率加起来。公平的话应该正好是 1，
  庄家要赚钱，所以通常比 1 多一点，多出来的就是抽水。例如 1.02 表示抽水约 2%。
  注意：国内网站说的“水位”是另一回事，是香港盘写法的赔率（0.85 = 欧洲赔率 1.85，见需求文档第 16 节“格式统一”），
  所以这里不用“水位”这个词，统一叫“赔率倒数之和”和“抽水”。src/clean/quality.py 的抽水检查也是这个定义。
- 去水：把隐含概率改成加起来正好等于 1 的概率，作为“市场认为的真实概率”。改的方法不止一种：
  1. 按比例（默认）：每个隐含概率都除以它们的和。最简单。
  2. Shin 法：假设市场里有一部分“知道内幕的人”，庄家为了防他们，对冷门多抽水。冷门的概率被压得更多。
  3. 幂函数法：每个隐含概率都取同一个次方 k（p = q 的 k 次方），选 k 让加起来等于 1。同样是冷门被压得更多。
  第 3 阶段比较这三种，哪种让标杆更准就用哪种。

所有函数都同时支持两种输入：
- 一组赔率（一维），例如胜平负 [2.0, 3.5, 4.0]，大小球或亚盘 [1.90, 1.95]；
- 很多组赔率（二维，每一行是一场比赛），回测时一次算几万场，速度快。
选项个数不限（两项、三项都行，以后比分玩法几十项也行）。
"""
from functools import lru_cache

import numpy as np

from src.common.config import load_settings

METHODS = ("proportional", "shin", "power")


@lru_cache(maxsize=1)
def _calc_settings() -> dict:
    """读 settings.yaml 的 calc 段。只读一次存起来：回测里会调用几十万次，每次读文件太慢。"""
    return load_settings()["calc"]


def _as_odds(odds) -> np.ndarray:
    """把输入整理成数字数组，并检查赔率合不合理。"""
    o = np.asarray(odds, dtype=float)
    if o.ndim not in (1, 2) or o.shape[-1] < 2:
        raise ValueError(f"赔率要是一组（一维）或很多组（二维，每行一场），每组至少 2 个选项；收到的形状是 {o.shape}")
    if np.isnan(o).any():
        raise ValueError("赔率里有空值，先把缺赔率的比赛去掉再算")
    if (o <= 1).any():
        raise ValueError("赔率必须大于 1（欧洲小数赔率）。如果是香港盘写法的 0.85，要先加 1 换成 1.85")
    return o


def implied_probs(odds) -> np.ndarray:
    """赔率 → 隐含概率（还带着抽水）：每个赔率取倒数。例：2.0 → 0.5，4.0 → 0.25。"""
    return 1.0 / _as_odds(odds)


def overround(odds):
    """赔率倒数之和。1.0 = 没有抽水；1.05 = 抽水约 5%。一组赔率返回一个数，很多组返回每场一个数。

    例：胜平负 2.0 / 3.5 / 4.0 → 1/2 + 1/3.5 + 1/4 = 0.5 + 0.2857 + 0.25 = 1.0357，抽水约 3.6%。
    """
    return implied_probs(odds).sum(axis=-1)


def fair_odds(probs) -> np.ndarray:
    """概率 → 公平赔率（不含抽水）：取倒数。例：概率 0.4 → 公平赔率 2.5。页面上“换算的公平赔率”用它。"""
    p = np.asarray(probs, dtype=float)
    if (p <= 0).any():
        raise ValueError("概率为 0 没有公平赔率（会是无穷大）")
    return 1.0 / p


def devig_proportional(odds) -> np.ndarray:
    """按比例去水（默认方法）：每个隐含概率除以隐含概率之和。

    例：2.0 / 3.5 / 4.0 → 隐含概率 0.5、0.2857、0.25，和 1.0357；
    各除以 1.0357 → 0.4828、0.2759、0.2414（正好是 14/29、8/29、7/29），加起来 = 1。
    """
    q = implied_probs(odds)
    return q / q.sum(axis=-1, keepdims=True)


def _bisect(sum_minus_one, lo, hi, iterations: int = 100):
    """二分法解方程：找参数 x，让 sum_minus_one(x) = 0（去水后的概率加起来正好是 1）。

    要求：sum_minus_one 在 lo 处 >= 0、在 hi 处 <= 0，中间随参数变大而变小。
    做法像猜数字：每次取中点，算出来还大于 0 就说明答案在右半边，否则在左半边，范围每次减半。
    lo、hi 可以是数组（每场比赛一个），所有比赛一起算。100 次后范围缩小到 2 的 100 次方分之一，比电脑能表示的最小差距还小。
    """
    lo = np.array(lo, dtype=float)
    hi = np.array(hi, dtype=float)
    for _ in range(iterations):
        mid = (lo + hi) / 2
        go_right = sum_minus_one(mid) > 0
        lo = np.where(go_right, mid, lo)
        hi = np.where(go_right, hi, mid)
    return (lo + hi) / 2


def _finish(p: np.ndarray, method: str) -> np.ndarray:
    """检查解出来的概率加起来是不是 1（精度 settings.yaml 的 calc.solver_tol），再消掉最后一点舍入误差。"""
    total = p.sum(axis=-1, keepdims=True)
    worst = float(np.abs(total - 1).max())
    if worst > _calc_settings()["solver_tol"]:
        raise ArithmeticError(f"{method} 去水没解出来：概率之和和 1 差 {worst:.2e}，检查赔率是否正常")
    return p / total


def _shin_probs(q: np.ndarray, z) -> np.ndarray:
    """给定内幕比例 z，按 Shin 公式算概率：p = (√(z² + 4(1−z)·q²/B) − z) / (2(1−z))，B 是隐含概率之和。"""
    z = np.asarray(z, dtype=float)[..., None]          # 每场一个 z，扩成和 q 一样能逐个相乘的形状
    b = q.sum(axis=-1, keepdims=True)
    return (np.sqrt(z * z + 4 * (1 - z) * q * q / b) - z) / (2 * (1 - z))


def devig_shin(odds, return_z: bool = False):
    """Shin 法去水。

    想法（Shin 1993）：市场里有比例为 z 的“内幕玩家”，他们知道结果、只买会赢的那个选项。庄家为了不亏，
    会把每个选项的价格抬高，冷门抬得更多（冷门被内幕玩家买中时庄家赔得多）。反过来，从赔率解出 z，再还原真实概率：
        p = (√(z² + 4(1−z)·q²/B) − z) / (2(1−z))     q = 隐含概率，B = 隐含概率之和
    z 选成让 p 加起来正好等于 1，用二分法解。

    几个性质（测试里都有检查）：
    - 没有抽水（B = 1）时 z = 0，p = q，和原概率一样。
    - 两个选项时，Shin 法和“每个选项减去同样多的概率”（加法去水）结果相同。
    - 赔率倒数之和小于 1（交易所、多家最高常见）时，解出的 z 是负数。负的“内幕比例”没有实际意义，
      这里照样算出来（概率仍然加起来等于 1），用的时候要知道这一点。

    return_z=True 时同时返回 z（一组赔率返回一个数，很多组返回每场一个数）。
    """
    q = implied_probs(odds)
    z = _bisect(lambda zz: _shin_probs(q, zz).sum(axis=-1) - 1,
                lo=np.full(q.shape[:-1], -1.0),          # z = −1 时每个 p ≥ 0.5，加起来 ≥ 1
                hi=np.full(q.shape[:-1], 1 - 1e-9))      # z 接近 1 时加起来 ≈ Σq²/B < 1
    p = _finish(_shin_probs(q, z), "Shin 法")
    return (p, z) if return_z else p


def devig_power(odds, return_k: bool = False):
    """幂函数法去水：p = q 的 k 次方，选 k 让 p 加起来正好等于 1。

    隐含概率都小于 1，取次方 k > 1 会让它们都变小，小的（冷门）变得相对更多。
    例：隐含概率 0.6、0.8（和 1.4）。k = 2 时 0.36 + 0.64 = 1，所以去水后是 0.36、0.64；
    按比例去水则是 0.429、0.571。冷门 0.6 → 0.36，被压得比按比例多。
    没有抽水时 k = 1，p = q；赔率倒数之和小于 1 时 k < 1。

    return_k=True 时同时返回 k。
    """
    q = implied_probs(odds)
    log_q = np.log(q)
    sum_minus_one = lambda k: np.exp(np.asarray(k)[..., None] * log_q).sum(axis=-1) - 1
    hi = np.full(q.shape[:-1], 2.0)
    for _ in range(60):                                   # 上界不够大就翻倍，直到所有比赛的和都小于 1
        if (sum_minus_one(hi) <= 0).all():
            break
        hi = np.where(sum_minus_one(hi) > 0, hi * 2, hi)
    k = _bisect(sum_minus_one, lo=np.full(q.shape[:-1], 1e-9), hi=hi)   # k 接近 0 时每个 q^k ≈ 1，和 ≈ 选项个数 ≥ 2
    p = _finish(q ** np.asarray(k)[..., None], "幂函数法")
    return (p, k) if return_k else p


def devig(odds, method: str | None = None) -> np.ndarray:
    """去水的统一入口。method 不写就用 settings.yaml 的 calc.devig_method（默认 proportional 按比例）。"""
    method = method or _calc_settings()["devig_method"]
    if method == "proportional":
        return devig_proportional(odds)
    if method == "shin":
        return devig_shin(odds)
    if method == "power":
        return devig_power(odds)
    raise ValueError(f"不认识的去水方法 {method!r}，可选：{METHODS}")
