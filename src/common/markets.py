"""比分矩阵 → 各玩法的概率（需求文档第 3 节：所有玩法都从同一个比分矩阵算出来，各玩法之间不会矛盾）。

比分矩阵是一张表：第 i 行第 j 列 = 主队进 i 球、客队进 j 球的概率。例如 matrix[1, 0] = 1–0 的概率。
所有格子加起来应该等于 1（所有可能的比分合起来一定会发生一个）。

截断尾部怎么处理：
- 进球数理论上没有上限，表只能做到有限大。按需求文档第 3 节，内部每队算到 10 球（settings.yaml 的 calc.max_goals）。
- 本文件的 poisson_score_matrix 把“10 球及以上”的概率全部并进最后一行/列，所以矩阵加起来正好是 1，不丢概率。
  代价：算玩法时把“10 球及以上”当成正好 10 球。预期进球 1.5 时，一队进 11 球以上的概率不到百万分之一，
  只影响“让 9 球以上”“大 19.5 球”这种不会出现的盘口，正常盘口完全不受影响。
- 别的模型（Dixon-Coles、负二项等）生成的矩阵也必须自己把尾部并进去或重新归一，保证加起来等于 1。
  这里的玩法函数不管矩阵是怎么来的，都先用 check_score_matrix 检查：加起来和 1 的差距超过
  settings.yaml 的 calc.prob_sum_tol（百万分之一）就报错，说明模型把尾部丢掉了或算错了；差距在范围内的再按比例补成正好 1。
  不悄悄归一的原因：丢了 1% 的概率却自动放大，会让所有玩法都偏一点，又不报错，很难发现。
"""
from functools import lru_cache

import numpy as np
from scipy.stats import poisson

from src.common.config import load_settings
from src.common.settlement import OUTCOMES, asian_handicap_margin, over_under_margin, profit, settle_margin


@lru_cache(maxsize=1)
def _calc_settings() -> dict:
    """读 settings.yaml 的 calc 段（只读一次）。"""
    return load_settings()["calc"]


def poisson_score_matrix(home_xg: float, away_xg: float, max_goals: int | None = None) -> np.ndarray:
    """独立泊松 → 比分矩阵。两队进球数各自服从泊松分布、互不影响（Dixon-Coles 在模型阶段再做）。

    泊松分布：预期进球 λ 时，进 k 球的概率 = e^(−λ) × λ^k ÷ k!。
    例：λ = 1 时，进 0 球 = e^(−1) ≈ 0.3679，进 1 球 = 0.3679，进 2 球 = 0.1839。
    两队互不影响，所以 主进 i 客进 j 的概率 = 主队进 i 球的概率 × 客队进 j 球的概率。
    矩阵大小 (max_goals + 1) × (max_goals + 1)，最后一行/列是“max_goals 球及以上”（见文件开头）。
    """
    if home_xg < 0 or away_xg < 0:
        raise ValueError("预期进球不能是负数")
    n = _calc_settings()["max_goals"] if max_goals is None else max_goals
    if n < 1:
        raise ValueError("max_goals 至少是 1")

    def one_team(lam: float) -> np.ndarray:
        p = poisson.pmf(np.arange(n + 1), lam)
        p[n] = poisson.sf(n - 1, lam)             # 最后一格 = 进 n 球及以上 = 1 − 进 0 到 n−1 球的概率
        return p

    return np.outer(one_team(home_xg), one_team(away_xg))


def check_score_matrix(matrix, tol: float | None = None) -> np.ndarray:
    """检查比分矩阵：二维、没有空值和负数、加起来和 1 差距在 tol 以内。通过后按比例补成正好 1 再返回。

    矩阵不要求是正方形（主队算到 10 球、客队算到 8 球也行）。tol 不写就用 settings.yaml 的 calc.prob_sum_tol。
    """
    m = np.asarray(matrix, dtype=float)
    if m.ndim != 2 or min(m.shape) < 2:
        raise ValueError(f"比分矩阵要是至少 2×2 的表，收到的形状是 {m.shape}")
    if np.isnan(m).any() or (m < 0).any():
        raise ValueError("比分矩阵里有空值或负数")
    tol = _calc_settings()["prob_sum_tol"] if tol is None else tol
    total = m.sum()
    if abs(total - 1) > tol:
        raise ValueError(f"比分矩阵加起来是 {total:.8f}，和 1 差太多（允许 {tol}）。"
                         "常见原因：截断到最大进球数时把尾部概率丢了，应该并进最后一行/列")
    return m / total


def result_probs(matrix) -> np.ndarray:
    """胜平负概率，顺序 [主胜, 平, 客胜]（和 src/evaluate/metrics.py 的 RPS 顺序一致）。

    主胜 = 主队进球比客队多的格子（对角线左下方）加起来；平 = 对角线（0–0、1–1……）；客胜 = 对角线右上方。
    """
    m = check_score_matrix(matrix)
    return np.array([np.tril(m, -1).sum(), np.trace(m), np.triu(m, 1).sum()])


def _sum_by(matrix: np.ndarray, value_of_cell) -> dict:
    """把矩阵里“某个数相同”的格子加起来。value_of_cell(主进球, 客进球) 给出这个数（例如净胜球、总进球）。

    返回 {这个数: 概率}。例：按总进球加，{0: P(0–0), 1: P(1–0)+P(0–1), 2: ...}。
    """
    h, a = np.indices(matrix.shape)
    values = value_of_cell(h, a).ravel()
    probs = matrix.ravel()
    return {int(v): float(probs[values == v].sum()) for v in np.unique(values)}


def win_share(probs: dict) -> float:
    """亚盘、大小球的“胜算”（需求文档第 3 节“方向怎么定”）：把赢半、输半折成半个赢、半个输，走盘不计。

    胜算 = (赢 + 赢半×0.5) ÷ (赢 + 赢半×0.5 + 输 + 输半×0.5)。
    为什么公平赔率 = 1 ÷ 胜算：期望盈亏 = (赢 + 赢半/2)×(o−1) − (输 + 输半/2)，令它等于 0，解出 o = 1 ÷ 胜算。
    全部是走盘时没有胜算，返回 nan。
    """
    up = probs["win"] + probs["half_win"] * 0.5
    down = probs["loss"] + probs["half_loss"] * 0.5
    return up / (up + down) if up + down > 0 else float("nan")


def expected_profit(probs: dict, odds: float) -> float:
    """期望盈亏（每 1 单位本金）= 每种结果的概率 × 这种结果的盈亏，加起来。用实际赔率算就含抽水。"""
    return sum(probs[o] * profit(o, odds) for o in OUTCOMES)


def _line_market(margin_probs: dict, to_margin, odds: float | None) -> dict:
    """亚盘和大小球共用：margin_probs 是 {原始的数: 概率}，to_margin 把它换成 (你这边的净胜球, 你这边的盘口)。"""
    probs = dict.fromkeys(OUTCOMES, 0.0)
    for value, p in margin_probs.items():
        probs[settle_margin(*to_margin(value))] += p
    share = win_share(probs)
    probs["win_share"] = share
    probs["fair_odds"] = 1 / share if share > 0 else float("nan")
    probs["expected_profit"] = None if odds is None else expected_profit(probs, odds)
    return probs


def asian_handicap(matrix, line: float, side: str = "home", odds: float | None = None) -> dict:
    """亚盘让球（任意盘口，含四分之一盘）。line 是主队的盘口（数据库写法，负数 = 主队让球），side 选买哪边。

    返回字典：win / half_win / push / half_loss / loss 五种结果的概率（加起来 = 1），
    win_share 胜算，fair_odds 公平赔率（= 1 ÷ 胜算），expected_profit 给了赔率时的期望盈亏（每 1 单位本金）。
    做法：先按净胜球（主 − 客）把矩阵格子加起来，再对每个净胜球用 settlement 的规则结算。
    """
    m = check_score_matrix(matrix)
    by_diff = _sum_by(m, lambda h, a: h - a)
    return _line_market(by_diff, lambda d: asian_handicap_margin(d, 0, line, side), odds)


def over_under(matrix, line: float, side: str = "over", odds: float | None = None) -> dict:
    """大小球（任意线，含 2.25、2.75 这种四分之一线）。返回内容和 asian_handicap 一样。"""
    m = check_score_matrix(matrix)
    by_total = _sum_by(m, lambda h, a: h + a)
    return _line_market(by_total, lambda t: over_under_margin(t, line, side), odds)


def btts(matrix) -> dict:
    """双方都进球（BTTS）：是 = 两队都至少进 1 球（去掉第 0 行和第 0 列后剩下的格子）；否 = 1 − 是。"""
    m = check_score_matrix(matrix)
    yes = float(m[1:, 1:].sum())
    return {"yes": yes, "no": 1 - yes}


def correct_score(matrix, home_goals: int, away_goals: int) -> float:
    """正确比分的概率，就是矩阵里那一格。注意最后一行/列是“最大进球数及以上”。"""
    m = check_score_matrix(matrix)
    if not (0 <= home_goals < m.shape[0] and 0 <= away_goals < m.shape[1]):
        raise ValueError(f"比分 {home_goals}–{away_goals} 超出矩阵范围 {m.shape}")
    return float(m[home_goals, away_goals])


def top_scores(matrix, n: int = 5) -> list[tuple[int, int, float]]:
    """最可能的 n 个比分，返回 [(主进球, 客进球, 概率), ...]，概率从大到小（页面“前 5 个最可能比分”用）。"""
    m = check_score_matrix(matrix)
    order = np.argsort(m, axis=None)[::-1][:n]
    return [(int(i), int(j), float(m[i, j])) for i, j in zip(*np.unravel_index(order, m.shape))]


# ======================= 从赔率反推市场比分矩阵（第 5 节“市场比分矩阵”、第 4 节步骤 8 的赔率特征共用） =======================
def poisson_matrices(lam_h, lam_a, n: int = 10) -> np.ndarray:
    """一批比赛的独立泊松比分矩阵，形状 (场数, n + 1, n + 1)；最后一行 / 列是 “n 球及以上”，每个矩阵加起来是 1。

    和 src/models/goals.py 的 score_matrices(lam_h, lam_a, rho=0, alpha=0, n) 结果一样（测试里核对），
    放在这里是因为取数层（src/features）也要用，而数据只能往一个方向流，取数层不能引用模型（src/models）。
    递推算：进 0 球 e^(−λ)，进 k+1 球 = 进 k 球 × λ ÷ (k + 1)。
    """
    def one_team(lam) -> np.ndarray:
        lam = np.atleast_1d(np.asarray(lam, dtype=float))[:, None]
        ratio = lam / (np.arange(n - 1) + 1)
        p = np.exp(-lam) * np.concatenate([np.ones_like(lam), np.cumprod(ratio, axis=1)], axis=1)   # 进 0 … n−1 球
        tail = np.clip(1 - p.sum(axis=1, keepdims=True), 0, None)                                  # n 球及以上
        return np.concatenate([p, tail], axis=1)

    m = one_team(lam_h)[:, :, None] * one_team(lam_a)[:, None, :]
    return m / m.sum(axis=(1, 2), keepdims=True)


def matrix_probs(lh, la, n: int = 10) -> np.ndarray:
    """一批比赛的独立泊松比分矩阵 → 每场 [主胜, 平, 客胜, 大 2.5 球] 四个概率。"""
    m = poisson_matrices(lh, la, n)
    i, j = np.indices(m.shape[1:])
    return np.stack([(m * (i > j)).sum(axis=(1, 2)), (m * (i == j)).sum(axis=(1, 2)),
                     (m * (i < j)).sum(axis=(1, 2)), (m * (i + j > 2)).sum(axis=(1, 2))], axis=1)


def _gauss_newton(resid, x, iters: int, tol: float) -> np.ndarray:
    """每场一组参数 x（场数 × k），让 resid(x)（场数 × 目标个数）的平方和最小。几万场同时走。

    每一步用数值斜率 J 解 (JᵀJ + 一点点阻尼) Δ = −Jᵀ r。走了以后误差变大的场，那一步减半再试（最多 20 次），
    保证每场误差只降不升。fit_market_lambdas、fit_market_dc 共用。
    """
    h = 1e-5
    k = x.shape[1]
    eye = np.eye(k)
    r = resid(x)
    for _ in range(iters):
        J = np.stack([(resid(x + h * eye[i]) - r) / h for i in range(k)], axis=2)        # 场数 × 目标个数 × k
        A = np.einsum("nki,nkj->nij", J, J) + 1e-9 * eye
        step = -np.linalg.solve(A, np.einsum("nki,nk->ni", J, r)[..., None])[..., 0]
        cost = (r ** 2).sum(axis=1)
        scale = np.ones(len(x))
        for _ in range(20):                                   # 误差变大就把步子减半
            x_new = x + step * scale[:, None]
            r_new = resid(x_new)
            worse = (r_new ** 2).sum(axis=1) > cost
            if not worse.any():
                break
            scale[worse] /= 2
        keep = ~worse
        x[keep], r[keep] = x_new[keep], r_new[keep]
        if np.abs(step * scale[:, None]).max() < tol:
            break
    return x


def fit_market_lambdas(target, n: int = 10, iters: int = 60, tol: float = 1e-10) -> tuple[np.ndarray, np.ndarray]:
    """每场找 (λ主, λ客)，让 matrix_probs 的四个概率和 target（每场一行四个数）的差的平方和最小。

    在 log λ 上做高斯-牛顿（_gauss_newton），从平均水平出发（主队 1.45、客队 1.15 个）。返回 (λ主, λ客)。
    """
    t = np.asarray(target, float).reshape(-1, 4)
    if len(t) == 0:
        return np.zeros(0), np.zeros(0)
    x = np.tile(np.log([1.45, 1.15]), (len(t), 1))
    x = _gauss_newton(lambda x: matrix_probs(np.exp(x[:, 0]), np.exp(x[:, 1]), n) - t, x, iters, tol)
    return np.exp(x[:, 0]), np.exp(x[:, 1])


def dc_rho_bounds(lam_h, lam_a) -> tuple[np.ndarray, np.ndarray]:
    """DC 修正的 rho 能取的范围（每场一个）：让 0-0、0-1、1-0、1-1 四格乘的数都不小于 0。
    1 − λ主·λ客·rho ≥ 0、1 + λ主·rho ≥ 0、1 + λ客·rho ≥ 0、1 − rho ≥ 0 → 下限 −1 / max(λ主, λ客)，上限 min(1, 1 / (λ主·λ客))。"""
    lh, la = np.atleast_1d(np.asarray(lam_h, float)), np.atleast_1d(np.asarray(lam_a, float))
    return -1 / np.maximum(lh, la), np.minimum(1.0, 1 / (lh * la))


def dc_matrices(lam_h, lam_a, rho, n: int = 10) -> np.ndarray:
    """一批比赛的泊松 + DC 小比分修正比分矩阵，rho 每场可以不一样（和 src/models/goals.py 的 score_matrices 同一个公式）。

    四格：0-0 乘 (1 − λ主·λ客·rho)，0-1 乘 (1 + λ主·rho)，1-0 乘 (1 + λ客·rho)，1-1 乘 (1 − rho)。
    这四格加减正好抵消（总和不变），所以大小球不变；平局多 2·rho·c、主胜和客胜各少 rho·c（c = λ主·λ客·e^(−λ主−λ客)）。
    """
    lh, la = np.atleast_1d(np.asarray(lam_h, float)), np.atleast_1d(np.asarray(lam_a, float))
    r = np.broadcast_to(np.asarray(rho, float), lh.shape)
    m = poisson_matrices(lh, la, n)
    m[:, 0, 0] *= np.maximum(1 - lh * la * r, 0)
    m[:, 0, 1] *= np.maximum(1 + lh * r, 0)
    m[:, 1, 0] *= np.maximum(1 + la * r, 0)
    m[:, 1, 1] *= np.maximum(1 - r, 0)
    return m / m.sum(axis=(1, 2), keepdims=True)


def fit_market_dc(target, n: int = 10, iters: int = 60, tol: float = 1e-10
                  ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """每场找 (λ主, λ客, rho)，让 DC 比分矩阵的 [主胜, 平, 客胜, 大 2.5 球] 和 target（每场一行四个数，胜平负加起来是 1）一样。

    rho 不改“主胜 − 客胜”和大 2.5 球（见 dc_matrices），所以分两步：
      1. 两个 λ：让独立泊松矩阵的“主胜 − 客胜”和大 2.5 球对上（高斯-牛顿，两个数解两个数）；
      2. rho：平局还差多少就补多少，rho = (泊松的平局 − 要的平局) ÷ (2c)；超出 dc_rho_bounds 的取边界（这场平局就对不上，报告里看还原误差）。
    返回 (λ主, λ客, rho)。
    """
    t = np.asarray(target, float).reshape(-1, 4)
    if len(t) == 0:
        return np.zeros(0), np.zeros(0), np.zeros(0)
    goal = np.stack([t[:, 0] - t[:, 2], t[:, 3]], axis=1)

    def resid(x):
        p = matrix_probs(np.exp(x[:, 0]), np.exp(x[:, 1]), n)
        return np.stack([p[:, 0] - p[:, 2], p[:, 3]], axis=1) - goal

    x = _gauss_newton(resid, np.tile(np.log([1.45, 1.15]), (len(t), 1)), iters, tol)
    lh, la = np.exp(x[:, 0]), np.exp(x[:, 1])
    c = lh * la * np.exp(-lh - la)
    rho = (matrix_probs(lh, la, n)[:, 1] - t[:, 1]) / (2 * c)
    lo, hi = dc_rho_bounds(lh, la)
    return lh, la, np.clip(rho, lo, hi)
