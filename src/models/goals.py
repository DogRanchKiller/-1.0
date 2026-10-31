"""第 2 阶段步骤 1–3：泊松、Dixon-Coles、DC + 负二项（需求文档第 4 节）。三个模型共用这一份代码，只差两个开关。

一、模型长什么样（三个都一样的部分）
  每个联赛单独一套参数：每队一个“进攻力 a”和“防守力 d”，联赛一个“基准 mu”和“主场优势 home”。
  一场比赛，主队预期进球 λ主、客队预期进球 λ客：
      log λ主 = mu + home + a[主队] − d[客队]
      log λ客 = mu        + a[客队] − d[主队]
  用 log 的原因：λ 必须是正数，取 log 以后可以随便加减；加 0.1 就是乘 e^0.1 ≈ 1.105，也就是“多进 10.5% 的球”。
  例：mu = 0.25（e^0.25 ≈ 1.28 球），home = 0.25，主队 a = 0.3，客队 d = −0.1 →
      λ主 = e^(0.25 + 0.25 + 0.3 + 0.1) = e^0.9 ≈ 2.46 球。
  d 越大防守越好（对手进得越少）。

二、三个模型的差别（第 4 节步骤 1–3）
  1. 泊松（poisson）：两队进球各自服从泊松分布，互不影响。比分矩阵 = 两队进球概率相乘（和 markets.poisson_score_matrix 一样）。
  2. Dixon-Coles（dc，开关 dc）：独立泊松会低估 0:0、1:1、高估 1:0、0:1（真实比赛里两队进球不完全独立，
     比分接近时双方都更保守）。Dixon & Coles（1997）只改这 4 个小比分的格子，乘一个系数 τ：
         0:0 × (1 − λ主·λ客·ρ)    0:1 × (1 + λ主·ρ)    1:0 × (1 + λ客·ρ)    1:1 × (1 − ρ)
     ρ 是每个联赛拟合出来的一个数，通常是小的负数（例如 −0.05 → 0:0 和 1:1 变多、1:0 和 0:1 变少）。
     泊松下这 4 个格子改完总和不变（加的和减的正好抵消），所以矩阵加起来仍是 1。
  3. DC + 负二项（dcnb，开关 nb）：泊松的“方差 = 平均数”，进球数的波动被固定死了。
     负二项多一个“离散度 α”：方差 = λ + α·λ²。α > 0 时大比分（一队进 4 球以上）的概率比泊松大。
     α 也是每个联赛拟合出来的；拟合出来接近 0，说明泊松的波动已经够了，大比分没被低估。
     负二项下 DC 的 4 个格子改完总和不再正好不变，所以拟合和出矩阵时都除以“改完以后的总和 Z”（见 _nll）。

三、时间衰减（第 4 节“不用历史简单平均”）
  拟合时每场比赛有一个权重：权重 = 0.5 ^ (这场离截止时间多少天 ÷ 半衰期)。
  例：半衰期 365 天 → 1 年前的比赛算半场，2 年前算 1/4 场。半衰期写成 null = 不衰减（窗口里每场一样重）。
  窗口（window_days）以外的比赛不用。

四、收缩（ridge）和升班马
  每队的 a、d 被往一个“中心”轻轻拉（惩罚 ridge/2 × (a − 中心)²，d 一样）。
  - 作用 1：没有这个拉力，“所有 a 加 1、mu 减 1”结果完全一样，参数定不下来；有了拉力，a、d 以 0（联赛平均）为中心。
  - 作用 2：只踢了几场的队（新赛季刚开始的升班马）不会被几场比赛的运气带跑。ridge = 3 大约相当于“先假设踢了几场平均水平的比赛”。
  - 中心：普通球队是 0（联赛平均）；这个赛季的升班马（上赛季不在这个联赛）是“升班马先验”。
    升班马先验 = 最近几个赛季（promoted_history_seasons）的升班马在上一次拟合里的 a、d 的平均；
    还没有上一次拟合时（第一次拟合）用 settings 里的默认值。用上一次拟合的结果，不会“自己拉自己”。
    注意：往年升班马里留下来、后来变强的队，现在的 a、d 会比它刚升上来时好，所以这个平均可能偏乐观一点。
  - 出预测时：球队上赛季或这个赛季在这个联赛踢过（而且有拟合出来的参数），用它自己的 a、d；
    否则（刚升上来还没踢、或者很多年前在过又升回来）用升班马先验。

五、只用截止前的数据
  fit 收到的 history 只有截止前已经能知道的比赛（取数层保证、回测框架再查一遍）；本文件再按 window_days 往前截。
  权重按回测框架给的 cutoff 算，不用“现在”的时间。

六、怎么拟合
  最大似然：找一组参数，让窗口里每场比赛“真实比分的概率”的对数 × 权重 加起来最大（再减去收缩惩罚）。
  用 scipy 的 L-BFGS-B（一种只要“函数值 + 每个参数的斜率”的找最小值方法），斜率是手推公式算的（_nll），
  tests/test_goals_models.py 用“把参数挪一点点看函数变多少”的笨办法核对过公式。
  每周重新拟合时，从上一次的参数出发（球队大多不变），几十步就收敛。

七、步骤 4：拟合实力时“进球”用什么（开关 target，需求文档第 4 节“步骤 4 事先定的比较”）
  - goals：实际进球（默认，就是步骤 1–3 的模型，代码走原来那条路，结果一模一样）。
  - shots：射门推算的进球 = a × 射正 + b × 射偏（射偏 = 射门 − 射正）。
    a、b 每个联赛每次拟合单独估：窗口里的比赛、同样的时间权重，进球对射正、射偏做加权最小二乘（不带常数项）。
    例：a = 0.33、b = 0.02，一队 5 个射正、8 个射偏 → 0.33×5 + 0.02×8 = 1.81 个“射门进球”。
  - xg：Understat 的 xG（只能用“研究实验”用途，见 data_usage.yaml）。
  - mix：0.5 × 实际进球 + 0.5 × xG。
  不是整数的“进球”照样用泊松的式子拟合（x·log λ − λ 对小数也能算；log(x!) 换成 log Γ(x+1)，和参数无关，不影响结果）。
  DC 的 rho 是关于“真实比分”的（0:0、1:1 多了多少），所以 target 不是 goals 时分两步（two_stage）：
    第 1 步：不带 DC，用 target 拟合 mu、home、a、d；
    第 2 步：固定这些，只用真实比分拟合 rho（同样有 rho 的合法范围）。
  target = goals 也可以分两步（two_stage: true），用来分清“换了数据”和“改成两步”各自的影响。
  require：拟合只用这几类数据都齐的比赛（例如实验一要求有 Understat xG），缺的不进拟合，不拿别的数据补。
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from scipy.optimize import minimize, minimize_scalar
from scipy.special import gammaln

from src.models.base import Model, window_start

ALPHA_MIN = 1e-6      # 负二项离散度的下限：0 就是泊松，但公式里要除以 α，所以留一点点（1e-6 和泊松的差别小到看不出）
TAU_FLOOR = 1e-10     # DC 系数 τ 的下限（防止极端参数下出现 0 或负数，取 log 出错）
SERIES_U = 1e-3       # α·λ 小于这个数时，用级数展开算（直接算会因为两个很接近的数相减而丢精度）
RHO_LIMIT = 0.3       # DC 的 rho 最多到 ±0.3（文献里的估计一般在 −0.2 到 0 之间）
TAU_MIN = 0.05        # DC 修正后，窗口里每场的 4 个小比分格子至少留原来概率的 5%（见 rho_range）


# ======================= 一队进球数的概率 =======================
def goal_probs(lam, alpha: float, n: int) -> np.ndarray:
    """一队进 0, 1, …, n 球的概率，最后一格 = “n 球及以上”（和 markets.poisson_score_matrix 的规则一样，加起来正好是 1）。

    lam 可以是一个数，也可以是一串数（每场一个），返回形状 (场数, n + 1)。alpha = 0 是泊松，> 0 是负二项。
    递推算（不用 scipy，快很多）：
        进 0 球：泊松 e^(−λ)；负二项 (1 + αλ)^(−1/α)
        进 k+1 球 = 进 k 球 × (1 + α·k) ÷ (k + 1) × λ ÷ (1 + αλ)       （α = 0 时就是泊松的 × λ ÷ (k + 1)）
    """
    lam = np.atleast_1d(np.asarray(lam, dtype=float))[:, None]
    k = np.arange(n - 1)
    if alpha > 0:
        u = alpha * lam
        p0 = np.exp(-np.log1p(u) / alpha)
        ratio = (1 + alpha * k) / (k + 1) * lam / (1 + u)
    else:
        p0 = np.exp(-lam)
        ratio = lam / (k + 1)
    p = p0 * np.concatenate([np.ones_like(lam), np.cumprod(ratio, axis=1)], axis=1)   # 进 0 … n−1 球
    tail = np.clip(1 - p.sum(axis=1, keepdims=True), 0, None)                       # n 球及以上
    return np.concatenate([p, tail], axis=1)


def score_matrices(lam_h, lam_a, rho: float = 0.0, alpha: float = 0.0, n: int = 10) -> np.ndarray:
    """一批比赛的比分矩阵，形状 (场数, n + 1, n + 1)，每个矩阵加起来是 1。

    第 i 行第 j 列 = 主队进 i 球、客队进 j 球的概率；最后一行 / 列是 “n 球及以上”。
    步骤：两队进球概率相乘 → DC 改 4 个小比分格子（rho = 0 就是不改）→ 除以总和（泊松下总和本来就是 1，负二项下差一点点）。
    """
    ph, pa = goal_probs(lam_h, alpha, n), goal_probs(lam_a, alpha, n)
    m = ph[:, :, None] * pa[:, None, :]
    if rho != 0.0:
        lh, la = np.atleast_1d(lam_h).astype(float), np.atleast_1d(lam_a).astype(float)
        m[:, 0, 0] *= np.maximum(1 - lh * la * rho, 0)
        m[:, 0, 1] *= np.maximum(1 + lh * rho, 0)
        m[:, 1, 0] *= np.maximum(1 + la * rho, 0)
        m[:, 1, 1] *= max(1 - rho, 0)
    return m / m.sum(axis=(1, 2), keepdims=True)


def time_weights(kickoffs: pd.Series, cutoff: str, half_life_days: float | None) -> np.ndarray:
    """每场比赛的权重 = 0.5 ^ (离截止时间的天数 ÷ 半衰期)。半衰期 None = 每场都是 1。"""
    if half_life_days is None:
        return np.ones(len(kickoffs))
    age = (pd.Timestamp(cutoff) - pd.to_datetime(kickoffs, format="%Y-%m-%d %H:%M:%S")).dt.total_seconds() / 86400
    return 0.5 ** (np.clip(age.to_numpy(), 0, None) / float(half_life_days))


# ======================= 拟合用的“对数似然 + 斜率” =======================
@dataclass
class FitData:
    """一个联赛拟合时用的数据，都是 numpy 数组，每场一个数。"""
    hi: np.ndarray       # 主队编号（0 … 队数−1）
    ai: np.ndarray       # 客队编号
    x: np.ndarray        # 主队进球
    y: np.ndarray        # 客队进球
    w: np.ndarray        # 时间衰减权重
    n_teams: int
    center_a: np.ndarray  # 每队进攻力往哪里拉（收缩中心）
    center_d: np.ndarray  # 每队防守力往哪里拉
    ridge: float
    dc: bool
    nb: bool
    max_goal: int = field(init=False)   # 窗口里最多进了几球（负二项查表用）

    def __post_init__(self):
        self.max_goal = int(max(self.x.max(initial=0), self.y.max(initial=0)))


def _unpack(theta: np.ndarray, n: int, dc: bool, nb: bool):
    """参数向量 → (mu, home, a, d, rho, alpha)。顺序：mu、home、n 个进攻力、n 个防守力、[rho]、[alpha]。"""
    mu, home = theta[0], theta[1]
    a, d = theta[2:2 + n], theta[2 + n:2 + 2 * n]
    i = 2 + 2 * n
    rho = theta[i] if dc else 0.0
    alpha = theta[i + int(dc)] if nb else 0.0
    return mu, home, a, d, rho, alpha


def _side(x: np.ndarray, lam: np.ndarray, alpha: float, nb: bool, kmax: int):
    """一队这一边：返回 (每场对数概率, 对 log λ 的斜率, 对 α 的斜率)。

    泊松：log P = x·log λ − λ − log(x!)；对 log λ 的斜率 = x − λ。
    负二项（方差 λ + αλ²），u = α·λ：
        log P = Σ_{k<x} log(1 + α·k) − log(x!) + x·log λ − (x + 1/α)·log(1 + u)
        对 log λ 的斜率 = (x − λ) ÷ (1 + u)
        对 α 的斜率 = Σ_{k<x} k ÷ (1 + α·k) + Q − x·λ ÷ (1 + u)，Q = log(1+u)/α² − λ/(α(1+u))
      Q 在 u 很小时是两个很接近的大数相减，改用级数 Q ≈ λ²(1/2 − 2u/3 + 3u²/4)。α → 0 时整个斜率 → ((x−λ)² − x)/2。
    """
    if not nb:
        return x * np.log(lam) - lam - gammaln(x + 1), x - lam, None
    u = alpha * lam
    k = np.arange(kmax)
    cum_log = np.concatenate([[0.0], np.cumsum(np.log1p(alpha * k))])      # Σ_{k<x} log(1+αk)，按 x 查表
    cum_frac = np.concatenate([[0.0], np.cumsum(k / (1 + alpha * k))])     # Σ_{k<x} k/(1+αk)
    l1p = np.log1p(u)
    ll = cum_log[x] - gammaln(x + 1) + x * np.log(lam) - (x + 1 / alpha) * l1p
    g_eta = (x - lam) / (1 + u)
    small = u < SERIES_U
    q = np.where(small, lam ** 2 * (0.5 - 2 * u / 3 + 0.75 * u ** 2),
                 l1p / alpha ** 2 - lam / (alpha * (1 + u)))
    g_alpha = cum_frac[x] + q - x * lam / (1 + u)
    return ll, g_eta, g_alpha


def _nll(theta: np.ndarray, f: FitData) -> tuple[float, np.ndarray]:
    """要最小化的数 = −(Σ 权重 × 每场真实比分的对数概率) + 收缩惩罚；同时返回每个参数的斜率。"""
    n = f.n_teams
    mu, home, a, d, rho, alpha = _unpack(theta, n, f.dc, f.nb)
    eta_h = mu + home + a[f.hi] - d[f.ai]
    eta_a = mu + a[f.ai] - d[f.hi]
    lh, la = np.exp(eta_h), np.exp(eta_a)
    ll_h, gh, galpha_h = _side(f.x, lh, alpha, f.nb, f.max_goal)
    ll_a, ga, galpha_a = _side(f.y, la, alpha, f.nb, f.max_goal)
    ll = ll_h + ll_a
    g_rho = np.zeros_like(ll)
    g_alpha = (galpha_h + galpha_a) if f.nb else None

    if f.dc:
        # ---- DC 的 4 个小比分格子：log τ 和它的斜率 ----
        tau = np.ones_like(ll)
        dt_h, dt_a, dt_r = np.zeros_like(ll), np.zeros_like(ll), np.zeros_like(ll)
        m00 = (f.x == 0) & (f.y == 0)
        m01 = (f.x == 0) & (f.y == 1)
        m10 = (f.x == 1) & (f.y == 0)
        m11 = (f.x == 1) & (f.y == 1)
        tau[m00] = 1 - lh[m00] * la[m00] * rho
        dt_h[m00] = dt_a[m00] = -lh[m00] * la[m00] * rho
        dt_r[m00] = -lh[m00] * la[m00]
        tau[m01] = 1 + lh[m01] * rho
        dt_h[m01] = lh[m01] * rho
        dt_r[m01] = lh[m01]
        tau[m10] = 1 + la[m10] * rho
        dt_a[m10] = la[m10] * rho
        dt_r[m10] = la[m10]
        tau[m11] = 1 - rho
        dt_r[m11] = -1.0
        # τ 被压到下限的场次：这一小段里对数似然是平的，斜率也要是 0（否则数值和斜率对不上，拟合程序会卡住）
        clipped = tau < TAU_FLOOR
        dt_h[clipped] = dt_a[clipped] = dt_r[clipped] = 0.0
        tau = np.maximum(tau, TAU_FLOOR)
        ll = ll + np.log(tau)
        gh = gh + dt_h / tau
        ga = ga + dt_a / tau
        g_rho = dt_r / tau
        if f.nb:
            # ---- 负二项下 4 个格子改完总和 Z ≠ 1，要除以 Z：log P 减 log Z ----
            # Z = 1 − ρ·D主·D客，D = λ·P(0 球) − P(1 球) = α·λ²·P(0 球) ÷ (1 + u)
            def dterm(lam):
                u = alpha * lam
                l1p = np.log1p(u)
                p0 = np.exp(-l1p / alpha)
                dd = alpha * lam ** 2 * p0 / (1 + u)
                dlog_eta = (2 + u - lam) / (1 + u)
                q = np.where(u < SERIES_U, lam ** 2 * (0.5 - 2 * u / 3 + 0.75 * u ** 2),
                             l1p / alpha ** 2 - lam / (alpha * (1 + u)))
                dd_alpha = lam ** 2 * p0 / (1 + u) * (1 + alpha * q - u / (1 + u))  # dD/dα，α→0 时 → λ²·P(0 球)
                return dd, dd * dlog_eta, dd_alpha
            dh, dh_eta, dh_alpha = dterm(lh)
            da, da_eta, da_alpha = dterm(la)
            z = 1 - rho * dh * da
            ll = ll - np.log(z)
            gh = gh + rho * dh_eta * da / z
            ga = ga + rho * dh * da_eta / z
            g_rho = g_rho + dh * da / z
            g_alpha = g_alpha + rho * (dh_alpha * da + dh * da_alpha) / z

    w = f.w
    value = -np.sum(w * ll)
    wgh, wga = w * gh, w * ga
    grad_a = -(np.bincount(f.hi, wgh, n) + np.bincount(f.ai, wga, n))
    grad_d = np.bincount(f.ai, wgh, n) + np.bincount(f.hi, wga, n)
    ra, rd = a - f.center_a, d - f.center_d
    value += 0.5 * f.ridge * (np.sum(ra ** 2) + np.sum(rd ** 2))
    grad = [np.array([-(wgh.sum() + wga.sum()), -wgh.sum()]), grad_a + f.ridge * ra, grad_d + f.ridge * rd]
    if f.dc:
        grad.append(np.array([-np.sum(w * g_rho)]))
    if f.nb:
        grad.append(np.array([-np.sum(w * g_alpha)]))
    return float(value), np.concatenate(grad)


def rho_range(theta: np.ndarray, f: FitData) -> tuple[float, float]:
    """按现在的参数，窗口里每一场的 4 个小比分格子都还留着至少 TAU_MIN 倍的概率时，rho 能取的范围。

    4 个系数：0:0 是 1 − λ主λ客ρ，0:1 是 1 + λ主ρ，1:0 是 1 + λ客ρ，1:1 是 1 − ρ。要每个都 ≥ TAU_MIN：
      ρ ≥ (TAU_MIN − 1) ÷ λ主、ρ ≥ (TAU_MIN − 1) ÷ λ客（ρ 是负数时，强队主场的 0:1 先撑不住）；
      ρ ≤ (1 − TAU_MIN) ÷ (λ主λ客)、ρ ≤ 1 − TAU_MIN。
    """
    n = f.n_teams
    mu, home, a, d, _, _ = _unpack(theta, n, f.dc, f.nb)
    lh = np.exp(mu + home + a[f.hi] - d[f.ai])
    la = np.exp(mu + a[f.ai] - d[f.hi])
    lo = float(np.max(np.maximum((TAU_MIN - 1) / lh, (TAU_MIN - 1) / la)))
    hi = float(min(np.min((1 - TAU_MIN) / (lh * la)), 1 - TAU_MIN))
    return lo, hi


def fit_league(f: FitData, start: np.ndarray, maxiter: int = 500) -> np.ndarray:
    """一个联赛的最大似然拟合。start 是起点（上一次拟合的参数或默认值）。rho、alpha 有上下限。

    Dixon-Coles 的 rho 还有一个“合法范围”（rho_range）：超出了，强弱悬殊的比赛里某个小比分格子会变成 0 甚至负数。
    只有真的出现过的比分才进似然，所以似然本身拦不住（例如强队主场 0:1 很少真的发生）。
    做法：拟合完检查一下，超出了就把 rho 的上下限收到合法范围，再拟合一次（球队实力几乎不变，一两次就够）。
    """
    n = f.n_teams
    rho_lo, rho_hi = -RHO_LIMIT, RHO_LIMIT
    x = np.asarray(start, dtype=float).copy()
    for _ in range(4):
        bounds = [(None, None)] * (2 + 2 * n)
        if f.dc:
            bounds.append((rho_lo, rho_hi))
            x[2 + 2 * n] = np.clip(x[2 + 2 * n], rho_lo, rho_hi)
        if f.nb:
            bounds.append((ALPHA_MIN, 1.0))
        res = minimize(_nll, x, args=(f,), jac=True, method="L-BFGS-B", bounds=bounds,
                       options={"maxiter": maxiter, "ftol": 1e-10, "gtol": 1e-6})
        x = res.x
        if not f.dc:
            return x
        lo, hi = rho_range(x, f)
        rho = x[2 + 2 * n]
        if lo - 1e-9 <= rho <= hi + 1e-9:
            return x
        rho_lo, rho_hi = max(rho_lo, lo), min(rho_hi, hi)
        if rho_lo > rho_hi:            # 不会发生（ρ = 0 总是合法的），防万一
            rho_lo = rho_hi = 0.0
    # 4 次还差一点（实测只在半衰期 90 天、有效场数很少时偶尔出现）：直接把 rho 放回合法范围
    lo, hi = rho_range(x, f)
    x[2 + 2 * n] = np.clip(x[2 + 2 * n], lo, hi)
    return x


# ======================= 步骤 4：拟合用的“进球”和两步拟合 =======================
TARGETS = ("goals", "shots", "xg", "mix")
# 每类“要求齐全”的数据 → 取数时要哪类数据（data_usage.yaml 的名字）、要哪几列都不是空的
REQUIRE = {
    "shots": ("射门、xG", ["home_shots", "away_shots", "home_shots_on_target", "away_shots_on_target"]),
    "understat": ("Understat xG（研究）", ["us_home_xg", "us_away_xg"]),
}
TARGET_NEEDS = {"goals": [], "shots": ["shots"], "xg": ["understat"], "mix": ["understat"]}


def shot_coefficients(part: pd.DataFrame, w: np.ndarray) -> tuple[float, float]:
    """射门推算进球的 a、b：进球 ≈ a × 射正 + b × 射偏，主客两边都算一行，加权最小二乘（不带常数项）。

    最小二乘的公式：[a, b] = (XᵀWX)⁻¹ XᵀWy，X 每行是 (射正, 射偏)，y 是进球，W 是时间权重。
    射偏 = 射门 − 射正；个别场次数据里射门比射正少（录入错误），射偏按 0 算。
    """
    sot = np.concatenate([part["home_shots_on_target"], part["away_shots_on_target"]]).astype(float)
    shots = np.concatenate([part["home_shots"], part["away_shots"]]).astype(float)
    off = np.clip(shots - sot, 0, None)
    goals = np.concatenate([part["home_goals"], part["away_goals"]]).astype(float)
    ww = np.concatenate([w, w])
    X = np.column_stack([sot, off])
    a, b = np.linalg.solve(X.T @ (X * ww[:, None]), X.T @ (goals * ww))
    return float(a), float(b)


def fit_targets(part: pd.DataFrame, target: str, coef: tuple[float, float] | None = None
                ) -> tuple[np.ndarray, np.ndarray]:
    """拟合实力时两队的“进球”（主队、客队），每场一个数。shots 要先算好 a、b（coef）。"""
    hg, ag = part["home_goals"].to_numpy(float), part["away_goals"].to_numpy(float)
    if target == "goals":
        return hg, ag
    if target == "shots":
        a, b = coef
        def side(sot, shots):
            sot, shots = part[sot].to_numpy(float), part[shots].to_numpy(float)
            return a * sot + b * np.clip(shots - sot, 0, None)
        x, y = side("home_shots_on_target", "home_shots"), side("away_shots_on_target", "away_shots")
    elif target == "xg":
        x, y = part["us_home_xg"].to_numpy(float), part["us_away_xg"].to_numpy(float)
    elif target == "mix":
        x = 0.5 * hg + 0.5 * part["us_home_xg"].to_numpy(float)
        y = 0.5 * ag + 0.5 * part["us_away_xg"].to_numpy(float)
    else:
        raise ValueError(f"target 只能是 {TARGETS}，现在是 {target!r}")
    # 泊松的式子要求“进球”不是负数。b 是负数时（射偏越多越少进），射正 0、射偏很多的一边会算出负数：按 0 算
    return np.clip(x, 0, None), np.clip(y, 0, None)


def fit_rho(f: FitData, theta: np.ndarray) -> float:
    """两步拟合的第 2 步：mu、home、a、d 固定（theta 是第 1 步的结果，不含 rho），只用真实比分找 rho。

    λ 固定时，泊松那部分和 rho 无关，要最大化的只剩 Σ 权重 × log τ（4 个小比分格子的修正系数）。
    rho 的范围：±RHO_LIMIT 和 rho_range（每场 4 个格子至少留 TAU_MIN）的交集。
    """
    n = f.n_teams
    mu, home, a, d = theta[0], theta[1], theta[2:2 + n], theta[2 + n:2 + 2 * n]
    lh = np.exp(mu + home + a[f.hi] - d[f.ai])
    la = np.exp(mu + a[f.ai] - d[f.hi])
    lo, hi = rho_range(np.concatenate([theta, [0.0]]), f)
    lo, hi = max(lo, -RHO_LIMIT), min(hi, RHO_LIMIT)
    if lo >= hi:
        return 0.0
    m00, m01 = (f.x == 0) & (f.y == 0), (f.x == 0) & (f.y == 1)
    m10, m11 = (f.x == 1) & (f.y == 0), (f.x == 1) & (f.y == 1)

    def neg(rho):
        tau = np.ones(len(f.x))
        tau[m00] = 1 - lh[m00] * la[m00] * rho
        tau[m01] = 1 + lh[m01] * rho
        tau[m10] = 1 + la[m10] * rho
        tau[m11] = 1 - rho
        return -float(np.sum(f.w * np.log(np.maximum(tau, TAU_FLOOR))))

    res = minimize_scalar(neg, bounds=(lo, hi), method="bounded", options={"xatol": 1e-7})
    return float(res.x)


# ======================= 模型 =======================
class GoalsModel(Model):
    """泊松 / Dixon-Coles / DC + 负二项（看 dc、nb 两个开关）。每个联赛一套参数，每周重新拟合（回测框架调 fit）。"""

    def __init__(self, name: str = "poisson", dc: bool = False, nb: bool = False, window_days: float = 1461,
                 half_life_days: float | None = 365, ridge: float = 3.0, promoted_attack: float = -0.2,
                 promoted_defence: float = -0.2, promoted_history_seasons: int = 3, min_matches: int = 200,
                 max_goals: int = 10, target: str = "goals", two_stage: bool = False, require=(),
                 usage: str = "A版输入"):
        self.name = name
        self.dc, self.nb = bool(dc), bool(nb)
        # ---- 步骤 4 的开关（见文件开头第七部分） ----
        if target not in TARGETS:
            raise ValueError(f"target 只能是 {TARGETS}，现在是 {target!r}")
        self.target = target
        self.two_stage = bool(two_stage) or target != "goals"     # 不用实际进球时一定分两步（rho 只看真实比分）
        if self.nb and self.two_stage:
            raise ValueError("负二项只支持 target = goals、一次拟合（离散度 α 是关于真实进球数的）")
        self.require = list(dict.fromkeys([*require, *TARGET_NEEDS[target]]))
        unknown = [r for r in self.require if r not in REQUIRE]
        if unknown:
            raise ValueError(f"require 只能是 {list(REQUIRE)}，现在有 {unknown}")
        self.data = ("比分", *(REQUIRE[r][0] for r in self.require))
        self.usage = usage
        self.coef_log: list[dict] = []       # target = shots 时每次拟合每个联赛的 a、b（回测完存下来给人看）
        self.window_days = float(window_days)
        self.half_life_days = None if half_life_days is None else float(half_life_days)
        self.ridge = float(ridge)
        self.promoted_default = (float(promoted_attack), float(promoted_defence))
        self.promoted_history_seasons = int(promoted_history_seasons)
        self.min_matches = int(min_matches)
        self.max_goals = int(max_goals)
        # ---- 状态 ----
        self.fits: dict[str, dict] = {}                        # 联赛 → 最近一次拟合的结果（见 fit）
        self.members: dict[tuple[str, int], set] = {}          # (联赛, 赛季) → 截止前已经知道在这个联赛踢过的队
        self.prior: dict[str, tuple[float, float]] = {}        # 联赛 → 升班马先验 (进攻力, 防守力)

    # ---------------- 升班马 ----------------
    def _promoted(self, comp: str, season: int) -> set:
        """这个赛季的升班马：这个赛季在、上赛季不在这个联赛的队。联赛第一个有数据的赛季没有“上赛季”，算没有升班马。"""
        prev = self.members.get((comp, season - 1))
        if prev is None:
            return set()
        return self.members.get((comp, season), set()) - prev

    def _new_prior(self, comp: str, season: int) -> tuple[float, float]:
        """升班马先验：最近 promoted_history_seasons 个赛季（不含 season）的升班马，在上一次拟合里的 a、d 的平均。"""
        old = self.fits.get(comp)
        if old is None:
            return self.prior.get(comp, self.promoted_default)
        # 按队名排好再求平均：集合（set）的顺序每次运行可能不同，加法顺序不同，最后几位小数就会差一点，
        # 回测结果会在 0.0001 左右来回变（2026-10-31 发现）；排好序后每次都完全一样
        vals = [(old["a"][t], old["d"][t]) for s in range(season - self.promoted_history_seasons, season)
                for t in sorted(self._promoted(comp, s)) if t in old["a"]]
        if not vals:
            return self.prior.get(comp, self.promoted_default)
        a, d = np.mean(vals, axis=0)
        return float(a), float(d)

    # ---------------- 拟合 ----------------
    def fit(self, history: pd.DataFrame, cutoff: str) -> None:
        if history.empty:
            return
        h = history[history["home_goals"].notna() & history["away_goals"].notna()]
        # 每个联赛每个赛季有哪些队（全部历史，不只是窗口里的；用来认升班马）
        self.members = {}
        for (comp, season), part in h.groupby(["competition", "season"]):
            self.members[(comp, int(season))] = set(part["home_team"]) | set(part["away_team"])
        recent = h[h["kickoff_earliest_utc"] >= window_start(cutoff, self.window_days)]
        # require：缺这几类数据的比赛不进拟合（球队名单、升班马还是按上面全部比分认，那个和拟合用什么无关）
        for r in self.require:
            cols = REQUIRE[r][1]
            missing = [c for c in cols if c not in recent.columns]
            if missing:
                raise ValueError(f"模型 {self.name} 要求「{REQUIRE[r][0]}」，但取来的数据里没有 {missing}")
            recent = recent[recent[cols].notna().all(axis=1)]
        new_fits = {}
        for comp, part in recent.groupby("competition"):
            if len(part) < self.min_matches:
                continue
            season = int(part["season"].max())
            prior = self._new_prior(comp, season)
            self.prior[comp] = prior
            promoted = self._promoted(comp, season)
            teams = sorted(set(part["home_team"]) | set(part["away_team"]))
            idx = {t: i for i, t in enumerate(teams)}
            ca = np.array([prior[0] if t in promoted else 0.0 for t in teams])
            cd = np.array([prior[1] if t in promoted else 0.0 for t in teams])
            w = time_weights(part["kickoff_earliest_utc"], cutoff, self.half_life_days)
            hi_, ai_ = part["home_team"].map(idx).to_numpy(int), part["away_team"].map(idx).to_numpy(int)
            goals_x, goals_y = part["home_goals"].to_numpy(int), part["away_goals"].to_numpy(int)
            if not self.two_stage:
                # 步骤 1–3 原来的做法：真实比分，一次把所有参数（含 rho、alpha）拟合出来
                f = FitData(hi=hi_, ai=ai_, x=goals_x, y=goals_y, w=w, n_teams=len(teams), center_a=ca,
                            center_d=cd, ridge=self.ridge, dc=self.dc, nb=self.nb)
                theta = fit_league(f, self._start(comp, teams, ca, cd, part))
            else:
                # 两步：第 1 步用 target 拟合实力（不带 DC），第 2 步固定实力、用真实比分找 rho
                coef = None
                if self.target == "shots":
                    coef = shot_coefficients(part, w)
                    self.coef_log.append({"cutoff": cutoff, "competition": comp, "a_sot": coef[0],
                                          "b_off": coef[1], "n": len(part)})
                tx, ty = fit_targets(part, self.target, coef)
                f1 = FitData(hi=hi_, ai=ai_, x=tx, y=ty, w=w, n_teams=len(teams), center_a=ca, center_d=cd,
                             ridge=self.ridge, dc=False, nb=False)
                theta = fit_league(f1, self._start(comp, teams, ca, cd, part, dc=False))
                if self.dc:
                    f2 = FitData(hi=hi_, ai=ai_, x=goals_x, y=goals_y, w=w, n_teams=len(teams), center_a=ca,
                                 center_d=cd, ridge=self.ridge, dc=True, nb=False)
                    theta = np.concatenate([theta, [fit_rho(f2, theta)]])
                f = f1
            mu, home, a, d, rho, alpha = _unpack(theta, len(teams), self.dc, self.nb)
            new_fits[comp] = {"mu": float(mu), "home": float(home), "rho": float(rho), "alpha": float(alpha),
                              "a": dict(zip(teams, map(float, a))), "d": dict(zip(teams, map(float, d))),
                              "season": season, "n": len(part), "n_eff": float(f.w.sum()), "prior": prior}
        self.fits = new_fits

    def _start(self, comp: str, teams: list, ca: np.ndarray, cd: np.ndarray, part: pd.DataFrame,
               dc: bool | None = None) -> np.ndarray:
        """拟合的起点：有上一次拟合就接着用（球队对得上的用旧值，新队用收缩中心）；没有就用场均进球。
        dc：起点里要不要放 rho（两步拟合的第 1 步不放）；不给就按模型的开关。"""
        dc = self.dc if dc is None else dc
        old = self.fits.get(comp)
        if old is None:
            mean_goals = (part["home_goals"].mean() + part["away_goals"].mean()) / 2
            mu, home, rho, alpha = np.log(max(mean_goals, 0.1)), 0.25, -0.05, 0.02
            a, d = ca.copy(), cd.copy()
        else:
            mu, home = old["mu"], old["home"]
            rho = old["rho"] if old["rho"] != 0 else -0.05
            alpha = old["alpha"] if old["alpha"] > 0 else 0.02
            a = np.array([old["a"].get(t, c) for t, c in zip(teams, ca)])
            d = np.array([old["d"].get(t, c) for t, c in zip(teams, cd)])
        parts = [[mu, home], a, d]
        if dc:
            parts.append([rho])
        if self.nb:
            parts.append([max(alpha, ALPHA_MIN)])
        return np.concatenate([np.asarray(p, dtype=float) for p in parts])

    # ---------------- 预测 ----------------
    def _strength(self, fit: dict, comp: str, season: int, team: str) -> tuple[float, float]:
        """球队的 (进攻力, 防守力)。这个赛季或上赛季在这个联赛踢过、又有拟合结果 → 用自己的；否则用升班马先验。"""
        known = team in self.members.get((comp, season), ()) or team in self.members.get((comp, season - 1), ())
        if known and team in fit["a"]:
            return fit["a"][team], fit["d"][team]
        return self._new_prior(comp, season)

    def expected_goals(self, targets: pd.DataFrame) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """每场 (λ主, λ客, 有没有拟合结果)。没有拟合结果的联赛（窗口里比赛太少）λ 是 nan。"""
        n = len(targets)
        lh, la, ok = np.full(n, np.nan), np.full(n, np.nan), np.zeros(n, bool)
        for i, (comp, season, home, away) in enumerate(targets[["competition", "season", "home_team", "away_team"]]
                                                       .itertuples(index=False, name=None)):
            fit = self.fits.get(comp)
            if fit is None:
                continue
            ah, dh = self._strength(fit, comp, int(season), home)
            aa, da = self._strength(fit, comp, int(season), away)
            lh[i] = np.exp(fit["mu"] + fit["home"] + ah - da)
            la[i] = np.exp(fit["mu"] + aa - dh)
            ok[i] = True
        return lh, la, ok

    def predict(self, targets: pd.DataFrame, cutoff: str) -> list:
        lh, la, ok = self.expected_goals(targets)
        out: list = [None] * len(targets)
        comps = targets["competition"].to_numpy()
        for comp in set(comps[ok]):
            rows = np.flatnonzero(ok & (comps == comp))
            fit = self.fits[comp]
            mats = score_matrices(lh[rows], la[rows], fit["rho"], fit["alpha"], self.max_goals)
            for r, m in zip(rows, mats):
                out[r] = m
        return out

    # ---------------- 给人看 ----------------
    def params(self) -> dict:
        out = {}
        for c, f in self.fits.items():
            out[c] = {"基准mu": round(f["mu"], 4), "主场优势home": round(f["home"], 4),
                      "两队都是平均水平时的预期进球（主/客）": [round(float(np.exp(f["mu"] + f["home"])), 3),
                                                   round(float(np.exp(f["mu"])), 3)],
                      "升班马先验(a,d)": [round(v, 3) for v in f["prior"]], "场数": f["n"],
                      "有效场数（衰减后）": round(f["n_eff"], 1)}
            if self.dc:
                out[c]["rho"] = round(f["rho"], 4)
            if self.nb:
                out[c]["负二项alpha"] = round(f["alpha"], 5)
        return out

    def ratings_table(self) -> pd.DataFrame:
        """最近一次拟合、当前赛季各队的进攻力和防守力（0 = 联赛平均；e^a = 进球是平均的几倍），按 a + d 从高到低。"""
        rows = []
        for c, f in self.fits.items():
            for t in sorted(self.members.get((c, f["season"]), ())):
                if t in f["a"]:
                    rows.append({"competition": c, "season": f["season"], "team": t,
                                 "attack": round(f["a"][t], 4), "defence": round(f["d"][t], 4),
                                 "total": round(f["a"][t] + f["d"][t], 4)})
        if not rows:
            return pd.DataFrame(columns=["competition", "season", "team", "attack", "defence", "total"])
        return pd.DataFrame(rows).sort_values(["competition", "total"], ascending=[True, False], ignore_index=True)
