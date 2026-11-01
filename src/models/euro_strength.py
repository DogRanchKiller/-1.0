"""第 8 阶段：欧战实力表（需求文档第 4 节“第 8 阶段怎么做”里的“欧战实力表”）。

要回答的问题：不同联赛的球队在欧战里碰头，怎么比较实力？
做法：在每队的联赛实力（来自 dyn 的 a 进攻、d 防守）之上，给每个联赛加一个偏移 δ（五大联赛各一个，外加“联赛外”一个）：
  log λ主 = μ + h·(非中立场) + (a主 + δ主) − (d客 + δ客)
  log λ客 = μ + (a客 + δ客) − (d主 + δ主)
  联赛外球队 a = d = 0，只靠它所在组的 δ。进球按独立泊松（rho = 0）。
注意：λ主里是 δ主 − δ客，λ客里是 δ客 − δ主，所以只有 δ 之间的差有意义；
所有 δ 同加一个数，预测一点不变。靠对 δ 的岭惩罚（先验均值 0、标准差 delta_prior_sd）定住，
对外报告时再减去五个联赛 δ 的平均值（deltas()）。

拟合（fit）：加权泊松对数似然 + δ 的岭惩罚，用 L-BFGS-B。
  权重和窗口沿用 models.goals 的半衰期和窗口（time_weights、window_start）；窗口里的欧战场数不到 min_matches 就不出。
  每场比赛两个进球数各算一项。

评分快照（add）：每场欧战比赛“进入”这个表时，按当时 dyn 的状态记下两队的 a、d（dyn_team_table 取）。
  调用的人要保证：这些比赛的比分在截止时间前已经能知道，而且 dyn 已经处理到同一个截止时间（先 dyn.fit 再 add）。
  快照比比赛本身晚几天（滚动时每 7 天进来一批），不会用到截止时间以后的数据。
  这是近似：用赛后几天内的评分代替开球那一刻的评分，每场比赛的评分以后不再更新。

本文件不碰数据库和取数层（和 src/models 其他模型一样），不改 dyn、goals、gbdt。
这个表不是预测联赛的模型，不进页面、不进融合。
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.optimize import minimize

from src.common import markets
from src.models.base import window_start
from src.models.goals import score_matrices, time_weights

OUTSIDE = '联赛外'     # 不在五大联赛的球队（含次级联赛球队）一律算这一组


def dyn_team_table(dyn, rating_leagues) -> dict:
    """dyn 现在的状态 → {球队: (联赛代码, a, d)}。只放 rating_leagues 里的联赛。

    一个联赛的球队 = 这个赛季和上个赛季踢过的队（赛季刚开始、有的队还没踢第一场，也算在里面）。
    联赛还没过 burn_in_seasons（dyn 头几个赛季只让估计稳定，不出预测）的，a、d 记 nan，add() 会跳过这样的比赛。
    调用前 dyn 要先 fit 过（fit 里会把平均值挪到 mu 上，a、d 才是“比联赛平均强多少”）。
    """
    out: dict = {}
    for comp in rating_leagues:
        lg = dyn.leagues.get(comp)
        if lg is None:
            continue
        ready = lg.season - lg.first_season >= dyn.burn_in_seasons
        prev = lg.members.get(lg.season - 1) or []
        cur = lg.members.get(lg.season) or []
        for team in dict.fromkeys([*prev, *cur]):
            if team not in lg.idx:
                continue
            i, j = lg.pos(team)
            out[team] = (comp, float(lg.m[i]), float(lg.m[j])) if ready else (comp, float('nan'), float('nan'))
    return out


class EuroStrength:
    """欧战实力表。with_delta = False 是对照版：δ 固定为 0，其余（μ、h、窗口、权重）完全一样。"""

    def __init__(self, rating_leagues, delta_prior_sd: float = 0.5, window_days: float = 1461.0,
                 half_life_days: float | None = 365.0, min_matches: int = 200, max_goals: int = 10,
                 with_delta: bool = True):
        self.leagues = list(rating_leagues)
        self.groups = [*self.leagues, OUTSIDE]
        self.sd = float(delta_prior_sd)
        self.window_days = float(window_days)
        self.half_life_days = None if half_life_days is None else float(half_life_days)
        self.min_matches = int(min_matches)
        self.max_goals = int(max_goals)
        self.with_delta = bool(with_delta)
        # 快照：(开球时间, match_id, 主队组, 客队组, a主 − d客, a客 − d主, 是不是主客场(1/0), 主进球, 客进球)
        self.rows: list[tuple] = []
        self.seen: set[int] = set()
        self.skipped = 0                                  # 因为联赛评分还拿不到而跳过的比赛数
        self.theta = np.zeros(2 + len(self.groups))       # [μ, h, δ 各组]
        self.theta[0], self.theta[1] = 0.2, 0.2
        self.ready = False
        self.n_used = 0                                   # 最近一次拟合用了多少场
        self.converged = True

    @classmethod
    def from_settings(cls, settings: dict, with_delta: bool = True) -> 'EuroStrength':
        """按 settings.yaml 造：euro 段 + models.goals 的窗口、半衰期 + models.min_matches + calc.max_goals。"""
        e, g = settings['euro'], settings['models']['goals']
        return cls(e['rating_leagues'], e['delta_prior_sd'], g['window_days'], g['half_life_days'],
                   settings['models']['min_matches'], settings['calc']['max_goals'], with_delta)

    # ---------------- 球队的组和评分 ----------------
    def _lookup(self, table: dict, team: str) -> tuple[int, float, float]:
        """球队 → (组编号, a, d)。表里没有的算联赛外：最后一组，a = d = 0。"""
        v = table.get(team)
        if v is None:
            return len(self.leagues), 0.0, 0.0
        return self.leagues.index(v[0]), float(v[1]), float(v[2])

    # ---------------- 记快照 ----------------
    def add(self, matches: pd.DataFrame, table: dict) -> int:
        """把新知道比分的欧战比赛记进来（match_id 已经记过的跳过）。matches 要有
        match_id、home_team、away_team、kickoff_earliest_utc、home_goals、away_goals、neutral（src/features/euro_matches.py 整理好的表）。
        table 是 dyn_team_table 的结果。返回这次新记了几场。"""
        if matches is None or len(matches) == 0:
            return 0
        n = 0
        for r in matches.sort_values(['kickoff_earliest_utc', 'match_id']).to_dict('records'):
            mid = int(r['match_id'])
            if mid in self.seen:
                continue
            self.seen.add(mid)
            gh, ah, dh = self._lookup(table, r['home_team'])
            ga, aa, da = self._lookup(table, r['away_team'])
            if np.isnan([ah, dh, aa, da]).any():
                self.skipped += 1
                continue
            self.rows.append((r['kickoff_earliest_utc'], mid, gh, ga, ah - da, aa - dh,
                              0.0 if bool(r['neutral']) else 1.0, int(r['home_goals']), int(r['away_goals'])))
            n += 1
        return n

    # ---------------- 拟合 ----------------
    def fit(self, cutoff) -> None:
        """用截止时间之前、窗口里的快照估 μ、h、δ。窗口里的场数不到 min_matches：ready = False，不出预测。"""
        cutoff = str(cutoff)
        start = window_start(cutoff, self.window_days)
        use = [r for r in self.rows if start <= r[0] <= cutoff]
        self.n_used = len(use)
        self.ready = len(use) >= self.min_matches
        if not self.ready:
            return
        w = np.asarray(time_weights(pd.Series([r[0] for r in use]), cutoff, self.half_life_days), dtype=float)
        gh = np.array([r[2] for r in use], dtype=int)
        ga = np.array([r[3] for r in use], dtype=int)
        off_h = np.array([r[4] for r in use], dtype=float)
        off_a = np.array([r[5] for r in use], dtype=float)
        nn = np.array([r[6] for r in use], dtype=float)
        x = np.array([r[7] for r in use], dtype=float)
        y = np.array([r[8] for r in use], dtype=float)
        k, sd2 = len(self.groups), self.sd ** 2

        def nll(th):
            mu, hm, dl = th[0], th[1], th[2:]
            g = dl[gh] - dl[ga]
            eh, ea = mu + hm * nn + off_h + g, mu + off_a - g
            lh, la = np.exp(eh), np.exp(ea)
            val = (w * (lh - x * eh + la - y * ea)).sum() + (dl ** 2).sum() / (2 * sd2)
            rh, ra = w * (lh - x), w * (la - y)
            dg = rh - ra
            grad = np.empty_like(th)
            grad[0] = (rh + ra).sum()
            grad[1] = (rh * nn).sum()
            grad[2:] = (np.bincount(gh, weights=dg, minlength=k) - np.bincount(ga, weights=dg, minlength=k)
                        + dl / sd2)
            return val, grad

        free = (None, None) if self.with_delta else (0.0, 0.0)      # 对照版：δ 锁在 0
        res = minimize(nll, self.theta, jac=True, method='L-BFGS-B', bounds=[(None, None)] * 2 + [free] * k)
        self.theta = res.x
        self.converged = bool(res.success)

    # ---------------- 看结果 ----------------
    @property
    def mu(self) -> float:
        return float(self.theta[0])

    @property
    def home(self) -> float:
        return float(self.theta[1])

    def deltas(self) -> dict:
        """每组的 δ（减去五个联赛 δ 的平均值，所以五个联赛的和是 0；联赛外是比这个平均高多少）。"""
        d = self.theta[2:]
        base = float(d[:len(self.leagues)].mean())
        return {g: float(v) - base for g, v in zip(self.groups, d)}

    def params(self) -> dict:
        return {'μ': round(self.mu, 4), '主场优势h': round(self.home, 4), 'δ': {g: round(v, 4) for g, v in self.deltas().items()},
                '窗口里的场数': self.n_used, '跳过（联赛评分拿不到）': self.skipped, '拟合收敛': self.converged}

    # ---------------- 预测 ----------------
    def expected_goals(self, home: str, away: str, neutral: bool, table: dict):
        """(λ主, λ客)。还没拟合出来（窗口里场数不够）或两队评分拿不到，返回 None。"""
        if not self.ready:
            return None
        gh, ah, dh = self._lookup(table, home)
        ga, aa, da = self._lookup(table, away)
        if np.isnan([ah, dh, aa, da]).any():
            return None
        dl = self.theta[2:]
        g = float(dl[gh] - dl[ga])
        eh = self.mu + (0.0 if neutral else self.home) + ah - da + g
        ea = self.mu + aa - dh - g
        return float(np.exp(eh)), float(np.exp(ea))

    def expected_margin(self, home: str, away: str, neutral: bool, table: dict):
        """预期净胜球（主队 − 客队）= λ主 − λ客（独立泊松，进球数的平均相减）。没有预测返回 None。"""
        v = self.expected_goals(home, away, neutral, table)
        return None if v is None else v[0] - v[1]

    def result_probs(self, home: str, away: str, neutral: bool, table: dict):
        """胜平负三个概率（比分矩阵推出来的，矩阵开到 max_goals 球）。没有预测返回 None。"""
        v = self.expected_goals(home, away, neutral, table)
        if v is None:
            return None
        return np.asarray(markets.result_probs(score_matrices(v[0], v[1], 0.0, 0.0, self.max_goals)[0]), dtype=float)
