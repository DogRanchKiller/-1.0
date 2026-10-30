"""第 0 步模型之二：Elo 评分（需求文档第 4 节步骤 0“Elo 用于实力评分”，以及“特殊时间点”）。

一、Elo 怎么算（每场赛后更新一次）
  每队一个分数，开始都是 1500（settings.yaml 的 models.elo.initial）。
  1. 赛前算“预期得分”：E_主 = 1 ÷ (1 + 10^(−(R_主 + H − R_客) ÷ 400))
     R 是评分，H 是主场优势（models.elo.home_advantage，默认 65 分）。实际得分：赢 = 1，平 = 0.5，输 = 0。
     例：两队都是 1500，主场 +65 → 差 65 分 → E_主 = 1 ÷ (1 + 10^(−0.1625)) ≈ 0.593。
  2. 赛后：R_主 新 = R_主 + K × (实际得分 − E_主)；客队减去同样多（两队加减的分一样，总分不变）。
     例：K = 20，主队赢了 → 主队 +20 × (1 − 0.593) ≈ +8.15，客队 −8.15。打平 → 主队 20 × (0.5 − 0.593) ≈ −1.85。
     冷门（弱队赢强队）加得多，强队赢弱队只加一点点。
  3. 可选（models.elo.goal_diff）：净胜球越多变得越多，K 乘一个系数：净胜 0–1 球 ×1，2 球 ×1.5，3 球以上 ×(11 + N) ÷ 8
     （这是 eloratings.net 国家队 Elo 的做法）。例：3–0 → ×1.75。

二、特殊时间点（第 4 节）
  - 新赛季：留在联赛里的球队，评分往“上赛季末的联赛平均”拉回一部分（models.elo.season_regression，默认 1/3）。
    例：上赛季末联赛平均 1500，某队 1560 → 1560 − (1560 − 1500) × 1/3 = 1540。
    理由：休赛期换人、换帅，上赛季的强弱只能部分延续到新赛季。
  - 升班马：初始分 = 上赛季末联赛平均 + “往年升班马的平均水平”。
    往年升班马的平均水平 = 最近几个赛季（models.elo.promoted_history_seasons）的升班马，在它们升上来那个赛季结束时，
    评分比当时的联赛平均低多少。还没有记录时用 models.elo.promoted_default（−100）。
    只用截止前已经结束的赛季，所以不偷看未来。
    第一个有数据的赛季（例如英超 1993/94）没有“上赛季”，所有队都从 1500 开始。
  - 降级的队离开这个联赛，评分就不用了；以后再升上来，按升班马重新给初始分，不接着用很多年前的旧分。
  - 每个联赛单独一套评分。次级联赛的评分不带到顶级联赛（第 2 阶段只回测顶级联赛；以后可以试“用次级联赛评分当升班马初始值”）。
  怎么知道“新赛季开始了”：某联赛第一次碰到新赛季的比赛（不管是更新评分还是给这个赛季的比赛做预测）时，
  上赛季一定已经全部踢完、结果都知道了，这时做“拉回”和“给升班马初始分”。赛程赛前就公布，知道这个赛季有哪些队不算偷看。

三、评分怎么变成胜平负概率
  Elo 只给“预期得分”，不分平局。用有序 logit（见 ordered_logit.py，那里写了为什么不用分箱）：
  拿截止前最近 5 年（models.elo.logit_window_days）每场比赛“赛前的评分差（含主场优势）”和真实结果，
  每个联赛单独拟合一组参数（各联赛平局多少不一样，例如法甲平局多）。
  每个联赛有数据的头两个赛季（models.elo.burn_in_seasons）评分还没稳定（大家都从 1500 开始），不用来拟合。

四、在回测里怎么跑
  update()：每次有新比赛结果能知道，就按开球顺序更新评分（便宜，每场都做）。
  fit()：每周一次（backtest.refit_days），先把还没处理的比赛更新进评分，再重新拟合“评分 → 概率”的参数。
  predict()：用截止时间那一刻的评分 + 最近一次拟合的参数。
"""
from __future__ import annotations

from collections import defaultdict

import numpy as np
import pandas as pd

from src.models import ordered_logit
from src.models.base import Model, window_start

RESULT_SCORE = {"H": 1.0, "D": 0.5, "A": 0.0}       # 主队的实际得分
RESULT_INDEX = {"H": 0, "D": 1, "A": 2}             # 和 metrics.py 一样：0 主胜、1 平、2 客胜


def expected_home(r_home: float, r_away: float, home_advantage: float) -> float:
    """主队的预期得分（0–1）。公式见文件开头第一部分。"""
    return 1.0 / (1.0 + 10 ** (-(r_home + home_advantage - r_away) / 400.0))


def goal_diff_multiplier(home_goals: int, away_goals: int) -> float:
    """净胜球系数：净胜 0–1 球 ×1，2 球 ×1.5，N ≥ 3 球 ×(11 + N) ÷ 8。"""
    n = abs(int(home_goals) - int(away_goals))
    if n <= 1:
        return 1.0
    if n == 2:
        return 1.5
    return (11 + n) / 8


class EloModel(Model):
    """每个联赛一套 Elo 评分 + 每个联赛一组有序 logit 参数。"""

    def __init__(self, name: str = "elo", initial: float = 1500, k: float = 20, home_advantage: float = 65,
                 goal_diff: bool = False, season_regression: float = 1 / 3, promoted_default: float = -100,
                 promoted_history_seasons: int = 5, burn_in_seasons: int = 2, logit_window_days: float = 1825,
                 min_matches: int = 200):
        self.name = name
        self.initial = initial
        self.k = k
        self.home_advantage = home_advantage
        self.goal_diff = goal_diff
        self.season_regression = season_regression
        self.promoted_default = promoted_default
        self.promoted_history_seasons = promoted_history_seasons
        self.burn_in_seasons = burn_in_seasons
        self.logit_window_days = logit_window_days
        self.min_matches = min_matches
        # ---- 状态 ----
        self.ratings: dict[tuple[str, str], float] = {}               # (联赛, 球队) → 当前评分
        self.team_season: dict[tuple[str, str], int] = {}             # (联赛, 球队) → 这个评分属于哪个赛季
        self.current_season: dict[str, int] = {}                      # 联赛 → 现在是哪个赛季
        self.first_season: dict[str, int] = {}                        # 联赛 → 第一个有数据的赛季
        self.members: dict[tuple[str, int], set] = defaultdict(set)   # (联赛, 赛季) → 这个赛季有哪些队
        self.season_base: dict[tuple[str, int], float] = {}           # (联赛, 赛季) → 赛季开始时用的“上赛季末联赛平均”
        self.promoted: dict[tuple[str, int], set] = defaultdict(set)  # (联赛, 赛季) → 这个赛季的升班马
        self.promoted_offsets: dict[str, list] = defaultdict(list)    # 联赛 → [(赛季, 升班马赛季末评分 − 联赛平均), ...]
        self.records: dict[str, list] = defaultdict(list)             # 联赛 → [(开球时间, 赛前评分差含主场, 结果 0/1/2), ...]
        self.processed: set[int] = set()                              # 已经更新进评分的比赛
        self.theta: dict[str, np.ndarray] = {}                        # 联赛 → 有序 logit 参数
        self.fit_n: dict[str, int] = {}                               # 联赛 → 拟合用了几场

    # ---------------- 新赛季、升班马 ----------------
    def _season_mean(self, comp: str, season: int) -> float:
        """某联赛某赛季所有队现在的平均评分（赛季结束后调用 = 赛季末联赛平均）。"""
        teams = self.members.get((comp, season))
        if not teams:
            return self.initial
        return float(np.mean([self.ratings[(comp, t)] for t in teams]))

    def _promoted_offset(self, comp: str, season: int) -> float:
        """升班马初始分比联赛平均低多少：最近几个已经结束的赛季的升班马的平均；没有记录用默认值。"""
        seasons = sorted({s for s, _ in self.promoted_offsets[comp] if s < season})[-self.promoted_history_seasons:]
        past = [off for s, off in self.promoted_offsets[comp] if s in seasons]
        return float(np.mean(past)) if past else float(self.promoted_default)

    def _roll_season(self, comp: str, season: int) -> None:
        """某联赛第一次碰到 season 赛季：记下上赛季升班马的表现，算出上赛季末的联赛平均，作为新赛季的基准。"""
        prev = self.current_season.get(comp)
        if prev is None:                                  # 这个联赛第一个有数据的赛季
            self.first_season[comp] = season
            self.season_base[(comp, season)] = float(self.initial)
        else:
            base = self._season_mean(comp, prev)
            for t in self.promoted.get((comp, prev), ()):  # 上赛季升班马：赛季末比联赛平均差多少
                self.promoted_offsets[comp].append((prev, self.ratings[(comp, t)] - base))
            self.season_base[(comp, season)] = base
        self.current_season[comp] = season

    def _ensure_team(self, comp: str, season: int, team: str) -> None:
        """球队第一次出现在这个联赛的这个赛季：给它这个赛季的初始评分（留队的拉回，升班马给初始分）。"""
        if self.current_season.get(comp) is None or season > self.current_season[comp]:
            self._roll_season(comp, season)
        key = (comp, team)
        if self.team_season.get(key) == season:
            return
        base = self.season_base[(comp, season)]
        if self.first_season[comp] == season:
            self.ratings[key] = float(self.initial)
        elif self.team_season.get(key) == season - 1:     # 上赛季也在这个联赛：往联赛平均拉回
            r = self.ratings[key]
            self.ratings[key] = r - (r - base) * self.season_regression
        else:                                             # 升班马（或者很多年前在过，又升回来）
            self.ratings[key] = base + self._promoted_offset(comp, season)
            self.promoted[(comp, season)].add(team)
        self.team_season[key] = season
        self.members[(comp, season)].add(team)

    # ---------------- 每场赛后更新 ----------------
    def _process(self, rows: pd.DataFrame) -> None:
        """把还没处理过的比赛按开球顺序更新进评分，同时记下“赛前评分差 + 结果”给有序 logit 拟合用。"""
        new = rows[~rows["match_id"].isin(self.processed)]
        new = new[new["result"].notna() & new["home_goals"].notna()]
        if new.empty:
            return
        new = new.sort_values(["kickoff_earliest_utc", "match_id"])
        for mid, comp, season, home, away, hg, ag, res, ko in new[
                ["match_id", "competition", "season", "home_team", "away_team", "home_goals", "away_goals",
                 "result", "kickoff_earliest_utc"]].itertuples(index=False, name=None):
            season = int(season)
            late = self.current_season.get(comp) is not None and season < self.current_season[comp]
            if late and ((comp, home) not in self.ratings or (comp, away) not in self.ratings):
                self.processed.add(int(mid))   # 极少见：上赛季的补赛结果在新赛季开始后才知道，而且有队没有评分 → 跳过
                continue
            if not late:
                self._ensure_team(comp, season, home)
                self._ensure_team(comp, season, away)
            # late（上赛季补赛）：直接用两队现在的评分更新，不再改“这个赛季有哪些队”
            rh, ra = self.ratings[(comp, home)], self.ratings[(comp, away)]
            diff = rh + self.home_advantage - ra
            if not late and season >= self.first_season[comp] + self.burn_in_seasons:
                self.records[comp].append((ko, diff, RESULT_INDEX[res]))
            e = expected_home(rh, ra, self.home_advantage)
            k = self.k * (goal_diff_multiplier(hg, ag) if self.goal_diff else 1.0)
            change = k * (RESULT_SCORE[res] - e)
            self.ratings[(comp, home)] = rh + change
            self.ratings[(comp, away)] = ra - change
            self.processed.add(int(mid))

    def update(self, new_rows: pd.DataFrame, cutoff: str) -> None:
        self._process(new_rows)

    def fit(self, history: pd.DataFrame, cutoff: str) -> None:
        """先补上还没处理的比赛，再按联赛重新拟合有序 logit（只用截止前最近 logit_window_days 天的比赛）。"""
        self._process(history)
        start = window_start(cutoff, self.logit_window_days)
        for comp, recs in self.records.items():
            use = [(d, y) for ko, d, y in recs if start <= ko <= cutoff]
            if len(use) < self.min_matches:
                self.theta.pop(comp, None)
                continue
            d, y = np.array(use, dtype=float).T
            self.theta[comp] = ordered_logit.fit_ordered_logit(d, y.astype(int), start=self.theta.get(comp))
            self.fit_n[comp] = len(use)

    # ---------------- 预测 ----------------
    def rating_diff(self, comp: str, season: int, home: str, away: str) -> float:
        """这场比赛赛前的评分差（主队评分 + 主场优势 − 客队评分）。没见过的队按新赛季规则给初始分。"""
        if self.current_season.get(comp) is not None and season < self.current_season[comp]:
            # 上赛季的补赛，开打时这个联赛已经进入新赛季：用两队现在的评分（没有评分的按新赛季基准）
            base = self.season_base[(comp, self.current_season[comp])]
            return self.ratings.get((comp, home), base) + self.home_advantage - self.ratings.get((comp, away), base)
        self._ensure_team(comp, season, home)
        self._ensure_team(comp, season, away)
        return self.ratings[(comp, home)] + self.home_advantage - self.ratings[(comp, away)]

    def predict(self, targets: pd.DataFrame, cutoff: str) -> list:
        out = []
        for comp, season, home, away in targets[["competition", "season", "home_team", "away_team"]].itertuples(
                index=False, name=None):
            theta = self.theta.get(comp)
            if theta is None:
                out.append(None)
                continue
            out.append(ordered_logit.result_probs(theta, self.rating_diff(comp, int(season), home, away))[0])
        return out

    # ---------------- 给人看 ----------------
    def params(self) -> dict:
        return {"设置": {"K": self.k, "主场优势": self.home_advantage, "净胜球系数": self.goal_diff,
                       "新赛季拉回": self.season_regression, "升班马默认": self.promoted_default},
                "有序logit": {c: {**ordered_logit.describe(t), "场数": self.fit_n.get(c)} for c, t in self.theta.items()},
                "升班马平均差": {c: round(self._promoted_offset(c, self.current_season[c]), 1)
                           for c in self.current_season}}

    def ratings_table(self) -> pd.DataFrame:
        """每个联赛当前赛季各队的评分，从高到低。"""
        rows = [{"competition": c, "season": s, "team": t, "rating": round(self.ratings[(c, t)], 1)}
                for (c, s), teams in self.members.items() if self.current_season.get(c) == s for t in teams]
        return pd.DataFrame(rows).sort_values(["competition", "rating"], ascending=[True, False], ignore_index=True)
