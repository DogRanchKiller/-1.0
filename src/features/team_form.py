"""步骤 6 的特征计算器（需求文档第 4 节“步骤 6 怎么做”）：pi-rating、积分榜、近期状态。

这三个都是“一场一场往前走”的计算器：
  view(...)    这场比赛赛前能知道的数（不改任何状态，预测时也能随便调用）；
  observe(...) 这场比赛踢完，把结果算进去。
用法永远是先 view、再 observe，所以一场比赛的特征里不会有这场自己的比分。
这里不碰数据库，也不拿取数层：比赛结果由调用方（src/models/gbdt.py）从取数层取好的表里一场一场交进来。

一、pi-rating（Constantinou、Fenton 2013，“Determining the level of ability of football teams by dynamic ratings”）
  每队两个分：主场分 H、客场分 A（0 = 联赛平均）。
  1. 分数换成“预期净胜球”：ψ(R) = 符号(R) × (b^(|R| ÷ c) − 1)。b = 10、c = 3：R = 0.5 → 约 0.47 球。
  2. 这场主队预期净胜球 = ψ(主队的 H) − ψ(客队的 A)；误差 e = |真实净胜球 − 预期净胜球|。
  3. 修正量 = c × log₁₀(1 + e)（误差大时也不会改得太猛），主队踢得比预期好就加、差就减。
  4. 主队：H += 修正量 × λ，A 跟着变 γ 倍；客队：A −= 修正量 × λ，H 跟着变 γ 倍。
     λ = 0.035（学多快）、γ = 0.7（主场表现有几成能说明客场）。都用原文的数，不调。
  例：两队都是 0，主队 3:0 赢 → 预期 0、误差 3 → 修正量 3 × log₁₀4 ≈ 1.81 → 主队 H +0.063、A +0.044；客队反过来。
  新赛季：留队的接着用；升班马（上个有数据的赛季不在这个联赛）= 最近 history_seasons 个赛季的升班马
  在“升上来那个赛季结束时”的平均分（还没有记录时用 0），和 dyn、Elo 的做法一样。

二、积分榜（本赛季、截止前的联赛比赛）
  踢了几场、场均积分、场均净胜球、名次比例（0 = 第一、1 = 最后，只算踢过的队）。
  名次按 积分、净胜球、进球 排，一样的算并列（不用相互战绩等细则；积分榜页面另有完整规则 src/features/standings.py）。

三、近期状态（这队在这个联赛里的最近几场）
  只看离这场 within_days 天以内的比赛（休了很久、刚升上来的队，就只有几场或者没有）。
  最近 5、10、20 场的场均：进球、失球、射正、被射正、积分；再加时间衰减的平均（半衰期按“场”算：5 场、20 场）。
  射正有空值（有的赛季没有射门数据）时，只用有数据的场次平均；一场都没有就是空值（梯度提升树能直接处理空值）。
"""
from __future__ import annotations

import math
from collections import deque

import numpy as np

STATS = ("gf", "ga", "sotf", "sota", "pts")     # 进球、失球、射正、被射正、积分


# ======================= 一、pi-rating =======================
class PiRating:
    def __init__(self, lam: float = 0.035, gamma: float = 0.7, b: float = 10.0, c: float = 3.0,
                 history_seasons: int = 3):
        self.lam, self.gamma, self.b, self.c = float(lam), float(gamma), float(b), float(c)
        self.history_seasons = int(history_seasons)
        self.r: dict[tuple[str, str], list[float]] = {}          # (联赛, 队) → [主场分, 客场分]
        self.members: dict[str, dict[int, set]] = {}             # 联赛 → 赛季 → 有哪些队
        self.promoted: dict[str, dict[int, list]] = {}           # 联赛 → 赛季 → 升班马
        self.end: dict[str, dict[int, list]] = {}                # 联赛 → 赛季 → 这季升班马赛季末的 [H, A]
        self.season: dict[str, int] = {}                         # 联赛 → 现在处在哪个赛季

    def psi(self, r: float) -> float:
        """分数 → 预期净胜球（第一部分第 1 条）。"""
        return math.copysign(self.b ** (abs(r) / self.c) - 1.0, r)

    # ---- 赛季、升班马 ----
    def _prev(self, comp: str, season: int) -> int | None:
        """这个联赛 season 之前、最近一个有队的赛季。"""
        earlier = [s for s, t in self.members.get(comp, {}).items() if s < season and t]
        return max(earlier) if earlier else None

    def is_promoted(self, comp: str, season: int, team: str) -> int:
        """1 = 升班马（上个有数据的赛季不在这个联赛）；联赛第一个赛季大家都算 0。"""
        prev = self._prev(comp, season)
        return 0 if prev is None else int(team not in self.members[comp][prev])

    def _end_of(self, comp: str, season: int) -> list:
        if season == self.season.get(comp):                      # 还没结束的赛季：用现在的分
            return [self.r[(comp, t)] for t in self.promoted.get(comp, {}).get(season, [])]
        return self.end.get(comp, {}).get(season, [])

    def _prior(self, comp: str, season: int) -> list[float]:
        vals = [v for s in range(season - self.history_seasons, season) for v in self._end_of(comp, s)]
        return [float(x) for x in np.mean(vals, axis=0)] if vals else [0.0, 0.0]

    def _start(self, comp: str, season: int, team: str):
        """这个赛季还没踢过的队从哪开始：留队的返回 None（接着用），升班马、第一个赛季返回 [H, A]。"""
        prev = self._prev(comp, season)
        if prev is None:
            return [0.0, 0.0]
        if team in self.members[comp][prev]:
            return None
        return self._prior(comp, season)

    def view(self, comp: str, season: int, team: str) -> tuple[float, float]:
        """赛前这队的 (主场分, 客场分)，不改状态。"""
        cur = self.season.get(comp)
        if cur is not None and season < cur:                     # 上赛季的补赛：用现在的分
            v = self.r.get((comp, team))
            return (float("nan"), float("nan")) if v is None else (v[0], v[1])
        if team in self.members.get(comp, {}).get(season, ()):
            v = self.r[(comp, team)]
            return v[0], v[1]
        start = self._start(comp, season, team)
        v = self.r[(comp, team)] if start is None else start
        return v[0], v[1]

    def _ensure(self, comp: str, season: int, team: str) -> None:
        seasons = self.members.setdefault(comp, {})
        cur = self.season.get(comp)
        if cur is None or season > cur:                          # 进入新赛季：记下上赛季升班马的赛季末分
            if cur is not None and self.promoted.get(comp, {}).get(cur):
                self.end.setdefault(comp, {})[cur] = [list(v) for v in self._end_of(comp, cur)]
            self.season[comp] = season
        seasons.setdefault(season, set())
        if team in seasons[season]:
            return
        start = self._start(comp, season, team)
        if start is not None:
            if self._prev(comp, season) is not None:
                self.promoted.setdefault(comp, {}).setdefault(season, []).append(team)
            self.r[(comp, team)] = list(start)
        seasons[season].add(team)

    def observe(self, comp: str, season: int, home: str, away: str, hg: int, ag: int) -> None:
        cur = self.season.get(comp)
        if cur is not None and season < cur:                     # 上赛季的补赛：两队都有分才更新
            if (comp, home) not in self.r or (comp, away) not in self.r:
                return
        else:
            self._ensure(comp, season, home)
            self._ensure(comp, season, away)
        rh, ra = self.r[(comp, home)], self.r[(comp, away)]
        gd = float(hg) - float(ag)
        egd = self.psi(rh[0]) - self.psi(ra[1])
        step = self.c * math.log10(1.0 + abs(gd - egd)) * (1.0 if gd > egd else -1.0)
        dh = step * self.lam                                     # 主队主场分的变化
        rh[0] += dh
        rh[1] += dh * self.gamma
        da = -step * self.lam                                    # 客队客场分的变化
        ra[1] += da
        ra[0] += da * self.gamma


# ======================= 二、积分榜 =======================
class Standings:
    def __init__(self):
        self.tables: dict[tuple[str, int], dict[str, list]] = {}   # (联赛, 赛季) → 队 → [场数, 积分, 进球, 失球]

    def view(self, comp: str, season: int, team: str) -> tuple[float, float, float, float]:
        """(踢了几场, 场均积分, 场均净胜球, 名次比例)。还没踢过：(0, 空, 空, 空)。"""
        t = self.tables.get((comp, season), {})
        row = t.get(team)
        if not row or row[0] == 0:
            return 0.0, float("nan"), float("nan"), float("nan")
        key = (row[1], row[2] - row[3], row[2])
        n = sum(1 for r in t.values() if r[0] > 0)
        better = sum(1 for r in t.values() if r[0] > 0 and (r[1], r[2] - r[3], r[2]) > key)
        frac = better / (n - 1) if n > 1 else float("nan")
        return float(row[0]), row[1] / row[0], (row[2] - row[3]) / row[0], frac

    def observe(self, comp: str, season: int, home: str, away: str, hg: int, ag: int) -> None:
        t = self.tables.setdefault((comp, season), {})
        hp, ap = (3, 0) if hg > ag else (1, 1) if hg == ag else (0, 3)
        for team, gf, ga, p in ((home, hg, ag, hp), (away, ag, hg, ap)):
            r = t.setdefault(team, [0, 0, 0, 0])
            r[0] += 1
            r[1] += p
            r[2] += int(gf)
            r[3] += int(ga)


# ======================= 三、近期状态 =======================
class Form:
    def __init__(self, windows=(5, 10, 20), half_lives=(5, 20), within_days: float = 365):
        self.windows = tuple(int(w) for w in windows)
        self.half_lives = tuple(float(h) for h in half_lives)
        self.within = float(within_days)
        self.keep = max(max(self.windows), int(4 * max(self.half_lives)))   # 再早的场次权重不到 1/16，不留
        self.hist: dict[tuple[str, str], deque] = {}                         # (联赛, 队) → [(天, gf, ga, sotf, sota, pts), ...]
        self.names = ["form_n"] + [f"form{w}_{s}" for w in self.windows for s in STATS] + \
                     [f"ewm{int(h)}_{s}" for h in self.half_lives for s in STATS]

    @staticmethod
    def _mean(a: np.ndarray, w: np.ndarray) -> np.ndarray:
        """按权重 w 对每一列求平均，空值不算；一列全是空值 → 空值。"""
        ok = ~np.isnan(a)
        tot = (np.where(ok, a, 0.0) * w[:, None]).sum(axis=0)
        wt = (ok * w[:, None]).sum(axis=0)
        with np.errstate(invalid="ignore", divide="ignore"):
            return np.where(wt > 0, tot / wt, np.nan)

    def view(self, comp: str, team: str, day: float) -> list[float]:
        """这队在 day 那天之前的近期状态，顺序和 self.names 一样。"""
        q = self.hist.get((comp, team))
        rows = [r for r in q if day - r[0] <= self.within] if q else []
        out = [float(len(rows))]
        if not rows:
            return out + [float("nan")] * (len(self.names) - 1)
        a = np.array([r[1:] for r in rows], dtype=float)        # 从早到晚
        for w in self.windows:
            out += list(self._mean(a[-w:], np.ones(min(w, len(a)))))
        age = np.arange(len(a))[::-1]                            # 最近一场 0、前一场 1……
        for h in self.half_lives:
            out += list(self._mean(a, 0.5 ** (age / h)))
        return out

    def observe(self, comp: str, team: str, day: float, gf, ga, sotf, sota) -> None:
        pts = 3.0 if gf > ga else 1.0 if gf == ga else 0.0
        q = self.hist.setdefault((comp, team), deque(maxlen=self.keep))
        q.append((day, float(gf), float(ga), float(sotf), float(sota), pts))
