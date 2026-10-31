"""第 4 节步骤 5：动态实力模型（状态空间模型，用“扩展卡尔曼滤波”的办法一场一场更新）。

一、和 dc 有什么不同
  dc 每周用最近 4 年的比赛重新拟合一次，越近的比赛权重越大。两次拟合之间踢完的比赛，要等下次拟合才算进去。
  动态模型换个想法：每队的进攻力 a、防守力 d 每天都在随机地漂一点，我们手里有的是“现在的估计 + 有多不确定”。
  每踢完一场，就按这场的比分把估计挪一挪；越不确定挪得越多，挪完更确定一些。

二、状态（每个联赛一套，见 _League）
  m：估计值，排成一列 [mu, home, a₁, d₁, a₂, d₂, …]（队按第一次出现的顺序排）。
  P：协方差矩阵。对角线上是每个数“有多不确定”（方差），其他格子是两个数之间互相牵连的程度。
     例：某队一直和另一队踢，两队的估计就会互相牵连；一队的估计变了，另一队也跟着变一点。
  进球的式子和 dc 一样：主队预期进球 λ主 = e^(mu + home + a主 − d客)，客队 λ客 = e^(mu + a客 − d主)，进球按泊松分布。

三、时间往前走（_advance）
  每过一天，每队 a、d 的方差加 q（q 越大 = 越相信实力变得快）；mu、home 的方差每天加 q_league（很小，几乎不变）。
  估计值本身不动（随机漂，往哪边漂的可能一样大）。

四、一场比赛踢完（_observe）
  把 log λ 看成状态的线性组合：log λ主 = H₁ · m，log λ客 = H₂ · m（H 只在这场两队和 mu、home 的位置上是 ±1）。
  在当前估计上按泊松似然走一步牛顿法（扩展卡尔曼滤波的常见近似）：
    g = (主进球 − λ主, 客进球 − λ客)            “比预期多进了几个”
    L = diag(λ主, λ客)                         泊松似然的“弯曲程度”
    K = P Hᵀ (I + L H P Hᵀ)⁻¹                  “增益”：每个状态该挪多少
    m ← m + K g，  P ← P − K L H P
  例：两队都是平均水平，主队预期 1.6 个、实际进了 4 个 → g主 = 2.4，主队 a 往上挪、客队 d 往下挪，
     挪多少看两队现在有多不确定（新升上来的队方差大，挪得多）。
  更新前把这场“赛前”的 λ主、λ客 记下来（_League.rec），每周拟合 rho 用。

五、每周拟合（fit）
  1. 先把还没处理的比赛按开球顺序处理完。
  2. 把各队 a、d 减去本赛季各队的平均值，加到 mu 上（_recentre）：预期进球一点不变，只是让 0 = 联赛平均，
     也防止所有队的数一起慢慢漂走（所有 a、d 同加一个数，预期进球不变，比分拦不住这种漂移）。
  3. 用窗口里（window_days、半衰期 half_life_days）每场赛前的 λ 和真实比分找 DC 的 rho（和步骤 4 两步拟合的第 2 步一样）。
  4. 窗口里比赛够 min_matches 场的联赛才出预测；而且要预测的比赛所在赛季前面至少有 burn_in_seasons 个赛季的数据
     （例：联赛从 2005 开始、burn_in_seasons = 2 → 2007 赛季起出预测。按“要预测的赛季”判断，新赛季第一周也不漏）。

六、新赛季（_new_season、_ensure）
  - 进入新赛季时，上赛季的队 a、d 的方差各加 season_bump（休赛期转会、换帅）。
  - 升班马（上赛季不在这个联赛）：估计值 = 最近 promoted_history_seasons 个赛季的升班马在“升上来那个赛季结束时”的平均
    （还没有记录时用 promoted_attack、promoted_defence），方差 promoted_var，和其他队的牵连清零。
    很多年前在这个联赛踢过的队也按升班马算，不接着用旧值（和 Elo、dc 一样）。
  - 联赛第一个有数据的赛季：mu = log(first_goals)、home = first_home、各队 0。
  - 怎么知道“新赛季开始了”：处理结果时第一次碰到某联赛新赛季的比赛。升班马在它第一场比赛处理时才加进状态
    （先把时间走到开球时间，再加：所以每个升班马第一场赛前的方差都正好是 promoted_var）。
  - 出预测（predict）不改动任何状态：新赛季的比赛、还没踢过的升班马，按上面的规则当场算出估计值来用。
    这样“什么时候出过预测”不会影响模型，每场赛后更新和每周更新的对照才公平。

七、per_match（每场赛后马上更新）
  true：回测框架每有新结果就交给 update()，马上处理。false：update() 什么都不做，等每周 fit() 时一起处理
  （对照组 dyn_weekly，用来分清“每场赛后马上更新”和“换成动态模型”各值多少）。
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from src.models.base import Model, window_start
from src.models.goals import fit_rho_lambdas, score_matrices, time_weights


def _days(text: str) -> float:
    """'YYYY-MM-DD HH:MM:SS'（UTC）→ 从 1970 年起的天数（只用来算两场之间隔了几天）。"""
    return pd.Timestamp(text).value / 86400e9


def new_result_rows(rows: pd.DataFrame, processed: set) -> list:
    """还没处理过（不在 processed 里）、有比分的比赛，按开球顺序：(match_id, 联赛, 赛季, 主队, 客队, 主进球, 客进球, 开球时间)。
    dyn 和 ada（src/models/adaptive.py）共用。"""
    if rows is None or rows.empty:
        return []
    rows = rows[rows["result"].notna() & rows["home_goals"].notna() & ~rows["match_id"].isin(processed)]
    if rows.empty:
        return []
    rows = rows.sort_values(["kickoff_earliest_utc", "match_id"])
    return list(rows[["match_id", "competition", "season", "home_team", "away_team", "home_goals", "away_goals",
                      "kickoff_earliest_utc"]].itertuples(index=False, name=None))


class _League:
    """一个联赛的状态（说明见文件开头第二部分）。"""

    def __init__(self, first_season: int, mu: float, home: float, var_mu: float, var_home: float):
        self.idx: dict[str, int] = {}               # 队名 → 第几个队（a 在 m 的 2 + 2k 位置，d 在 3 + 2k）
        self.m = np.array([mu, home], dtype=float)
        self.P = np.diag([var_mu, var_home]).astype(float)
        self.first_season = first_season
        self.season = first_season                  # 现在处在哪个赛季
        self.members: dict[int, list[str]] = {first_season: []}    # 每个赛季有哪些队（按第一次出现排）
        self.promoted: dict[int, list[str]] = {}    # 每个赛季的升班马
        self.end_strength: dict[int, list] = {}     # 赛季 → 这个赛季的升班马在赛季结束时的 (a, d)
        self.t: float | None = None                 # 上次处理到的时间（天）
        self.rec: list[tuple] = []                  # 每场：(开球时间, 赛前 λ主, 赛前 λ客, 主进球, 客进球)
        self.rho = 0.0
        self.ready = False                          # 窗口里比赛够不够（每周拟合时判断；赛季够不够在出预测时判断）

    def pos(self, team: str) -> tuple[int, int]:
        k = self.idx[team]
        return 2 + 2 * k, 3 + 2 * k


class DynamicModel(Model):
    """动态实力模型。设置都从 settings.yaml 的 models.dynamic 和 models.dynamic_variants 读（见 registry）。"""

    def __init__(self, name: str = "dyn", q: float = 3e-5, q_league: float = 1e-6, season_bump: float = 0.0,
                 promoted_var: float = 0.1, first_var: float = 0.1, first_var_mu: float = 0.1,
                 first_var_home: float = 0.05, first_goals: float = 1.35, first_home: float = 0.25,
                 promoted_attack: float = -0.2, promoted_defence: float = -0.2, promoted_history_seasons: int = 3,
                 burn_in_seasons: int = 2, window_days: float = 1461, half_life_days: float | None = 365,
                 per_match: bool = True, min_matches: int = 200, max_goals: int = 10):
        self.name = name
        self.q, self.q_league, self.season_bump = float(q), float(q_league), float(season_bump)
        self.promoted_var, self.first_var = float(promoted_var), float(first_var)
        self.first_var_mu, self.first_var_home = float(first_var_mu), float(first_var_home)
        self.first_goals, self.first_home = float(first_goals), float(first_home)
        self.promoted_default = (float(promoted_attack), float(promoted_defence))
        self.promoted_history_seasons = int(promoted_history_seasons)
        self.burn_in_seasons = int(burn_in_seasons)
        self.window_days = float(window_days)
        self.half_life_days = None if half_life_days is None else float(half_life_days)
        self.per_match = bool(per_match)
        self.min_matches = int(min_matches)
        self.max_goals = int(max_goals)
        self.leagues: dict[str, _League] = {}
        self.processed: set[int] = set()

    # ---------------- 球队进出 ----------------
    def _league(self, comp: str, season: int) -> _League:
        lg = self.leagues.get(comp)
        if lg is None:
            lg = self.leagues[comp] = _League(season, np.log(self.first_goals), self.first_home,
                                              self.first_var_mu, self.first_var_home)
        return lg

    def _end_strength(self, lg: _League, season: int) -> list:
        """某赛季的升班马在赛季结束时的 (a, d)：已经结束的赛季查记录；还没进入下赛季的（就是现在这个赛季）用现在的估计。"""
        if season == lg.season:
            return [(lg.m[i], lg.m[j]) for i, j in map(lg.pos, lg.promoted.get(season, []))]
        return lg.end_strength.get(season, [])

    def _prior(self, lg: _League, season: int) -> tuple[float, float]:
        """升班马的初始估计：最近几个赛季的升班马在升上来那个赛季结束时的平均 (a, d)。"""
        vals = [v for s in range(season - self.promoted_history_seasons, season) for v in self._end_strength(lg, s)]
        if not vals:
            return self.promoted_default
        a, d = np.mean(vals, axis=0)
        return float(a), float(d)

    def _new_season(self, lg: _League, season: int) -> None:
        """联赛进入新赛季：记下上赛季升班马的赛季末实力；上赛季的队方差加 season_bump。"""
        prev = lg.season
        if lg.promoted.get(prev):
            lg.end_strength[prev] = self._end_strength(lg, prev)
        if self.season_bump > 0:
            for t in lg.members.get(prev, []):
                i, j = lg.pos(t)
                lg.P[i, i] += self.season_bump
                lg.P[j, j] += self.season_bump
        lg.season = season
        lg.members.setdefault(season, [])

    def _start(self, lg: _League, season: int, team: str):
        """这个赛季还没踢过的队，该从哪开始（第六部分）：返回 (估计值 (a, d), 方差, 是不是升班马)；上赛季就在的返回 None（接着用）。
        season 可以是联赛还没进入的下个赛季（出预测时用）。"""
        earlier = [s for s in lg.members if s < season and lg.members[s]]
        if not earlier:                                   # 联赛第一个赛季
            return (0.0, 0.0), self.first_var, False
        if team in lg.members[max(earlier)]:              # 上赛季就在：接着用
            return None
        return self._prior(lg, season), self.promoted_var, True   # 升班马（包括很多年前在这个联赛踢过的队）

    def _ensure(self, lg: _League, season: int, team: str) -> None:
        """这个赛季第一次碰到这个队：按 _start 给初始估计，加进状态。"""
        if team in lg.members[season]:
            return
        start = self._start(lg, season, team)
        lg.members[season].append(team)
        if start is None:
            return
        mean, var, promoted = start
        if promoted:
            lg.promoted.setdefault(season, []).append(team)
        if team not in lg.idx:                            # 状态里加两格（a、d）
            lg.idx[team] = len(lg.idx)
            n = len(lg.m)
            lg.m = np.concatenate([lg.m, [0.0, 0.0]])
            P = np.zeros((n + 2, n + 2))
            P[:n, :n] = lg.P
            lg.P = P
        i, j = lg.pos(team)
        lg.P[[i, j], :] = 0.0                             # 和其他队的牵连清零
        lg.P[:, [i, j]] = 0.0
        lg.P[i, i] = lg.P[j, j] = var
        lg.m[i], lg.m[j] = mean

    def _advance(self, lg: _League, t: float) -> None:
        """时间从上次处理的时刻走到 t：方差加 q × 天数（第三部分）。估计值不动。"""
        if lg.t is not None and t > lg.t:
            dt = t - lg.t
            n = len(lg.m)
            lg.P[np.arange(2, n), np.arange(2, n)] += self.q * dt
            lg.P[0, 0] += self.q_league * dt
            lg.P[1, 1] += self.q_league * dt
        if lg.t is None or t > lg.t:
            lg.t = t

    def _lambdas(self, lg: _League, home: str, away: str) -> tuple[float, float]:
        ah, dh = lg.pos(home)
        aa, da = lg.pos(away)
        m = lg.m
        return float(np.exp(m[0] + m[1] + m[ah] - m[da])), float(np.exp(m[0] + m[aa] - m[dh]))

    def _strength(self, lg: _League, season: int, team: str):
        """出预测用：这个队在 season 赛季的 (a, d) 估计（不改状态）。拿不到（上赛季的补赛、队不在状态里）返回 None。"""
        if season < lg.season:
            if team not in lg.idx:
                return None
        elif team not in lg.members.get(season, []):
            start = self._start(lg, season, team)
            if start is not None:
                return start[0]
        i, j = lg.pos(team)
        return float(lg.m[i]), float(lg.m[j])

    def _observe(self, lg: _League, ko: str, home: str, away: str, hg: int, ag: int) -> None:
        """一场比赛踢完：记下赛前的 λ，再按第四部分的式子更新 m、P。"""
        lh, la = self._lambdas(lg, home, away)
        lg.rec.append((ko, lh, la, int(hg), int(ag)))
        ah, dh = lg.pos(home)
        aa, da = lg.pos(away)
        n = len(lg.m)
        H = np.zeros((2, n))
        H[0, [0, 1, ah, da]] = (1.0, 1.0, 1.0, -1.0)     # log λ主 = mu + home + a主 − d客
        H[1, [0, aa, dh]] = (1.0, 1.0, -1.0)             # log λ客 = mu + a客 − d主
        lam = np.array([lh, la])
        g = np.array([hg, ag], dtype=float) - lam
        PH = lg.P @ H.T                                  # n × 2
        S = np.eye(2) + lam[:, None] * (H @ PH)          # I + L H P Hᵀ
        K = np.linalg.solve(S.T, PH.T).T                 # P Hᵀ S⁻¹
        lg.m = lg.m + K @ g
        lg.P = lg.P - K @ (lam[:, None] * PH.T)
        lg.P = (lg.P + lg.P.T) / 2                       # 去掉计算误差，保持对称

    # ---------------- 处理比赛结果 ----------------
    def _prepare(self, comp: str, season: int, home: str, away: str, ko: str):
        """一场比赛更新之前要做的：换赛季、把时间走到开球、加升班马。返回这个联赛的状态；这场要跳过时返回 None。
        （单独写出来，是因为 src/models/adaptive.py 要在更新前先看每个 dyn 赛前的预期进球。）"""
        lg = self._league(comp, season)
        if season > lg.season:
            self._new_season(lg, season)
        if season < lg.season:
            # 上赛季的补赛，结果在新赛季开始后才知道：用两队现在的估计更新；有队不在状态里就跳过（极少见）
            if home not in lg.idx or away not in lg.idx:
                return None
            self._advance(lg, _days(ko))
        else:
            self._advance(lg, _days(ko))                 # 先把时间走到开球，再加升班马（第六部分）
            self._ensure(lg, season, home)
            self._ensure(lg, season, away)
        return lg

    def _process(self, rows: pd.DataFrame) -> None:
        """按开球顺序处理还没处理过的比赛（有比分的才算）。"""
        for mid, comp, season, home, away, hg, ag, ko in new_result_rows(rows, self.processed):
            self.processed.add(int(mid))
            lg = self._prepare(comp, int(season), home, away, ko)
            if lg is not None:
                self._observe(lg, ko, home, away, hg, ag)

    def update(self, new_rows: pd.DataFrame, cutoff: str) -> None:
        if self.per_match:
            self._process(new_rows)

    def _recentre(self, lg: _League) -> None:
        """本赛季各队 a、d 的平均值挪到 mu 上（第五部分第 2 条）。所有队（包括不在本赛季的）一起挪，预期进球不变。"""
        teams = lg.members.get(lg.season) or []
        if len(teams) < 2:
            prev = [s for s in lg.members if s < lg.season]
            teams = lg.members[max(prev)] if prev else teams
        if len(teams) < 2:
            return
        n = len(lg.m)
        ia = np.array([lg.pos(t)[0] for t in teams])
        T = np.eye(n)
        all_a, all_d = np.arange(2, n, 2), np.arange(3, n, 2)
        # 新 a_i = a_i − 平均 a；新 d_i = d_i − 平均 d；新 mu = mu + 平均 a − 平均 d
        T[np.ix_(all_a, ia)] -= 1.0 / len(ia)
        T[np.ix_(all_d, ia + 1)] -= 1.0 / len(ia)
        T[0, ia] += 1.0 / len(ia)
        T[0, ia + 1] -= 1.0 / len(ia)
        lg.m = T @ lg.m
        lg.P = T @ lg.P @ T.T
        lg.P = (lg.P + lg.P.T) / 2

    def fit(self, history: pd.DataFrame, cutoff: str) -> None:
        """每周一次（第五部分）：补上没处理的比赛、挪平均值、拟合 rho、判断能不能出预测。"""
        self._process(history)
        start = window_start(cutoff, self.window_days)
        for comp in sorted(self.leagues):
            lg = self.leagues[comp]
            self._recentre(lg)
            use = [r for r in lg.rec if start <= r[0] <= cutoff]
            lg.ready = len(use) >= self.min_matches
            if not lg.ready:
                continue
            ko, lh, la, x, y = zip(*use)
            w = time_weights(pd.Series(ko), cutoff, self.half_life_days)
            lg.rho = fit_rho_lambdas(np.array(lh), np.array(la), np.array(x), np.array(y), w)

    # ---------------- 预测 ----------------
    def predict(self, targets: pd.DataFrame, cutoff: str) -> list:
        out: list = [None] * len(targets)
        by_comp: dict[str, list] = {}
        for r, (comp, season, home, away) in enumerate(
                targets[["competition", "season", "home_team", "away_team"]].itertuples(index=False, name=None)):
            lg = self.leagues.get(comp)
            if lg is None or not lg.ready:
                continue
            season = int(season)
            if season - lg.first_season < self.burn_in_seasons:   # 头几个赛季只让估计稳定，不出预测
                continue
            sh, sa = self._strength(lg, season, home), self._strength(lg, season, away)
            if sh is None or sa is None:                 # 上赛季的补赛、有队不在状态里：不出（极少见）
                continue
            mu, hm = lg.m[0], lg.m[1]
            by_comp.setdefault(comp, []).append((r, float(np.exp(mu + hm + sh[0] - sa[1])), float(np.exp(mu + sa[0] - sh[1]))))
        for comp, items in by_comp.items():
            rows, lh, la = zip(*items)
            mats = score_matrices(np.array(lh), np.array(la), self.leagues[comp].rho, 0.0, self.max_goals)
            for r, m in zip(rows, mats):
                out[r] = m
        return out

    # ---------------- 步骤 6 用：赛前的预期进球和它的不确定 ----------------
    def pre_match_view(self, comp: str, season: int, home: str, away: str, ko: str):
        """这场比赛开球那一刻（处理这场之前）的 (log λ主, log λ客, log λ主 的标准差, log λ客 的标准差, 主场优势)。不改任何状态。

        算法和“先 _prepare（换赛季、时间走到开球、加升班马）再看状态”完全一样，只是不真的改：
        只取这场用得到的 6 个数（mu、home、主队 a、d、客队 a、d）和它们的协方差，按 _prepare 的顺序在小矩阵上算。
        log λ 的方差 = hᵀ C h（h 是第四部分里的 H 那一行）。为什么用 log λ 不用 a、d：每周挪平均值（_recentre）会改 a、d、mu，
        但不改 log λ，也不改它的方差，所以不同时间算出的数能放在一起比。
        拿不到（上赛季的补赛、有队不在状态里；联赛还没有数据）返回 None。步骤 6（src/models/gbdt.py）训练、预测都用它。
        """
        lg = self.leagues.get(comp)
        if lg is None:
            return None
        t = _days(ko)
        dt = t - lg.t if lg.t is not None and t > lg.t else 0.0
        late = season < lg.season
        rolling = season > lg.season
        slots = []                                       # 每队：(估计值 (a, d), 方差) 是新的；None = 接着用状态里的
        for team in (home, away):
            if late or team in lg.members.get(season, []):
                if team not in lg.idx:
                    return None
                slots.append(None)
                continue
            start = self._start(lg, season, team)
            slots.append(None if start is None else (start[0], start[1]))
            if start is None and team not in lg.idx:     # 不会发生（上赛季在的队一定在状态里），保险起见
                return None
        idx = [0, 1]
        for team, slot in zip((home, away), slots):
            idx += list(lg.pos(team)) if slot is None else [-1, -1]
        mean = np.zeros(6)
        C = np.zeros((6, 6))
        keep = [k for k, i in enumerate(idx) if i >= 0]
        src = [idx[k] for k in keep]
        mean[keep] = lg.m[src]
        C[np.ix_(keep, keep)] = lg.P[np.ix_(src, src)]
        C[0, 0] += self.q_league * dt
        C[1, 1] += self.q_league * dt
        for n, (team, slot) in enumerate(zip((home, away), slots)):
            k = 2 + 2 * n
            if slot is None:
                bump = self.season_bump if rolling and team in lg.members.get(lg.season, []) else 0.0
                C[k, k] += self.q * dt + bump
                C[k + 1, k + 1] += self.q * dt + bump
            else:                                        # 新加进来的队：和其他数没有牵连，方差 = 初始方差
                (a, d), var = slot
                mean[k], mean[k + 1] = a, d
                C[k, :] = C[:, k] = C[k + 1, :] = C[:, k + 1] = 0.0
                C[k, k] = C[k + 1, k + 1] = var
        hh = np.array([1.0, 1.0, 1.0, 0.0, 0.0, -1.0])   # log λ主 = mu + home + a主 − d客
        ha = np.array([1.0, 0.0, 0.0, -1.0, 1.0, 0.0])   # log λ客 = mu + a客 − d主
        return (float(hh @ mean), float(ha @ mean), float(np.sqrt(max(hh @ C @ hh, 0.0))),
                float(np.sqrt(max(ha @ C @ ha, 0.0))), float(mean[1]))

    # ---------------- 给人看 ----------------
    def params(self) -> dict:
        out = {}
        for c in sorted(self.leagues):
            lg = self.leagues[c]
            teams = lg.members.get(lg.season, [])
            sd = [float(np.sqrt(lg.P[i, i])) for t in teams for i in lg.pos(t)]
            out[c] = {"基准mu": round(float(lg.m[0]), 4), "主场优势home": round(float(lg.m[1]), 4),
                      "rho": round(lg.rho, 4), "处理过的比赛": len(lg.rec), "赛季": lg.season,
                      "本赛季各队 a、d 的平均不确定（标准差）": round(float(np.mean(sd)), 4) if sd else None,
                      "升班马先验(a,d)": [round(v, 3) for v in self._prior(lg, lg.season)]}
        return out

    def ratings_table(self) -> pd.DataFrame:
        """每个联赛当前赛季各队的进攻力、防守力和它们的不确定（标准差），按 a + d 从高到低。"""
        rows = []
        for c in sorted(self.leagues):
            lg = self.leagues[c]
            for t in sorted(lg.members.get(lg.season, [])):
                i, j = lg.pos(t)
                rows.append({"competition": c, "season": lg.season, "team": t,
                             "attack": round(float(lg.m[i]), 4), "defence": round(float(lg.m[j]), 4),
                             "total": round(float(lg.m[i] + lg.m[j]), 4),
                             "attack_sd": round(float(np.sqrt(lg.P[i, i])), 4),
                             "defence_sd": round(float(np.sqrt(lg.P[j, j])), 4)})
        if not rows:
            return pd.DataFrame(columns=["competition", "season", "team", "attack", "defence", "total"])
        return pd.DataFrame(rows).sort_values(["competition", "total"], ascending=[True, False], ignore_index=True)
