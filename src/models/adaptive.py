"""第 4 节步骤 5 补充：每队自己的实力变化速度（ada，“动态模型平均”）。

一、为什么要做
  dyn（src/models/dynamic.py）里，同一个联赛所有队的实力变化速度 q 都一样。
  但有的队很稳定（同一个教练、阵容多年不变），有的队变动大（换帅、大换血）。
  这一步让每队自己决定“多信历史”：稳定的队多信历史，变动大的队多信最近几场。

二、怎么做（Raftery、Kárný、Ettler 2010 的“动态模型平均”，用在每队身上）
  1. 同时跑 K 个 dyn（每个叫一个“候选”），只有 q 不一样（settings 里 candidates_q，例如 3e-6、1e-5、3e-5、1e-4）。
     每个候选自己照常更新，互不影响，和单独跑一个 dyn 完全一样。
  2. 每队对 K 个候选各有一个权重，开始都是 1/K。另外整个联赛也有一套权重（给 mu、主场优势用）。
  3. 一场比赛踢完：
     a. 先“忘掉一点”：权重开 α 次方再归一（α = 1 不忘；α 越小，越往 1/K 拉回，越只看最近几场）。
     b. 用忘掉之后的权重算出这场的“赛前预期进球”（第 4 条），记下来，拟合 rho 用。
     c. 看每个候选赛前给这场真实比分多大的概率：泊松概率 P(主进球) × P(客进球)。
        两队的权重、联赛的权重都乘上这个概率再归一：哪个候选猜得准，权重往哪个挪。
     d. 每个候选按这场比分照常更新自己的状态。
     （程序里权重存成 log，乘法变加法，不会因为数太小变成 0。）
  4. 出预测：每队的进攻力 a、防守力 d = K 个候选的估计按这队的权重加权平均；mu、主场优势按联赛的权重加权平均。
     平均之前先把每个候选的数换成“0 = 参照的那些队的平均”（第三部分），不同候选的数才能直接平均。
     然后和 dyn 一样：预期进球 → 加 DC 的 rho → 比分矩阵。
  5. 升班马（这个赛季升上来的队）：第一场比赛之前权重回到 1/K（以前在这个联赛踢过也一样，和 dyn 的升班马规则一致）。
  per_team = false（对照组 ada_league）：一个联赛所有队共用联赛那一套权重，不分队。

三、为什么要“换成 0 = 平均”再平均
  dyn 里所有队的 a、d 同时加一个数、mu 减掉它，预期进球一点不变（比分拦不住这种漂移，dyn 每周挪一次平均值）。
  不同候选可能漂得不一样多，直接平均会把漂移混进来。换算：
    a' = a − 参照队 a 的平均，d' = d − 参照队 d 的平均，mu' = mu + 参照队 a 的平均 − 参照队 d 的平均。
  每个候选自己的预期进球不变；所有候选都换成同一种“0 点”，平均才有意义。
  参照的那些队：本赛季已经踢过的队；本赛季刚开始、踢过的队比上赛季的队少时，用上赛季的队（数字更稳）。

四、和 dyn 的关系（检查里验证）
  只有一个候选时，ada 和 dyn 的预测完全一样；几个候选的 q 都一样时，也和 dyn 完全一样。

五、什么地方帮不上
  所有候选的主场优势变化速度一样（q_league），所以 2020/21 空场那种“主场优势突然变小”，ada 和 dyn 一样跟得慢。
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from src.models.base import Model, window_start
from src.models.dynamic import DynamicModel, _League, new_result_rows
from src.models.goals import fit_rho_lambdas, score_matrices, time_weights

LEAGUE_KEY = "__联赛__"        # 联赛那一套权重在 logw 里用的名字（不会和队名重）


class AdaptiveDynamicModel(Model):
    """每队自己的变化速度（说明见文件开头）。设置从 settings.yaml 的 models.adaptive 和 models.adaptive_variants 读：
    candidates_q（几个候选的 q）、alpha（忘得多快）、per_team（是否每队一套权重），其余设置交给每个候选 dyn（见 registry）。"""

    def __init__(self, name: str = "ada", candidates_q=(3e-6, 1e-5, 3e-5, 1e-4), alpha: float = 0.98,
                 per_team: bool = True, **dyn_settings):
        self.name = name
        self.qs = [float(q) for q in candidates_q]
        if not self.qs:
            raise ValueError("candidates_q 至少要有一个 q")
        self.alpha = float(alpha)
        if not 0 < self.alpha <= 1:
            raise ValueError(f"alpha 要在 0 到 1 之间（含 1），收到 {alpha}")
        self.per_team = bool(per_team)
        dyn_settings.pop("q", None)                      # q 由候选给
        dyn_settings["per_match"] = True                 # 每个候选由这里一场一场喂，自己的 per_match 用不上
        self.filters = [DynamicModel(name=f"{name}_q{q:g}", q=q, **dyn_settings) for q in self.qs]
        f0 = self.filters[0]
        self.window_days, self.half_life_days = f0.window_days, f0.half_life_days
        self.min_matches, self.burn_in_seasons, self.max_goals = f0.min_matches, f0.burn_in_seasons, f0.max_goals
        self.logw: dict[str, dict[str, np.ndarray]] = {}    # 联赛 → {队名或 LEAGUE_KEY: K 个 log 权重}
        self.reset_done: dict[str, set] = {}                # 联赛 → {(赛季, 队)}：升班马的权重已经回到 1/K
        self.rec: dict[str, list] = {}                      # 联赛 → 每场 (开球时间, 赛前 λ主, 赛前 λ客, 主进球, 客进球)
        self.rho: dict[str, float] = {}
        self.ready: dict[str, bool] = {}
        self.processed: set[int] = set()

    # ---------------- 权重 ----------------
    def _key(self, team: str) -> str:
        return team if self.per_team else LEAGUE_KEY

    def _fresh(self, comp: str, season: int, team: str) -> bool:
        """这个队在 season 赛季是不是“权重该从 1/K 开始”的：升班马（或联赛第一个赛季的队）、而且权重还没回到 1/K。"""
        lg0 = self.filters[0].leagues.get(comp)
        if lg0 is None or (season, team) in self.reset_done.get(comp, set()):
            return False
        if season >= lg0.season and team not in lg0.members.get(season, []):
            start = self.filters[0]._start(lg0, season, team)
            return start is not None                     # 不是 None = 升班马 / 第一个赛季（None = 上赛季就在，接着用）
        return season == lg0.season and team in lg0.promoted.get(season, [])

    def _weights(self, comp: str, key: str, fresh: bool = False) -> np.ndarray:
        """忘掉一点之后的权重（第二部分 3a）。没有记录或 fresh 时是 1/K。"""
        lw = None if fresh else self.logw.get(comp, {}).get(key)
        if lw is None:
            return np.full(len(self.qs), 1.0 / len(self.qs))
        x = self.alpha * lw
        w = np.exp(x - x.max())
        return w / w.sum()

    def _team_weights(self, comp: str, season: int, team: str) -> np.ndarray:
        if not self.per_team:
            return self._weights(comp, LEAGUE_KEY)
        return self._weights(comp, team, fresh=self._fresh(comp, season, team))

    # ---------------- 换成“0 = 平均”再平均 ----------------
    @staticmethod
    def _reference(lg: _League) -> list[str]:
        """参照的那些队（第三部分）：本赛季踢过的队；比上赛季的队少时用上赛季的队。"""
        now = lg.members.get(lg.season, [])
        prev = [s for s in lg.members if s < lg.season and lg.members[s]]
        before = lg.members[max(prev)] if prev else []
        teams = now if len(now) >= len(before) else before
        return [t for t in teams if t in lg.idx]

    def _centre(self, lg: _League) -> tuple[float, float]:
        """参照队 a 的平均、d 的平均。"""
        teams = self._reference(lg)
        if not teams:
            return 0.0, 0.0
        ia = np.array([lg.pos(t)[0] for t in teams])
        return float(lg.m[ia].mean()), float(lg.m[ia + 1].mean())

    def _mixed(self, comp: str, season: int, home: str, away: str):
        """用忘掉之后的权重，把 K 个候选平均成这场的 (λ主, λ客)；有候选拿不到两队估计时返回 None。
        同时返回每个候选自己的 (λ主, λ客)，更新权重用。"""
        wl = self._weights(comp, LEAGUE_KEY)
        wh, wa = self._team_weights(comp, season, home), self._team_weights(comp, season, away)
        mu = hm = ah = dh = aa = da = 0.0
        own = []
        for k, f in enumerate(self.filters):
            lg = f.leagues[comp]
            sh, sa = f._strength(lg, season, home), f._strength(lg, season, away)
            if sh is None or sa is None:
                return None
            ca, cd = self._centre(lg)
            mu_k = lg.m[0] + ca - cd
            own.append((np.exp(lg.m[0] + lg.m[1] + sh[0] - sa[1]), np.exp(lg.m[0] + sa[0] - sh[1])))
            mu += wl[k] * mu_k
            hm += wl[k] * lg.m[1]
            ah += wh[k] * (sh[0] - ca)
            dh += wh[k] * (sh[1] - cd)
            aa += wa[k] * (sa[0] - ca)
            da += wa[k] * (sa[1] - cd)
        return float(np.exp(mu + hm + ah - da)), float(np.exp(mu + aa - dh)), np.array(own)

    # ---------------- 处理比赛结果 ----------------
    def _process(self, rows: pd.DataFrame) -> None:
        """按开球顺序一场一场处理（第二部分第 3 条）。"""
        for mid, comp, season, home, away, hg, ag, ko in new_result_rows(rows, self.processed):
            self.processed.add(int(mid))
            season = int(season)
            lgs = [f._prepare(comp, season, home, away, ko) for f in self.filters]
            if any(lg is None for lg in lgs):            # 上赛季的补赛、有队不在状态里：每个候选都一样跳过
                continue
            logw = self.logw.setdefault(comp, {})
            if self.per_team:
                for t in (home, away):                   # 升班马第一场之前权重回到 1/K（第二部分第 5 条）
                    if self._fresh(comp, season, t):
                        logw[t] = np.zeros(len(self.qs))
                        self.reset_done.setdefault(comp, set()).add((season, t))
            mixed = self._mixed(comp, season, home, away)
            if mixed is not None:
                lh, la, own = mixed
                self.rec.setdefault(comp, []).append((ko, lh, la, int(hg), int(ag)))
                # 每个候选赛前给真实比分的泊松 log 概率（去掉和候选无关的 log(x!)）
                ll = hg * np.log(own[:, 0]) - own[:, 0] + ag * np.log(own[:, 1]) - own[:, 1]
                keys = {LEAGUE_KEY} | ({home, away} if self.per_team else set())
                for key in keys:
                    x = self.alpha * logw.get(key, np.zeros(len(self.qs))) + ll
                    logw[key] = x - (x.max() + np.log(np.exp(x - x.max()).sum()))   # 归一（log 里减掉总和的 log）
            for f, lg in zip(self.filters, lgs):
                f._observe(lg, ko, home, away, hg, ag)

    def update(self, new_rows: pd.DataFrame, cutoff: str) -> None:
        self._process(new_rows)

    def fit(self, history: pd.DataFrame, cutoff: str) -> None:
        """每周一次：补上没处理的比赛；每个候选挪平均值（和 dyn 一样）；用加权后的赛前预期进球拟合 rho。"""
        self._process(history)
        start = window_start(cutoff, self.window_days)
        for comp in sorted(self.filters[0].leagues):
            for f in self.filters:
                f._recentre(f.leagues[comp])
            use = [r for r in self.rec.get(comp, []) if start <= r[0] <= cutoff]
            self.ready[comp] = len(use) >= self.min_matches
            if not self.ready[comp]:
                continue
            ko, lh, la, x, y = zip(*use)
            w = time_weights(pd.Series(ko), cutoff, self.half_life_days)
            self.rho[comp] = fit_rho_lambdas(np.array(lh), np.array(la), np.array(x), np.array(y), w)

    # ---------------- 预测 ----------------
    def predict(self, targets: pd.DataFrame, cutoff: str) -> list:
        out: list = [None] * len(targets)
        by_comp: dict[str, list] = {}
        for r, (comp, season, home, away) in enumerate(
                targets[["competition", "season", "home_team", "away_team"]].itertuples(index=False, name=None)):
            lg0 = self.filters[0].leagues.get(comp)
            if lg0 is None or not self.ready.get(comp):
                continue
            season = int(season)
            if season - lg0.first_season < self.burn_in_seasons:
                continue
            mixed = self._mixed(comp, season, home, away)
            if mixed is None:
                continue
            by_comp.setdefault(comp, []).append((r, mixed[0], mixed[1]))
        for comp, items in by_comp.items():
            rows, lh, la = zip(*items)
            mats = score_matrices(np.array(lh), np.array(la), self.rho.get(comp, 0.0), 0.0, self.max_goals)
            for r, m in zip(rows, mats):
                out[r] = m
        return out

    # ---------------- 给人看 ----------------
    def params(self) -> dict:
        out = {}
        for comp in sorted(self.filters[0].leagues):
            wl = self._weights(comp, LEAGUE_KEY)
            out[comp] = {"rho": round(self.rho.get(comp, 0.0), 4), "处理过的比赛": len(self.rec.get(comp, [])),
                         "联赛权重（每个 q）": {f"{q:g}": round(float(w), 3) for q, w in zip(self.qs, wl)},
                         "联赛权重下的 q": float(f"{float(np.dot(wl, self.qs)):.3g}")}
        return out

    def ratings_table(self) -> pd.DataFrame:
        """每个联赛当前赛季各队：加权后的进攻力、防守力，每个候选的权重，权重下的平均 q（越大 = 越信最近几场）。"""
        rows = []
        for comp in sorted(self.filters[0].leagues):
            lg0 = self.filters[0].leagues[comp]
            centres = [self._centre(f.leagues[comp]) for f in self.filters]
            for t in sorted(lg0.members.get(lg0.season, [])):
                w = self._team_weights(comp, lg0.season, t)
                a = d = 0.0
                for k, f in enumerate(self.filters):
                    i, j = f.leagues[comp].pos(t)
                    a += w[k] * (f.leagues[comp].m[i] - centres[k][0])
                    d += w[k] * (f.leagues[comp].m[j] - centres[k][1])
                rows.append({"competition": comp, "season": lg0.season, "team": t, "attack": round(a, 4),
                             "defence": round(d, 4), "total": round(a + d, 4),
                             "q_weighted": float(f"{float(np.dot(w, self.qs)):.3g}"),
                             **{f"w_q{q:g}": round(float(x), 3) for q, x in zip(self.qs, w)}})
        if not rows:
            return pd.DataFrame(columns=["competition", "season", "team", "attack", "defence", "total"])
        return pd.DataFrame(rows).sort_values(["competition", "total"], ascending=[True, False], ignore_index=True)
