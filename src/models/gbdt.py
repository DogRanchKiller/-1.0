"""第 4 节步骤 6：特征表 + 梯度提升树 A 版（需求文档第 4 节“步骤 6 怎么做”“步骤 6 事先定的比较”）。

一、想法
  前面的模型各自只用一种信息：dyn 用比分推实力，Elo 只看输赢……
  这里把“赛前能知道的各种数”排成一张表（特征表），交给梯度提升树（LightGBM）自己去找“哪些数、怎么组合能猜准进球”。
  梯度提升树 = 很多棵小决策树加起来：第一棵树粗略地猜，第二棵专门修正第一棵猜错的地方，第三棵再修正……

二、特征表（每场两行：主队看客队一行、客队看主队一行；“这队” = own，“对手” = opp）
  联赛、是不是主场；
  dyn：这队预期进球的 log、对手（= 这队预期丢球）的 log、两个数的不确定（标准差）、联赛主场优势；
  Elo（elo_gd 的设置）：这队、对手比本赛季基准高多少分，评分差（含主场）；
  pi-rating：这队这场用的那个分（主场分或客场分）、对手的、预期净胜球；
  是不是升班马（这队、对手）；休息天数、欧战后（这队、对手，回测框架算好交进来，见 src/features/rest_days.py）；
  积分榜：踢了几场、场均积分、场均净胜球、名次比例（这队、对手）；
  近期状态：最近 5、10、20 场和时间衰减的进球、失球、射正、被射正、积分（这队、对手）。
  算法见 src/features/team_form.py。所有特征都是“这场开球前”的：每场先算特征、后把比分交给各个计算器。

三、训练和预测
  - 目标：这队这场进几个球，泊松回归（树输出的就是预期进球）。主队、客队两行一起训练。
  - 训练用：从每个联赛第 3 个有数据的赛季起（前两季 dyn 还没稳定，和 dyn 出预测的规则一样）、截止前已经踢完的全部比赛。
  - 每 retrain_days 天（91 天）重新训练一次：先拿训练数据里最后 valid_days 天当验证集，找“种多少棵树”（验证集泊松损失
    early_stopping_rounds 轮不降就停），再用全部训练数据种这么多棵。两次训练之间，特征照样每场赛后更新。
    分不出验证集（最后 valid_days 天没有比赛，或者之前的比赛不到 min_train_rows 行）就先不重新训练：
    还没训练过就不出预测，训练过就接着用上次的树，下次再试。
  - 出预测：两行分别得到主队、客队预期进球，再用 dyn 这个联赛的 rho 做 DC 小比分修正，得到比分矩阵。
    只在 dyn 能出预测的比赛上出预测（两个模型比同一批比赛）。
  - target = result（对照组 gb_1x2）：每场只用主队那一行，直接输出胜平负三个概率（多分类）。

四、里面的 dyn、Elo
  这里自己带一个 dyn 和一个 Elo，每场赛后照常更新（和单独跑时一样），dyn 每周 fit 一次（挪平均值、拟合 rho）。
  Elo 的“评分 → 概率”这里用不上，不拟合。
"""
from __future__ import annotations

import lightgbm as lgb
import numpy as np
import pandas as pd

from src.features.team_form import Form, PiRating, Standings
from src.models.base import Model, window_start
from src.models.dynamic import DynamicModel, _days
from src.models.elo import EloModel
from src.models.goals import score_matrices

RESULT_INDEX = {"H": 0, "D": 1, "A": 2}
SHOT_COLS = ["home_shots_on_target", "away_shots_on_target"]
REST_COLS = ["home_rest_days", "away_rest_days", "home_after_euro", "away_after_euro"]
TABLE = ["played", "ppg", "gdpg", "rank"]


class GBDTModel(Model):
    data = ("比分", "射门、xG")
    usage = "A版输入"
    uses_rest_days = True            # 回测框架看到这个，会给比赛加上休息天数四列（src/evaluate/backtest.py）

    def __init__(self, name: str, dyn: dict, elo: dict, pi: dict, form: dict, leagues, features: str = "all",
                 target: str = "goals", num_leaves: int = 15, min_data_in_leaf: int = 100,
                 learning_rate: float = 0.03, feature_fraction: float = 0.8, bagging_fraction: float = 0.8,
                 lambda_l2: float = 1.0, max_rounds: int = 2000, early_stopping_rounds: int = 50,
                 valid_days: float = 365, retrain_days: float = 91, min_train_rows: int = 2000,
                 seed: int = 0, max_goals: int = 10, min_matches: int = 200):
        if features not in ("all", "nodyn", "dynonly"):
            raise ValueError(f"features 只能是 all / nodyn / dynonly，收到 {features!r}")
        if target not in ("goals", "result"):
            raise ValueError(f"target 只能是 goals / result，收到 {target!r}")
        self.name = name
        self.dyn = DynamicModel(name="dyn", **dyn)
        self.elo = EloModel(name="elo", **elo)
        self.pi = PiRating(**pi)
        self.table = Standings()
        self.form = Form(**form)
        self.league_code = {c: float(i) for i, c in enumerate(sorted(leagues))}
        self.features, self.target = features, target
        self.lgb_params = {
            "objective": "poisson" if target == "goals" else "multiclass", "learning_rate": float(learning_rate),
            "num_leaves": int(num_leaves), "min_data_in_leaf": int(min_data_in_leaf),
            "feature_fraction": float(feature_fraction), "bagging_fraction": float(bagging_fraction),
            "bagging_freq": 1, "lambda_l2": float(lambda_l2), "seed": int(seed), "deterministic": True,
            "force_row_wise": True, "num_threads": 1, "verbose": -1}
        if target == "result":
            self.lgb_params["num_class"] = 3
        self.max_rounds, self.early_stopping_rounds = int(max_rounds), int(early_stopping_rounds)
        self.valid_days, self.retrain_days = float(valid_days), float(retrain_days)
        self.min_train_rows, self.max_goals = int(min_train_rows), int(max_goals)
        side = lambda s: [f"{n}_own" for n in s] + [f"{n}_opp" for n in s]
        self.names = (["league", "is_home", "dyn_loglam_for", "dyn_loglam_against", "dyn_sd_for", "dyn_sd_against",
                       "dyn_home", "elo_own", "elo_opp", "elo_diff", "pi_own", "pi_opp", "pi_egd"]
                      + [f"{n}_{w}" for n in ("promoted", "rest_days", "after_euro") for w in ("own", "opp")]
                      + side([f"table_{t}" for t in TABLE])
                      + side(self.form.names))
        if features == "all":
            self.use = list(range(len(self.names)))
        elif features == "nodyn":
            self.use = [i for i, n in enumerate(self.names) if not n.startswith("dyn_")]
        else:
            self.use = [i for i, n in enumerate(self.names) if n in ("league", "is_home") or n.startswith("dyn_")]
        self.processed: set[int] = set()
        self.rows: list[np.ndarray] = []         # 训练用的特征（每场两行：主队一行、客队一行）
        self.goals: list[int] = []               # 每行：这队进了几个
        self.results: list[int] = []             # 每行：这场的结果 0 主胜 / 1 平 / 2 客胜（多分类用，只看主队那一行）
        self.kickoffs: list[str] = []
        self.booster: lgb.Booster | None = None
        self.last_train: pd.Timestamp | None = None
        self.train_log: list[dict] = []

    # ---------------- 一场比赛的特征（不改状态，Elo 的换赛季除外，见下面） ----------------
    def _features(self, comp: str, season: int, home: str, away: str, ko: str, rest) -> tuple | None:
        """这场比赛开球前的特征：(主队那一行, 客队那一行)；dyn 给不出（上赛季补赛里有队不在状态里等）返回 None。"""
        v = self.dyn.pre_match_view(comp, season, home, away, ko)
        if v is None:
            return None
        llh, lla, sdh, sda, hm = v
        # Elo：碰到新赛季先换赛季、给新队初始分（换赛季只用上赛季已经知道的结果，不偷看）。
        # 和单独跑 Elo 有一点不同：单独跑的 Elo 拟合好“评分 → 概率”以后出预测时才换赛季，这里只要 dyn 能出预测就换。
        # 所以上赛季的补赛如果在新赛季开始后才知道结果，这里的赛季基准可能和单独跑的差一点（极少见）
        self.elo.rating_diff(comp, season, home, away)
        base = self.elo.season_base.get((comp, self.elo.current_season.get(comp)), self.elo.initial)
        eh = self.elo.ratings.get((comp, home), base) - base
        ea = self.elo.ratings.get((comp, away), base) - base
        H = self.elo.home_advantage
        ph, pa = self.pi.view(comp, season, home), self.pi.view(comp, season, away)
        egd = self.pi.psi(ph[0]) - self.pi.psi(pa[1])            # 主队这场的预期净胜球
        prh, pra = float(self.pi.is_promoted(comp, season, home)), float(self.pi.is_promoted(comp, season, away))
        rh, ra, euh, eua = rest
        th, ta = self.table.view(comp, season, home), self.table.view(comp, season, away)
        day = _days(ko)
        fh, fa = self.form.view(comp, home, day), self.form.view(comp, away, day)
        lc = self.league_code.get(comp, -1.0)
        hrow = [lc, 1.0, llh, lla, sdh, sda, hm, eh, ea, eh - ea + H, ph[0], pa[1], egd,
                prh, pra, rh, ra, euh, eua, *th, *ta, *fh, *fa]
        arow = [lc, 0.0, lla, llh, sda, sdh, hm, ea, eh, ea - eh - H, pa[1], ph[0], -egd,
                pra, prh, ra, rh, eua, euh, *ta, *th, *fa, *fh]
        return np.array(hrow, dtype=float), np.array(arow, dtype=float)

    # ---------------- 处理比赛结果 ----------------
    def _process(self, rows: pd.DataFrame) -> None:
        """按开球顺序：每场先算赛前特征（存进训练表），再把比分交给 dyn、Elo、pi-rating、积分榜、近期状态。

        同一时刻开球的几场（同一轮的同时开球、只有日期的比赛）：先把这几场的特征都算完，再一场一场更新。
        否则 match_id 大的那场的特征会用到同时开球的别场的比分（积分榜名次、dyn 的联赛平均会动），
        出预测时这些比分还不知道，训练和预测就对不上了。更新的顺序（开球时间、match_id）和单独跑 dyn、Elo 一样。
        只管“完全同一时刻”：比这场早几小时开球、但截止时间（T-24 等）还不知道结果的比赛，训练表里照样算进去了。
        这是所有时点共用一张训练表的代价（影响很小；出预测时不会偷看），要去掉得每个时点各建一张训练表。
        """
        if rows is None or rows.empty:
            return
        r = rows[rows["result"].notna() & rows["home_goals"].notna() & ~rows["match_id"].isin(self.processed)]
        if r.empty:
            return
        r = r.sort_values(["kickoff_earliest_utc", "match_id"])
        num = {c: pd.to_numeric(r[c], errors="coerce").to_numpy(float) if c in r else np.full(len(r), np.nan)
               for c in SHOT_COLS + REST_COLS}
        items = list(r[["match_id", "competition", "season", "home_team", "away_team", "home_goals", "away_goals",
                        "result", "kickoff_earliest_utc"]].itertuples(index=False, name=None))
        start = 0
        while start < len(items):
            end = start
            while end < len(items) and items[end][8] == items[start][8]:   # 同一个开球时间的一组
                end += 1
            feats = [self._features(it[1], int(it[2]), it[3], it[4], it[8], tuple(num[c][k] for c in REST_COLS))
                     for k, it in enumerate(items[start:end], start)]
            for k, f in zip(range(start, end), feats):
                self._observe_row(items[k], f, num["home_shots_on_target"][k], num["away_shots_on_target"][k])
            start = end

    def _observe_row(self, item, feats, sh, sa) -> None:
        """一场比赛踢完：交给各个计算器更新，再把赛前特征存进训练表（这场的比分不会进这场的特征）。"""
        mid, comp, season, home, away, hg, ag, res, ko = item
        mid, season, hg, ag = int(mid), int(season), int(hg), int(ag)
        self.processed.add(mid)
        # dyn 的更新和单独跑 dyn 时 _process 做的一样
        self.dyn.processed.add(mid)
        lg = self.dyn._prepare(comp, season, home, away, ko)
        if lg is not None:
            self.dyn._observe(lg, ko, home, away, hg, ag)
        self.elo.process_row(mid, comp, season, home, away, hg, ag, res, ko)
        self.pi.observe(comp, season, home, away, hg, ag)
        self.table.observe(comp, season, home, away, hg, ag)
        day = _days(ko)
        self.form.observe(comp, home, day, hg, ag, sh, sa)
        self.form.observe(comp, away, day, ag, hg, sa, sh)
        lgs = self.dyn.leagues.get(comp)
        if feats is None or lgs is None or season - lgs.first_season < self.dyn.burn_in_seasons:
            return
        self.rows += [feats[0], feats[1]]
        self.goals += [hg, ag]
        self.results += [RESULT_INDEX[res]] * 2
        self.kickoffs += [ko, ko]

    def update(self, new_rows: pd.DataFrame, cutoff: str) -> None:
        self._process(new_rows)

    # ---------------- 训练 ----------------
    def _train(self, cutoff: str) -> None:
        full = np.vstack(self.rows)
        ko = np.array(self.kickoffs)
        if self.target == "goals":
            X, y = full[:, self.use], np.array(self.goals, dtype=float)
        else:                                                    # 多分类：每场只要主队那一行（第 2 列 is_home = 1）
            keep = full[:, 1] == 1.0
            X, ko, y = full[keep][:, self.use], ko[keep], np.array(self.results, dtype=float)[keep]
        if len(X) < self.min_train_rows:
            return
        names = [self.names[i] for i in self.use]
        cat = ["league"] if "league" in names else []
        valid = ko >= window_start(cutoff, self.valid_days)
        if valid.sum() == 0 or (~valid).sum() < self.min_train_rows:
            return                    # 分不出验证集就不知道种多少棵树：先不训练（不出预测），等数据够了再说
        dtr = lgb.Dataset(X[~valid], y[~valid], feature_name=names, categorical_feature=cat, free_raw_data=False)
        dva = lgb.Dataset(X[valid], y[valid], reference=dtr, free_raw_data=False)
        b = lgb.train(self.lgb_params, dtr, num_boost_round=self.max_rounds, valid_sets=[dva],
                      callbacks=[lgb.early_stopping(self.early_stopping_rounds, verbose=False)])
        best = max(int(b.best_iteration or self.max_rounds), 1)
        dall = lgb.Dataset(X, y, feature_name=names, categorical_feature=cat, free_raw_data=False)
        self.booster = lgb.train(self.lgb_params, dall, num_boost_round=best)
        self.last_train = pd.Timestamp(cutoff)
        self.train_log.append({"cutoff": cutoff, "rows": int(len(X)), "trees": best})

    def fit(self, history: pd.DataFrame, cutoff: str) -> None:
        """每周一次：补上没处理的比赛；dyn 挪平均值、拟合 rho；离上次训练满 retrain_days 天就重新训练树。"""
        self._process(history)
        self.dyn.fit(history, cutoff)
        cut = pd.Timestamp(cutoff)
        if self.rows and (self.last_train is None or cut - self.last_train >= pd.Timedelta(days=self.retrain_days)):
            self._train(cutoff)

    # ---------------- 预测 ----------------
    def predict(self, targets: pd.DataFrame, cutoff: str) -> list:
        out: list = [None] * len(targets)
        if self.booster is None:
            return out
        base = self.dyn.predict(targets, cutoff)                # 只在 dyn 能出预测的比赛上出
        rest = np.column_stack([pd.to_numeric(targets[c], errors="coerce").to_numpy(float) if c in targets
                                else np.full(len(targets), np.nan) for c in REST_COLS])
        items = []
        for r, (comp, season, home, away, ko) in enumerate(targets[
                ["competition", "season", "home_team", "away_team", "kickoff_earliest_utc"]].itertuples(
                index=False, name=None)):
            if base[r] is None:
                continue
            f = self._features(comp, int(season), home, away, ko, tuple(rest[r]))
            if f is not None:
                items.append((r, comp, f))
        if not items:
            return out
        if self.target == "result":
            p = self.booster.predict(np.vstack([f[0] for _, _, f in items])[:, self.use])
            for (r, _, _), q in zip(items, p):
                out[r] = q / q.sum()
            return out
        X = np.vstack([row for _, _, f in items for row in f])[:, self.use]
        lam = self.booster.predict(X).reshape(-1, 2)
        by_comp: dict[str, list] = {}
        for (r, comp, _), (lh, la) in zip(items, lam):
            by_comp.setdefault(comp, []).append((r, lh, la))
        for comp, its in by_comp.items():
            rr, lh, la = zip(*its)
            mats = score_matrices(np.array(lh), np.array(la), self.dyn.leagues[comp].rho, 0.0, self.max_goals)
            for r, m in zip(rr, mats):
                out[r] = m
        return out

    # ---------------- 给人看 ----------------
    def importance(self) -> pd.DataFrame:
        """最后一次训练里每个特征的“增益”（树用它分叉时，损失一共降了多少），从大到小，占比加起来是 1。"""
        if self.booster is None:
            return pd.DataFrame(columns=["feature", "gain_share"])
        g = self.booster.feature_importance("gain")
        df = pd.DataFrame({"feature": self.booster.feature_name(), "gain_share": g / max(g.sum(), 1e-12)})
        return df.sort_values("gain_share", ascending=False, ignore_index=True)

    def params(self) -> dict:
        imp = self.importance().head(15)
        return {"特征": self.features, "目标": self.target, "训练次数": len(self.train_log),
                "最后一次训练": self.train_log[-1] if self.train_log else None,
                "每次训练的树数": [t["trees"] for t in self.train_log],
                "用得最多的特征（增益占比）": {f: round(float(v), 4) for f, v in zip(imp["feature"], imp["gain_share"])},
                "dyn": self.dyn.params()}
