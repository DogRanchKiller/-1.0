"""第 11 阶段：国家队 Elo（需求文档第 4 节“第 11 阶段：国际赛进数据库和开发回测”的“模型和研究（INC）”）。

只给研究程序 src/evaluate/intl_study.py 用：不登记进模型注册表，不上线，不改 elo.py、ordered_logit.py。

一、国家队 Elo 怎么算（按 eloratings.net 公开的算法，维基百科 World Football Elo Ratings 条目，2026-11-05 核对过）
  每支国家队一个评分，第一次出现时 1500（settings 的 intl_study.start_rating）。每场比赛踢完更新一次：
  1. 预期得分（主队的）：We = 1 ÷ (10^(−dr/400) + 1)，dr = 主队评分 − 客队评分 + 主场优势。
     主场优势 100 分（intl_study.home_advantage）；中立场（martj42 标了 neutral）不加。
     例：两队都是 1500、不是中立场 → dr = 100 → We = 1 ÷ (10^(−0.25) + 1) ≈ 0.640。中立场 → dr = 0 → We = 0.5。
  2. 实际得分 W：主队赢 1、平 0.5、输 0。加时赛分出胜负照样算赢输；点球大战算平（martj42 的比分本来就是加时后的平局，
     所以直接按比分算就是平，不用再看谁赢了点球）。
  3. 这场比赛有多重要 K（按赛事分档，intl_study.k_by_tournament；赛事名是 martj42 的写法）：
     世界杯正赛、奥运会（1908–1980）60；洲际杯正赛和洲际之间的比赛 50；世界杯、洲际杯预选赛和大型赛事 40；其他赛事 30；友谊赛 20。
     赛事名没写进设置的直接报错（不猜它属于哪一档）。
  4. 净胜球系数 G：平局或净胜 1 球 1；2 球 1.5；N ≥ 3 球 (11 + N) ÷ 8（3 球 1.75、4 球 1.875、5 球 2……）。
     这就是 elo.py 里的 goal_diff_multiplier，直接拿来用（只 import，不改 elo.py）。
  5. 评分变化 P = K × G × (W − We)。维基百科条目写明 P 先四舍五入到整数再加减（intl_study.round_points）。
     主队加 P，客队减 P（两队加减一样多）。
     例：两队 1500、主场、友谊赛（K 20）、主队 2:0 赢 → G 1.5，P = 20 × 1.5 × (1 − 0.640) = 10.8 → 11；主队 1511，客队 1489。
  友谊赛照样更新评分（K 小一点），只是研究里不给友谊赛打分（轮换阵容，平局多少不一样）。

二、两个版本（事先写下的“中立场比较”）
  - intl_elo：中立场不加主场优势（Elo 更新和换成胜平负都不加）；
  - intl_elo_home：不管中立场，一律加主场优势 100（Elo 更新和换成胜平负都加），其余完全一样。
  两版的评分会慢慢分开（同一场比赛的 We 不一样，加减的分就不一样），所以要各算一套。

三、评分 → 胜平负：有序 logit（和第 2 阶段 Elo 一样的公式，复用 ordered_logit.py 的拟合函数，不改它）
  x = 赛前评分差（含主场优势：intl_elo 中立场不加，intl_elo_home 一律加）。
  每次重新拟合（研究程序每 7 天一次）只用：截止时间前已经知道比分的、比赛日期在截止时间往前 8 年（intl_study.fit_years）以内的
  正式比赛（不含友谊赛）。拿每场“赛前评分差”和真实结果拟合斜率、两个切点。每次都从同一个起点拟合，
  这样从哪一年开始跑，同一个截止时间拟合出来的数都一样。
  能用的比赛少于 min_fit_n 场时不拟合、不出预测（国际赛 8 年有几千场，正常不会碰到，只是防止拿很少的比赛硬拟合）。

四、不偷看
  这个类自己不取数据：研究程序只把截止时间前已经能知道比分的比赛交给 update（取数层保证），
  每场还带着“比分什么时候能知道”（known_at），拟合时再查一遍：known_at 晚于截止时间的比赛不用。
"""
from __future__ import annotations

import bisect

import numpy as np
import pandas as pd

from src.models import ordered_logit
from src.models.elo import expected_home, goal_diff_multiplier

RESULT_INDEX = {"H": 0, "D": 1, "A": 2}       # 和 metrics.py 一样：0 主胜、1 平、2 客胜
SCORE = {"H": 1.0, "D": 0.5, "A": 0.0}         # 主队的实际得分 W


class UnknownTournamentError(ValueError):
    """赛事名没写进 intl_study.k_by_tournament：不知道 K 是哪一档，不猜，直接停下。"""


def result_of(home_goals: int, away_goals: int) -> str:
    """比分 → 'H' / 'D' / 'A'。点球大战的比赛 martj42 记的是加时后的平局，所以这里自然就是平局。"""
    if home_goals > away_goals:
        return "H"
    return "D" if home_goals == away_goals else "A"


def round_half_away(x: float) -> float:
    """四舍五入到整数，0.5 往离 0 远的方向进（10.5 → 11，−10.5 → −11），主客两边对称。
    Python 自带的 round 是“银行家舍入”（10.5 → 10、11.5 → 12），两边会不对称，所以不用它。"""
    return float(np.sign(x) * np.floor(abs(x) + 0.5))


def k_table(k_by_tournament: dict) -> dict[str, float]:
    """设置里的 {K: [赛事名, ...]} 换成 {赛事名: K}。同一个赛事名写进两档直接报错（说明设置写错了）。"""
    out: dict[str, float] = {}
    for k, names in k_by_tournament.items():
        for name in names or []:
            if name in out:
                raise ValueError(f"赛事「{name}」在 k_by_tournament 里写了两次（K {out[name]:g} 和 {float(k):g}），请只留一档")
            out[str(name)] = float(k)
    return out


class IntlElo:
    """一套国家队 Elo 评分 + 一组有序 logit 参数。use_neutral=False 就是 intl_elo_home（一律加主场优势）。"""

    def __init__(self, k_by_tournament: dict, home_advantage: float = 100, start_rating: float = 1500,
                 use_neutral: bool = True, round_points: bool = True, friendly: str = "Friendly",
                 fit_years: float = 8, min_fit_n: int = 200, name: str = "intl_elo"):
        self.name = name
        self.k_of = k_table(k_by_tournament)
        self.home_advantage = float(home_advantage)
        self.start_rating = float(start_rating)
        self.use_neutral = bool(use_neutral)
        self.round_points = bool(round_points)
        self.friendly = friendly
        if not float(fit_years).is_integer() or fit_years <= 0:
            raise ValueError(f"fit_years 要是正整数（几年），收到 {fit_years}")
        self.fit_years = int(fit_years)
        self.min_fit_n = int(min_fit_n)
        # ---- 状态 ----
        self.ratings: dict[str, float] = {}      # 球队 → 现在的评分
        self.n_matches: dict[str, int] = {}      # 球队 → 已经算进评分的比赛场数（给人看）
        # 给有序 logit 用的记录（只记正式比赛），按更新顺序（= 比赛日期顺序）排好：
        self.rec_date: list[str] = []            # 比赛日期 'YYYY-MM-DD'
        self.rec_known: list[str] = []           # 比分能知道的时间（UTC 文字）
        self.rec_diff: list[float] = []          # 赛前评分差（含这一版的主场优势）
        self.rec_y: list[int] = []               # 结果 0 主胜 / 1 平 / 2 客胜
        self.last_date = ""                      # 最后更新的那场比赛的日期
        self.out_of_order = 0                    # 比赛日期比已经更新过的还早的场数（正常是 0，研究报告里列出来）
        self.n_updates = 0
        self.theta: np.ndarray | None = None     # 最近一次拟合的参数
        self.fit_n = 0                           # 最近一次拟合用了几场
        self.fit_cutoff: str | None = None       # 最近一次拟合的截止时间

    @classmethod
    def from_settings(cls, cfg: dict, use_neutral: bool, name: str) -> "IntlElo":
        """从 settings.yaml 的 intl_study 段建一个。"""
        return cls(cfg["k_by_tournament"], cfg["home_advantage"], cfg["start_rating"], use_neutral,
                   cfg.get("round_points", True), cfg["friendly_tournament"], cfg["fit_years"],
                   cfg.get("min_fit_n", 200), name)

    # ---------------- 小工具 ----------------
    def k_for(self, tournament: str) -> float:
        """这个赛事的 K。没写进设置就报错，不猜。"""
        try:
            return self.k_of[tournament]
        except KeyError:
            raise UnknownTournamentError(
                f"赛事「{tournament}」没写进 settings.yaml 的 intl_study.k_by_tournament，不知道 K 是哪一档。"
                f"请按维基百科的分档规则补上再跑") from None

    def home_adv(self, neutral) -> float:
        """这场比赛加多少主场优势：intl_elo 中立场加 0；intl_elo_home 一律加。"""
        return 0.0 if (self.use_neutral and bool(neutral)) else self.home_advantage

    def rating(self, team: str) -> float:
        """球队现在的评分；第一次出现的队按起始分。"""
        return self.ratings.get(team, self.start_rating)

    def rating_diff(self, home: str, away: str, neutral) -> float:
        """赛前评分差 dr = 主队评分 + 主场优势（看版本和中立场）− 客队评分。"""
        return self.rating(home) + self.home_adv(neutral) - self.rating(away)

    def is_official(self, tournament: str) -> bool:
        """正式比赛 = 不是友谊赛。"""
        return tournament != self.friendly

    # ---------------- 每场赛后更新 ----------------
    def update(self, match_date: str, home: str, away: str, home_goals: int, away_goals: int, neutral,
               tournament: str, known_at: str) -> float:
        """一场比赛更新进评分，返回主队加的分 P（客队减同样多）。研究程序按比赛日期顺序一场一场调用。"""
        k = self.k_for(tournament)                        # 赛事名不认识：在改评分之前就报错
        hg, ag = int(home_goals), int(away_goals)
        res = result_of(hg, ag)
        rh, ra = self.rating(home), self.rating(away)
        adv = self.home_adv(neutral)
        if self.is_official(tournament):                  # 先记下赛前评分差（给有序 logit 用），再改评分
            self.rec_date.append(str(match_date))
            self.rec_known.append(str(known_at))
            self.rec_diff.append(rh + adv - ra)
            self.rec_y.append(RESULT_INDEX[res])
        we = expected_home(rh, ra, adv)                   # = 1 ÷ (10^(−dr/400) + 1)
        p = k * goal_diff_multiplier(hg, ag) * (SCORE[res] - we)
        if self.round_points:
            p = round_half_away(p)
        self.ratings[home] = rh + p
        self.ratings[away] = ra - p
        self.n_matches[home] = self.n_matches.get(home, 0) + 1
        self.n_matches[away] = self.n_matches.get(away, 0) + 1
        if str(match_date) < self.last_date:
            self.out_of_order += 1
        self.last_date = max(self.last_date, str(match_date))
        self.n_updates += 1
        return p

    def update_frame(self, rows: pd.DataFrame) -> int:
        """一批比赛（列 match_date、home_team、away_team、home_goals、away_goals、neutral、tournament、known_at_utc）
        按比赛日期、match_id 排好，一场一场更新。返回更新了几场。"""
        if rows.empty:
            return 0
        rows = rows.sort_values(["match_date", "match_id"])
        for d, h, a, hg, ag, nt, tn, kn in rows[["match_date", "home_team", "away_team", "home_goals", "away_goals",
                                                 "neutral", "tournament", "known_at_utc"]].itertuples(index=False, name=None):
            self.update(d, h, a, hg, ag, nt, tn, kn)
        return len(rows)

    # ---------------- 有序 logit ----------------
    def fit_window(self, cutoff: str) -> tuple[np.ndarray, np.ndarray]:
        """拟合用哪些比赛：正式比赛、比分在截止时间前已经知道、比赛日期在截止时间往前 fit_years 年以内。
        返回 (评分差, 结果)。记录是按比赛日期排好的，先二分查找找到 8 年前那一场，再往后筛。"""
        start = (pd.Timestamp(cutoff) - pd.DateOffset(years=int(self.fit_years))).strftime("%Y-%m-%d")
        i =bisect.bisect_left(self.rec_date, start) if self.out_of_order == 0 else 0
        dates = np.array(self.rec_date[i:], dtype=object)
        known = np.array(self.rec_known[i:], dtype=object)
        ok = (dates >= start) & (known <= cutoff) if len(dates) else np.zeros(0, bool)
        return np.asarray(self.rec_diff[i:], float)[ok], np.asarray(self.rec_y[i:], int)[ok]

    def fit(self, cutoff: str) -> bool:
        """在截止时间 cutoff 重新拟合有序 logit。比赛够了返回 True；不够返回 False（这时不出预测）。"""
        d, y = self.fit_window(cutoff)
        self.fit_cutoff, self.fit_n = cutoff, len(d)
        if len(d) < self.min_fit_n:
            self.theta = None
            return False
        self.theta = ordered_logit.fit_ordered_logit(d, y)   # 每次都从同一个起点拟合（见文件开头第三部分）
        return True

    def predict(self, home: str, away: str, neutral) -> np.ndarray | None:
        """这场比赛赛前的 [主胜, 平, 客胜]；还没拟合（或比赛不够）就是 None。"""
        if self.theta is None:
            return None
        return ordered_logit.result_probs(self.theta, self.rating_diff(home, away, neutral))[0]

    # ---------------- 给人看 ----------------
    def params(self) -> dict:
        out = {"版本": self.name, "中立场不加主场优势": self.use_neutral, "主场优势": self.home_advantage,
               "起始分": self.start_rating, "P四舍五入": self.round_points, "更新场数": self.n_updates,
               "日期顺序不对的场数": self.out_of_order, "最近一次拟合截止": self.fit_cutoff, "拟合场数": self.fit_n}
        if self.theta is not None:
            out["有序logit"] = ordered_logit.describe(self.theta)
        return out

    def top(self, n: int = 10) -> list[tuple[str, float]]:
        """评分最高的 n 支队（给人看）。"""
        return sorted(((t, round(r, 1)) for t, r in self.ratings.items()), key=lambda x: -x[1])[:n]
