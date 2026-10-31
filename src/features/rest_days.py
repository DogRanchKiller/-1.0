"""休息天数和“欧战后”标签（需求文档第 4 节“步骤 6 怎么做”、第 17 节数据用途“杯赛欧战赛程”）。

一、算什么
  每场比赛、每个队：
    rest_days   离这队上一场正式比赛（联赛、杯赛、欧战都算）隔了几天，按日期算（周六踢、下周三踢 = 4 天），最多记 cap_days 天；
                上一场找不到（数据里第一场、休了很久）也记 cap_days。
    after_euro  上一场是不是欧战（settings 里 euro_competitions 那几个赛事），而且隔了 euro_within_days 天以内：是 1，不是 0。
  主队的两列叫 home_rest_days、home_after_euro，客队叫 away_rest_days、away_after_euro。

二、为什么这样就不偷看未来
  赛程只从取数层用「休息天数和欧战后标签」取，取到的只有截止时间之前已经开球的比赛（取数层按最晚可能的开球时间判断）。
  这里只把赛程算成天数，杯赛、欧战的对手、比分都不交给模型（数据用途字典：杯赛欧战赛程不能直接当模型输入）。
  “上一场”只在这场比赛日期之前的比赛里找，同一天的不算（同一队一天不会踢两场）。
  已知的偏差：T-24 时，开球前 24 小时内踢的比赛（例如前一天晚上的杯赛）还不知道踢没踢，上一场会找成再前一场，休息天数偏多。
  训练用的历史比赛没有这个问题（那时赛程早就知道了），所以预测时少数比赛的休息天数和训练时的算法略有不同。

三、怎么用（回测框架 src/evaluate/backtest.py）
  每次截止时间往前走，就把新开球的赛程交给 add()；给模型的历史比赛、要预测的比赛先用 annotate() 加上上面四列。
"""
from __future__ import annotations

import bisect

import numpy as np
import pandas as pd

COLUMNS = ["home_rest_days", "away_rest_days", "home_after_euro", "away_after_euro"]


class RestDays:
    """每队一张按日期排好的赛程表：[(日期序号, match_id, 赛事), ...]。"""

    def __init__(self, euro_competitions, cap_days: int = 30, euro_within_days: int = 4):
        self.euro = set(euro_competitions)
        self.cap = int(cap_days)
        self.euro_within = int(euro_within_days)
        self.games: dict[str, list[tuple[int, int, str]]] = {}
        self.seen: set[int] = set()

    @staticmethod
    def _day(match_date) -> int:
        """'YYYY-MM-DD' → 从 1970-01-01 起第几天（只用来相减）。"""
        return int(pd.Timestamp(str(match_date)[:10]).value // 86_400_000_000_000)

    def add(self, schedule: pd.DataFrame) -> None:
        """把取数层 FeatureStore.schedule 取到的赛程加进来（重复的跳过）。"""
        if schedule is None or schedule.empty:
            return
        for mid, comp, d, home, away in schedule[["match_id", "competition", "match_date", "home_team",
                                                  "away_team"]].itertuples(index=False, name=None):
            mid = int(mid)
            if mid in self.seen:
                continue
            self.seen.add(mid)
            day = self._day(d)
            for team in (home, away):
                bisect.insort(self.games.setdefault(team, []), (day, mid, comp))

    def team(self, team: str, match_date) -> tuple[float, int]:
        """这队在 match_date 这天之前的上一场：(休息天数, 欧战后 0/1)。"""
        day = self._day(match_date)
        g = self.games.get(team)
        if not g:
            return float(self.cap), 0
        k = bisect.bisect_left(g, (day, -1, ""))      # 日期 < 这天的最后一场
        if k == 0:
            return float(self.cap), 0
        prev_day, _, comp = g[k - 1]
        gap = day - prev_day
        return float(min(gap, self.cap)), int(comp in self.euro and gap <= self.euro_within)

    def annotate(self, df: pd.DataFrame) -> pd.DataFrame:
        """给一张比赛表（每场一行，有 match_date、home_team、away_team）加上四列，返回新表（不改原表）。"""
        out = df.copy()
        if out.empty:
            for c in COLUMNS:
                out[c] = pd.Series(dtype=float)
            return out
        vals = np.array([(*self.team(h, d), *self.team(a, d)) for d, h, a in
                         out[["match_date", "home_team", "away_team"]].itertuples(index=False, name=None)],
                        dtype=float)
        out["home_rest_days"], out["home_after_euro"] = vals[:, 0], vals[:, 1]
        out["away_rest_days"], out["away_after_euro"] = vals[:, 2], vals[:, 3]
        return out
