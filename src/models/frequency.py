"""第 0 步模型之一：历史频率基线（需求文档第 4 节步骤 0“最低标准”）。

两个版本：
1. FrequencyModel（名字 frequency）：按联赛数主胜、平、客胜的比例。
   例：截止前 3 年英超踢了 1140 场，主胜 520、平 290、客胜 330 → 每场都预测 [45.6%, 25.4%, 28.9%]。
   它完全不管谁对谁，曼城主场打降级队和降级队主场打曼城给的概率一样。
   所以它是“最低标准”：后面任何模型连它都比不过，说明模型有问题。
2. FrequencyPoissonModel（名字 frequency_poisson）：按联赛的场均进球 → 独立泊松比分矩阵。
   例：截止前 3 年英超主队场均 1.53 球、客队 1.21 球 → 每场都用“主 1.53、客 1.21”的泊松比分矩阵。
   好处是能出比分矩阵，大小球、双方进球、亚盘都能从它算（第 3 节“所有玩法从同一个矩阵算”）。
   平局会估偏，方向看联赛。两个相反的原因叠在一起：
   ① 每场都用同一个联赛平均，相当于假设每场都是“两队差不多强”，两队差不多强最容易打平 → 往高估平局的方向偏；
   ② 两队进球独立的泊松本身会低估 0:0、1:1（Dixon-Coles 在第 4 节步骤 2 修正）→ 往低估平局的方向偏。
   2005/06–2023/24 实测（预测平局平均 对 实际）：英超 25.4% 对 24.2%、西甲 25.4% 对 24.9%（①占上风，高估）；
   德甲 24.2% 对 25.1%、意甲 25.5% 对 26.4%、法甲 26.7% 对 27.7%（②占上风，低估）。

只用截止前的数据：fit 收到的 history 本身就只有截止前已经能知道的比赛；这里再按 window_days 只取最近几年
（settings.yaml 的 models.frequency.window_days）。主场优势在变（2020/21 多数空场，主胜降到 40.8%），太久远的比例不代表现在。
窗口里的比赛少于 models.min_matches 场时，这个联赛不出预测（返回 None）。
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from src.common.markets import poisson_score_matrix
from src.models.base import Model, window_start


def _recent(history: pd.DataFrame, cutoff: str, days: float) -> pd.DataFrame:
    """history 里开球在 [截止 − days 天, 截止] 之间、有比分的比赛。"""
    h = history[history["kickoff_earliest_utc"] >= window_start(cutoff, days)]
    return h[h["result"].notna() & h["home_goals"].notna() & h["away_goals"].notna()]


class FrequencyModel(Model):
    """按联赛的主胜 / 平 / 客胜比例。"""

    name = "frequency"

    def __init__(self, window_days: float = 1095, min_matches: int = 200):
        self.window_days = window_days
        self.min_matches = min_matches
        self.table: dict[str, np.ndarray] = {}   # 联赛代码 → [主胜, 平, 客胜]
        self.counts: dict[str, int] = {}          # 联赛代码 → 用了几场

    def fit(self, history: pd.DataFrame, cutoff: str) -> None:
        self.table, self.counts = {}, {}
        recent = _recent(history, cutoff, self.window_days)
        for comp, part in recent.groupby("competition"):
            if len(part) < self.min_matches:
                continue
            share = part["result"].value_counts(normalize=True)
            self.table[comp] = np.array([share.get("H", 0.0), share.get("D", 0.0), share.get("A", 0.0)])
            self.counts[comp] = len(part)

    def predict(self, targets: pd.DataFrame, cutoff: str) -> list:
        return [self.table.get(c) for c in targets["competition"]]

    def params(self) -> dict:
        return {c: {"主胜": round(float(p[0]), 4), "平": round(float(p[1]), 4), "客胜": round(float(p[2]), 4),
                    "场数": self.counts[c]} for c, p in self.table.items()}


class FrequencyPoissonModel(Model):
    """按联赛的场均进球 → 独立泊松比分矩阵（每个联赛一个矩阵，同一联赛每场都一样）。"""

    name = "frequency_poisson"

    def __init__(self, window_days: float = 1095, min_matches: int = 200):
        self.window_days = window_days
        self.min_matches = min_matches
        self.means: dict[str, tuple[float, float, int]] = {}   # 联赛 → (主队场均进球, 客队场均进球, 场数)
        self.matrix: dict[str, np.ndarray] = {}

    def fit(self, history: pd.DataFrame, cutoff: str) -> None:
        self.means, self.matrix = {}, {}
        recent = _recent(history, cutoff, self.window_days)
        for comp, part in recent.groupby("competition"):
            if len(part) < self.min_matches:
                continue
            hg, ag = float(part["home_goals"].mean()), float(part["away_goals"].mean())
            self.means[comp] = (hg, ag, len(part))
            # 同一个矩阵对象给这个联赛的每场比赛用；回测按对象缓存推出来的玩法，不重复算
            self.matrix[comp] = poisson_score_matrix(hg, ag)

    def predict(self, targets: pd.DataFrame, cutoff: str) -> list:
        return [self.matrix.get(c) for c in targets["competition"]]

    def params(self) -> dict:
        return {c: {"主队场均进球": round(h, 4), "客队场均进球": round(a, 4), "场数": n}
                for c, (h, a, n) in self.means.items()}
