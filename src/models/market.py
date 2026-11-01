"""步骤 8 的主比较对象 market_pre：直接用赛前赔率去水后的胜平负当预测（需求文档第 4 节“步骤 8 怎么做”）。

不训练、不看比分：回测框架按“本场”接好赔率特征（src/features/market_odds.py，看 uses_pre_odds 去接），
这里只把 mo_p_home、mo_p_draw、mo_p_away 三个数拿出来。哪家的赔率、怎么去水，都在 market_odds 那边定
（平博 → 多家平均，按比例去水），和 gb_b 拿到的是同一组数，所以两者的差只来自“树在赔率之外又用了什么”。
出不了比分矩阵（只有胜平负），所以 goals_backtest 里没有比分 log loss、大小球、双方进球。
这场没有赛前赔率就返回 None（不打分；回测框架只比每个模型都有预测的比赛）。
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from src.models.base import Model, check_result_probs

COLS = ["mo_p_home", "mo_p_draw", "mo_p_away"]


class MarketModel(Model):
    data = ("比分",)
    uses_pre_odds = True

    def __init__(self, name: str, usage: str = "B版输入"):
        if usage == "A版输入":
            raise ValueError("A 版不用赔率：market 模型的用途要是「B版输入」（或评估类用途）")
        self.name, self.usage = name, usage

    def fit(self, history: pd.DataFrame, cutoff: str) -> None:
        """没有要拟合的参数。"""

    def predict(self, targets: pd.DataFrame, cutoff: str) -> list:
        if not all(c in targets for c in COLS):
            return [None] * len(targets)
        p = targets[COLS].apply(pd.to_numeric, errors="coerce").to_numpy(float)
        return [check_result_probs(row) if np.isfinite(row).all() else None for row in p]

    def params(self) -> dict:
        return {"说明": "赛前赔率去水后的胜平负（设置见 settings.yaml 的 market_odds）"}
