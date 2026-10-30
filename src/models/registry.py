"""模型登记表：按名字造模型对象，参数都从 settings.yaml 的 models 段读（不写死在代码里）。

现在能用的名字：
  frequency          历史频率基线（按联赛的主胜 / 平 / 客胜比例）
  frequency_poisson  按联赛场均进球的独立泊松比分矩阵
  elo                Elo 默认设置（models.elo）
  elo_k30 等         models.elo_variants 里事先写好的几组 Elo 设置（只写和默认不同的数）
以后加泊松、DC：在这里加一个名字，回测框架不用改。
"""
from __future__ import annotations

from src.common.config import load_settings
from src.models.base import Model
from src.models.elo import EloModel
from src.models.frequency import FrequencyModel, FrequencyPoissonModel


def available(settings: dict | None = None) -> list[str]:
    """所有能用的模型名字。"""
    s = (settings or load_settings())["models"]
    return ["frequency", "frequency_poisson", "elo", *s.get("elo_variants", {})]


def build_model(name: str, settings: dict | None = None) -> Model:
    """按名字造一个新的模型对象（每次回测都要新造，模型里有状态，不能两次回测共用一个）。"""
    s = (settings or load_settings())["models"]
    min_matches = s["min_matches"]
    if name == "frequency":
        return FrequencyModel(window_days=s["frequency"]["window_days"], min_matches=min_matches)
    if name == "frequency_poisson":
        return FrequencyPoissonModel(window_days=s["frequency_poisson"]["window_days"], min_matches=min_matches)
    variants = s.get("elo_variants", {}) or {}
    if name == "elo" or name in variants:
        cfg = {**s["elo"], **(variants.get(name) or {})}
        return EloModel(name=name, min_matches=min_matches, **cfg)
    raise ValueError(f"没有叫「{name}」的模型。能用的：{'、'.join(available(settings))}")
