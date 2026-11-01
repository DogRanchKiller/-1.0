"""模型登记表：按名字造模型对象，参数都从 settings.yaml 的 models 段读（不写死在代码里）。

现在能用的名字：
  frequency          历史频率基线（按联赛的主胜 / 平 / 客胜比例）
  frequency_poisson  按联赛场均进球的独立泊松比分矩阵
  elo                Elo 默认设置（models.elo）
  elo_k30 等         models.elo_variants 里事先写好的几组 Elo 设置（只写和默认不同的数）
  poisson、dc、dcnb 等  models.goals_variants 里事先写好的泊松 / Dixon-Coles / DC + 负二项（src/models/goals.py），
                     默认设置在 models.goals，每组只写和默认不同的数
  dyn_q3e5_b0 等      models.dynamic_variants 里事先写好的动态实力模型（src/models/dynamic.py），默认设置在 models.dynamic
  ada_a98 等          models.adaptive_variants 里事先写好的“每队自己的变化速度”（src/models/adaptive.py），
                     默认设置在 models.adaptive；每个候选 dyn 的其余设置 = models.dynamic + adaptive.base 那一组
  gb_l15 等           models.gbdt_variants 里事先写好的特征表 + 梯度提升树（src/models/gbdt.py），默认设置在 models.gbdt；
                     里面带的 dyn、Elo 用 gbdt.dyn、gbdt.elo 那两组的设置
  market_pre         models.market_variants：直接用赛前赔率去水后的胜平负（src/models/market.py，步骤 8 的比较对象）
加新模型：在这里加一个名字，回测框架不用改。
"""
from __future__ import annotations

from src.common.config import load_settings
from src.models.adaptive import AdaptiveDynamicModel
from src.models.base import Model
from src.models.dynamic import DynamicModel
from src.models.elo import EloModel
from src.models.frequency import FrequencyModel, FrequencyPoissonModel
from src.models.gbdt import GBDTModel
from src.models.goals import GoalsModel
from src.models.market import MarketModel


def available(settings: dict | None = None) -> list[str]:
    """所有能用的模型名字。"""
    s = (settings or load_settings())["models"]
    return ["frequency", "frequency_poisson", "elo", *s.get("elo_variants", {}), *(s.get("goals_variants") or {}), *(s.get("dynamic_variants") or {}), *(s.get("adaptive_variants") or {}),
            *(s.get("gbdt_variants") or {}), *(s.get("market_variants") or {})]


def model_config(name: str, settings: dict | None = None) -> dict:
    """这个名字的模型实际用的全部设置（存预测时记下来，设置一改就能看出是不同版本的模型）。"""
    full = settings or load_settings()
    s = full["models"]
    if name in ("frequency", "frequency_poisson"):
        return {"window_days": s[name]["window_days"], "min_matches": s["min_matches"]}
    variants = s.get("elo_variants", {}) or {}
    if name == "elo" or name in variants:
        return {**s["elo"], **(variants.get(name) or {}), "min_matches": s["min_matches"]}
    goals = s.get("goals_variants") or {}
    if name in goals:
        return {**s["goals"], **(goals[name] or {}), "min_matches": s["min_matches"],
                "max_goals": full["calc"]["max_goals"]}
    dyn = s.get("dynamic_variants") or {}
    if name in dyn:
        return {**s["dynamic"], **(dyn[name] or {}), "min_matches": s["min_matches"],
                "max_goals": full["calc"]["max_goals"]}
    ada = s.get("adaptive_variants") or {}
    if name in ada:
        a = {**s["adaptive"], **(ada[name] or {})}
        base = a.pop("base")
        cfg = {**s["dynamic"], **(dyn[base] or {}), **a, "min_matches": s["min_matches"],
               "max_goals": full["calc"]["max_goals"]}
        cfg.pop("q")                                     # q 由 candidates_q 给，不记一个用不上的数
        return cfg
    gb = s.get("gbdt_variants") or {}
    if name in gb:
        g = {**s["gbdt"], **(gb[name] or {})}
        dyn_name, elo_name = g.pop("dyn"), g.pop("elo")
        g["dyn"] = model_config(dyn_name, full)
        g["elo"] = model_config(elo_name, full)
        lg = full["leagues"]
        return {**g, "leagues": sorted({c for grp in lg.values() for c in grp}), "min_matches": s["min_matches"],
                "max_goals": full["calc"]["max_goals"]}
    mk = s.get("market_variants") or {}
    if name in mk:                                       # 赔率怎么选、怎么去水在 market_odds 段，一起记下来
        return {**(mk[name] or {}), "market_odds": full["market_odds"]}
    raise ValueError(f"没有叫「{name}」的模型。能用的：{'、'.join(available(settings))}")


def build_model(name: str, settings: dict | None = None) -> Model:
    """按名字造一个新的模型对象（每次回测都要新造，模型里有状态，不能两次回测共用一个）。"""
    cfg = model_config(name, settings)
    if name == "frequency":
        return FrequencyModel(**cfg)
    if name == "frequency_poisson":
        return FrequencyPoissonModel(**cfg)
    if name in ((settings or load_settings())["models"].get("goals_variants") or {}):
        return GoalsModel(name=name, **cfg)
    if name in ((settings or load_settings())["models"].get("dynamic_variants") or {}):
        return DynamicModel(name=name, **cfg)
    if name in ((settings or load_settings())["models"].get("adaptive_variants") or {}):
        return AdaptiveDynamicModel(name=name, **cfg)
    if name in ((settings or load_settings())["models"].get("gbdt_variants") or {}):
        return GBDTModel(name=name, **cfg)
    if name in ((settings or load_settings())["models"].get("market_variants") or {}):
        return MarketModel(name=name, usage=cfg["usage"])
    return EloModel(name=name, **cfg)
