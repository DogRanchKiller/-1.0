"""取数层（第 2 阶段）：按截止时间、用途、运行模式取数据，拦住“偷看未来”和“用错数据”。

以后所有模型和回测都从这里拿数据，不直接写 SQL 查数据库。详细说明见 store.py 开头。
"""
from src.features.errors import FeatureAccessError, LookAheadError, SeasonLockedError, UsageNotAllowedError
from src.features.store import FeatureStore, open_readonly

__all__ = ["FeatureStore", "open_readonly", "FeatureAccessError", "LookAheadError", "SeasonLockedError",
           "UsageNotAllowedError"]
