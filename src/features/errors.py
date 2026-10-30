"""取数层的报错类型。

取数层发现“这样取数据会偷看未来 / 用错地方”时，不悄悄返回空表，而是直接停下来报错，
报错信息用中文讲清楚为什么不行、该怎么改。这样回测不会在不知不觉中算出一个虚高的成绩。

三种报错都属于 FeatureAccessError，调用方想“一网打尽”时可以只接这一种。
"""


class FeatureAccessError(Exception):
    """取数层所有报错的总类。"""


class UsageNotAllowedError(FeatureAccessError):
    """这类数据不允许用在声明的用途上（查数据用途字典 config/data_usage.yaml 的结果）。

    例子：A 版模型要读赔率；本场的收盘赔率拿来当 B 版输入；第 10 阶段前把角球当模型输入。
    """


class LookAheadError(FeatureAccessError):
    """截止时间不对，会看到预测那一刻还不知道的东西。

    例子：给本场做赛前预测，截止时间却设在开球之后。
    """


class SeasonLockedError(FeatureAccessError):
    """当前运行模式不能读这个赛季（第 17 节“三种运行模式”）。

    例子：开发模式下取到了期末考试赛季（2024/25、2025/26）的比赛，或者把考试赛季的比赛当预测目标。
    """
