"""所有模型共用的接口（第 2 阶段起，频率、Elo、泊松、DC……都按这个写，回测框架不用为每个模型改代码）。

一个模型就是一个对象，有三个动作：

    fit(history, cutoff)      —— “重新拟合”：用截止时间之前的全部历史比赛，重新估计参数。回测里每隔几天调用一次
                                 （settings.yaml 的 backtest.refit_days，默认 7 天），不是每场都调用（需求文档第 5 节“回测的计算量”）。
    update(new_rows, cutoff)  —— “增量更新”：两次重新拟合之间，又有新的比赛结果能知道了，把它们交给模型。
                                 便宜的更新放这里（例如 Elo 每场赛后加减分）；什么都不做也可以（频率模型就不做）。
    predict(targets, cutoff)  —— 给一批比赛出预测。每场返回下面三种之一：
                                   - 比分矩阵（二维数组，第 i 行第 j 列 = 主队进 i 球、客队进 j 球的概率），
                                     回测用 src/common/markets.py 从它推出胜平负、大小球、双方进球；
                                   - 胜平负三个概率 [主胜, 平, 客胜]（一维、长度 3），模型出不了比分矩阵时用（例如 Elo）；
                                   - None：这场出不了预测（例如截止前的比赛太少），回测里记下来，不打分。

两条规矩（保证不偷看未来）：
1. 模型自己不碰数据库，也不拿取数层（FeatureStore）。数据都由回测框架通过取数层取好，以表格的形式交给模型：
   history / new_rows 里只有截止时间之前已经能知道的比赛；targets 里只有赛前就知道的比赛信息（谁对谁、哪天），没有比分。
   为什么不把取数层交给模型：取数层只能检查“截止时间之前”，但截止时间是调用方给的；
   模型如果自己拿着取数层、随手填一个更晚的截止时间，取数层是看不出来的。所以干脆不给。
2. 拟合用的“截止时间”就是回测框架给的 cutoff，模型里不要用“现在”的时间。

模型要声明：
    name：名字，出现在回测结果表里（例如 'elo'）；
    data：要哪几类数据，按 config/data_usage.yaml 的名字（第 0 步模型只要“比分”）；
    usage：用途，第 2 阶段的模型都是“A版输入”（不用赔率的模型）。
"""
from __future__ import annotations

import numpy as np
import pandas as pd


class Model:
    """模型的基类。新模型继承它，至少写 fit 和 predict。"""

    name: str = "model"
    data: tuple[str, ...] = ("比分",)
    usage: str = "A版输入"

    def fit(self, history: pd.DataFrame, cutoff: str) -> None:
        """用截止时间之前的全部历史比赛重新拟合。history 按开球时间从早到晚排，列见 FeatureStore.historical_matches。"""
        raise NotImplementedError

    def update(self, new_rows: pd.DataFrame, cutoff: str) -> None:
        """两次拟合之间新能知道的比赛结果。默认什么都不做（下次 fit 时一起用）。"""

    def predict(self, targets: pd.DataFrame, cutoff: str) -> list:
        """给 targets 里每场比赛出预测，返回和 targets 一样长的列表（每项：比分矩阵 / 胜平负三个概率 / None）。"""
        raise NotImplementedError

    def params(self) -> dict:
        """当前参数（存进回测结果，以后复盘能知道“这版预测用的是什么参数”，需求文档第 16 节“预测记录”）。"""
        return {}


def check_result_probs(p) -> np.ndarray:
    """检查胜平负三个概率：长度 3、在 0–1 之间、加起来是 1（差得太多说明模型算错了，报错不悄悄改）。"""
    a = np.asarray(p, dtype=float)
    if a.shape != (3,) or np.isnan(a).any() or (a < 0).any():
        raise ValueError(f"胜平负概率要是 3 个 0–1 之间的数，收到 {p!r}")
    if abs(a.sum() - 1) > 1e-6:
        raise ValueError(f"胜平负概率加起来是 {a.sum():.8f}，不是 1")
    return a / a.sum()


def window_start(cutoff: str, days: float) -> str:
    """截止时间往前 days 天，写成和数据库一样的 UTC 文字（'YYYY-MM-DD HH:MM:SS'，按文字比大小就是按时间比）。"""
    return (pd.Timestamp(cutoff) - pd.Timedelta(days=float(days))).strftime("%Y-%m-%d %H:%M:%S")
