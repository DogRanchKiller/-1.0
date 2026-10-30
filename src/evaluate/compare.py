"""两个模型谁更准（需求文档第 5 节“怎么算真的更准”）：每场分数差 → 成块 bootstrap 可信区间 → 更准 / 没差别 / 更差。

一、为什么不能只看平均分
  两个模型在 7000 场上平均 log loss 差 0.002，可能只是这批比赛碰巧对其中一个有利。
  要问的是：如果换一批“同样性质”的比赛，差距还在不在？我们没有另一批比赛，就用 bootstrap 模拟。

二、bootstrap 是什么
  从已有的比赛里“有放回地”重新抽一批，和原来一样多（有的比赛被抽到两次，有的一次没抽到），算一次平均分数差；
  重复 2000 次（settings.yaml 的 compare.n_boot），得到 2000 个平均差。
  把它们从小到大排，去掉两头各 2.5%，中间 95%（compare.confidence）的范围就是“可信区间”。
  - 整个区间都小于 0（新模型分数更小）→ 更准；
  - 整个区间都大于 0 → 更差；
  - 区间跨过 0 → 没差别（第 5 节：当作没有差别，保留简单的那个）。

三、为什么按周成块抽（compare.block: week），不一场一场抽
  一场一场抽，假设每场比赛互不相关。其实同一周的比赛不完全独立：
  - 回测里参数每周重新拟合一次（backtest.refit_days = 7），同一周的预测都用同一组参数，参数偏了，这一周一起偏；
  - 同一个周末常有共同的因素：天气、国家队比赛日前后、赛季末“没有战意”的那几轮；
  - 一支球队一周踢两场（周中、周末）时，它“被低估”会同时影响这两场。
  这些都会让同一周的分数差往同一个方向偏。一场一场抽会把这种“一起偏”当成独立的运气，区间算得太窄，更容易误判“更准”。
  按周抽：把同一周（周一到周日，按比赛日期）的比赛当成一块，整块一起抽。周内关联大时区间会宽一些，更诚实。
  也可以按比赛日（block: day）或每场单独抽（block: match）对比看看差多少。
  实测（英超 2005/06–2023/24，elo 对 frequency）：按周、按天、按单场的区间宽度 0.0177 / 0.0177 / 0.0181，几乎一样。
  因为比的是同一场上两个模型的分数差，两个模型共同的坏运气（冷门多的周末）相减后大部分抵消了。按周抽是保险，不吃亏。

四、看哪个分数
  第 5 节的指标有三个（log loss、Brier、RPS），三个都算、都给结论；主要看 compare.primary_metric（默认 log loss：
  第 5 节“三个检验”的“整体谁更准”也用 log loss）。三个结论不一致时，说明差距很小、取决于怎么打分，按“没差别”对待更稳妥。

用法：
  .venv\\Scripts\\python -m src.evaluate.compare data/backtest/<运行名> --base frequency --new elo
  不写 --base/--new 时按 default_pairs 的几组比（每个模型 vs frequency、Elo 变体 vs elo、T-1 vs T-24、平博收盘 vs 模型）。
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from src.common.config import load_settings, resolve_path

METRICS = ("log_loss", "brier", "rps")
BETTER, SAME, WORSE = "更准", "没差别", "更差"


def block_ids(dates: pd.Series, block: str) -> np.ndarray:
    """每场比赛属于哪一块。week：比赛日期所在的那一周（周一开始）；day：比赛日期；match：每场自己一块。"""
    d = pd.to_datetime(dates)
    if block == "week":
        return (d - pd.to_timedelta(d.dt.day_of_week, unit="D")).dt.strftime("%Y-%m-%d").to_numpy()
    if block == "day":
        return d.dt.strftime("%Y-%m-%d").to_numpy()
    if block == "match":
        return np.arange(len(d)).astype(str)
    raise ValueError(f"block 只能是 week / day / match，收到 {block!r}")


def bootstrap_mean_diff(diff, blocks, n_boot: int, confidence: float, seed: int) -> tuple[float, float, float]:
    """成块 bootstrap：返回 (平均差, 区间下限, 区间上限)。

    每次有放回地抽和原来一样多的“块”，这批块里所有比赛的分数差加起来 ÷ 场数 = 这一次的平均差。
    先把每块的“差之和”和“场数”算好，抽样时只抽块号，2000 次也很快。
    """
    diff = np.asarray(diff, dtype=float)
    if len(diff) == 0:
        raise ValueError("没有可比较的比赛")
    _, inv = np.unique(np.asarray(blocks), return_inverse=True)
    sums = np.bincount(inv, weights=diff)
    counts = np.bincount(inv).astype(float)
    k = len(sums)
    rng = np.random.default_rng(seed)
    means = np.empty(n_boot)
    step = max(1, 2_000_000 // k)          # 一次抽多少轮：控制内存（每轮 k 个块号）
    for i in range(0, n_boot, step):
        idx = rng.integers(0, k, size=(min(step, n_boot - i), k))
        means[i:i + len(idx)] = sums[idx].sum(axis=1) / counts[idx].sum(axis=1)
    alpha = (1 - confidence) / 2
    lo, hi = np.quantile(means, [alpha, 1 - alpha])
    return float(diff.mean()), float(lo), float(hi)


def verdict(lo: float, hi: float) -> str:
    """分数越小越好，差 = 新 − 旧：区间整个小于 0 → 更准；整个大于 0 → 更差；跨过 0 → 没差别。"""
    if hi < 0:
        return BETTER
    if lo > 0:
        return WORSE
    return SAME


def paired(pred: pd.DataFrame, base: str, new: str, base_horizon: str, new_horizon: str) -> pd.DataFrame:
    """两边都有分数的比赛，每场一行：match_id、match_date、base_<分数>、new_<分数>。"""
    cols = ["match_id", "match_date", *METRICS]
    a = pred[(pred["model"] == base) & (pred["horizon"] == base_horizon)][cols].dropna(subset=list(METRICS))
    b = pred[(pred["model"] == new) & (pred["horizon"] == new_horizon)][cols].dropna(subset=list(METRICS))
    if a.empty or b.empty:
        raise ValueError(f"找不到 {base}@{base_horizon} 或 {new}@{new_horizon} 的分数")
    return a.merge(b.drop(columns="match_date"), on="match_id", suffixes=("_base", "_new"))


def compare(pred: pd.DataFrame, base: str, new: str, base_horizon: str = "T-24", new_horizon: str | None = None,
            settings: dict | None = None, block: str | None = None) -> pd.DataFrame:
    """new 比 base 怎么样。每个分数一行：场数、块数、两边平均分、平均差（新 − 旧）、可信区间、结论。"""
    c = (settings or load_settings())["compare"]
    block = block or c["block"]
    new_horizon = new_horizon or base_horizon
    p = paired(pred, base, new, base_horizon, new_horizon)
    blocks = block_ids(p["match_date"], block)
    rows = []
    for m in METRICS:
        d = p[f"{m}_new"] - p[f"{m}_base"]
        mean, lo, hi = bootstrap_mean_diff(d, blocks, c["n_boot"], c["confidence"], c["seed"])
        rows.append({"base": f"{base}@{base_horizon}", "new": f"{new}@{new_horizon}", "metric": m,
                     "n": len(p), "blocks": len(np.unique(blocks)), "block": block,
                     "base_mean": p[f"{m}_base"].mean(), "new_mean": p[f"{m}_new"].mean(),
                     "diff": mean, "ci_low": lo, "ci_high": hi, "verdict": verdict(lo, hi),
                     "primary": m == c["primary_metric"]})
    return pd.DataFrame(rows)


def load_predictions(run_dir) -> pd.DataFrame:
    path = Path(run_dir)
    path = path if path.is_absolute() else resolve_path(str(path))
    return pd.read_csv(path / "predictions.csv", encoding="utf-8-sig", low_memory=False)  # 一次读完，避免“同一列类型混杂”的警告


def default_pairs(pred: pd.DataFrame) -> list[tuple[str, str, str, str]]:
    """默认比较（都是 T-24，T-6 / T-1 用 --horizon 指定）：
    1. 每个模型 vs frequency（第 4 节“最低标准”）；
    2. 每个 elo_ 开头的变体 vs 默认 elo（事先写好的几组设置，第 5 节）；
    3. 每个模型的 T-1 vs 它自己的 T-24（晚一点预测有没有用）；
    4. 两个标杆 vs 每个模型的 T-24（离庄家多远）：平博收盘；多家平均收盘；多家平均“赛前代替”（2018/19 及以前）
       分开比，不混在一起；
    5. 平博收盘 vs 多家平均收盘（两个都有的比赛，2019/20 起）：平博是不是真的比多家平均准。"""
    benches = {"pinnacle_close", "market_avg"}
    models = [m for m in pred["model"].unique() if m not in benches]
    horizons = set(pred["horizon"].unique())
    pairs = [("frequency", m, "T-24", "T-24") for m in models if m != "frequency"]
    pairs += [("elo", m, "T-24", "T-24") for m in models if m.startswith("elo_") and "elo" in models]
    if {"T-1", "T-24"} <= horizons:
        pairs += [(m, m, "T-24", "T-1") for m in models]
    have = set(pred[["model", "horizon"]].drop_duplicates().itertuples(index=False, name=None))
    for bench, hz in [("pinnacle_close", "收盘"), ("market_avg", "收盘"), ("market_avg", "赛前代替")]:
        if (bench, hz) in have:
            pairs += [(m, bench, "T-24", hz) for m in models]
    if {("pinnacle_close", "收盘"), ("market_avg", "收盘")} <= have:
        pairs.append(("market_avg", "pinnacle_close", "收盘", "收盘"))
    return pairs


def main() -> None:
    ap = argparse.ArgumentParser(description="两个模型谁更准（成块 bootstrap）")
    ap.add_argument("run_dir", help="回测结果文件夹，例如 data/backtest/E0_2005-2023")
    ap.add_argument("--base", help="旧模型（基准）")
    ap.add_argument("--new", help="新模型")
    ap.add_argument("--horizon", default="T-24", help="基准的时点，默认 T-24")
    ap.add_argument("--new-horizon", default=None, help="新模型的时点，默认和基准一样（平博收盘写 收盘）")
    ap.add_argument("--block", default=None, help="week / day / match，默认 settings 里的")
    a = ap.parse_args()
    pred = load_predictions(a.run_dir)
    pairs = [(a.base, a.new, a.horizon, a.new_horizon or a.horizon)] if a.base and a.new else default_pairs(pred)
    out = pd.concat([compare(pred, b, n, bh, nh, block=a.block) for b, n, bh, nh in pairs], ignore_index=True)
    pd.set_option("display.width", 220)
    print(out.to_string(index=False, float_format=lambda v: f"{v:.5f}"))
    path = Path(a.run_dir)
    path = path if path.is_absolute() else resolve_path(str(path))
    out.to_csv(path / "compare.csv", index=False, encoding="utf-8-sig", float_format="%.6f")
    print(f"\n存在 {path / 'compare.csv'}")


if __name__ == "__main__":
    main()
