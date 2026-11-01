"""步骤 8 的“另外报告”：B 版几组分联赛、分赛季、按和赛前赔率分歧大小分开比，再看每组离平博收盘多远
（需求文档第 4 节“步骤 8 事先定的比较”）。

一、比什么
  回测（goals_backtest）跑完以后读 predictions.csv。每一对（旧:新）都用 compare.py 同一套办法：
  每场分数差（新 − 旧）→ 按周成块 bootstrap → 95% 区间 → 更准 / 没差别 / 更差。这里只是把比赛分成几份分别比，
  规则一点不变。只报告，不改结论（结论看全部比赛的主比较，事先定的）。

二、怎么分
  1. 分联赛（5 个）、分赛季（10 个）。
  2. 按分歧大小分三份：分歧 = B 版（--gap 的第一个模型，默认 gb_b）和赛前赔率（第二个，默认 market_pre）
     主胜、平、客胜三个概率里差得最多的那个（绝对值）；分界用 settings.yaml 的 market_checks.gap_bins
     （3% 以内 / 3%–6% / 6% 以上，和第 5 节检验二一样）。分组只用两边赛前就有的预测，不看比分。
  3. 离平博收盘多远：--models 里每个模型 T-24 和平博收盘比（全部比赛），差越小越接近庄家最后的价格。

用法（先跑回测）：
  .venv\\Scripts\\python -m src.evaluate.b_study --run s8_eval \\
      --pairs market_pre:gb_b gb_l7_b:gb_b market_pre:gb_mkt gb_mkt:gb_b --models gb_l7_b gb_b gb_mkt market_pre
结果存在 data/backtest/<run>/b_study.csv
"""
from __future__ import annotations

import argparse

import numpy as np
import pandas as pd

from src.common.config import load_settings, resolve_path
from src.evaluate.compare import compare, load_predictions

P = ["p_home", "p_draw", "p_away"]
PIN, CLOSE = "pinnacle_close", "收盘"


def gap_groups(pred: pd.DataFrame, a: str, b: str, bins, horizon: str = "T-24") -> pd.Series:
    """每场（index = match_id）属于哪一份分歧组。两边有一边没有预测的比赛不在结果里。"""
    p = pred[pred["horizon"] == horizon]
    pa = p[p["model"] == a].drop_duplicates("match_id").set_index("match_id")[P].dropna()
    pb = p[p["model"] == b].drop_duplicates("match_id").set_index("match_id")[P].dropna()
    both = pa.index.intersection(pb.index)
    gap = (pa.loc[both] - pb.loc[both]).abs().max(axis=1)
    lo, hi = float(bins[0]), float(bins[1])
    names = np.where(gap < lo, f"分歧 {lo:.0%} 以内", np.where(gap < hi, f"分歧 {lo:.0%}–{hi:.0%}", f"分歧 {hi:.0%} 以上"))
    return pd.Series(names, index=both)


def run(pred: pd.DataFrame, pairs: list[tuple[str, str]], models: list[str], gap: tuple[str, str], settings: dict,
        horizon: str = "T-24") -> pd.DataFrame:
    groups = gap_groups(pred, *gap, settings["market_checks"]["gap_bins"], horizon)
    t = pred[pred["horizon"] == horizon]
    t = t.assign(分歧组=t["match_id"].map(groups))
    parts = [("全部", "全部", t)]
    parts += [("联赛", c, s) for c, s in t.groupby("competition")]
    parts += [("赛季", str(y), s) for y, s in t.groupby("season")]
    parts += [("分歧组", g, s) for g, s in t.groupby("分歧组")]
    rows = []
    for kind, name, sub in parts:
        for base, new in pairs:
            rows.append(compare(sub, base, new, horizon, settings=settings).assign(分法=kind, 哪一份=name))
    pin = pred[(pred["model"] == PIN) & (pred["horizon"] == CLOSE)]
    for m in models:                      # 离平博收盘多远（全部比赛）
        rows.append(compare(pd.concat([t, pin]), m, PIN, horizon, CLOSE, settings=settings)
                    .assign(分法="离平博收盘", 哪一份="全部"))
    out = pd.concat(rows, ignore_index=True)
    first = ["分法", "哪一份"]
    return out[[*first, *[c for c in out.columns if c not in first]]]


def main() -> None:
    settings = load_settings()
    ap = argparse.ArgumentParser(description="步骤 8：B 版几组分联赛、分赛季、按分歧大小分开比，离平博收盘多远")
    ap.add_argument("--run", required=True, help="回测结果文件夹名（data/backtest/ 下面）")
    ap.add_argument("--pairs", nargs="+", required=True, help="旧:新，例如 market_pre:gb_b")
    ap.add_argument("--models", nargs="*", default=[], help="和平博收盘比的模型")
    ap.add_argument("--gap", default="gb_b:market_pre", help="分歧组按哪两个模型算")
    ap.add_argument("--horizon", default="T-24")
    a = ap.parse_args()
    run_dir = resolve_path(settings["backtest"]["output_dir"]) / a.run
    pairs = [tuple(p.split(":")) for p in a.pairs]
    out = run(load_predictions(run_dir), pairs, a.models, tuple(a.gap.split(":")), settings, a.horizon)
    out.to_csv(run_dir / "b_study.csv", index=False, encoding="utf-8-sig", float_format="%.6f")
    pd.set_option("display.width", 250)
    pd.set_option("display.max_columns", 20)
    show = out[out["primary"]].drop(columns=["primary", "block", "blocks"])
    print(show.to_string(index=False, float_format=lambda v: f"{v:.4f}"))
    print(f"结果存在 {run_dir / 'b_study.csv'}")


if __name__ == "__main__":
    main()
