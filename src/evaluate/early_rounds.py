"""每个赛季“前 10 轮”和后面的比赛分开比（需求文档第 4 节“步骤 5 事先定的比较”）。

一、为什么要分开看
  新赛季开始时，球队夏天换了人、换了教练，上赛季末的实力不一定还对。
  动态模型（dyn）每场赛后马上更新，想法是它在赛季初“跟得更快”；赛季中段大家都有足够的比赛，差别可能变小。
  所以把每个赛季的前 10 轮单独拿出来比一次，再把剩下的比赛比一次。

二、“前 10 轮”怎么数
  不用网站上的“第几轮”（改期、补赛会让轮次乱掉），按开球时间排：
  每个联赛每个赛季，球队数 ÷ 2 × 10 场（20 个队的联赛 = 前 100 场，18 个队 = 前 90 场）。
  球队数 = 这个联赛这个赛季预测里出现过的队（主队、客队合起来去重）。
  同一时间开球的按比赛编号排，结果固定。

三、怎么比
  和 compare.py 一样：每场分数差（新 − 旧）→ 按周成块 bootstrap → 95% 区间 → 更准 / 没差别 / 更差。
  这一步只报告，不改结论（结论看全部比赛）。

用法（先跑回测）：
  .venv\\Scripts\\python -m src.evaluate.early_rounds --run step5_eval --pairs dc:dyn_q1e5_b01 dc:dyn_weekly
结果存在 data/backtest/<run>/early_rounds.csv
"""
from __future__ import annotations

import argparse

import pandas as pd

from src.common.config import load_settings, resolve_path
from src.evaluate.compare import compare, load_predictions

ROUNDS = 10        # 前几轮（需求文档写的是 10，写在这里不放 settings：这是这一次比较事先定下的规则，不是可调参数）


def early_flags(pred: pd.DataFrame, rounds: int = ROUNDS) -> pd.Series:
    """每场比赛（按 match_id）是不是这个联赛这个赛季的“前 rounds 轮”。返回 index = match_id 的 True / False。"""
    m = pred.drop_duplicates("match_id")[["match_id", "competition", "season", "match_date", "kickoff_utc",
                                          "home_team", "away_team"]].copy()
    m["order_time"] = pd.to_datetime(m["kickoff_utc"]).fillna(pd.to_datetime(m["match_date"]))  # 早年没有开球时间，用比赛日期
    m = m.sort_values(["competition", "season", "order_time", "match_id"])
    m["nth"] = m.groupby(["competition", "season"]).cumcount()     # 这个联赛这个赛季的第几场（从 0 数）
    teams = m.melt(id_vars=["competition", "season"], value_vars=["home_team", "away_team"])
    n_teams = teams.groupby(["competition", "season"])["value"].nunique().rename("n_teams")
    m = m.join(n_teams, on=["competition", "season"])
    m["early"] = m["nth"] < m["n_teams"] // 2 * rounds
    return m.set_index("match_id")["early"]


def run(pred: pd.DataFrame, pairs: list[tuple[str, str]], settings: dict, horizon: str = "T-24") -> pd.DataFrame:
    flags = early_flags(pred)
    pred = pred.assign(early=pred["match_id"].map(flags))
    rows = []
    for part, sub in [(f"前 {ROUNDS} 轮", pred[pred["early"]]), ("后面的比赛", pred[~pred["early"]]), ("全部", pred)]:
        for base, new in pairs:
            rows.append(compare(sub, base, new, horizon, settings=settings).assign(部分=part))
    out = pd.concat(rows, ignore_index=True)
    return out[["部分", *[c for c in out.columns if c != "部分"]]]


def main() -> None:
    settings = load_settings()
    ap = argparse.ArgumentParser(description="每个赛季前 10 轮和后面的比赛分开比")
    ap.add_argument("--run", required=True, help="回测结果文件夹名（data/backtest/ 下面）")
    ap.add_argument("--pairs", nargs="+", required=True, help="旧:新，例如 dc:dyn_q1e5_b01")
    ap.add_argument("--horizon", default="T-24")
    a = ap.parse_args()
    run_dir = resolve_path(settings["backtest"]["output_dir"]) / a.run
    pairs = [tuple(p.split(":")) for p in a.pairs]
    out = run(load_predictions(run_dir), pairs, settings, a.horizon)
    out.to_csv(run_dir / "early_rounds.csv", index=False, encoding="utf-8-sig", float_format="%.6f")
    pd.set_option("display.width", 250)
    pd.set_option("display.max_columns", 20)
    show = out[out["primary"]].drop(columns=["primary", "block"])
    print(show.to_string(index=False, float_format=lambda v: f"{v:.4f}"))
    print(f"结果存在 {run_dir / 'early_rounds.csv'}")


if __name__ == "__main__":
    main()
