"""步骤 7 的“另外报告”：伤停组和 gb_l7_r 分联赛、分赛季、按主力伤得多不多分开比（需求文档第 4 节“步骤 7 事先定的比较”）。

一、比什么
  回测（src.evaluate.backtest 或 goals_backtest）跑完以后，读 predictions.csv，
  每一对（旧:新）都用 compare.py 同一套办法：每场分数差（新 − 旧）→ 按周成块 bootstrap → 95% 区间 → 更准 / 没差别 / 更差。
  这里只是把比赛分成几份分别比，规则一点不变。只报告，不改结论（结论看全部比赛，事先定的）。

二、怎么分
  1. 分联赛（5 个）、分赛季（10 个）。
  2. 按“两队主力伤得多不多”分三份。主力伤得多不多 = 两队里“最贵 11 人掉了多少”（inj_top_loss）大的那个：
     - 两队主力都没伤（= 0）；
     - 有主力伤，但不多（大于 0，但不到“有主力伤的比赛”里的第 90 百分位）；
     - 主力伤得多（“有主力伤的比赛”里第 90 百分位以上，最重的一成）。
     第 90 百分位是这一次报告事先定的分法（写在这里，不放 settings：不是可调参数）。

三、伤停数从哪来，为什么不经过取数层
  分组用的是 data/research/injury_features.csv（src/clean/injury_features.py 生成）里每场开球前 24 小时的伤停。
  这只是“事后把已经打完分的比赛分组”，不给模型、不改预测，所以不经过取数层。
  为了确认分组用的就是模型当时看到的那份数，程序检查每场的 as_of_utc（这份伤停是哪个时间点的）不晚于回测的截止时间，
  不满足就停下报错。

用法（先跑回测）：
  .venv\\Scripts\\python -m src.evaluate.injury_study --run s7_eval --pairs gb_l7_r:gb_inj gb_l7_r:gb_val gb_l7_r:gb_inj_val
结果存在 data/backtest/<run>/injury_study.csv
"""
from __future__ import annotations

import argparse

import numpy as np
import pandas as pd

from src.common.config import load_settings, resolve_path
from src.evaluate.compare import compare, load_predictions

HEAVY_QUANTILE = 0.9     # “主力伤得多”= 第 90 百分位以上（事先定的分法，见上面第二条）


def injury_groups(pred: pd.DataFrame, injuries: pd.DataFrame, quantile: float = HEAVY_QUANTILE) -> pd.Series:
    """每场比赛（按 match_id）属于哪一份。返回 index = match_id 的文字；这场没有伤停数的是“没有伤停数”。"""
    inj = injuries.set_index("match_id")
    m = pred.drop_duplicates("match_id").set_index("match_id")
    # 两队里大的那个；一队是空值时用另一队的（np.fmax 跳过空值）
    loss = pd.Series(np.fmax(inj["home_inj_top_loss"], inj["away_inj_top_loss"]), index=inj.index).reindex(m.index)
    # 分组用的伤停必须是模型当时能看到的：as_of_utc 不晚于截止时间（两边都是 "YYYY-MM-DD HH:MM:SS" 文字，直接比）
    as_of = inj["as_of_utc"].reindex(m.index)
    late = as_of.notna() & (as_of.astype(str) > m["cutoff_utc"].astype(str))
    if late.any():
        raise ValueError(f"有 {int(late.sum())} 场的伤停快照晚于截止时间，不能拿来分组（例如 match_id {late[late].index[0]}）")
    cut = loss[loss > 0].quantile(quantile) if (loss > 0).any() else np.inf
    out = pd.Series("没有伤停数", index=m.index)
    out[loss == 0] = "两队主力都没伤"
    out[(loss > 0) & (loss < cut)] = "有主力伤，不多"
    out[loss >= cut] = f"主力伤得多（第 {round(quantile * 100)} 百分位以上）"
    return out


def run(pred: pd.DataFrame, injuries: pd.DataFrame, pairs: list[tuple[str, str]], settings: dict,
        horizon: str = "T-24") -> pd.DataFrame:
    pred = pred[pred["horizon"] == horizon]
    groups = injury_groups(pred, injuries)
    pred = pred.assign(伤停分组=pred["match_id"].map(groups))
    parts = [("全部", "全部", pred)]
    parts += [("联赛", c, s) for c, s in pred.groupby("competition")]
    parts += [("赛季", str(y), s) for y, s in pred.groupby("season")]
    parts += [("伤停分组", g, s) for g, s in pred.groupby("伤停分组")]
    rows = []
    for kind, name, sub in parts:
        for base, new in pairs:
            rows.append(compare(sub, base, new, horizon, settings=settings).assign(分法=kind, 哪一份=name))
    out = pd.concat(rows, ignore_index=True)
    first = ["分法", "哪一份"]
    return out[[*first, *[c for c in out.columns if c not in first]]]


def main() -> None:
    settings = load_settings()
    ap = argparse.ArgumentParser(description="步骤 7：伤停组分联赛、分赛季、按主力伤得多不多分开比")
    ap.add_argument("--run", required=True, help="回测结果文件夹名（data/backtest/ 下面）")
    ap.add_argument("--pairs", nargs="+", required=True, help="旧:新，例如 gb_l7_r:gb_inj")
    ap.add_argument("--horizon", default="T-24")
    a = ap.parse_args()
    run_dir = resolve_path(settings["backtest"]["output_dir"]) / a.run
    injuries = pd.read_csv(resolve_path(settings["paths"]["injury_features"]), encoding="utf-8-sig")
    pairs = [tuple(p.split(":")) for p in a.pairs]
    out = run(load_predictions(run_dir), injuries, pairs, settings, a.horizon)
    out.to_csv(run_dir / "injury_study.csv", index=False, encoding="utf-8-sig", float_format="%.6f")
    pd.set_option("display.width", 250)
    pd.set_option("display.max_columns", 20)
    show = out[out["primary"]].drop(columns=["primary", "block", "blocks"])
    print(show.to_string(index=False, float_format=lambda v: f"{v:.4f}"))
    print(f"结果存在 {run_dir / 'injury_study.csv'}")


if __name__ == "__main__":
    main()
