"""定时任务“四本账对账”（需求文档第 16 节“四本账怎么算（第 2 步）”）：每天把四本账重新算一遍。

运行：.venv\\Scripts\\python -m src.jobs.ledger_daily
定时：src/jobs/register_tasks.ps1 登记的“足球预测_四本账对账”，每天 09:40（在 09:10 的赛后复盘之后）。

做的事和手动运行 src.evaluate.ledger 一模一样（同一个 build、同一个 write），结果写到 settings.yaml ledger.output_dir
（data/backtest/ledger/，data/ 不进 Git）：ledger_matches.csv、ledger_summary.csv。
另外每次运行开始、结束各写一次心跳文件（src/jobs/heartbeat.py，任务名 ledger），数据状态页“定时任务”那一行能看到：
最近一次跑没跑成、对上账的预测几条、下结论的时点（ledger.main_kind，T-24）每个通道对账了几场、差的平均是多少。
差 = 模型 − 市场（log loss 一类的分数，小于 0 = 模型更准）；场数不够 ledger.min_matches 时只列数、不下结论。
只读：数据库只读打开，不改存档、模型、设置。出错不推送（settings.yaml notify.push_jobs 里没有它），只在状态页显示。
"""
from __future__ import annotations

import math

import pandas as pd

from src.common.config import load_settings, resolve_path
from src.evaluate.ledger import build, write
from src.features import FeatureStore, open_readonly
from src.jobs.heartbeat import job_run, state_dir


def heartbeat_summary(rows: pd.DataFrame, summary: pd.DataFrame, settings: dict) -> dict:
    """心跳文件的“摘要”：状态页只显示数字和 40 个字以内的短文字、最多 6 项，所以写
    “对上账的预测 n 条” + 下结论的时点每个通道一项（例“胜平负 12 场，差 +0.031”）。"""
    out = {"对上账的预测": int(rows["prediction_id"].nunique()) if len(rows) else 0}
    main = settings["ledger"]["main_kind"]
    for _, r in summary[summary["时点"] == main].iterrows():
        n, d = int(r["对账的场数"]), r.get("差的平均")
        if n and isinstance(d, (int, float)) and not math.isnan(d):
            out[f"{main} {r['通道']}"] = f"{n} 场，差 {d:+.3f}"
        else:
            out[f"{main} {r['通道']}"] = f"{n} 场"
    return out


def main() -> None:
    settings = load_settings()
    with job_run("ledger", state_dir(settings)) as run:
        con = open_readonly(resolve_path(settings["paths"]["database"]))
        try:
            fs = FeatureStore(con, "实时", settings)     # 和手动运行一样：存档的比赛是上线后的赛季，用实时模式
            rows, summary = build(fs, con, settings)
        finally:
            con.close()
        out = write(rows, summary, settings)
        run.summary = heartbeat_summary(rows, summary, settings)
    print(f"写到 {out}")
    for k, v in run.summary.items():
        print(f"  {k}：{v}")


if __name__ == "__main__":
    main()
