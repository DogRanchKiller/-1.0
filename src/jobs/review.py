"""定时任务“赛后复盘”（需求文档第 15 节）：每次把全部已经踢完、有预测的比赛重新对账一遍。

运行：.venv\\Scripts\\python -m src.jobs.review
定时：src/jobs/register_tasks.ps1 登记的“足球预测_赛后复盘”，每天两次。

输出（都在 settings.yaml 的 review.output_dir，data/ 不进 Git）：
- scores.csv：每场每个版本的分数（接口 3，列名和顺序固定），见 src/review/reconcile.py；
- matches/<match_id>.json：单场复盘，见 src/review/match_review.py；
- reports/周报_YYYY-Www.md、月报_YYYY-MM.md：见 src/review/reports.py。

每次运行开始、结束各写一次心跳文件（src/jobs/heartbeat.py，任务名 review），状态页能看到。
数据库只读打开；比分、收盘赔率经过取数层（实时模式，用途「复盘」）。结果只用来记录和展示，不改任何参数。
"""
from __future__ import annotations

import os
from datetime import datetime, timezone
from pathlib import Path

from src.common.config import load_settings, resolve_path
from src.features import FeatureStore
from src.features.known_time import to_utc_text
from src.features.store import open_readonly
from src.jobs.heartbeat import job_run, state_dir
from src.review.match_review import attach_labels, build_review, team_names_zh, write_match_files
from src.review.reconcile import SOURCE_CLOSE, SOURCE_FUSION, SOURCE_MODEL, collect, score_rows
from src.review.reports import write_reports


def write_scores(path: Path, df) -> None:
    """scores.csv：Excel 能直接打开的 UTF-8（带 BOM），分数保留 6 位小数；先写临时文件再改名。"""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".csv.tmp")
    df.to_csv(tmp, index=False, encoding="utf-8-sig", float_format="%.6f")
    os.replace(tmp, path)


def run_review(settings: dict, con, now: datetime | None = None, out_dir: Path | None = None) -> dict:
    """做一次完整复盘，返回摘要（写进心跳文件、打印给人看）。con 是数据库连接（只读即可）。"""
    now = now or datetime.now(timezone.utc)
    out_dir = Path(out_dir) if out_dir else resolve_path(settings["review"]["output_dir"])
    fs = FeatureStore(con, "实时", settings)
    col = collect(con, fs, settings, now)
    scores = score_rows(col.matches, settings)
    labels = attach_labels(fs, col.matches, settings)
    names = team_names_zh(con, [t for m in col.matches for t in (m["home_team"], m["away_team"])])
    generated = to_utc_text(now)
    reviews = [build_review(m, labels.get(m["match_id"]), names, settings, generated) for m in col.matches]
    write_scores(out_dir / "scores.csv", scores)
    n_files = write_match_files(out_dir, reviews)
    reports = write_reports(out_dir, reviews, scores, settings, now)
    count = scores["source"].value_counts()
    return {
        "对账场数": len(col.matches),
        "模型行": int(count.get(SOURCE_MODEL, 0)), "融合行": int(count.get(SOURCE_FUSION, 0)),
        "收盘赔率行": int(count.get(SOURCE_CLOSE, 0)),
        "单场文件": n_files, "报告": [p.name for p in reports],
        "跳过": col.skipped, "提醒": col.warnings[:20],
        "标签算不出": sum(1 for v in labels.values() if v and "算不出" in v),
        "误差归类": {t: sum(1 for r in reviews if r["误差归类"]["类型"] == t)
                     for t in sorted({r["误差归类"]["类型"] for r in reviews})},
        "输出文件夹": str(out_dir),
    }


def print_summary(s: dict) -> None:
    print(f"赛后复盘：对账 {s['对账场数']} 场（模型行 {s['模型行']}、融合行 {s['融合行']}、收盘赔率行 {s['收盘赔率行']}）")
    print(f"  单场复盘文件 {s['单场文件']} 个，报告 {len(s['报告'])} 份：{'、'.join(s['报告']) or '无'}")
    for why, n in s["跳过"].items():
        print(f"  跳过：{why} —— {n}")
    if s["误差归类"]:
        print("  误差归类：" + "、".join(f"{k} {v} 场" for k, v in s["误差归类"].items()))
    if s["标签算不出"]:
        print(f"  有 {s['标签算不出']} 场标签算不出（原因写在单场文件里）")
    for w in s["提醒"]:
        print(f"  提醒：{w}")
    print(f"  输出在 {s['输出文件夹']}")


def main() -> None:
    settings = load_settings()
    with job_run("review", state_dir(settings)) as run:
        con = open_readonly(resolve_path(settings["paths"]["database"]))
        try:
            summary = run_review(settings, con)
        finally:
            con.close()
        run.summary = {k: v for k, v in summary.items() if k not in ("提醒", "报告")} | {"报告份数": len(summary["报告"])}
    print_summary(summary)


if __name__ == "__main__":
    main()
