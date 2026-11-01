"""定时预测的定时任务入口（需求文档第 9 节“完整链路的分工和接口”，K1）。

每 15 分钟由 Windows 任务计划程序运行一次：
    .venv\\Scripts\\python -m src.jobs.scheduled_predict
做的事都在 src/predict/scheduled.py 的 run_scheduled 里（哪些比赛到点了、出预测、存进预测存档）。这里只管“外壳”：
  1. 用 src/jobs/heartbeat.py 的 job_run("scheduled_predict") 包住整次运行：开始、结束、成功没有、摘要都写进
     data/jobs/scheduled_predict.json（状态页、提醒读它）。出错时心跳文件记下错误，程序照样以失败结束。
  2. 先拿锁（settings.yaml 的 schedule.lock_file）：上一次还没跑完就跳过这次，免得同一场比赛存两行。
  3. 取数层用只读连接，存档用另一个能写的连接（和预览页一样）。

测试或补跑时可以指定“现在”：
    .venv\\Scripts\\python -m src.jobs.scheduled_predict --now "2026-11-07 14:00"
（--now 只改截止时间；存档里的写入时间仍是电脑的真实时间。）
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
from pathlib import Path

from src.clean.schema import create_tables
from src.common.config import load_settings, resolve_path
from src.common.db import connect
from src.features import FeatureStore, open_readonly
from src.jobs.heartbeat import job_run, state_dir
from src.predict.scheduled import lock_from_settings, parse_utc, run_scheduled

JOB_NAME = "scheduled_predict"     # 心跳文件的任务名（接口 2 定的）


def run_job(settings: dict, now: datetime, db_path: Path | None = None, folder: Path | None = None,
            model=None, code: str | None = None) -> dict:
    """跑一次定时预测，返回写进心跳文件的摘要。

    db_path：数据库（不给就是 settings.yaml 的 paths.database）；folder：心跳文件夹（不给就是 jobs.state_dir）。
    model、code 原样交给 run_scheduled（测试里换成假模型）。
    """
    db_path = Path(db_path) if db_path else resolve_path(settings["paths"]["database"])
    with job_run(JOB_NAME, folder or state_dir(settings)) as run:
        with lock_from_settings(settings) as lock:
            run.summary["清掉残留锁"] = lock.cleared_stale
            if not lock.acquired:
                # 别的运行正在跑：这次什么都不做，算成功（不是出错），摘要里写明
                run.summary["跳过"] = "上次运行还没结束"
                return run.summary
            con = connect(db_path)              # 能写的连接：存预测
            try:
                create_tables(con)              # 第一次运行时建 predictions 表（已有就不动）
                ro = open_readonly(db_path)     # 只读连接：取数层只读不写
                try:
                    run.summary.update(run_scheduled(FeatureStore(ro, "实时", settings), con, settings, now,
                                                     model=model, code=code))
                finally:
                    ro.close()
            finally:
                con.close()
    return run.summary


def main(argv: list[str] | None = None) -> None:
    from src.predict.preview import code_version   # 拿 Git 版本要跑 git 命令，只在真正运行时做

    ap = argparse.ArgumentParser(description="定时出 T-24 / T-6 / T-1 预测")
    ap.add_argument("--now", help="把“现在”当作这个时间（UTC，例如 2026-11-07 14:00），测试、补跑用")
    args = ap.parse_args(argv)
    now = parse_utc(args.now) if args.now else datetime.now(timezone.utc)
    s = run_job(load_settings(), now, code=code_version())
    if s.get("跳过"):
        print(f"定时预测：跳过（{s['跳过']}）")
        return
    print(f"定时预测：窗口内 {s.get('窗口内比赛', 0)} 场，新存 {s.get('新存', 0)} 行"
          f"（T-24 {s.get('T-24', 0)}、T-6 {s.get('T-6', 0)}、T-1 {s.get('T-1', 0)}），"
          f"已出过 {s.get('已出过', 0)}，还没到点 {s.get('还没到点', 0)}，跳过的时点 {len(s.get('跳过的时点', []))} 场")


if __name__ == "__main__":
    main()
