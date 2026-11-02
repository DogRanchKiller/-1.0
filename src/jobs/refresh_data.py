"""每小时更新懂球帝数据：原始采集成功后，再把已有原始文件导入数据库。

运行：.venv\\Scripts\\python -m src.jobs.refresh_data
复用现有采集器的低频限制、共享请求间隔、内容检查和导入程序，不另写一套取数逻辑。
每次写 refresh_data 心跳；采集或导入失败照常抛出，任务计划程序能看到失败。
"""
from __future__ import annotations

import logging

from src.clean.dongqiudi_import import run as import_raw
from src.collect.dongqiudi import Collector, setup_logging
from src.common.config import load_settings, resolve_path
from src.common.db import connect
from src.jobs.heartbeat import job_run, state_dir

log = logging.getLogger("dongqiudi")


class RefreshCollector(Collector):
    """只收紧自动更新的失败判定，不改历史回补时“缺一场继续抓”的原有做法。"""

    def fetch(self, url, data_type, target, check, once, retry_bad=False):
        result = super().fetch(url, data_type, target, check, once, retry_bad=retry_bad)
        status, _ = result
        # 原采集器对部分网络失败、可疑赔率返回状态而不抛异常。自动更新不能把这类情况
        # 记成成功后继续导入旧文件，否则看起来定时任务正常，预测实际仍在使用旧数据。
        # 单场最终赔率/详情的 404 可能只是来源本来没有，保留计数；赛程 404 则不能算成功。
        if status in ("failed", "suspicious") or (status == "not_found" and data_type == "fixtures"):
            raise RuntimeError(f"懂球帝采集未完成：{data_type} 返回 {status}，本次未开始导入（{target.name}）")
        return result


def run_refresh(settings: dict | None = None) -> dict:
    """完整更新一次。测试传入临时设置，并替换采集器、导入函数；不读取任何密钥。"""
    settings = settings or load_settings()
    with job_run("refresh_data", state_dir(settings)) as run:
        run.summary["阶段"] = "采集"
        con = connect(resolve_path(settings["paths"]["database"]))
        collector = None
        try:
            raw = resolve_path(settings["paths"]["raw"])
            collector = RefreshCollector(con, raw, settings["dongqiudi"])
            try:
                collector.upcoming(list(settings["dongqiudi"]["league_ids"]))
            finally:
                # 异常时也留下请求数量，便于区别“没变化”和“根本没抓成功”。
                run.summary["采集请求"] = dict(collector.counts)
            run.summary["阶段"] = "导入"
            result = import_raw(con, raw, resolve_path(settings["paths"]["reports"]), settings)
            run.summary["写入"] = result["counts"]
            run.summary["阶段"] = "完成"
        finally:
            try:
                if collector is not None:
                    collector.session.close()
            finally:
                con.close()
    return dict(run.summary)


def main() -> dict:
    settings = load_settings()
    setup_logging(resolve_path(settings["paths"]["logs"]))
    try:
        summary = run_refresh(settings)
    except Exception:
        log.exception("每小时采集和导入失败；详情同时记在 refresh_data 心跳里")
        raise
    log.info("每小时采集和导入完成：%s", summary)
    return summary


if __name__ == "__main__":
    main()
