"""每小时更新懂球帝数据：原始采集成功后，再把已有原始文件导入数据库。

运行：.venv\\Scripts\\python -m src.jobs.refresh_data
复用现有采集器的低频限制、共享请求间隔、内容检查和导入程序，不另写一套取数逻辑。
五大联赛那一套做完以后，settings.yaml 有 dongqiudi_jc 段时，再用同一个采集器抓、导竞彩联赛（leagues.jc，
src/clean/dongqiudi_jc_import.py）。顺序固定在后面：竞彩联赛出问题时，五大联赛这一小时的数据已经导进去了。
每次写 refresh_data 心跳；采集或导入失败照常抛出，任务计划程序能看到失败。
"""
from __future__ import annotations

import logging

from src.clean.dongqiudi_import import run as import_raw
from src.clean.dongqiudi_jc_import import run as import_jc
from src.collect.dongqiudi import Collector, jc_config, setup_logging
from src.common.config import load_settings, resolve_path
from src.common.db import connect
from src.jobs.data_lock import data_write_lock, source_refresh_lock
from src.jobs.heartbeat import job_run, state_dir

log = logging.getLogger("dongqiudi")


class RefreshCollector(Collector):
    """只收紧自动更新的失败判定，不改历史回补时“缺一场继续抓”的原有做法。"""

    def fetch(self, url, data_type, target, check, once, retry_bad=False, judge=None):
        result = super().fetch(url, data_type, target, check, once, retry_bad=retry_bad, judge=judge)
        status, _ = result
        # 首发页只给网页显示用、不进模型：一页没抓到不能耽误赔率导入和预测。
        # 失败、可疑照样记在抓取记录里（数据状态页能看到），这里只写一条提醒，不算整次失败。
        # 网站拒绝访问（403/429）在采集器里已经抛 Blocked，照常停下。
        if data_type in ("lineup_live", "lineup_final"):
            if status in ("failed", "suspicious"):
                log.warning("首发页没抓到（%s 返回 %s），不影响赔率导入：%s", data_type, status, url)
            return result
        # 原采集器对部分网络失败、可疑赔率返回状态而不抛异常。自动更新不能把这类情况
        # 记成成功后继续导入旧文件，否则看起来定时任务正常，预测实际仍在使用旧数据。
        # 单场最终赔率/详情的 404 可能只是来源本来没有，保留计数；赛程 404 则不能算成功。
        if status in ("failed", "suspicious") or (status == "not_found" and data_type == "fixtures"):
            raise RuntimeError(f"懂球帝采集未完成：{data_type} 返回 {status}，本次未开始导入（{target.name}）")
        return result


def run_refresh(settings: dict | None = None) -> dict:
    """完整更新一次。测试传入临时设置，并替换采集器、导入函数；不读取任何密钥。"""
    settings = settings or load_settings()
    with source_refresh_lock(settings, "dongqiudi"), job_run("refresh_data", state_dir(settings)) as run:
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
            with data_write_lock(settings):
                result = import_raw(con, raw, resolve_path(settings["paths"]["reports"]), settings)
            run.summary["写入"] = result["counts"]
            if "dongqiudi_jc" in settings:
                run_jc(con, raw, settings, run.summary)
            run.summary["阶段"] = "完成"
        finally:
            try:
                if collector is not None:
                    collector.session.close()
            finally:
                con.close()
    return dict(run.summary)


def run_jc(con, raw, settings: dict, summary: dict) -> None:
    """竞彩联赛：同一套“采集成功才导入”。采集失败、导入失败照常抛出（整次任务记失败），五大联赛已经导好的不受影响。"""
    summary["阶段"] = "竞彩联赛采集"
    cfg = jc_config(settings)
    collector = RefreshCollector(con, raw, cfg)
    try:
        collector.upcoming(list(cfg["league_ids"]))
    finally:
        summary["竞彩联赛采集请求"] = dict(collector.counts)
        collector.session.close()
    summary["阶段"] = "竞彩联赛导入"
    with data_write_lock(settings):
        result = import_jc(con, raw, resolve_path(settings["paths"]["reports"]), settings)
    summary["竞彩联赛写入"] = result["counts"]


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
