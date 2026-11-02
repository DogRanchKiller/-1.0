"""低频刷新当季football-data，验证原始凭据后全量导入；失败不伪装完成。"""
from __future__ import annotations

import json
import logging
import os
from datetime import date, datetime, timezone

from src.clean.football_data_import import run as import_raw
from src.collect.football_data import BASE_URL, SOURCE, Downloader, current_season, season_code
from src.common.config import load_settings, resolve_path
from src.common.db import connect
from src.jobs.data_lock import data_write_lock, source_refresh_lock
from src.jobs.heartbeat import job_run, state_dir


def run_refresh(settings: dict | None = None) -> dict:
    """当前季必需文件都成功才导入；unchanged 也重试前次尚未完成的入库。"""
    settings = settings or load_settings()
    folder = state_dir(settings)
    with source_refresh_lock(settings, "football_data"), job_run("refresh_football_data", folder) as run:
        run.summary["阶段"] = "检查范围"
        registered = [*settings["leagues"]["top"], *settings["leagues"]["second"]]
        configuration = settings.get("football_data", {})
        leagues = configuration["refresh_leagues"] if "refresh_leagues" in configuration else registered
        if (not isinstance(leagues, list) or not leagues
                or any(not isinstance(code, str) or code not in registered for code in leagues)
                or len(set(leagues)) != len(leagues)):
            raise ValueError("football-data必需联赛为空、重复或未登记")
        leagues = list(leagues)
        season = current_season(date.today())
        run.summary.update(阶段="采集", 赛季=season, leagues={})
        con = connect(resolve_path(settings["paths"]["database"]))
        downloader = None
        try:
            raw = resolve_path(settings["paths"]["raw"])
            downloader = Downloader(con, raw, settings["scrape"]["min_interval_seconds"])
            for league in leagues:
                url = f"{BASE_URL}/mmz4281/{season_code(season)}/{league}.csv"
                target = raw / SOURCE / league / season_code(season)
                status = downloader.fetch(url, "league_csv", target, snapshot_dir=target, retry_missing=True)
                run.summary["采集请求"] = dict(downloader.counts)
                run.summary["leagues"][league] = {"season": season, "status": status}
                if status not in ("saved", "unchanged"):
                    raise RuntimeError(f"football-data {league} 必需文件返回{status}，未开始导入")
                checked = con.execute("select fetched_at_utc from fetch_log where source=? and url=? "
                                      "and data_type='league_csv' and status=? order by id desc limit 1",
                                      (SOURCE, url, status)).fetchone()
                if checked is None:
                    raise RuntimeError(f"football-data {league} 缺少成功检查记录")
                run.summary["leagues"][league]["last_checked_utc"] = checked[0]
            run.summary["阶段"] = "等待写锁"
            with data_write_lock(settings):
                run.summary["阶段"] = "验证与导入"
                result = import_raw(settings, con, required_season=season, required_leagues=leagues)
            completed = datetime.now(timezone.utc).isoformat(timespec="seconds")
            for league in leagues:
                run.summary["leagues"][league].update(result["leagues"][league], last_import_success_utc=completed)
            run.summary["写入"] = {key: result[key] for key in ("files", "matches", "odds_rows", "dropped_odds")}
            manifest = {"schema": 1, "source": SOURCE, "leagues": run.summary["leagues"]}
            path = folder / "football_data_sources.json"
            temp = path.with_suffix(".json.tmp")
            temp.write_text(json.dumps(manifest, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")
            os.replace(temp, path)
            run.summary["阶段"] = "完成"
        finally:
            try:
                if downloader is not None:
                    downloader.session.close()
            finally:
                con.close()
    return dict(run.summary)


def main():
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    result = run_refresh()
    logging.getLogger(__name__).info("football-data 当前季更新完成：%s", result)
    return result


if __name__ == "__main__":
    main()
