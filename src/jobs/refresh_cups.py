"""每日只检查当前赛季已配置杯赛/欧战；页面完整才导入，失败保留旧证据。

独立运行：python -m src.jobs.refresh_cups。只更新采集/比赛资料，不训练模型、不重算
标签或预测、不重跑研究/考试。抓齐链接页面不代表外部赛历完整，状态中明确保留未知。
"""
from __future__ import annotations

import copy
import json
import logging
import os
import uuid
from datetime import date, datetime, timezone
from pathlib import Path

from src.clean.wikipedia_import import import_collected_scope
from src.collect.wikipedia import WikiDownloader, current_season, titles_for
from src.common.config import load_settings, resolve_path
from src.common.db import connect
from src.jobs.data_lock import data_write_lock, source_refresh_lock
from src.jobs.heartbeat import job_run, state_dir

log = logging.getLogger("wikipedia")


def _now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S")


def _utc(value: str) -> str:
    stamp = datetime.fromisoformat(value.replace("Z", "+00:00"))
    if stamp.tzinfo is None:
        stamp = stamp.replace(tzinfo=timezone.utc)
    return stamp.astimezone(timezone.utc).strftime("%Y-%m-%d %H:%M:%S")


def _read_manifest(path: Path) -> dict:
    if not path.exists():
        return {"schema": 1, "source": "wikipedia", "competitions": {}}
    data = json.loads(path.read_text(encoding="utf-8"))
    if data.get("schema") != 1 or not isinstance(data.get("competitions"), dict):
        raise ValueError("杯赛来源清单格式错误，不能覆盖已有来源证据")
    return data


def _write_manifest(path: Path, data: dict) -> None:
    """同一数据库写锁内原子替换状态；原始页面和 fetch_log 仍是独立不可覆盖证据。"""
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.name + "." + uuid.uuid4().hex + ".tmp")
    temp.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
    os.replace(temp, path)


def run_refresh(settings: dict | None = None) -> dict:
    settings = settings or load_settings()
    with source_refresh_lock(settings, "cups"), job_run("refresh_cups", state_dir(settings)) as run:
        run.summary["阶段"] = "检查范围"
        cfg = settings.get("cup_refresh", {})
        codes = cfg.get("competitions")
        comps = settings["wikipedia"]["competitions"]
        today = date.today()
        season = current_season(today)
        if cfg.get("enabled") is not True:
            raise ValueError("杯赛日更未启用")
        if (not isinstance(codes, list) or not codes or len(set(codes)) != len(codes)
                or any(c not in comps or not titles_for(comps[c], season) for c in codes)):
            raise ValueError("杯赛日更赛事范围必须非空、无重复且都是已有的当前赛季赛事")
        run.summary.update({"赛季": season, "赛事": list(codes), "competitions": {}, "status": "running"})
        path = state_dir(settings) / "cup_sources.json"
        manifest = _read_manifest(path)
        con = connect(resolve_path(settings["paths"]["database"]))
        collector = None
        failures = []
        try:
            collector = WikiDownloader(con, resolve_path(settings["paths"]["raw"]), settings)
            for code in codes:
                run.summary["阶段"] = f"采集 {code} {season}"
                previous = manifest["competitions"].get(code, {})
                entry = copy.deepcopy(previous)
                if previous.get("season") != season:
                    # 上一季的成功证据保存在 last_success，不能挂在本季字段下。
                    entry = {"last_success": copy.deepcopy(previous.get("last_success"))}
                # 失败时保留旧成功清单。latest_attempt 是这次实际页面证据，不能和旧清单混用。
                entry.update({"season": season, "status": "failed", "complete": False,
                              "calendar_complete": None, "last_checked_utc": _now(),
                              "import_status": "blocked"})
                for field in ("last_changed_utc", "last_import_success_utc", "raw_file", "last_success"):
                    entry.setdefault(field, None)
                entry.setdefault("raw_files", [])
                attempt = {"competition": code, "season": season, "status": "failed", "complete": False,
                           "pages": [], "errors": []}
                try:
                    attempt = collector.collect_scope(code, comps[code], season, today)
                    entry["complete"] = attempt["complete"]
                    entry["status"] = attempt["status"]
                    if attempt["complete"] and attempt["status"] == "complete":
                        run.summary["阶段"] = f"导入 {code} {season}"
                        entry["import_status"] = "failed"
                        with data_write_lock(settings):
                            result = import_collected_scope(settings, con, attempt, today)
                        entry["import_status"] = "imported"
                        entry["import_result"] = result
                        entry["last_import_transaction_utc"] = _now()
                        entry["preserved_rows"] = result["preserved_rows"]
                        if result["preserved_rows"] or result["skipped_no_score"]:
                            entry["status"] = "partial"
                            attempt["errors"].append("本轮仍有未重新识别或缺比分的旧行，已保留，尚未验证消失原因")
                        else:
                            entry["status"] = "ok"
                            entry["last_import_success_utc"] = entry["last_import_transaction_utc"]
                            good_pages = [p for p in attempt["pages"] if p["status"] in ("saved", "unchanged")]
                            files = sorted({p["raw_file"] for p in good_pages})
                            entry["raw_files"] = files
                            entry["raw_file"] = files[0] if len(files) == 1 else None
                            changed = [_utc(p["fetched_at_utc"]) for p in good_pages if p["status"] == "saved"]
                            if result.get("source_changed_at_utc"):
                                entry["last_changed_utc"] = result["source_changed_at_utc"]
                            elif changed:
                                entry["last_changed_utc"] = max(changed)
                            entry["last_success"] = {"season": season,
                                                     "imported_at_utc": entry["last_import_success_utc"],
                                                     "raw_files": files, "pages": copy.deepcopy(good_pages)}
                except Exception as exc:
                    entry["status"] = "failed"
                    attempt["errors"].append(f"{type(exc).__name__}: {exc}")
                    log.exception("%s %s 刷新失败，保留旧来源与预测", code, season)
                entry["last_checked_utc"] = _now()
                entry["pages"] = {"attempted": len(attempt["pages"]),
                                  "verified": sum(p["status"] in ("saved", "unchanged") for p in attempt["pages"]),
                                  "failed": sum(p["status"] not in ("saved", "unchanged") for p in attempt["pages"])}
                entry["latest_attempt"] = attempt
                manifest["competitions"][code] = entry
                manifest["updated_at_utc"] = _now()
                _write_manifest(path, manifest)
                run.summary["competitions"][code] = {"status": entry["status"], "complete": entry["complete"],
                                                     "pages": entry["pages"], "import_status": entry["import_status"],
                                                     "errors": attempt["errors"]}
                if entry["status"] != "ok":
                    failures.append(code)
            run.summary["采集请求"] = dict(collector.counts)
            if failures:
                states = [v["status"] for v in run.summary["competitions"].values()]
                run.summary["status"] = "partial" if any(s in ("ok", "partial") for s in states) else "failed"
                raise RuntimeError(f"杯赛日更未全部完成：{', '.join(failures)}；旧数据及成功清单已保留")
            run.summary["阶段"] = "完成"
            run.summary["status"] = "ok"
        finally:
            if collector is not None:
                run.summary["采集请求"] = dict(collector.counts)
                try:
                    collector.session.close()
                finally:
                    con.close()
            else:
                con.close()
    return dict(run.summary)


def main() -> dict:
    settings = load_settings()
    folder = resolve_path(settings["paths"]["logs"])
    folder.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s",
                        handlers=[logging.FileHandler(folder / "refresh_cups.log", encoding="utf-8"),
                                  logging.StreamHandler()])
    return run_refresh(settings)


if __name__ == "__main__":
    main()
