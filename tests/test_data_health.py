"""数据健康不能从任务成功推导。只用内存库和临时原始文件，不联网。"""
import hashlib
import json
import sqlite3
from datetime import datetime, timedelta, timezone

import pytest

from src.clean.schema import create_tables
from src.common.config import load_settings
from src.common.db import FETCH_LOG_SCHEMA
from src.jobs import status_page as sp

NOW = datetime(2026, 11, 2, 8, tzinfo=timezone.utc)
BOOKS = ["平博 Pinnacle", "多家平均"]


@pytest.fixture
def env(tmp_path):
    con = sqlite3.connect(":memory:")
    con.executescript(FETCH_LOG_SCHEMA)
    create_tables(con)
    settings = load_settings()
    settings["paths"]["raw"] = str(tmp_path / "raw")
    settings["jobs"]["state_dir"] = str(tmp_path / "jobs")
    settings["review"]["output_dir"] = str(tmp_path / "review")
    settings["football_data"]["refresh_leagues"] = ["E0", "D1"]
    settings["dongqiudi"]["league_ids"] = {"E0": 4, "D1": 5}
    settings["cup_refresh"]["competitions"] = []
    settings["status"]["data_health"] = {
        "football_data_stale_hours": 36, "dongqiudi_stale_hours": 8,
        "cup_stale_hours": 36, "result_lookback_days": 7, "result_due_hours": 3,
        "upcoming_days": 8, "closing_snapshot_stale_hours": 30,
    }
    yield con, settings, tmp_path
    con.close()


def health():
    from src.jobs import data_health
    return data_health


def match(con, mid, day, status="finished", league="E0", season=2026, sources=(), kickoff=None,
          precision="day"):
    con.execute("insert into matches(match_id,competition,season,home_team,away_team,match_date,"
                "kickoff_utc,status,date_precision) values(?,?,?,?,?,?,?,?,?)",
                (mid, league, season, f"H{mid}", f"A{mid}", day, kickoff, status, precision))
    for source in sources:
        con.execute("insert into match_stats(match_id,source,home_goals,away_goals) values(?,?,2,1)",
                    (mid, source))
        con.execute("insert into match_sources(match_id,source,source_key,raw_file,fetched_at_utc) "
                    "values(?,?,?,?,?)", (mid, source, f"{source}/{mid}", "source.csv", NOW.isoformat()))


def fetch(con, source, data_type, url, when, status="saved", path=None):
    con.execute("insert into fetch_log(source,data_type,url,fetched_at_utc,status,file_path) "
                "values(?,?,?,?,?,?)", (source, data_type, url, when.isoformat(), status, str(path) if path else None))


def csv_file(tmp, league="E0", pinnacle=False):
    path = tmp / "raw" / "football-data" / league / "2627" / "2026-11-01.csv"
    path.parent.mkdir(parents=True, exist_ok=True)
    head = "Div,Date,HomeTeam,AwayTeam,FTHG,FTAG,AvgCH,AvgCD,AvgCA,AvgC>2.5,AvgC<2.5"
    if pinnacle:
        head += ",PSCH,PSCD,PSCA,PC>2.5,PC<2.5"
    path.write_text(head + "\n", encoding="utf-8")
    return path


def manifest(tmp, rows, name="football_data_sources", group="leagues"):
    path = tmp / "jobs" / f"{name}.json"
    path.parent.mkdir(exist_ok=True)
    path.write_text(json.dumps({"schema": 1, group: rows}), encoding="utf-8")


def source_row(checks, source="football-data", league="E0"):
    return next(c for c in checks if c.key == f"data_{source}_{league}_2026")


def test_database_completed_date_cannot_advance_with_future_fixtures(env):
    con, _, _ = env
    match(con, 1, "2026-11-01", sources=["dongqiudi"])
    match(con, 2, "2027-02-10", "scheduled")
    got = sp.check_database(con, NOW)
    assert "最近已完成赛果 2026-11-01" in got.text
    assert "未赛赛程最远 2027-02-10" in got.text


def test_empty_or_old_schema_is_not_green(env):
    con, _, _ = env
    assert sp.check_database(con, NOW).level != "ok"
    con.execute("drop table match_stats")
    assert sp.check_database(con, NOW).level != "ok"


def test_future_finished_record_does_not_advance_completed_date(env):
    con, _, _ = env
    match(con, 1, "2026-11-01", sources=["dongqiudi"])
    match(con, 2, "2027-02-10", sources=["dongqiudi"])
    got = sp.check_database(con, NOW)
    assert "最近已完成赛果 2026-11-01" in got.text and "未来日期却标已赛 1 场" in got.text
    assert got.level == "warn"


def test_current_season_uses_same_beijing_date_as_refresh_jobs(env):
    con, settings, _ = env
    july_first_beijing = datetime(2026, 6, 30, 16, 30, tzinfo=timezone.utc)
    fetch(con, "football-data", "league_csv", "https://football-data.co.uk/mmz4281/2627/E0.csv", july_first_beijing)
    rows = health().check_sources(con, settings, july_first_beijing)
    assert any(c.key == "data_football-data_E0_2026" and "最近成功检查 07-01 00:30" in c.text for c in rows)


def test_notes_old_season_and_unchanged_do_not_forge_content_changes(env):
    con, settings, tmp = env
    path = csv_file(tmp)
    changed = NOW - timedelta(days=5)
    fetch(con, "football-data", "league_csv", "https://football-data.co.uk/mmz4281/2627/E0.csv", changed, path=path)
    fetch(con, "football-data", "league_csv", "https://football-data.co.uk/mmz4281/2627/E0.csv", NOW, "unchanged")
    fetch(con, "football-data", "notes", "https://football-data.co.uk/notes.txt", NOW)
    fetch(con, "football-data", "league_csv", "https://football-data.co.uk/mmz4281/2526/D1.csv", NOW)
    rows = health().check_sources(con, settings, NOW)
    e0, d1 = source_row(rows), source_row(rows, league="D1")
    assert "最近成功检查 11-02 16:00" in e0.text
    assert "内容保存 10-28 16:00" in e0.text
    assert "导入时间未知" in e0.text
    assert d1.level != "ok" and "本季尚无成功检查" in d1.text


def test_one_fresh_league_cannot_hide_other_league_staleness(env):
    con, settings, _ = env
    for lg, lid, delta in [("E0", 4, 1), ("D1", 5, 48)]:
        fetch(con, "dongqiudi", "fixtures",
              f"https://api.dongqiudi.com/data/tab/league/new/{lid}?start=2026-10-30%2000%3A00%3A00",
              NOW - timedelta(hours=delta))
    rows = health().check_sources(con, settings, NOW)
    assert "检查已过期" not in source_row(rows, "dongqiudi").text
    assert "检查已过期" in source_row(rows, "dongqiudi", "D1").text


def test_failed_request_does_not_refresh_success_time(env):
    con, settings, tmp = env
    path = csv_file(tmp)
    url = "https://football-data.co.uk/mmz4281/2627/E0.csv"
    fetch(con, "football-data", "league_csv", url, NOW - timedelta(days=4), path=path)
    fetch(con, "football-data", "league_csv", url, NOW, "failed")
    got = source_row(health().check_sources(con, settings, NOW))
    assert got.level == "warn" and "检查已过期" in got.text and "最近请求 failed" in got.text


def test_import_time_is_separate_and_missing_columns_are_not_missing_download(env):
    con, settings, tmp = env
    path = csv_file(tmp)
    fetch(con, "football-data", "league_csv", "https://football-data.co.uk/mmz4281/2627/E0.csv", NOW, path=path)
    manifest(tmp, {"E0": {"season": 2026, "status": "saved", "raw_file": str(path),
                          "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                          "last_checked_utc": NOW.isoformat(), "last_changed_utc": NOW.isoformat(),
                          "last_import_success_utc": (NOW - timedelta(minutes=2)).isoformat()}})
    got = source_row(health().check_sources(con, settings, NOW))
    assert "导入成功 11-02 15:58" in got.text
    assert "平博 Pinnacle 胜平负：文件缺列" in got.text
    assert "多家平均 胜平负：列齐全" in got.text
    assert "字段存在不代表本次复盘已覆盖" in got.text
    assert "尚无联赛文件" not in got.text


def test_previous_season_manifest_cannot_supply_current_import_time(env):
    con, settings, tmp = env
    manifest(tmp, {"E0": {"season": 2025, "last_import_success_utc": NOW.isoformat()}})
    got = source_row(health().check_sources(con, settings, NOW))
    assert "导入时间未知" in got.text


def test_current_success_manifest_resolves_file_after_unchanged_refresh(env):
    con, settings, tmp = env
    path = csv_file(tmp)
    url = "https://football-data.co.uk/mmz4281/2627/E0.csv"
    old = NOW - timedelta(days=2)
    fetch(con, "football-data", "league_csv", url, old, path="old/location/E0.csv")
    fetch(con, "football-data", "league_csv", url, NOW, "unchanged")
    manifest(tmp, {"E0": {"season": 2026, "status": "unchanged", "raw_file": str(path),
                          "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                          "last_checked_utc": NOW.isoformat(), "last_changed_utc": old.isoformat(),
                          "last_import_success_utc": NOW.isoformat()}})
    got = source_row(health().check_sources(con, settings, NOW))
    assert "多家平均 胜平负：列齐全" in got.text


def test_previous_season_nested_success_is_not_a_current_import(env):
    con, settings, tmp = env
    settings["cup_refresh"]["competitions"] = ["UCL"]
    manifest(tmp, {"UCL": {"season": 2026, "status": "partial", "last_checked_utc": NOW.isoformat(),
                          "last_import_success_utc": NOW.isoformat(), "last_success": {"season": 2025}}},
             "cup_sources", "competitions")
    got = source_row(health().check_sources(con, settings, NOW), "wikipedia", "UCL")
    assert "导入时间未知" in got.text


def test_raw_file_outside_source_root_is_not_read(env):
    con, settings, tmp = env
    outside = tmp / "outside.csv"
    outside.write_text("Div,PSCH,PSCD,PSCA\n", encoding="utf-8")
    fetch(con, "football-data", "league_csv", "https://football-data.co.uk/mmz4281/2627/E0.csv", NOW, path=outside)
    got = source_row(health().check_sources(con, settings, NOW))
    assert "文件路径不属于本联赛本季" in got.text and "列齐全" not in got.text


def test_results_distinguish_fallback_from_missing_and_exclude_wrong_season(env):
    con, settings, _ = env
    match(con, 1, "2026-11-01", sources=["dongqiudi"])
    match(con, 2, "2026-11-01", sources=["football-data", "dongqiudi"])
    match(con, 3, "2026-11-01")
    match(con, 4, "2026-11-01", "postponed")
    match(con, 5, "2026-11-01", "scheduled", kickoff="2026-11-01 12:00:00")
    match(con, 6, "2026-11-01", season=2025)
    got = next(c for c in health().check_results(con, settings, NOW) if c.key == "results_E0_2026")
    assert "懂球帝补位 1 场" in got.text and "两源均缺比分 2 场" in got.text
    assert "已知已赛或应结束 4 场" in got.text and got.level == "warn"


def test_round_precision_is_not_a_definite_overdue_result(env):
    con, settings, _ = env
    match(con, 1, "2026-11-01", "scheduled", kickoff="2026-11-01 12:00:00", precision="round")
    got = next(c for c in health().check_results(con, settings, NOW) if c.key == "results_E0_2026")
    assert "两源均缺比分 0 场" in got.text


def test_future_zero_is_unverified_not_collection_failure(env):
    con, settings, _ = env
    match(con, 1, "2026-11-05", "scheduled", kickoff="2026-11-05 12:00:00")
    match(con, 2, "2027-02-10", "scheduled", kickoff="2027-02-10 12:00:00")
    rows = health().check_future(con, settings, NOW)
    e0 = next(c for c in rows if c.key == "future_E0_2026")
    d1 = next(c for c in rows if c.key == "future_D1_2026")
    assert "已知赛程 1 场" in e0.text and "完整性未验证" in e0.text
    assert "已知赛程 0 场" in d1.text and d1.level == "unknown"
    assert "休赛" in d1.text


def coverage(tmp, n=38, q=0, when=NOW):
    payload = {"generated_at": when.isoformat(), "evaluation_cutoff": when.isoformat(),
               "closing_coverage": {"match_count": n, "qualified_match_count": q,
                   "coverage": q / n if n else None, "requested_match_ids": list(range(1, n + 1)),
                   "by_bookmaker_market": [
                       {"bookmaker": b, "market": market, "line": 2.5 if market == "ou" else None,
                        "match_count": n, "qualified_match_count": q, "coverage": q / n if n else None,
                        "absence_reasons": {"no_record": n - q} if n != q else {}}
                       for b in BOOKS for market in ["1x2", "ou"]]}}
    path = tmp / "review" / "live_metrics.json"
    path.parent.mkdir(exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path, payload


def test_closing_uses_saved_cohort_four_groups_and_warns_even_if_job_succeeded(env):
    _, settings, tmp = env
    coverage(tmp)
    rows = health().check_closing(settings, NOW)
    assert len(rows) == 4 and all(c.level == "warn" for c in rows)
    assert all("0/38" in c.text and "无合格收盘证据" in c.text for c in rows)
    assert {c.name for c in rows} == {f"合格收盘：{b} {m}" for b in BOOKS for m in ["胜平负", "大小2.5"]}


def test_empty_closing_cohort_has_no_percentage_or_fault_popup(env):
    _, settings, tmp = env
    coverage(tmp, n=0)
    rows = health().check_closing(settings, NOW)
    assert all(c.level == "unknown" and "无可评估样本" in c.text and "0%" not in c.text for c in rows)
    assert sp.new_alerts(rows, tmp / "alerts.json") == []
    page = sp.render_html(rows, {}, {}, [], [], NOW)
    assert "一切正常" not in page and "无可评估样本" in page


@pytest.mark.parametrize("damage", ["old", "missing_time", "future", "bad_json", "wrong_denominator", "missing_group", "bool_count"])
def test_stale_or_malformed_closing_snapshots_are_unknown(env, damage):
    _, settings, tmp = env
    path, payload = coverage(tmp, when=NOW - timedelta(days=3) if damage == "old" else NOW)
    if damage == "missing_time":
        del payload["evaluation_cutoff"]
    elif damage == "future":
        payload["generated_at"] = (NOW + timedelta(days=1)).isoformat()
    elif damage == "wrong_denominator":
        payload["closing_coverage"]["by_bookmaker_market"][0]["match_count"] = 40
    elif damage == "missing_group":
        payload["closing_coverage"]["by_bookmaker_market"].pop()
    elif damage == "bool_count":
        payload["closing_coverage"]["qualified_match_count"] = True
    path.write_text("{" if damage == "bad_json" else json.dumps(payload), encoding="utf-8")
    rows = health().check_closing(settings, NOW)
    assert all(c.level != "ok" and "覆盖未知" in c.text for c in rows)


def test_cup_failed_attempt_does_not_claim_full_coverage(env):
    con, settings, tmp = env
    settings["cup_refresh"]["competitions"] = ["UCL"]
    manifest(tmp, {"UCL": {"season": 2026, "status": "partial", "last_checked_utc": NOW.isoformat(),
                          "last_changed_utc": None, "last_import_success_utc": None,
                          "complete": False, "import_status": "blocked"}}, "cup_sources", "competitions")
    got = source_row(health().check_sources(con, settings, NOW), "wikipedia", "UCL")
    assert got.level == "warn" and "partial" in got.text and "导入时间未知" in got.text


def test_health_checks_are_readonly_and_tolerate_missing_tables(env):
    con, settings, _ = env
    before = con.total_changes
    con.execute("pragma query_only=on")
    rows = health().build_checks(con, settings, NOW)
    assert rows and con.total_changes == before
    empty = sqlite3.connect(":memory:")
    empty.execute("pragma query_only=on")
    assert any(c.level != "ok" for c in health().build_checks(empty, settings, NOW))
    empty.close()
