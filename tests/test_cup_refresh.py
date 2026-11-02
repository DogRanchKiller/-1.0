"""杯赛日更的保全边界：假网页、临时库，不联网、不读写正式产物。"""
import datetime as dt
import gzip
import hashlib
import json
import sqlite3
from collections import Counter
from contextlib import contextmanager
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import pytest
import requests

from src.clean import wikipedia_import as W
from src.clean.schema import create_tables
from src.collect import wikipedia as C
from src.common.db import FETCH_LOG_SCHEMA
from src.jobs.heartbeat import read_state


DAY = dt.date(2026, 11, 2)
FETCHED = "2026-11-01T08:00:00+00:00"
PAGE = b'<html><div class="mw-parser-output">ok</div></html>'


@pytest.fixture(autouse=True)
def offline(monkeypatch):
    def denied(*args, **kwargs):
        raise AssertionError("测试禁止网络")
    monkeypatch.setattr(requests.sessions.Session, "request", denied)


@pytest.fixture
def env(tmp_path, monkeypatch):
    monkeypatch.setattr(C, "PROJECT_ROOT", tmp_path)
    settings = {"paths": {"raw": str(tmp_path / "data/raw"), "database": str(tmp_path / "db.sqlite")},
                "jobs": {"state_dir": str(tmp_path / "jobs")},
                "cup_refresh": {"enabled": True, "competitions": ["UCL"]},
                "wikipedia": {"user_agent": "Test/1", "min_interval_seconds": 0,
                              "retry_waits": [], "base_url": "https://en.wikipedia.org/wiki/",
                              "skip_subpages": [], "max_subpages": 5, "first_season": 2000,
                              "competitions": {"UCL": {"国家": "欧洲", "titles": [[2000, 2100, "{season} Cup"]]},
                                               "FAC": {"国家": "英格兰", "titles": [[2000, 2100, "{season} FA Cup"]]}}}}
    con = sqlite3.connect(":memory:")
    con.execute(FETCH_LOG_SCHEMA)
    create_tables(con)
    yield settings, con, tmp_path
    con.close()


def collector(env, replies):
    settings, con, _ = env
    urls = []
    def get(url, timeout):
        urls.append(url)
        code, body = replies.pop(0)
        return SimpleNamespace(status_code=code, content=body, headers={})
    session = SimpleNamespace(headers={}, get=get, urls=urls, close=lambda: None)
    return C.WikiDownloader(con, Path(settings["paths"]["raw"]), settings, session=session)


@pytest.mark.parametrize("only,seasons", [([], [2026]), (["UCL"], []), (["UNKNOWN"], [2026])])
def test_empty_or_unknown_scope_cannot_download_history(env, only, seasons):
    d = collector(env, [])
    with pytest.raises(ValueError, match="范围|赛事|赛季"):
        d.run(env[0], DAY, only=only, seasons=seasons)
    assert d.session.urls == []


def test_same_day_changed_page_keeps_both_originals(env):
    d = collector(env, [(200, PAGE), (200, PAGE.replace(b"ok", b"new"))])
    folder = d.raw_dir / "UCL" / "2026"
    d.today = DAY
    d.fetch("2026 Cup", folder, True)
    first = {p: p.read_bytes() for p in folder.rglob("*.html.gz")}
    d.fetch("2026 Cup", folder, True)
    assert all(p.read_bytes() == content for p, content in first.items())
    assert len(list(folder.rglob("*.html.gz"))) == 2


def test_main_success_and_failed_child_are_partial_not_complete(env):
    main = b'<div class="mw-parser-output"><a href="/wiki/2026%E2%80%9327_Cup_group_stage">x</a></div>'
    d = collector(env, [(200, main), (404, b"")])
    report = d.collect_scope("UCL", env[0]["wikipedia"]["competitions"]["UCL"], 2026, DAY)
    assert report["status"] == "partial" and report["complete"] is False
    assert [p["status"] for p in report["pages"]] == ["saved", "not_found"]
    assert len(d.session.urls) == 2


def match_page(score="1–0", day="2026-11-01"):
    return (f'<div class="mw-parser-output"><h2>Group stage</h2>'
            f'<div class="footballbox"><div class="fdate"><span class="bday">{day}</span></div>'
            '<table><tr><th class="fhome"><a title="A">A</a></th>'
            f'<th class="fscore">{score}</th><th class="faway"><a title="B">B</a></th>'
            '</tr></table></div></div>').encode()


def save_page(env, content=PAGE, *, logged_hash=None):
    settings, con, root = env
    path = Path(settings["paths"]["raw"]) / "wikipedia/UCL/2026/2026–27_Cup/2026-11-01.html.gz"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(gzip.compress(content))
    rel = path.relative_to(root).as_posix()
    con.execute("insert into fetch_log(url,source,data_type,fetched_at_utc,status,file_path,sha256,note)"
                " values(?,'wikipedia','wiki_page',?,'saved',?,?,?)",
                (C.page_url(settings["wikipedia"]["base_url"], "2026–27 Cup"), FETCHED, rel,
                 logged_hash or hashlib.sha256(content).hexdigest(), "title=2026–27 Cup"))
    con.commit()
    return path, rel


def test_import_path_uses_same_project_root_as_fetch_log(env):
    settings, _, root = env
    path, rel = save_page(env, match_page())
    rows = W.load_season(Path(settings["paths"]["raw"]), "UCL", {}, 2026, ["2026–27 Cup"], [])
    assert len(rows) == 1 and rows[0].raw_file == rel


def test_provenance_requires_original_content_hash_not_latest_time(env):
    settings, con, _ = env
    _, rel = save_page(env)
    con.execute("insert into fetch_log(url,source,data_type,fetched_at_utc,status,file_path,sha256)"
                " values('x','wikipedia','wiki_page','2026-11-02T08:00:00+00:00','saved',?,'wrong')", (rel,))
    con.commit()
    got = W.verified_fetch_times(con, [rel], Path(settings["paths"]["raw"]))
    assert got == {rel: FETCHED}
    path = Path(settings["paths"]["raw"]) / "wikipedia/UCL/2026/2026–27_Cup/2026-11-01.html.gz"
    path.write_bytes(gzip.compress(b"changed without evidence"))
    assert W.verified_fetch_times(con, [rel], Path(settings["paths"]["raw"])) == {}


@pytest.mark.parametrize("column,value", [("url", "https://en.wikipedia.org/wiki/Other_page"),
                                           ("data_type", "not_a_wiki_page"),
                                           ("note", "title=Other page")])
def test_matching_hash_is_not_enough_without_correct_page_evidence(env, column, value):
    settings, con, _ = env
    _, rel = save_page(env)
    con.execute(f"update fetch_log set {column}=?", (value,))
    con.commit()
    assert W.verified_fetch_times(con, [rel], Path(settings["paths"]["raw"])) == {}


def test_one_time_metadata_repair_preserves_all_match_and_prediction_values(env):
    settings, con, _ = env
    _, rel = save_page(env)
    old = row(raw_file=rel.removeprefix("data/"))
    W.write_database(con, [old], teams(), settings["wikipedia"]["competitions"], {})
    before = records(con)
    result = W.repair_source_provenance(settings, con)
    after = records(con)
    assert result["updated"] == 1 and result["unverified"] == 0
    assert after["match_sources"][0][3:] == (rel, FETCHED)
    assert {k: v for k, v in after.items() if k != "match_sources"} == {k: v for k, v in before.items() if k != "match_sources"}
    assert W.repair_source_provenance(settings, con)["updated"] == 0


def test_one_time_repair_leaves_unverified_old_source_unchanged(env):
    settings, con, _ = env
    _, rel = save_page(env, logged_hash="incorrect")
    W.write_database(con, [row(raw_file=rel.removeprefix("data/"))], teams(),
                     settings["wikipedia"]["competitions"], {})
    before = records(con)
    result = W.repair_source_provenance(settings, con)
    assert result["updated"] == 0 and result["unverified"] == 1
    assert records(con) == before


def test_legacy_path_repair_does_not_keep_an_unproven_earlier_time(env):
    settings, con, _ = env
    _, rel = save_page(env)
    old = rel.removeprefix("data/")
    W.write_database(con, [row(raw_file=old)], teams(), settings["wikipedia"]["competitions"],
                     {old: "1999-01-01 00:00:00"})
    result = W.repair_source_provenance(settings, con)
    assert con.execute("select fetched_at_utc from match_sources").fetchone()[0] == FETCHED
    assert result["timestamps_repaired"] == 1


def test_repair_rejects_other_competition_page_even_with_valid_log(env):
    settings, con, _ = env
    _, rel = save_page(env)
    W.write_database(con, [row(raw_file=rel.removeprefix("data/"), competition="FAC")], teams(),
                     settings["wikipedia"]["competitions"], {})
    before = records(con)
    result = W.repair_source_provenance(settings, con)
    assert result["updated"] == 0 and result["unverified"] == 1
    assert records(con) == before


def teams():
    return W.TeamMap({x: x for x in "ABC"}, {}, {x: {"英文": x, "国家": "英格兰"} for x in "ABC"})


def row(season=2026, home="A", away="B", **kw):
    r = W.Row("UCL", season, "G", 0, home, away, f"{season}-11-01", "day", None,
              "finished", 1, 0, "H", 0, None, None, None, None, "", "page.html.gz", "p", home, away, "", "")
    return replace(r, **kw)


def records(con):
    return {t: con.execute(f"select * from {t} order by 1,2").fetchall()
            for t in ("matches", "match_sources", "match_stats", "cup_details", "predictions", "match_source_state")}


def test_scoped_update_preserves_absent_fixture_historical_rows_and_predictions(env):
    settings, con, _ = env
    originals = [row(2025), row(), row(home="B", away="C")]
    W.write_database(con, originals, teams(), settings["wikipedia"]["competitions"], {"page.html.gz": FETCHED})
    con.execute("insert into predictions(match_id,competition,season,home_team,away_team,match_date,kind,model,"
                "model_version,cutoff_utc,created_at_utc,detail) select match_id,competition,season,home_team,away_team,"
                "match_date,'T-24','test','v1','2026-10-31','2026-10-31','original' from matches where season=2026")
    con.commit()
    before = records(con)
    result = W.write_scoped_database(con, [replace(originals[1], home_goals=2)], teams(),
                                     settings["wikipedia"]["competitions"], {"page.html.gz": FETCHED}, [("UCL", 2026)])
    after = records(con)
    assert after["predictions"] == before["predictions"]
    assert after["matches"] == before["matches"] and after["match_sources"] == before["match_sources"]
    assert result["preserved_rows"] == 1 and after["match_source_state"] == []
    changed = [r for r in after["match_stats"] if r not in before["match_stats"]]
    assert len(changed) == 1 and changed[0][2] == 2


def test_scoped_unverified_or_outside_rows_fail_before_any_write(env):
    settings, con, _ = env
    before = records(con)
    for rows, fetched, scope in [([row()], {}, [("UCL", 2026)]),
                                  ([row(2025)], {"page.html.gz": FETCHED}, [("UCL", 2026)]),
                                  ([row()], {"page.html.gz": FETCHED}, [])]:
        with pytest.raises(ValueError):
            W.write_scoped_database(con, rows, teams(), settings["wikipedia"]["competitions"], fetched, scope)
        assert records(con) == before


def test_finished_fixture_does_not_regress_when_page_loses_score(env):
    settings, con, _ = env
    original = row()
    W.write_database(con, [original], teams(), settings["wikipedia"]["competitions"], {"page.html.gz": FETCHED})
    before = records(con)
    result = W.write_scoped_database(con, [replace(original, status="scheduled", home_goals=None, away_goals=None)],
                                     teams(), settings["wikipedia"]["competitions"], {"page.html.gz": FETCHED}, [("UCL", 2026)])
    assert records(con) == before and result["preserved_rows"] == 1


def test_collected_scope_parses_only_observed_pages_and_preserves_missing_score(env, monkeypatch):
    settings, con, _ = env
    monkeypatch.setattr(W, "load_team_map", lambda s: teams())
    d = collector(env, [(200, match_page())])
    collected = d.collect_scope("UCL", settings["wikipedia"]["competitions"]["UCL"], 2026, DAY)
    result = W.import_collected_scope(settings, con, collected, DAY)
    assert result["matches"] == 1
    before = records(con)
    # 即使目录里另有旧页，本轮清单不含它就不能混进去作为本轮成功。
    extra = d.raw_dir / "UCL/2026/stale.html.gz"
    extra.write_bytes(gzip.compress(match_page("9–9", "2026-10-01")))
    d = collector(env, [(200, match_page(""))])
    collected = d.collect_scope("UCL", settings["wikipedia"]["competitions"]["UCL"], 2026, DAY)
    result = W.import_collected_scope(settings, con, collected, DAY)
    assert result["skipped_no_score"] == 1 and result["preserved_rows"] == 1
    assert records(con) == before


def test_empty_wiki_content_does_not_claim_a_successful_import(env, monkeypatch):
    settings, con, _ = env
    monkeypatch.setattr(W, "load_team_map", lambda s: teams())
    d = collector(env, [(200, PAGE)])
    collected = d.collect_scope("UCL", settings["wikipedia"]["competitions"]["UCL"], 2026, DAY)
    before = records(con)
    with pytest.raises(ValueError, match="没有|暂无|未解析"):
        W.import_collected_scope(settings, con, collected, DAY)
    assert records(con) == before


def job_env(env, monkeypatch):
    from src.jobs import refresh_cups as job
    settings, _, root = env
    settings["data_refresh"] = {"lock_timeout_seconds": 0, "lock_poll_seconds": 0.01}
    calls = []
    control = {"status": "complete", "import_error": None, "preserved_rows": 0}
    class FakeCollector:
        def __init__(self, *args, **kw):
            self.counts = {"saved": 1}
            self.session = SimpleNamespace(close=lambda: calls.append("close"))
        def collect_scope(self, code, comp, year, today):
            calls.append(("collect", code, year))
            assert year == C.current_season(today)
            status = control["status"].get(code, "complete") if isinstance(control["status"], dict) else control["status"]
            return {"competition": code, "season": year, "status": status, "complete": status == "complete",
                    "errors": [] if status == "complete" else ["子页404"],
                    "pages": [{"title": "page", "status": "saved", "raw_file": "data/raw/page.html.gz",
                               "fetched_at_utc": "2026-11-02T01:00:00+00:00", "sha256": "abc"}]}
    def importer(settings, con, collected, today):
        calls.append("import")
        if control["import_error"]:
            raise control["import_error"]
        return {"matches": 1, "preserved_rows": control["preserved_rows"], "skipped_no_score": 0,
                "parsed": 1, "problems": 0, "calendar_complete": None}
    monkeypatch.setattr(job, "WikiDownloader", FakeCollector)
    monkeypatch.setattr(job, "import_collected_scope", importer)
    return job, settings, root, calls, control


def test_job_uses_current_season_and_records_complete_success(env, monkeypatch):
    job, settings, root, calls, _ = job_env(env, monkeypatch)
    result = job.run_refresh(settings)
    assert calls[0] == ("collect", "UCL", C.current_season(dt.date.today()))
    assert calls[-2:] == ["import", "close"]
    state = read_state("refresh_cups", root / "jobs")
    source = json.loads((root / "jobs/cup_sources.json").read_text(encoding="utf-8"))["competitions"]["UCL"]
    assert state["成功"] is True and state["摘要"] == result
    assert source["status"] == "ok" and source["complete"] is True and source["calendar_complete"] is None
    assert source["last_import_success_utc"] and source["last_changed_utc"] == "2026-11-02 01:00:00"


@pytest.mark.parametrize("failure", ["partial", "failed", "import"])
def test_failed_job_preserves_last_success_manifest_and_blocks_partial_import(env, monkeypatch, failure):
    job, settings, root, calls, control = job_env(env, monkeypatch)
    job.run_refresh(settings)
    path = root / "jobs/cup_sources.json"
    previous = json.loads(path.read_text(encoding="utf-8"))["competitions"]["UCL"]
    calls.clear()
    if failure == "import":
        control["import_error"] = ValueError("哈希不一致")
    else:
        control["status"] = failure
    with pytest.raises(RuntimeError):
        job.run_refresh(settings)
    if failure != "import":
        assert "import" not in calls
    source = json.loads(path.read_text(encoding="utf-8"))["competitions"]["UCL"]
    assert source["last_success"] == previous["last_success"]
    assert source["last_import_success_utc"] == previous["last_import_success_utc"]
    assert source["raw_files"] == previous["raw_files"]
    assert source["status"] != "ok" and source["latest_attempt"]
    assert read_state("refresh_cups", root / "jobs")["成功"] is False


def test_job_empty_scope_cannot_create_collector(env, monkeypatch):
    job, settings, root, calls, _ = job_env(env, monkeypatch)
    settings["cup_refresh"]["competitions"] = []
    with pytest.raises(ValueError, match="范围|赛事"):
        job.run_refresh(settings)
    assert calls == [] and read_state("refresh_cups", root / "jobs")["成功"] is False


def test_new_season_failure_does_not_present_old_season_as_current_success(env, monkeypatch):
    job, settings, root, _, control = job_env(env, monkeypatch)
    folder = root / "jobs"
    folder.mkdir()
    last = {"season": C.current_season(dt.date.today()) - 1, "raw_files": ["old.html.gz"],
            "imported_at_utc": "2025-11-01 00:00:00"}
    entry = {"season": last["season"], "last_success": last, "raw_files": last["raw_files"],
             "raw_file": "old.html.gz", "last_import_success_utc": last["imported_at_utc"],
             "last_changed_utc": last["imported_at_utc"]}
    (folder / "cup_sources.json").write_text(json.dumps({"schema": 1, "competitions": {"UCL": entry}}), encoding="utf-8")
    control["status"] = "failed"
    with pytest.raises(RuntimeError):
        job.run_refresh(settings)
    current = json.loads((folder / "cup_sources.json").read_text(encoding="utf-8"))["competitions"]["UCL"]
    assert current["last_import_success_utc"] is None and current["raw_files"] == []
    assert current["last_changed_utc"] is None and current["raw_file"] is None
    assert current["last_success"] == last


def test_partial_collection_is_reported_partial_even_when_only_one_competition(env, monkeypatch):
    job, settings, root, _, control = job_env(env, monkeypatch)
    control["status"] = "partial"
    with pytest.raises(RuntimeError):
        job.run_refresh(settings)
    state = read_state("refresh_cups", root / "jobs")
    assert state["摘要"]["status"] == "partial"


def test_scoped_transaction_rolls_back_prior_rows_if_later_write_fails(env):
    settings, con, _ = env
    before = records(con)
    con.execute("create trigger fail_second before insert on matches when NEW.home_team='B'"
                " begin select raise(abort,'synthetic rollback'); end")
    with pytest.raises(sqlite3.IntegrityError, match="synthetic rollback"):
        W.write_scoped_database(con, [row(), row(home="B", away="C")], teams(),
                                settings["wikipedia"]["competitions"], {"page.html.gz": FETCHED}, [("UCL", 2026)])
    assert records(con) == before and con.execute("select count(*) from teams").fetchone()[0] == 0


def test_page_cap_and_network_exception_are_explicit_incomplete_attempts(env):
    main = b'<div class="mw-parser-output"><a href="/wiki/2026%E2%80%9327_Cup_group_stage">x</a></div>'
    env[0]["wikipedia"]["max_subpages"] = 0
    d = collector(env, [(200, main)])
    result = d.collect_scope("UCL", env[0]["wikipedia"]["competitions"]["UCL"], 2026, DAY)
    assert result["status"] == "partial" and len(d.session.urls) == 1 and result["errors"]


def test_metadata_repair_rollback_on_any_row_failure(env):
    settings, con, _ = env
    _, rel = save_page(env)
    old = rel.removeprefix("data/")
    W.write_database(con, [row(raw_file=old), row(home="B", away="C", raw_file=old)], teams(),
                     settings["wikipedia"]["competitions"], {})
    before = records(con)
    second = before["match_sources"][1][0]
    con.execute("create trigger fail_repair before update on match_sources when NEW.match_id=" + str(second)
                + " begin select raise(abort,'repair rollback'); end")
    with pytest.raises(sqlite3.IntegrityError, match="repair rollback"):
        W.repair_source_provenance(settings, con)
    assert records(con) == before


def test_job_does_not_hold_database_lock_while_waiting_for_source(env, monkeypatch):
    job, settings, _, calls, _ = job_env(env, monkeypatch)
    held = {"write": False}
    @contextmanager
    def short_lock(settings):
        held["write"] = True
        try:
            yield
        finally:
            held["write"] = False
    real_collect = job.WikiDownloader.collect_scope
    real_import = job.import_collected_scope
    def collect(self, *args):
        assert not held["write"], "等待网络不能占用其他来源的写库锁"
        return real_collect(self, *args)
    def import_scope(*args):
        assert held["write"], "每项赛事导入事务应持有共享写锁"
        return real_import(*args)
    monkeypatch.setattr(job, "data_write_lock", short_lock)
    monkeypatch.setattr(job.WikiDownloader, "collect_scope", collect)
    monkeypatch.setattr(job, "import_collected_scope", import_scope)
    job.run_refresh(settings)
    assert "import" in calls and held["write"] is False


def test_one_failed_competition_does_not_hide_another_success(env, monkeypatch):
    job, settings, root, calls, control = job_env(env, monkeypatch)
    settings["cup_refresh"]["competitions"] = ["UCL", "FAC"]
    control["status"] = {"UCL": "failed", "FAC": "complete"}
    with pytest.raises(RuntimeError):
        job.run_refresh(settings)
    sources = json.loads((root / "jobs/cup_sources.json").read_text(encoding="utf-8"))["competitions"]
    assert sources["UCL"]["status"] == "failed" and sources["FAC"]["status"] == "ok"
    assert calls.count("import") == 1 and read_state("refresh_cups", root / "jobs")["摘要"]["status"] == "partial"
