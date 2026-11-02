"""当前赛季更新的行为验证：假HTTP、临时数据库，不联网、不读正式数据库。"""
import copy
import hashlib
import json
import sqlite3
from contextlib import contextmanager
from datetime import date
from pathlib import Path
from types import SimpleNamespace

import pytest
import requests

from src.common.config import load_settings
from src.common.db import connect
from src.jobs.heartbeat import read_state

HEADER = "Div,Date,Time,HomeTeam,AwayTeam,FTHG,FTAG,FTR,AvgCH,AvgCD,AvgCA,AvgC>2.5,AvgC<2.5\n"
ROW = "E0,01/11/2026,15:00,Arsenal,Chelsea,2,1,H,2.0,3.5,4.0,1.9,2.0\n"
ROW2 = "E0,02/11/2026,15:00,Liverpool,Everton,1,0,H,1.8,3.6,4.2,1.9,2.0\n"


@pytest.fixture(autouse=True)
def no_network(monkeypatch):
    def forbidden(*args, **kwargs):
        raise AssertionError("测试禁止真实网络")
    monkeypatch.setattr(requests.sessions.Session, "request", forbidden)


@pytest.fixture
def env(tmp_path, monkeypatch):
    from src.clean import football_data_import as importer
    from src.collect import football_data as collector

    settings = copy.deepcopy(load_settings())
    settings["paths"].update(database=str(tmp_path / "football.db"), raw=str(tmp_path / "data/raw"))
    settings["jobs"]["state_dir"] = str(tmp_path / "jobs")
    settings["leagues"] = {"top": {"E0": "英超"}, "second": {}}
    settings["football_data"]["refresh_leagues"] = ["E0"]
    settings["scrape"]["min_interval_seconds"] = 0
    monkeypatch.setattr(importer, "PROJECT_ROOT", tmp_path)
    monkeypatch.setattr(collector, "PROJECT_ROOT", tmp_path)
    con = connect(Path(settings["paths"]["database"]))
    yield settings, con, tmp_path
    con.close()


def snapshot(env, filename, rows, when="2026-11-02T00:00:00+00:00", *, logged=True, digest=None, season="2627", header=HEADER):
    settings, con, root = env
    path = root / "data/raw/football-data/E0" / season / filename
    path.parent.mkdir(parents=True, exist_ok=True)
    content = (header + rows).encode()
    path.write_bytes(content)
    if logged:
        con.execute("insert into fetch_log(url,source,data_type,fetched_at_utc,status,file_path,sha256) values(?,?,?,?,?,?,?)",
                    (f"https://football-data.co.uk/mmz4281/{season}/E0.csv", "football-data", "league_csv", when,
                     "saved", path.relative_to(root).as_posix(), digest or hashlib.sha256(content).hexdigest()))
        con.commit()
    return path


def imported(env):
    from src.clean.football_data_import import run
    settings, con, _ = env
    return run(settings, con, required_season=2026, required_leagues=["E0"])


def source_rows(con):
    return con.execute("select m.home_team,s.raw_file,s.fetched_at_utc from match_sources s join matches m using(match_id) "
                       "where s.source='football-data' order by m.home_team").fetchall()


def database_rows(con):
    return {name: con.execute(f"select * from {name} order by 1,2").fetchall()
            for name in ("matches", "match_sources", "match_stats", "odds", "predictions")}


def test_unchanged_match_keeps_verified_first_source_when_snapshot_grows(env):
    _, con, _ = env
    first = snapshot(env, "2026-11-02.csv", ROW)
    imported(env)
    before = source_rows(con)
    snapshot(env, "2026-11-03.csv", ROW + ROW2, "2026-11-03T00:00:00+00:00")
    imported(env)
    after = source_rows(con)
    assert after[0] == before[0]
    assert after[1][2] == "2026-11-03T00:00:00+00:00"
    assert first.exists()


def test_changed_odds_receive_new_source_time_and_fail_earlier_asof(env):
    from src.features import FeatureStore
    from src.review.closing_evidence import collect_closing_evidence

    settings, con, _ = env
    snapshot(env, "2026-11-02.csv", ROW)
    imported(env)
    snapshot(env, "2026-11-03.csv", ROW.replace("2.0,3.5,4.0", "2.1,3.5,4.0"), "2026-11-03T00:00:00+00:00")
    imported(env)
    assert source_rows(con)[0][2] == "2026-11-03T00:00:00+00:00"
    fs = FeatureStore(con, "实时", settings)
    mid = con.execute("select match_id from matches").fetchone()[0]
    before = collect_closing_evidence(fs, [mid], "2026-11-02 12:00", bookmakers=["多家平均"])
    assert not before["qualified"].any() and "fetched_after_cutoff" in before.iloc[0]["reasons"]
    after = collect_closing_evidence(fs, [mid], "2026-11-03 12:00", bookmakers=["多家平均"])
    assert after["qualified"].all()


@pytest.mark.parametrize("problem", ["no_log", "wrong_hash", "wrong_source", "wrong_url"])
def test_unverified_current_snapshot_cannot_replace_database(env, problem):
    _, con, _ = env
    snapshot(env, "2026-11-02.csv", ROW)
    imported(env)
    before = database_rows(con)
    path = snapshot(env, "2026-11-03.csv", ROW + ROW2, "2026-11-03T00:00:00+00:00",
                    logged=problem != "no_log", digest="0" * 64 if problem == "wrong_hash" else None)
    if problem in ("wrong_source", "wrong_url"):
        con.execute(f"update fetch_log set {'source' if problem == 'wrong_source' else 'url'}='wrong' where id=(select max(id) from fetch_log)")
        con.commit()
    with pytest.raises(ValueError, match="凭据|核验|hash|下载"):
        imported(env)
    assert database_rows(con) == before and path.exists()


@pytest.mark.parametrize("bad_rows", [ROW2, ROW + ROW2.replace("Liverpool", "Mystery FC"), ROW.replace(",2,1,H,", ",,,,")])
def test_incomplete_current_source_does_not_replace_existing_data(env, bad_rows):
    _, con, _ = env
    snapshot(env, "2026-11-02.csv", ROW)
    imported(env)
    before = database_rows(con)
    snapshot(env, "2026-11-03.csv", bad_rows, "2026-11-03T00:00:00+00:00")
    with pytest.raises(ValueError):
        imported(env)
    assert database_rows(con) == before


def test_current_refresh_preserves_historical_source_and_old_prediction(env):
    _, con, _ = env
    snapshot(env, "2026-11-02.csv", ROW)
    historical = ROW.replace("01/11/2026", "01/11/2023")
    snapshot(env, "2023-11-02.csv", historical, "2023-11-02T00:00:00+00:00", season="2324")
    imported(env)
    mid = con.execute("select match_id from matches where season=2026").fetchone()[0]
    con.execute("insert into predictions(match_id,competition,season,home_team,away_team,match_date,kind,model,model_version,"
                "cutoff_utc,created_at_utc,p_home,p_draw,p_away,detail) values(?,'E0',2026,'Arsenal','Chelsea','2026-11-01',"
                "'T-24','frozen','v1','2026-10-31 15:00','2026-10-31 15:01',.5,.3,.2,'{\"original\":true}')", (mid,))
    con.commit()
    before = con.execute("select * from predictions").fetchall()
    old = con.execute("select s.* from match_sources s join matches m using(match_id) where m.season=2023").fetchall()
    snapshot(env, "2026-11-03.csv", ROW + ROW2, "2026-11-03T00:00:00+00:00")
    imported(env)
    assert con.execute("select * from predictions").fetchall() == before
    assert con.execute("select s.* from match_sources s join matches m using(match_id) where m.season=2023").fetchall() == old


def test_missing_historical_raw_file_cannot_delete_historical_source(env):
    _, con, _ = env
    snapshot(env, "2026-11-02.csv", ROW)
    old = snapshot(env, "2023-11-02.csv", ROW.replace("01/11/2026", "01/11/2023"),
                   "2023-11-02T00:00:00+00:00", season="2324")
    imported(env)
    before = database_rows(con)
    old.unlink()
    with pytest.raises(ValueError, match="缺失"):
        imported(env)
    assert database_rows(con) == before


def test_corrupted_old_raw_file_cannot_lend_early_timestamp(env):
    _, con, _ = env
    old = snapshot(env, "2026-11-02.csv", ROW)
    imported(env)
    old.write_bytes((HEADER + ROW.replace("2.0,3.5,4.0", "2.1,3.5,4.0")).encode())
    snapshot(env, "2026-11-03.csv", ROW, "2026-11-03T00:00:00+00:00")
    imported(env)
    assert source_rows(con)[0][2] == "2026-11-03T00:00:00+00:00"


@pytest.mark.parametrize("loss", ["columns", "whole_market", "one_price"])
def test_current_snapshot_cannot_remove_previously_known_market_evidence(env, loss):
    _, con, _ = env
    snapshot(env, "2026-11-02.csv", ROW)
    imported(env)
    before = database_rows(con)
    header = HEADER
    if loss == "columns":
        header = ",".join(HEADER.split(",")[:8]) + "\n"
        rows = ",".join(ROW.split(",")[:8]) + "\n"
    else:
        rows = ROW.replace("2.0,3.5,4.0", ",," if loss == "whole_market" else ",3.5,4.0")
    snapshot(env, "2026-11-03.csv", rows, "2026-11-03T00:00:00+00:00", header=header)
    with pytest.raises(ValueError, match="赔率"):
        imported(env)
    assert database_rows(con) == before


@pytest.mark.parametrize("loss", ["columns", "one_value"])
def test_current_snapshot_cannot_remove_previously_known_match_statistics(env, loss):
    _, con, _ = env
    header = HEADER.rstrip() + ",HST,AST\n"
    snapshot(env, "2026-11-02.csv", ROW.rstrip() + ",5,3\n", header=header)
    imported(env)
    before = database_rows(con)
    if loss == "columns":
        header, rows = HEADER, ROW
    else:
        rows = ROW.rstrip() + ",,3\n"
    snapshot(env, "2026-11-03.csv", rows, "2026-11-03T00:00:00+00:00", header=header)
    with pytest.raises(ValueError, match="统计"):
        imported(env)
    assert database_rows(con) == before


@pytest.mark.parametrize("bad_rows", [ROW.rstrip().rsplit(",", 1)[0] + "\n", ROW.replace("2.0,3.5,4.0", "inf,3.5,4.0")])
def test_truncated_row_or_infinite_price_cannot_replace_current_source(env, bad_rows):
    _, con, _ = env
    snapshot(env, "2026-11-02.csv", ROW)
    imported(env)
    before = database_rows(con)
    snapshot(env, "2026-11-03.csv", bad_rows, "2026-11-03T00:00:00+00:00")
    with pytest.raises(ValueError):
        imported(env)
    assert database_rows(con) == before


def test_import_transaction_failure_keeps_all_old_rows(env):
    _, con, _ = env
    snapshot(env, "2026-11-02.csv", ROW)
    imported(env)
    before = database_rows(con)
    con.execute("create trigger fail_new_odds before insert on odds begin select raise(abort, 'atomic failure'); end")
    con.commit()
    snapshot(env, "2026-11-03.csv", ROW + ROW2, "2026-11-03T00:00:00+00:00")
    with pytest.raises(sqlite3.IntegrityError, match="atomic failure"):
        imported(env)
    assert database_rows(con) == before


@pytest.fixture
def job_env(env, monkeypatch):
    lock_events = []
    @contextmanager
    def fake_lock(settings):
        assert lock_events and lock_events[-1] == "source_enter"
        lock_events.append("write_enter")
        try:
            yield
        finally:
            lock_events.append("write_exit")
    @contextmanager
    def fake_source_lock(settings, source):
        assert source == "football_data"
        lock_events.append("source_enter")
        try:
            yield
        finally:
            lock_events.append("source_exit")
    from src.jobs import refresh_football_data as job
    from src.collect import football_data as collector
    class FixedDate(date):
        @classmethod
        def today(cls):
            return cls(2026, 11, 2)
    monkeypatch.setattr(job, "date", FixedDate)
    monkeypatch.setattr(collector, "date", FixedDate)
    monkeypatch.setattr(job, "data_write_lock", fake_lock)
    monkeypatch.setattr(job, "source_refresh_lock", fake_source_lock, raising=False)
    actual_import = job.import_raw
    def locked_import(*args, **kwargs):
        assert lock_events[-1] == "write_enter"
        return actual_import(*args, **kwargs)
    monkeypatch.setattr(job, "import_raw", locked_import)
    pages = {"E0": (200, (HEADER + ROW).encode())}
    requests_seen = []
    class Session:
        headers = {}
        def get(self, url, timeout):
            assert lock_events[-1] == "source_enter", "网络阶段不能占用数据库共享写锁"
            requests_seen.append(url)
            code, content = pages[url.rsplit("/", 1)[-1].removesuffix(".csv")]
            return SimpleNamespace(status_code=code, content=content)
        def close(self):
            pass
    monkeypatch.setattr(requests, "Session", Session)
    return job, env, pages, requests_seen, lock_events


def test_refresh_saved_and_unchanged_both_import_with_truthful_manifest(job_env):
    job, (settings, con, root), _, requests_seen, lock_events = job_env
    first = job.run_refresh(settings)
    source = source_rows(con)
    second = job.run_refresh(settings)
    assert len(requests_seen) == 2 and all("/2627/E0.csv" in url for url in requests_seen)
    assert source_rows(con) == source
    assert first["阶段"] == second["阶段"] == "完成"
    assert second["leagues"]["E0"]["status"] == "unchanged"
    manifest = json.loads((root / "jobs/football_data_sources.json").read_text(encoding="utf-8"))
    assert manifest["schema"] == 1 and manifest["leagues"] == second["leagues"]
    entry = manifest["leagues"]["E0"]
    assert entry["last_import_success_utc"] and entry["sha256"] and entry["match_count"] == 1
    assert entry["last_changed_utc"] == first["leagues"]["E0"]["last_changed_utc"]
    assert read_state("refresh_football_data", root / "jobs")["成功"] is True
    assert lock_events == ["source_enter", "write_enter", "write_exit", "source_exit"] * 2


@pytest.mark.parametrize("status,content", [(403, b"denied"), (404, b"missing"), (200, b"<html>blocked</html>")])
def test_required_fetch_failure_does_not_import_or_replace_manifest(job_env, status, content):
    job, (settings, con, root), pages, _, _ = job_env
    job.run_refresh(settings)
    before = database_rows(con)
    manifest = (root / "jobs/football_data_sources.json").read_bytes()
    pages["E0"] = status, content
    with pytest.raises(RuntimeError):
        job.run_refresh(settings)
    assert database_rows(con) == before
    assert (root / "jobs/football_data_sources.json").read_bytes() == manifest
    assert read_state("refresh_football_data", root / "jobs")["成功"] is False


def test_unchanged_download_retries_import_after_previous_failure(job_env, monkeypatch):
    job, (settings, con, root), _, _, _ = job_env
    actual = job.import_raw
    monkeypatch.setattr(job, "import_raw", lambda *a, **kw: (_ for _ in ()).throw(RuntimeError("injected import failure")))
    with pytest.raises(RuntimeError, match="injected"):
        job.run_refresh(settings)
    assert not (root / "jobs/football_data_sources.json").exists()
    monkeypatch.setattr(job, "import_raw", actual)
    result = job.run_refresh(settings)
    assert result["leagues"]["E0"]["status"] == "unchanged"
    assert len(source_rows(con)) == 1 and read_state("refresh_football_data", root / "jobs")["成功"] is True


def test_manifest_replace_failure_is_failed_job_and_old_manifest_survives(job_env, monkeypatch):
    job, (settings, _, root), _, _, _ = job_env
    job.run_refresh(settings)
    path = root / "jobs/football_data_sources.json"
    old = path.read_bytes()
    actual = job.os.replace
    def failing(source, target):
        if Path(target) == path:
            raise OSError("manifest unavailable")
        return actual(source, target)
    monkeypatch.setattr(job.os, "replace", failing)
    with pytest.raises(OSError, match="manifest unavailable"):
        job.run_refresh(settings)
    assert path.read_bytes() == old
    assert read_state("refresh_football_data", root / "jobs")["成功"] is False


@pytest.mark.parametrize("leagues", [[], ["E0", "E0"], ["BAD"], "E0"])
def test_invalid_required_leagues_rejected_before_http(job_env, leagues):
    job, (settings, _, root), _, requests_seen, _ = job_env
    settings["football_data"]["refresh_leagues"] = leagues
    with pytest.raises(ValueError, match="联赛"):
        job.run_refresh(settings)
    assert requests_seen == []
    assert read_state("refresh_football_data", root / "jobs")["成功"] is False


def test_refresh_downloads_only_the_ten_registered_current_season_files(job_env):
    job, (settings, con, root), pages, requests_seen, _ = job_env
    settings["leagues"] = copy.deepcopy(load_settings()["leagues"])
    pairs = {"E0": ("Arsenal", "Chelsea"), "SP1": ("Barcelona", "Real Madrid"),
             "D1": ("Bayern Munich", "Dortmund"), "I1": ("Juventus", "Inter"),
             "F1": ("Paris SG", "Lyon"), "E1": ("Blackburn", "Derby"),
             "SP2": ("Cadiz", "Zaragoza"), "D2": ("Hannover", "Karlsruhe"),
             "I2": ("Bari", "Palermo"), "F2": ("Amiens", "Caen")}
    settings["football_data"].pop("refresh_leagues")  # 缺省才使用已登记全部联赛。
    for code, (home, away) in pairs.items():
        row = ROW.replace("E0", code).replace("Arsenal", home).replace("Chelsea", away)
        pages[code] = 200, (HEADER + row).encode()
    result = job.run_refresh(settings)
    assert requests_seen == [f"https://football-data.co.uk/mmz4281/2627/{code}.csv" for code in pairs]
    assert set(result["leagues"]) == set(pairs)
    assert con.execute("select count(*) from match_sources where source='football-data'").fetchone()[0] == 10
    assert len(json.loads((root / "jobs/football_data_sources.json").read_text(encoding="utf-8"))["leagues"]) == 10


def test_later_required_league_failure_preserves_pre_refresh_database_and_manifest(job_env):
    job, (settings, con, root), pages, requests_seen, lock_events = job_env
    job.run_refresh(settings)
    before = database_rows(con)
    manifest = (root / "jobs/football_data_sources.json").read_bytes()
    settings["leagues"]["top"]["SP1"] = "西甲"
    settings["football_data"]["refresh_leagues"] = ["E0", "SP1"]
    pages["E0"] = 200, (HEADER + ROW + ROW2).encode()
    pages["SP1"] = 403, b"denied"
    with pytest.raises(RuntimeError, match="SP1"):
        job.run_refresh(settings)
    assert len(requests_seen) == 3
    assert database_rows(con) == before
    assert (root / "jobs/football_data_sources.json").read_bytes() == manifest
    assert con.execute("select count(*) from fetch_log where status='saved'").fetchone()[0] == 2
    assert lock_events.count("write_enter") == 1  # 第二轮在任何入库开始前已停止。


def test_duplicate_start_cannot_overwrite_active_heartbeat(job_env, monkeypatch):
    job, (settings, _, root), _, requests_seen, _ = job_env
    job.run_refresh(settings)
    path = root / "jobs/refresh_football_data.json"
    old_heartbeat = path.read_bytes()
    @contextmanager
    def busy(*args):
        raise TimeoutError("same source still running")
        yield
    monkeypatch.setattr(job, "source_refresh_lock", busy)
    with pytest.raises(TimeoutError, match="same source"):
        job.run_refresh(settings)
    assert path.read_bytes() == old_heartbeat
    assert len(requests_seen) == 1
