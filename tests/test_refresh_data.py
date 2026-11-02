"""每小时更新入口：采集成功才导入，异常必须留失败心跳。全部临时数据、禁止联网。"""
from pathlib import Path
from types import SimpleNamespace

import pytest
import requests

from src.jobs.heartbeat import read_state


@pytest.fixture(autouse=True)
def no_network(monkeypatch):
    def forbidden(*args, **kwargs):
        raise AssertionError("测试不能联网")

    monkeypatch.setattr(requests.sessions.Session, "request", forbidden)


@pytest.fixture
def env(tmp_path, monkeypatch):
    from src.jobs import refresh_data as job

    settings = {
        "paths": {"database": str(tmp_path / "football.db"), "raw": str(tmp_path / "raw"),
                  "reports": str(tmp_path / "reports")},
        "jobs": {"state_dir": str(tmp_path / "jobs")},
        "dongqiudi": {"league_ids": {"E0": 8, "SP1": 7}},
    }
    calls = []
    control = {"collect_error": None, "import_error": None}

    class FakeCollector:
        def __init__(self, con, raw, cfg):
            self.con, self.raw, self.cfg = con, raw, cfg
            self.counts = {}
            self.session = SimpleNamespace(close=lambda: calls.append("close_session"))

        def upcoming(self, leagues):
            assert leagues == ["E0", "SP1"]
            assert self.raw == Path(settings["paths"]["raw"])
            calls.append("collect")
            self.counts = {"saved": 2, "unchanged": 1}
            if control["collect_error"]:
                self.counts["failed"] = 1
                raise control["collect_error"]

    def fake_import(con, raw, reports, s, *, teams_yaml=None):
        assert calls == ["collect"]
        assert s is settings and teams_yaml is None
        assert raw == Path(settings["paths"]["raw"])
        assert reports == Path(settings["paths"]["reports"])
        calls.append("import")
        if control["import_error"]:
            raise control["import_error"]
        con.execute("create table imported (n integer)")
        con.execute("insert into imported values (2)")
        con.commit()
        return {"counts": {"新增比赛": 2}}

    monkeypatch.setattr(job, "RefreshCollector", FakeCollector)
    monkeypatch.setattr(job, "import_raw", fake_import)
    return job, settings, calls, control, tmp_path


def test_collect_then_import_and_success_heartbeat(env):
    job, settings, calls, _, tmp = env
    summary = job.run_refresh(settings)
    assert calls == ["collect", "import", "close_session"]
    state = read_state("refresh_data", tmp / "jobs")
    assert state["成功"] is True and state["错误"] is None
    assert state["摘要"] == summary
    assert summary["阶段"] == "完成"
    assert summary["采集请求"] == {"saved": 2, "unchanged": 1}
    assert summary["写入"] == {"新增比赛": 2}


def test_collection_failure_does_not_import_and_is_not_swallowed(env):
    job, settings, calls, control, tmp = env
    control["collect_error"] = RuntimeError("赛程响应无效")
    with pytest.raises(RuntimeError, match="赛程响应无效"):
        job.run_refresh(settings)
    assert calls == ["collect", "close_session"]
    state = read_state("refresh_data", tmp / "jobs")
    assert state["成功"] is False and "赛程响应无效" in state["错误"]
    assert state["摘要"]["阶段"] == "采集"
    assert state["摘要"]["采集请求"]["failed"] == 1


def test_import_failure_preserves_previous_success_and_closes_resources(env):
    job, settings, calls, control, tmp = env
    job.run_refresh(settings)
    previous = read_state("refresh_data", tmp / "jobs")["上次成功"]
    calls.clear()
    control["import_error"] = ValueError("赔率导入失败")
    with pytest.raises(ValueError, match="赔率导入失败"):
        job.run_refresh(settings)
    assert calls == ["collect", "import", "close_session"]
    state = read_state("refresh_data", tmp / "jobs")
    assert state["成功"] is False and "赔率导入失败" in state["错误"]
    assert state["上次成功"] == previous
    assert state["摘要"]["阶段"] == "导入"


@pytest.mark.parametrize("status,data_type", [
    ("failed", "odds_live"), ("suspicious", "odds_live"), ("not_found", "fixtures"),
])
def test_bad_fetch_cannot_look_like_a_successful_refresh(monkeypatch, tmp_path, status, data_type):
    """原采集器会正常返回 failed；更新入口必须把它变成失败，不能据此导入旧数据。"""
    from src.collect.dongqiudi import Collector
    from src.jobs.refresh_data import RefreshCollector

    monkeypatch.setattr(Collector, "fetch", lambda *args, **kwargs: (status, None))
    collector = object.__new__(RefreshCollector)
    with pytest.raises(RuntimeError, match=status):
        collector.fetch("https://example.invalid/fixture", data_type, tmp_path / "raw.json.gz", None, False)


@pytest.mark.parametrize("status,data_type,content", [
    ("saved", "fixtures", b"data"), ("unchanged", "odds_live", b"data"),
    ("not_found", "odds_final", None), ("skipped", "match_detail", None),
])
def test_unchanged_or_optional_missing_data_does_not_break_refresh(monkeypatch, tmp_path, status, data_type, content):
    """赔率没变化仍是成功；历史单场赔率或详情源本来没有，也不该永远卡住每小时入库。"""
    from src.collect.dongqiudi import Collector
    from src.jobs.refresh_data import RefreshCollector

    monkeypatch.setattr(Collector, "fetch", lambda *args, **kwargs: (status, content))
    collector = object.__new__(RefreshCollector)
    assert collector.fetch("https://example.invalid/fixture", data_type, tmp_path / "raw.json.gz", None, False) == (status, content)
