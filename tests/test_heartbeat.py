"""心跳文件（src/jobs/heartbeat.py）的检查。"""
import pytest

from src.jobs.heartbeat import job_run, read_all, read_state


def test_success_writes_state(tmp_path):
    with job_run("demo", folder=tmp_path) as run:
        # 运行中：文件已经在，"结束" 是空的
        mid = read_state("demo", tmp_path)
        assert mid["结束"] is None and mid["成功"] is None
        run.summary["新存"] = 3
    s = read_state("demo", tmp_path)
    assert s["成功"] is True and s["摘要"] == {"新存": 3}
    assert s["上次成功"] == s["结束"] and s["错误"] is None


def test_failure_keeps_last_success(tmp_path):
    with job_run("demo", folder=tmp_path):
        pass
    ok_time = read_state("demo", tmp_path)["上次成功"]
    with pytest.raises(ValueError):
        with job_run("demo", folder=tmp_path):
            raise ValueError("坏了")
    s = read_state("demo", tmp_path)
    assert s["成功"] is False and "坏了" in s["错误"]
    assert s["上次成功"] == ok_time        # 失败时保留上一次成功的时间


def test_read_all_skips_broken(tmp_path):
    with job_run("a", folder=tmp_path):
        pass
    (tmp_path / "b.json").write_text("{坏的", encoding="utf-8")
    assert set(read_all(tmp_path)) == {"a"}
    assert read_state("never", tmp_path) is None
