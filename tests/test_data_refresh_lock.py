"""真实进程验证三个来源共享写锁；不访问网络或正式数据库。"""
import subprocess
import sys

import pytest


def settings_for(tmp_path):
    return {"paths": {"database": str(tmp_path / "data" / "football.db")},
            "data_refresh": {"lock_timeout_seconds": 0, "lock_poll_seconds": 0.01}}


def child_code(settings):
    return ("from src.jobs.data_lock import data_write_lock\n"
            f"settings = {settings!r}\n"
            "with data_write_lock(settings):\n"
            "    print('acquired', flush=True)\n")


def test_different_processes_cannot_refresh_the_same_database(tmp_path):
    from src.jobs.data_lock import data_write_lock

    settings = settings_for(tmp_path)
    command = [sys.executable, "-X", "utf8", "-c", child_code(settings)]
    with data_write_lock(settings):
        failed = subprocess.run(command, capture_output=True, text=True, timeout=15)
        assert failed.returncode != 0 and "TimeoutError" in failed.stderr
        assert "acquired" not in failed.stdout
    succeeded = subprocess.run(command, capture_output=True, text=True, timeout=15)
    assert succeeded.returncode == 0, succeeded.stderr
    assert succeeded.stdout.strip() == "acquired"


def test_exception_releases_lock_and_different_databases_do_not_block(tmp_path):
    from src.jobs.data_lock import data_write_lock

    settings = settings_for(tmp_path)
    other = settings_for(tmp_path / "other")
    with pytest.raises(ValueError, match="import failed"):
        with data_write_lock(settings), data_write_lock(other):
            raise ValueError("import failed")
    with data_write_lock(settings):
        pass


def test_terminated_owner_does_not_leave_a_permanent_lock(tmp_path):
    from src.jobs.data_lock import data_write_lock

    settings = settings_for(tmp_path)
    code = child_code(settings).replace("    print('acquired', flush=True)",
        "    print('acquired', flush=True)\n    import sys\n    sys.stdin.read()")
    proc = subprocess.Popen([sys.executable, "-X", "utf8", "-c", code], stdin=subprocess.PIPE,
                            stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    try:
        assert proc.stdout.readline().strip() == "acquired"
        with pytest.raises(TimeoutError):
            with data_write_lock(settings):
                pass
    finally:
        proc.terminate()
        proc.communicate(timeout=10)
    with data_write_lock(settings):
        pass


def test_source_job_lock_blocks_duplicate_job_but_not_other_source(tmp_path):
    from src.jobs.data_lock import data_write_lock, source_refresh_lock

    settings = settings_for(tmp_path)
    code = ("from src.jobs.data_lock import source_refresh_lock\n"
            f"with source_refresh_lock({settings!r}, 'cups'):\n    print('acquired')")
    with source_refresh_lock(settings, "cups"):
        with data_write_lock(settings):
            pass
        failed = subprocess.run([sys.executable, "-X", "utf8", "-c", code],
                                capture_output=True, text=True, timeout=15)
        assert failed.returncode != 0 and 'TimeoutError' in failed.stderr
        with source_refresh_lock(settings, "football"):
            pass


@pytest.mark.parametrize("key,value", [("lock_timeout_seconds", -1),
    ("lock_timeout_seconds", float("nan")), ("lock_poll_seconds", 0)])
def test_invalid_lock_settings_are_rejected(tmp_path, key, value):
    from src.jobs.data_lock import data_write_lock

    settings = settings_for(tmp_path)
    settings["data_refresh"][key] = value
    with pytest.raises(ValueError):
        with data_write_lock(settings):
            pass
